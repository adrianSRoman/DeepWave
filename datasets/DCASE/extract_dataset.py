# ############################################################################
# extract_dataset.py
# ==================
# Adapted from extract_dataset.py by Sepand KASHANI
# ############################################################################

r"""
Form per-frequency APGD datasets from FRIDA .wav files.

Datasets stored under ./dataset

All datasets can then be merged using:

cd ./dataset
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    for start in cold warm; do
        python3 ../../scripts/merge_dataset.py --out="D_freq${idx_freq}_${start}.npz" "D_*_freq${idx_freq}_${start}.npz";
    done;
done
"""

import argparse
import logging
import pathlib
import sys
import os

import numpy as np
import scipy.constants as constants
import scipy.io.wavfile as wavfile
import scipy.linalg as linalg
import skimage.util as skutil

from helpers import *

import deepwave.apgd as apgd
import deepwave.nn as nn
import deepwave.tools.data_gen.source as source
import deepwave.tools.data_gen.statistics as statistics
import imot_tools.math.sphere.grid as grid
import imot_tools.phased_array as phased_array


def parse_args():
    parser = argparse.ArgumentParser(description='Extract multi-spectral datasets from DCASE2022 time-series',
                                     epilog="""
    Example
    -------
    python3 extract_dataset.py --data=1-3-5.wav --metatada=/path/to/metadata_csv_dir/
                """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--data',
                        help='.wav audio file containing Pyramic measurements.',
                        required=True,
                        type=str)
    parser.add_argument('--metadata',
                        help='Directory containing .csv for labeled data.',
                        required=True,
                        type=str)
    parser.add_argument('--warm_start',
                        help='Use warm-starting for APGD.',
                        action='store_true')

    args = parser.parse_args()
    args.data = pathlib.Path(args.data).expanduser().absolute()
    return vars(args)

def get_track_data(file_path, gt_path):
    """
    Parameters
    ----------
    file_name : :py:class:`~pathlib.Path`
        .wav file to process
    src_map : dict
        (src_name[str], src_xyz[np.ndarray]) map.

    Returns
    -------
    rate : int
        Sample rate [Hz].
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) samples. (float64)
    sky_model : :py:class:`~deepwave.tools.data_gen.source.SkyModel`
        (N_src,) ground truth for .wav file
    """
    if not (isinstance(file_path, pathlib.Path) and
            file_path.exists() and
            file_path.is_file() and
            file_path.suffix == '.wav'):
        raise ValueError('Parameter[file_name] must point to a .wav file.')

    rate, data = wavfile.read(file_path)
    data = data.astype(np.float64) / np.iinfo(data.dtype).max

    if gt_path:
        gt_dict = load_output_format_file(gt_path)
    else:
        nsamps = int(10 * (data.shape[0] / rate))
        gt_dict = dict()
        for i in range(nsamps):
            gt_dict[i] = [[0, 0]]

    gt_dict_cart = convert_format_polar_to_cartesian(gt_dict)
    max_frame = max(gt_dict_cart.keys()) # get total number of frames
    sky_model_list = []
    for k in range(max_frame):
        if k not in gt_dict_cart:
            src_xyz = [[1e-6]*3]
        else:
            src_xyz = gt_dict_cart[k]
        s_l = len(src_xyz)
        src_map = {k: v for (k, v) in zip([i for i in range(s_l)], np.array(src_xyz))} 
        N_src = s_l
        s_xyz = np.stack([src_map[k] for k in range(s_l)], axis=1)
        s_I = np.ones((N_src,))
        sky_model = source.SkyModel(s_xyz, s_I)
        sky_model_list.append(sky_model)

    return rate, data, sky_model_list

def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    """
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    """
    S_sti = (statistics.TimeSeries(data, rate)
             .extract_visibilities(T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti) + 1
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
         .squeeze(axis=(1, 2))
         .sum(axis=1))
    return S


def process_track(file_path, gt_path, mic='ambeo'):
    """
    Parameters
    ----------
    rate : int
        Sample rate [Hz].
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) samples. (float64)
    sky_model : :py:class:`~deepwave.tools.data_gen.source.SkyModel`
        (N_src,) ground truth for synthesized data stream.

    Returns
    -------
    D : dict(freq_idx -> :py:class:`~deepwave.nn.DataSet`)
    """
    xyz = get_xyz(mic) # get xyz coordinates of mic channels
    dev_xyz =  np.array(xyz).T
    N_antenna = dev_xyz.shape[1]
    # Ground truth labels
    track_filename = str(file_path).split("/")[-1]
    gt_filename = track_filename.split(".")[0] + ".csv"
    rate, data, sky_model_list = get_track_data(file_path, gt_path+gt_filename)
    freq, bw = (skutil  # Center frequencies to form images
                .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
                .mean(axis=-1)), 50.0  # [Hz]
    T_sti = 12.5e-3
    T_stationarity = 8 * T_sti  # Choose to have frame_rate = 10.
    N_freq = len(freq)

    wl_min = 345.08736 / (freq.max() + 500)
    sh_order = phased_array.nyquist_rate(dev_xyz, wl_min) # Maximum order of complex plane waves that can be imaged by the instrument.
    R = grid.fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(30))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    N_px = R.shape[1]
    sampler = nn.Sampler(N_antenna, N_px)
    for idx_freq in range(N_freq):
        wl = 345.08736 / freq[idx_freq]
        A = phased_array.steering_operator(dev_xyz, R, wl)

        S = form_visibility(data, rate, freq[idx_freq], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]
        apgd_gamma = 0.5
        apgd_lambda_ = np.zeros((N_sample,))
        apgd_N_iter = np.zeros((N_sample,), dtype=int)
        apgd_tts = np.zeros((N_sample,))
        apgd_data = np.zeros((N_sample, sampler._N_cell))
        I_prev = np.zeros((N_px,))
        for idx_s in range(N_sample):
            logging.info(f'Processing freq_idx={idx_freq + 1:02d}/{N_freq:02d}, '
                         f'sample_idx={idx_s + 1:03d}/{N_sample:03d}')

            # Normalize visibilities
            S_D, S_V = linalg.eigh(S[idx_s])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T

            I_apgd = apgd.solve(S_norm, A, gamma=apgd_gamma, x0=I_prev.copy(), verbosity='NONE')
            apgd_data[idx_s] = sampler.encode(S=S_norm,
                                              I=I_apgd['backtrace'][-1],
                                              I_prev=I_apgd['backtrace'][0])
            apgd_lambda_[idx_s] = I_apgd['lambda_']
            apgd_N_iter[idx_s] = I_apgd['niter']
            apgd_tts[idx_s] = I_apgd['time']

        D_ = nn.DataSet(data=apgd_data,
                        XYZ=dev_xyz,
                        R=R,
                        wl=wl,
                        ground_truth=sky_model_list[:N_sample],
                        lambda_=apgd_lambda_,
                        gamma=apgd_gamma,
                        N_iter=apgd_N_iter,
                        tts=apgd_tts)
        d_name = f'./dataset/{track_filename.split(".")[0]}_freq{idx_freq}_singletrack.npz' 
        logging.info(f'Storing dataset to {d_name}')
        D_.to_file(d_name)


if __name__ == '__main__':
    args = parse_args()
    warm_suffix = 'warm' if args['warm_start'] else 'cold'
    log_fname = (pathlib.Path(f'./dataset/D_{args["data"].stem}_{warm_suffix}.log')
                 .expanduser().absolute())
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s | %(message)s',
                        filename=log_fname,
                        filemode='w')
    # Setup logging to stdout.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(console_formatter)
    logging.getLogger(__name__).addHandler(console)
    logging.info('START')

    process_track(args['data'], gt_path=args['metadata']) #"/Volumes/T7/DCASE2022-Task3/DCASE2022_SELD_synth_data/metadata/")
    logging.info('DONE')
