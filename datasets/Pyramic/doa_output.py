# ############################################################################
# color_plot.py
# =============
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

"""
Plot multi-frequency datasets in RGB colors.
"""

import argparse
import collections.abc as abc
import os
import csv
import pathlib


import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pkg_resources as pkg
import tqdm

import deepwave.nn as nn
import deepwave.nn.crnn as crnn
import deepwave.spectral as spectral
import deepwave.tools.math.func as func
import deepwave.tools.math.graph as graph
import deepwave.tools.math.linalg as pylinalg
import imot_tools.math.sphere.transform as transform
import imot_tools.phased_array as phased_array

import mpl_toolkits.basemap as basemap
from sklearn.cluster import KMeans
from itertools import combinations

def determine_similar_location(azi_rad1, lon_rad1, azi_rad2, lon_rad2, thresh_unify=15.5):
    return distance_between_spherical_coordinates_rad(azi_rad1, lon_rad1, azi_rad2, lon_rad2) < thresh_unify

def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    The function implemenets the angukla distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Parameters :
    ---------------
    :param az1: azimuth angle 1
    :param az2: azimuth angle 2
    :param ele1: elevation angle 1 
    :param ele2: elevation angle 2
    Return:
    ----------------
    :return dist: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist

def sort_coord_pairs(gt_coords_list, pred_coords_list, threshold=30):
    sorted_idxs = []
    for gt_coord in gt_coords_list:
        abs_delta = [np.abs(pred_coord[0]-gt_coord[0]) + np.abs(pred_coord[1]-gt_coord[1]) for pred_coord in pred_coords_list]
        min_idx = np.argmin(np.array(abs_delta))
        if abs_delta[min_idx] < threshold and min_idx not in sorted_idxs:
            sorted_idxs.append(min_idx)
        else:
            sorted_idxs.append(None)
    pred_coords_list = [pred_coords_list[i] if i is not None else None for i in sorted_idxs]
    return pred_coords_list, sorted_idxs

def parse_args():
    parser = argparse.ArgumentParser(description='Produce DAS/RNN true-color plots.',
                                     epilog=r"""
    Example
    -------
    python3 doa_output.py --datasets D_freq[0-8].npz         \
                          --parameters D_freq[0-8]_train.npz
                """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--datasets',
                        help=('Multi-frequency datasets. It is assumed files once sorted go from '
                              'smallest to largest frequency.'),
                        nargs='+',
                        type=str,
                        required=True,)
    parser.add_argument('--parameters',
                        help=('Trained network parameter files. It is assumed files once sorted '
                              'go from smallest to largest frequency.'),
                        nargs='+',
                        type=str,
                        required=True,)
    parser.add_argument('--img_type',
                        help='Type of image to produce.',
                        required=True,
                        type=str,
                        choices=['APGD', 'RNN', 'DAS'])
    parser.add_argument('--img_idx',
                        help=('Image indices to export. Interpreted as Python code. '
                              'If left unspecified, export all images.'),
                        default=None)
    parser.add_argument('--lon_ticks',
                        help='in degrees',
                        default=np.linspace(-180, 180, 5))
    parser.add_argument('--gt_out',
                        help='path to ground truth .csv file',
                        default='./gt_labels.csv')
    parser.add_argument('--pred_out',
                        help='path to predicted doa labels .csv file',
                        default='./pred_labels.csv')                    

    args = parser.parse_args()

    datasets = []
    for f in args.datasets:
        ff = pathlib.Path(f).expanduser().absolute()
        if not ff.exists():
            raise ValueError(f'File "{str(ff)}" does not exist.')
        datasets.append(ff)
    args.datasets = datasets

    parameters = []
    for f in args.parameters:
        ff = pathlib.Path(f).expanduser().absolute()
        if not ff.exists():
            raise ValueError(f'File "{str(ff)}" does not exist.')
        parameters.append(ff)
    args.parameters = parameters

    # Make sure (dataset, parameter) are conformant (light check)
    if not (len(args.datasets) == len(args.parameters)):
        raise ValueError('Parameters[datasets, parameters] have different number of arguments.')
    args.datasets = sorted(args.datasets)
    args.parameters = sorted(args.parameters)

    if args.img_idx is None:
        D = nn.DataSet.from_file(str(args.datasets[0]))
        N_sample = len(D)
        img_idx = np.arange(N_sample)
    else:
        img_idx = np.unique(eval(args.img_idx))
    for f in args.datasets:
        D = nn.DataSet.from_file(str(f))
        N_sample = len(D)
        if not np.all((img_idx >= 0) & (img_idx < N_sample)):
            raise ValueError('Some image indices are out of range.')
    args.img_idx = img_idx

    if isinstance(args.lon_ticks, str):  # user-specified values
        lon_ticks = eval(args.lon_ticks)
    else:
        lon_ticks = np.unique(args.lon_ticks)
    if not ((-180 <= lon_ticks.min()) &
            (lon_ticks.max() <= 180)):
        raise ValueError('Parameter[lon_ticks] is out of range.')
    args.lon_ticks = lon_ticks

    return vars(args)


def get_field(D, P, idx_img, img_type):
    """
    Parameters
    ----------
    D : list(:py:class:`~deepwave.nn.DataSet`)
        (9,) multi-frequency datasets.
    P : list(:py:class:`~deepwave.nn.crnn.Parameter`)
        (9,) multi-frequency trained parameters.
    idx_img : int
        Image index
    img_type : str
        One of ['APGD', 'RNN', 'DAS']

    Returns
    -------
    I : :py:class:`~numpy.ndarray`
        (9, N_px) frequency intensities of specified image.
    """
    I = []
    for idx_freq in range(9):
        Df, Pf = D[idx_freq], P[idx_freq]

        N_antenna = Df.XYZ.shape[1]
        N_px = Df.R.shape[1]
        K = int(Pf['K'])
        parameter = crnn.Parameter(N_antenna, N_px, K)
        sampler = Df.sampler()

        A = phased_array.steering_operator(Df.XYZ, Df.R, Df.wl)
        if img_type == 'APGD':
            _, I_apgd, _ = sampler.decode(Df[idx_img])
            I.append(I_apgd)
        elif img_type == 'RNN':
            Ln, _ = graph.laplacian_exp(Df.R, normalized=True)
            afunc = lambda _: func.retanh(Pf['tanh_lin_limit'], _)
            p_opt = Pf['p_opt'][np.argmin(Pf['v_loss'])]
            S, _, I_prev = sampler.decode(Df[idx_img])
            N_layer = Pf['N_layer']
            rnn_eval = crnn.Evaluator(N_layer, parameter, p_opt, Ln, afunc)
            I_rnn = rnn_eval(S, I_prev)
            I.append(I_rnn)
        elif img_type == 'DAS':
            S, _, _ = sampler.decode(Df[idx_img])
            alpha = 1 / (2 * pylinalg.eighMax(A))
            beta = 2 * Df.lambda_[idx_img] * alpha * (1 - Df.gamma) + 1

            I_das = spectral.DAS(Df.XYZ, S, Df.wl, Df.R) * 2 * alpha / beta
            I.append(I_das)
        else:
            raise ValueError(f'Parameter[img_type] invalid.')

    I = np.stack(I, axis=0)
    return I

def wrapped_rad2deg(lat_r, lon_r):
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].

    Parameters
    ----------
    lat_r : :py:class:`~numpy.ndarray`
    lon_r : :py:class:`~numpy.ndarray`

    Returns
    -------
    lat_d : :py:class:`~numpy.ndarray`
    lon_d : :py:class:`~numpy.ndarray`
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


def get_intensity_coords_deg(I, R, catalog=None, N_max=50):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """
    max_idx = I.argsort()[-N_max:][::-1]
    _, R_el, R_az = transform.cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])
    bm = basemap.Basemap(projection='mill',
                        llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                        llcrnrlon=R_az_min, urcrnrlon=R_az_max)
    R_x, R_y = bm(R_az, R_el)

    sky_lon, sky_lat  = None, None
    centroid_lon, centroid_lat = None, None
    if catalog is not None:
        _, sky_el, sky_az = transform.cart2eq(*catalog.xyz)
        sky_el, sky_az = wrapped_rad2deg(sky_el, sky_az)
        sky_x, sky_y = bm(sky_az, sky_el)
        sky_lon, sky_lat = bm(sky_x, sky_y, inverse=True)

    if len(sky_lon) == 0:
        return centroid_lon, centroid_lat, sky_lon, sky_lat

    # Create Kmeans clusters
    K = 3 
    for _k in range(K, 0, -1):
        x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
        km_res = KMeans(n_clusters=_k).fit(x_y)
        clusters = km_res.cluster_centers_
        centroid_lon, centroid_lat = bm(clusters[:,0], clusters[:,1], inverse=True)

        centroid_lon_rad = centroid_lon * np.pi / 180
        centroid_lat_rad = centroid_lat * np.pi / 180

        all_centroids_pairs = combinations(np.arange(_k), 2)
        centroids_overlap = False
        for _cent_pair in all_centroids_pairs:
            location_overlapping = determine_similar_location(centroid_lon_rad[_cent_pair[0]], centroid_lat_rad[_cent_pair[0]],
                                                              centroid_lon_rad[_cent_pair[1]], centroid_lat_rad[_cent_pair[1]])
            if location_overlapping: # keep looping if overlap between centroids
                centroids_overlap = True
                break
        if not centroids_overlap:   
            break  # done computing K-means centroids

    return centroid_lon, centroid_lat, sky_lon, sky_lat

def regression_label_format_to_output_format(_doa_labels, nb_locations):
    """
    The function converts the doa labels predicted to dcase output format.
    Paremeters:
    -----------------
    :param _doa_labels: DOA labels matrix polar: [nb_frames, 2*nb_locations] or catesian: [nb_frames, 3*nb_locations]
    :param _nb_locations: number of intensity locations to be classified.
    Return:
    ------------------  
    :return: _output_dict: returns a dict containing dcase output format
    """

    _nb_classes = len(self._unique_classes)
    _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
    _azi_labels, _ele_labels = None, None
    _x, _y, _z = None, None, None
    if _is_polar:
        _azi_labels = _doa_labels[:, :_nb_locations]
        _ele_labels = _doa_labels[:, _nb_locations:]

    _output_dict = {}
    for _frame_ind in range(_doa_labels.shape[0]):
        _output_dict[_frame_ind] = []
        for _loc in range(nb_locations):
            if _is_polar:
                _output_dict[_frame_ind].append([_azi_labels[_frame_ind, _loc], _ele_labels[_frame_ind, _loc]])
            # TODO: if _cartesian:
    return _output_dict

def convert_polar_to_cartesian(azi_polar, ele_polar):
    ele_rad = ele_polar*np.pi/180.0
    azi_rad = azi_polar*np.pi/180.0
    tmp_label = np.cos(ele_rad)
    x = np.cos(azi_rad) * tmp_label
    y = np.sin(azi_rad) * tmp_label
    z = np.sin(ele_rad)
    return x, y, z

if __name__ == '__main__':
    args = parse_args()

    D = [nn.DataSet.from_file(str(_)) for _ in args['datasets']]
    P = [np.load(_) for _ in args['parameters']]

    R = D[0].R
    _output_dict = {}
    _gt_dict = {}
    for idx_img in tqdm.tqdm(args['img_idx']):
        _output_dict[idx_img] = []
        _gt_dict[idx_img] = []

        I = get_field(D, P, idx_img, img_type=args['img_type'])
        I_sum = I.sum(axis=0)

        # Filter field to lie in specified interval
        _, R_lat, R_lon = transform.cart2eq(*R)
        _, R_lon_d = wrapped_rad2deg(R_lat, R_lon)
        min_lon, max_lon = args['lon_ticks'].min(), args['lon_ticks'].max()
        mask_lon = (min_lon <= R_lon_d) & (R_lon_d <= max_lon)

        R_field = transform.eq2cart(1, R_lat[mask_lon], R_lon[mask_lon])
        I_sum = I_sum[mask_lon]

        sky_model = D[0].ground_truth[idx_img]
        pred_lon, pred_lat, sky_lon_label, sky_lat_label = get_intensity_coords_deg(I_sum, R_field, catalog=sky_model, N_max=50)

        for _ind_loc in range(len(sky_lon_label)): # store ground truth doa labels
             _gt_dict[idx_img].append([sky_lon_label[_ind_loc], sky_lat_label[_ind_loc]])

        for _ind_loc in range(len(pred_lon)): # store predicted doa labels (1 <= pred_doa <= 3)
            _output_dict[idx_img].append([pred_lon[_ind_loc], pred_lat[_ind_loc]])

    nb_locations = 3
    with open(args['gt_out'], mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx_img in tqdm.tqdm(args['img_idx']):
            for _loc in range(len(_gt_dict[idx_img])):
                row = [idx_img, 0, 0, _gt_dict[idx_img][_loc][0], _gt_dict[idx_img][_loc][1]]
                csv_writer.writerow(row)
    
    with open(args['pred_out'], mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx_img in tqdm.tqdm(args['img_idx']):
            for _loc in range(len(_output_dict[idx_img])): # iterate through number of predicted DOAs
                x, y, z = convert_polar_to_cartesian(_output_dict[idx_img][_loc][0], _output_dict[idx_img][_loc][1])
                row = [idx_img, 0, 0, x, y, z]
                csv_writer.writerow(row)
