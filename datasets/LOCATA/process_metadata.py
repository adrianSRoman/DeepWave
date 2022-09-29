# Implementation adapted from https://github.com/audiofhrozen/locata_wrapper/blob/master/locata_wrapper/utils/process.py

from argparse import Namespace
import h5py
import glob
import logging
import numpy as np
import pandas as pd
import os
import timeit
import csv

from load_metadata import GetTruth
from load_metadata import LoadData

from matplotlib import pyplot as plt


def ElapsedTime(time_array):
    n_steps = time_array.shape[0]
    elapsed_time = np.zeros([n_steps])
    for i in range(1, n_steps):
        elapsed_time[i] = (time_array[i] - time_array[i - 1]).total_seconds()
    return np.cumsum(elapsed_time)


def ProcessTaskMetadata(this_task, arrays, data_dir, is_dev):
    task_dir = os.path.join(data_dir, 'task{}'.format(this_task))
    # Read all recording IDs available for this task:
    recordings = sorted(glob.glob(os.path.join(task_dir, '*')))
    print("recordings", recordings)
    truth_list = []
    # Parse through all recordings within this task:
    for this_recording in recordings:
        recording_id = int(this_recording.split('recording')[1])
        # Read all recording IDs available for this task:
        array_names = glob.glob(os.path.join(this_recording, '*'))
        for array_dir in array_names:
            this_array = os.path.basename(array_dir)
            if this_array not in arrays:
                continue
            print('Processing task {}, recording {}, array {}.'.format(this_task, recording_id, this_array))
            # Load metadata from csv
            position_array, position_source, required_time = LoadData(
                array_dir, None, None, is_dev)
            print('Processing Complete!')
            # Extract ground truth
            # position_array stores all optitrack measurements.
            # Extract valid measurements only (specified by required_time.valid_flag).
            truth = GetTruth(this_array, position_array, position_source, required_time, recording_id, is_dev)
            truth_list.append(truth)
    return truth_list



def Locata2DecaseFormat(tasks, data_dir, arrays=["eigenmike"], is_dev=True, coord_system="cartesian"):
    FS_POS = 120 # Position labeling done at 120Hz
    for task_id in tasks:
        truth_list = ProcessTaskMetadata(task_id, arrays, data_dir, is_dev)
        for truth in truth_list:
            out_filename = f'./metadata/task{task_id}_recording{truth.recording_id}.csv'
            with open(out_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                print("Processing {}".format(out_filename))
                for iframe in range(0, truth.frames, FS_POS//10): # sample every 100msec
                    for speaker in truth.source:
                        csv_row = [iframe//12]
                        if coord_system == "cartesian":
                            csv_row.extend(truth.source[speaker].cart_pos[iframe])
                        elif coord_system == "polar":
                            csv_row.extend(truth.source[speaker].polar_pos[iframe])
                        csv_writer.writerow(csv_row)


Locata2DecaseFormat(["1", "2", "3", "4"], "/Volumes/T7/LOCATA-dev", arrays=["eigenmike"], is_dev=True, coord_system="polar")