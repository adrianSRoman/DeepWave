# Implementation adapted from https://github.com/audiofhrozen/locata_wrapper/blob/master/locata_wrapper/utils/process.py

from argparse import Namespace
import h5py
import glob
import logging
import numpy as np
import pandas as pd
import os
import timeit

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
            truth = GetTruth(this_array, position_array, position_source, required_time, is_dev)

            # NOTE: we will get one truth per file, and so one .csv per truth!



def Locata2DecaseFormat(tasks, metadata_path):
    # TODO:
    # - Get truths list (for all recordings under a task)
    # - for each truth
    #   - for each speaker, get the polar positions
    #       -  dump data into csv format

    # Pseudo code:
    # for truth in truth_list:
    #     for task in tasks:
    #         for speaker in truth.source:
    #             print(truth.source[speaker].polar_pos)
    #             print("#####")


ProcessTaskMetadata(2, ["eigenmike"], "/Volumes/T7/LOCATA-dev", True)