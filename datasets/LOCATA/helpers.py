import csv
import math
import numpy as np

ambeovr_raw = {
    # colatitude (deg), azimuth (deg), radius (m)
    "Ch1:FLU": [55, 45, 0.01],
    "Ch2:FRD": [125, -45, 0.01],
    "Ch3:BLD": [125, 135, 0.01],
    "Ch4:BRU": [55, -135, 0.01],
}

tetra_raw = {
    # colatitude (deg), azimuth (deg), radius (m)
    "Ch1:FLU": [55, 45, 0.042],
    "Ch2:FRD": [125, -45, 0.042],
    "Ch3:BLD": [125, 135, 0.042],
    "Ch4:BRU": [55, -135, 0.042],
}

eigenmike_raw = {
    # colatitude, azimuth, radius
    # (degrees, degrees, meters)
    "1": [69, 0, 0.042],
    "2": [90, 32, 0.042],
    "3": [111, 0, 0.042],
    "4": [90, 328, 0.042],
    "5": [32, 0, 0.042],
    "6": [55, 45, 0.042],
    "7": [90, 69, 0.042],
    "8": [125, 45, 0.042],
    "9": [148, 0, 0.042],
    "10": [125, 315, 0.042],
    "11": [90, 291, 0.042],
    "12": [55, 315, 0.042],
    "13": [21, 91, 0.042],
    "14": [58, 90, 0.042],
    "15": [121, 90, 0.042],
    "16": [159, 89, 0.042],
    "17": [69, 180, 0.042],
    "18": [90, 212, 0.042],
    "19": [111, 180, 0.042],
    "20": [90, 148, 0.042],
    "21": [32, 180, 0.042],
    "22": [55, 225, 0.042],
    "23": [90, 249, 0.042],
    "24": [125, 225, 0.042],
    "25": [148, 180, 0.042],
    "26": [125, 135, 0.042],
    "27": [90, 111, 0.042],
    "28": [55, 135, 0.042],
    "29": [21, 269, 0.042],
    "30": [58, 270, 0.042],
    "31": [122, 270, 0.042],
    "32": [159, 271, 0.042],
}

def _deg2rad(coords_dict):
    """
    Take a dictionary with microphone array
    capsules and 3D polar coordinates to
    convert them from degrees to radians
    colatitude, azimuth, and radius (radius
    is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }

def _polar2cart(coords_dict, units=None):
    """
    Take a dictionary with microphone array
    capsules and polar coordinates and convert
    to cartesian
    Parameters:
        units: (str) indicating 'degrees' or 'radians'
    """
    if units == None or units != "degrees" and units != "radians":
        raise ValueError("you must specify units of 'degrees' or 'radians'")
    elif units == "degrees":
        coords_dict = _deg2rad(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }

def get_xyz(mic='ambeo'):
    mic_coords = None
    if mic == 'ambeo':
        mic_coords = _polar2cart(ambeovr_raw, units='degrees')
    elif mic == 'tetra':
        mic_coords = _polar2cart(tetra_raw, units='degrees')
    elif mic == 'eigenmike':
        mic_coords = _polar2cart(eigenmike_raw, units='degrees')

    if mic_coords == None:
        raise ValueError("you must specify a valid microphone: 'ambeo', 'tetra', 'eigenmike'")

    xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords] 

    return xyz

def load_output_format_file(_output_format_file, coord_system="cartesian"):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary with coordinates only, we discard classes info (consider same class so far)
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        _output_dict[_frame_ind].append([float(_words[1]), float(_words[2]), float(_words[3])])
    _fid.close()
    return _output_dict

def convert_format_polar_to_cartesian(in_dict, dist=2): 
    # NOTE: dist=2, for now we will assume all objects are located at 2m away from the speaker
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                ele_rad = tmp_val[1]*np.pi/180.
                azi_rad = tmp_val[0]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = dist * np.cos(azi_rad) * tmp_label
                y = dist * np.sin(azi_rad) * tmp_label
                z = dist * np.sin(ele_rad)

                out_dict[frame_cnt].append([x, y, z])
    return out_dict
