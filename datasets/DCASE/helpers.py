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
    xyz = None
    if mic == 'ambeo':
        mic_coords = _polar2cart(ambeovr_raw, units='degrees')
        xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords]
    elif mic == 'tetra':
        mic_coords = _polar2cart(tetra_raw, units='degrees')
        xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords] 

    if xyz == None:
        raise ValueError("you must specify a valid microphone: 'ambeo', 'tetra'")
    return xyz

def load_output_format_file(_output_format_file):
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
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([float(_words[3]), float(_words[4]), float(_words[5])])
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
