import json
import numpy as np
from logbook import Logger
import cv2



log = Logger('camera')

def get_projection_matrix(calibration):
    """
    Calculates projection matrix from camera calibration.

    :param calibration: camera calibration dictionary
    :return: projection matrix
    """
    R = cv2.Rodrigues(calibration['rotationVector'])[0]
    t = calibration['translationVector']
    Rt = np.concatenate([R, t], axis=1)
    C = calibration['cameraMatrix']
    P = np.matmul(C, Rt)
    return P


def load_calibration(file):
    with open(file) as fd:
        log.info("Loading camera calibration: {}".format(file))
        data = json.load(fd)
        calib = {}
        for k, v in data.items():
            calib[k] = np.asarray(v)
        calib['projection'] = get_projection_matrix(calib)
        return calib
