import json
import numpy as np
from logbook import Logger


log = Logger('camera')


def load_calibration(file):
    with open(file) as fd:
        log.info("Loading camera calibration: {}".format(file))
        data = json.load(fd)
        calib = {}
        for k, v in data.items():
            calib[k] = np.asarray(v)
        return calib
