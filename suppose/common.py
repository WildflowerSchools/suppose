from functools import wraps
from itertools import zip_longest
import time
from math import ceil, sqrt
from logbook import Logger
import numpy as np
import cv2
from tf_pose.common import CocoPairsRender, CocoColors, CocoPart


log = Logger('common')


TIMESTAMP_REGEX = r"^.*\/capucine\/videos\/(?P<camera_id>\w{8}-\w{4}-\w{4}-\w{4}-\w{12})\/(?P<date>(?P<year>\d{4})\/(?P<month>\d{2})\/(?P<day>\d{2}))\/(?P<hour>\d{2})\/(?P<filename>(?P<minute>\d{2})-(?P<second>\d{2}).mp4)$"


def timing(f):
    @wraps(f)
    def _timing(*args, **kwargs):
        t0 = time.time()
        try:
            return f(*args, **kwargs)
        finally:
            t1 = time.time()
            dt = t1 - t0
            argnames = f.__code__.co_varnames[:f.__code__.co_argcount]
            fname = f.__name__
            named_positional_args = list(zip(argnames, args[:len(argnames)]))
            extra_args = [("args", list(args[len(argnames):]))]
            keyword_args = [("kwargs", kwargs)]
            arg_info = named_positional_args + extra_args + keyword_args
            msg = "{}({}) took {:.3f} seconds".format(fname,
                                                      ', '.join('{}={}'.format(entry[0], entry[1])[:20] for entry in arg_info),
                                                      dt)
            log.info(msg)
    return _timing


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def stack_images(images, canvas_width=480*2, force_n=None):
    """
    Stack images into a grid with max_width.

    Automatically resize images and place in grid.

    Will force grid to have equal number of columns and rows (or 1 less row than column).
    """
    if force_n is None:
        n = len(images)
    else:
        n = force_n

    columns = int(ceil(sqrt(n)))
    rows = ceil(n/columns)

    height, width, _ = images[0].shape

    # assume all images same size for now, easy to adapt to mutiple sizes later and pad if needed
    scale = canvas_width/(width * columns)

    # create canvas
    canvas_height = int((height*scale)*rows)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    dsize_width = int(width*scale)
    dsize_height = int(height*scale)
    dsize = (dsize_width, dsize_height)

    r = 0
    c = 0
    for idx, im in enumerate(images):
        im_resized = cv2.resize(im, dsize)
        x = c*dsize[0]
        y = r*dsize[1]
        canvas[y:y+dsize_height, x:x+dsize_width] = im_resized
        c += 1
        if c >= columns:
            c = 0
            r += 1
    return canvas

NUM_BODY_PARTS = 18

KEYPOINT_RMSE_WEIGHTS = np.array([
    1,      # Nose = 0
    1,      # Neck = 1
    1,      # RShoulder = 2
    0.5,      # RElbow = 3
    0.5,      # RWrist = 4
    1,      # LShoulder = 5
    0.5,      # LElbow = 6
    0.5,      # LWrist = 7
    0.5,      # RHip = 8
    0.1,      # RKnee = 9
    0.1,      # RAnkle = 10
    0.5,      # LHip = 11
    0.1,      # LKnee = 12
    0.1,      # LAnkle = 13
    1,      # REye = 14
    1,      # LEye = 15
    1,      # REar = 16
    1,      # LEar = 17
], dtype=np.float64)

xKEYPOINT_RMSE_WEIGHTS = np.array([
    1,      # Nose = 0
    1,      # Neck = 1
    1,      # RShoulder = 2
    0.0,      # RElbow = 3
    0.0,      # RWrist = 4
    1,      # LShoulder = 5
    0.0,      # LElbow = 6
    0.0,      # LWrist = 7
    0.0,      # RHip = 8
    0.0,      # RKnee = 9
    0.0,      # RAnkle = 10
    0.0,      # LHip = 11
    0.0,      # LKnee = 12
    0.0,      # LAnkle = 13
    1,      # REye = 14
    1,      # LEye = 15
    1,      # REar = 16
    1,      # LEar = 17
], dtype=np.float64)

xKEYPOINT_RMSE_WEIGHTS = np.array([
    0,      # Nose = 0
    1,      # Neck = 1
    1,      # RShoulder = 2
    0.0,      # RElbow = 3
    0.0,      # RWrist = 4
    1,      # LShoulder = 5
    0.0,      # LElbow = 6
    0.0,      # LWrist = 7
    0.0,      # RHip = 8
    0.0,      # RKnee = 9
    0.0,      # RAnkle = 10
    0.0,      # LHip = 11
    0.0,      # LKnee = 12
    0.0,      # LAnkle = 13
    0,      # REye = 14
    0,      # LEye = 15
    0,      # REar = 16
    0,      # LEar = 17
], dtype=np.float64)

# DEBUG
#KEYPOINT_RMSE_WEIGHTS = np.ones_like(KEYPOINT_RMSE_WEIGHTS).astype(np.float64)

#KEYPOINT_RMSE_WEIGHTS = KEYPOINT_RMSE_WEIGHTS / np.sum(KEYPOINT_RMSE_WEIGHTS)


def rmse(p1, p2, weights=None):
    diff = p1 - p2
    #print(repr(diff))
    #print()
    #print(repr(weights))
    if weights is not None:
        diff = diff * weights[:, None]
    return np.linalg.norm(diff)/np.sqrt(p1.shape[0])


LIMB_COLORS = dict(zip(CocoPairsRender, CocoColors))

NECK_INDEX = CocoPart.Neck.value
SHOULDER_INDICES = [CocoPart.RShoulder.value, CocoPart.LShoulder.value]
# (0, 1, 2, 5, 8, 11, 14 , 15, 16, 17)
HEAD_AND_TORSO_INDICES = [
    CocoPart.Nose.value,
    CocoPart.Neck.value,
    CocoPart.RShoulder.value,
    CocoPart.LShoulder.value,
    CocoPart.RHip.value,
    CocoPart.LHip.value,
    CocoPart.REye.value,
    CocoPart.LEye.value,
    CocoPart.REar.value,
    CocoPart.LEar.value,
]

LOWER_BODY_INDICES = [
    CocoPart.RHip.value,
    CocoPart.RKnee.value,
    CocoPart.RAnkle.value,
    CocoPart.LHip.value,
    CocoPart.LKnee.value,
    CocoPart.LAnkle.value,
]

UPPER_BODY_INDICES = [
    CocoPart.Nose.value,
    CocoPart.Neck.value,
    CocoPart.RShoulder.value,
    CocoPart.RElbow.value,
    CocoPart.RWrist.value,
    CocoPart.LShoulder.value,
    CocoPart.LElbow.value,
    CocoPart.LWrist.value,
    CocoPart.REye.value,
    CocoPart.LEye.value,
    CocoPart.REar.value,
    CocoPart.LEar.value,
]