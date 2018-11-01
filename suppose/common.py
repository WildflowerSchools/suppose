from functools import wraps
from itertools import zip_longest
import time
from math import ceil, sqrt
from logbook import Logger
import numpy as np
import cv2


log = Logger('common')


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