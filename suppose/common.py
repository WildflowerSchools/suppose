from functools import wraps
import time
from logbook import Logger


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
