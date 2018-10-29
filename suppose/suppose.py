import pickle
import glob
import time
import os
import cv2
from functools import wraps
from tqdm import tqdm
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from logbook import Logger, RotatingFileHandler
import numpy as np
import pandas as pd


log_handler = RotatingFileHandler('suppose.log')
log_handler.push_application()
log = Logger('poser')


MAX_NUM_BODY_PARTS = 18


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
                                                       ', '.join('{}={}'.format(entry[0], entry[1]) for entry in arg_info),
                                                       dt)
            log.info(msg)
    return _timing

def pose_to_array(pose):
    all_humans = []
    for human in pose:
        body_parts = human.body_parts
        parts = []
        for idx in range(MAX_NUM_BODY_PARTS):
            try:
                body_part = body_parts[idx]
                part = (body_part.x, body_part.y, body_part.score)
            except KeyError:
                part = (0, 0, 0)
            parts.append(part)
        all_humans.append(np.array(parts, dtype=np.float32))

    pose_converted = np.array(all_humans, dtype=np.float32)
    return pose_converted


@timing
def tfpose_to_pandas(poses):
    all_poses = [pose_to_array(pose) for pose in poses]
    df = pd.DataFrame(all_poses, columns=["poses"])
    return df


@timing
def extract_poses(video, e, model_name, write_output=True):
    log.info("Processing {}".format(video))
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    poses = []
    time0 = time.time()
    if cap.isOpened() is False:
        raise IOError("Error reading file {}".format(video))

    pbar = tqdm(total=video_length)
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        count += 1
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        poses.append(humans)
        pbar.update(1)
    time1 = time.time()
    log.info("{} fps".format((count / (time1 - time0))))
    log.info("{} s".format(time1 - time0))

    if write_output:
        output_filename = '{}__poses-{}.pickle'.format(video, model_name)
        log.info("Writing output to {}".format(output_filename))
        with open(output_filename, 'wb') as fd:
            pickle.dump(poses, fd)

        df_poses_pickle_filename = '{}__poses_df-{}.pickle.xz'.format(video, model_name)
        df_poses_json_filename = '{}__poses_df-{}.json.xz'.format(video, model_name)
        df_poses = tfpose_to_pandas(poses)
        df_poses.to_pickle(df_poses_pickle_filename, compression="xz")
        df_poses.to_json(df_poses_json_filename, compression="xz")

    cap.release()
    pbar.close()


def _run(videos, model, resolution, write_output, display_progress):
    log.info("Starting Pose Extractor")
    log.info("videos: {}".format(videos))
    log.info("model: {}".format(model))
    log.info("resolution: {}".format(resolution))
    log.debug("Initializing {} : {}".format(model, get_graph_path(model)))

    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    files = glob.glob(videos)
    files.sort()
    length = len(files)
    log.info("Processing {} files".format(length))
    disable_tqdm = not display_progress
    with tqdm(files, disable=disable_tqdm) as tqdm_files:
        for idx, f in enumerate(tqdm_files):
            display_filename = os.path.join(*f.rsplit("/", maxsplit=4)[-2:])
            log.info("File {} / {} - {}".format(idx+1, length, f))
            tqdm_files.set_description(display_filename)
            extract_poses(f, e, model, write_output)

    log.info("Done!")
