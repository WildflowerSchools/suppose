import pickle
import glob
import time
import os
import cv2
from tqdm import tqdm
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from logbook import Logger, RotatingFileHandler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .common import timing


log_handler = RotatingFileHandler('suppose.log')
log_handler.push_application()
log = Logger('pose2d')


MAX_NUM_BODY_PARTS = 18


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
    all_poses = { idx: {"poses": pose_to_array(pose)} for idx, pose in enumerate(poses)}
    df = pd.DataFrame.from_dict(all_poses, orient="index")
    return df


def parse_datetime(a, format):
    return datetime.strptime(a, format)


@timing
def extract_poses(video, e, model_name, write_output=True, datetime_start=None):
    log.info("Processing {}".format(video))
    if datetime_start is None:
        datetime_start = datetime.fromtimestamp(0)
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    poses = []
    time0 = time.time()
    if cap.isOpened() is False:
        raise IOError("Error reading file {}".format(video))

    timestamps = []
    frame_numbers = []
    pbar = tqdm(total=video_length)
    while cap.isOpened():
        offset = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret_val, image = cap.read()
        if not ret_val:
            break
        timestamp = datetime_start + timedelta(milliseconds=int(offset))
        timestamps.append(timestamp)
        frame_numbers.append(count)
        count += 1
        humans = e.inference(image, resize_to_default=True, upsample_size=4.0)
        poses.append(humans)
        pbar.update(1)
    time1 = time.time()
    log.info("{} fps".format((count / (time1 - time0))))
    log.info("{} s".format(time1 - time0))

    if write_output:
        df_poses_pickle_filename = '{}__poses_df-{}.pickle.xz'.format(video, model_name)
        df_poses_json_filename = '{}__poses_df-{}.json.xz'.format(video, model_name)
        df_poses = tfpose_to_pandas(poses)
        df_poses.index = timestamps
        df_poses['file'] = os.path.basename(video)
        df_poses['model'] = model_name
        df_poses['frame'] = frame_numbers
        for column in ('file', 'model'):
            df_poses[column] = df_poses[column].astype('category')

        log.info("Writing to: {}".format(df_poses_pickle_filename))
        df_poses.to_pickle(df_poses_pickle_filename, compression="xz")
        log.info("Writing to: {}".format(df_poses_json_filename))
        df_poses.to_json(df_poses_json_filename, compression="xz")

    cap.release()
    pbar.close()


@timing
def extract(videos, model, resolution, write_output, display_progress, file_datetime_format):
    log.info("Starting Pose Extractor")
    log.info("videos: {}".format(videos))
    log.info("model: {}".format(model))
    log.info("resolution: {}".format(resolution))
    log.debug("Initializing {} : {}".format(model, get_graph_path(model)))

    w, h = model_wh(resolution)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    files = glob.glob(videos)
    files.sort()
    log.info("Files to process: {}".format(files))
    length = len(files)
    log.info("Processing {} files".format(length))
    disable_tqdm = not display_progress
    with tqdm(files, disable=disable_tqdm) as tqdm_files:
        for idx, f in enumerate(tqdm_files):
            display_filename = os.path.join(*f.rsplit("/", maxsplit=4)[-2:])
            log.info("File {} / {} - {}".format(idx+1, length, f))
            tqdm_files.set_description(display_filename)
            # get timestamp of first frame of video via filename
            if file_datetime_format != "":
                datetime_start = parse_datetime(os.path.basename(f), file_datetime_format)
            else:
                datetime_start = None
            extract_poses(f, e, model, write_output, datetime_start)

    log.info("Done!")


@timing
def combine(files_glob, output_filename):
    log.info("Combine serialized Pandas dataframes")
    log.info("files_glob: {}".format(files_glob))
    log.info("output_filename: {}".format(output_filename))
    files = glob.glob(files_glob)
    files.sort()
    log.info("Files to process: {}".format(files))
    dfs = []
    for f in files:
        log.info("Reading file: {}".format(f))
        df = pd.read_pickle(f)
        dfs.append(df)
    ef = pd.concat(dfs, copy=False)
    ef = ef[~ef.index.duplicated(keep='last')]
    ef.sort_index(inplace=True)
    for column in ('file', 'model'):
        ef[column] = ef[column].astype('category')
    output_pickle_filename = '{}.pickle.xz'.format(output_filename)
    output_json_filename = '{}.json.xz'.format(output_filename)
    log.info("Writing to: {}".format(output_pickle_filename))
    ef.to_pickle(output_pickle_filename, compression="xz")
    log.info("Writing to: {}".format(output_json_filename))
    ef.to_json(output_json_filename, compression="xz")


@timing
def bundle_multiview(camera_poses, output):
    log.info("Bundling multiple camera views together into single file")
    data = {}
    for cp in camera_poses:
        name = cp['name']
        file = cp['file']
        log.info("{} - {}".format(name, file))
        data[name] = pd.read_pickle(file)
    df = pd.concat(data.values(), axis=1, keys=data.keys())
    output_pickle_filename = '{}.pickle.xz'.format(output)
    output_json_filename = '{}.json.xz'.format(output)
    log.info("Writing to: {}".format(output_pickle_filename))
    df.to_pickle(output_pickle_filename, compression="xz")
    log.info("Writing to: {}".format(output_json_filename))
    df.to_json(output_json_filename, compression="xz")
