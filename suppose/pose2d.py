import pickle
import glob
import time
import os
import cv2
from tqdm import tqdm
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from logbook import Logger
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .common import timing
from suppose import suppose_pb2
from suppose import proto


log = Logger('pose2d')


MAX_NUM_BODY_PARTS = 18


def pose_to_array(pose, frame_width, frame_height):
    all_humans = []
    for human in pose:
        body_parts = human.body_parts
        parts = []
        for idx in range(MAX_NUM_BODY_PARTS):
            try:
                body_part = body_parts[idx]
                part = (body_part.x * frame_width, body_part.y * frame_height, body_part.score)
            except KeyError:
                part = (0, 0, 0)
            parts.append(part)
        all_humans.append(np.array(parts, dtype=np.float32))

    pose_converted = np.array(all_humans, dtype=np.float32)
    return pose_converted


@timing
def tfpose_to_pandas(poses, frame_width, frame_height):
    all_poses = { idx: {"poses": pose_to_array(pose, frame_width, frame_height)} for idx, pose in enumerate(poses)}
    df = pd.DataFrame.from_dict(all_poses, orient="index")
    return df


def human_to_pose2d(human, frame_width, frame_height):
    body_parts = human.body_parts
    pose2d = proto.Pose2D()
    for idx in range(MAX_NUM_BODY_PARTS):
        keypoint = proto.Keypoint2D()
        try:
            body_part = body_parts[idx]
        except KeyError:
            pass
        else:
            keypoint.point.x = body_part.x * frame_width
            keypoint.point.y = body_part.y * frame_height
            keypoint.score = body_part.score
        pose2d.keypoints.append(keypoint)
    return pose2d


def parse_datetime(a, format):
    return datetime.strptime(a, format)


@timing
def extract_poses(input_file, e, model_name, datetime_start):
    log.info("Processing {}".format(input_file))
    if datetime_start is None:
        datetime_start = datetime.fromtimestamp(0)
    cap = cv2.VideoCapture(input_file)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    poses = []
    if cap.isOpened() is False:
        raise IOError("Error reading file {}".format(input_file))

    timestamps = []
    frame_numbers = []

    processed_video = proto.ProcessedVideo(width=width, height=height, file=input_file, model=model_name)

    pbar = tqdm(total=video_length)
    time0 = time.time()
    while cap.isOpened():
        # debug
        #if count >= 10:
        #    break
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
        frame = proto.Frame(timestamp=timestamp.timestamp())
        for human in humans:
            frame.poses.append(human_to_pose2d(human, width, height))
        processed_video.frames.append(frame)
        pbar.update(1)
    time1 = time.time()
    log.info("{} fps".format((count / (time1 - time0))))
    log.info("{} s".format(time1 - time0))
    cap.release()
    pbar.close()
    log.info("Converting to protobuf")
    return processed_video




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
            processed_video = extract_poses(f, e, model, datetime_start)
            if write_output:
                output_filename = "{}__proto-ProcessedVideo-{}.pb".format(f, model)
                processed_video.to_proto_file(output_filename)
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
