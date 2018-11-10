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


def humans_to_Frame(frame, humans, frame_width, frame_height):
    #frame = suppose_pb2.Frame()
    for human in humans:
        body_parts = human.body_parts
        #keypoints = []
        pose2d = frame.poses.add()
        #pose2d.keypoints = keypoints
        for idx in range(MAX_NUM_BODY_PARTS):
            keypoint = pose2d.keypoints.add()
            try:
                body_part = body_parts[idx]
                keypoint.point.x = body_part.x * frame_width
                keypoint.point.y = body_part.y * frame_height
                keypoint.score = body_part.score

                #keypoint = suppose_pb2.Pose2D.Keypoint2D(x=body_part.x * frame_width,
                #                                       y=body_part.y *  frame_height,
                #                                       score=body_part.score)
            except KeyError:
                #keypoint = suppose_pb2.Pose2D.Keypoint2D(x=0, y=0, score=0)
                pass
            #keypoints.append(keypoint)

    #return frame

@timing
def tfposes_to_ProcessedVideo(poses, timestamps, frame_width, frame_height, input_file, model_name):
    video = suppose_pb2.ProcessedVideo()
    video.width = frame_width
    video.height = frame_height
    video.file = input_file
    video.model = model_name
    for timestamp, pose in zip(timestamps, poses):
        frame = video.frames.add()
        frame.timestamp = timestamp.timestamp()
        humans_to_Frame(frame, pose, frame_width, frame_height)
    return video


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
    time0 = time.time()
    if cap.isOpened() is False:
        raise IOError("Error reading file {}".format(input_file))

    timestamps = []
    frame_numbers = []
    pbar = tqdm(total=video_length)
    while cap.isOpened():
        # debug
        if count > 10:
            break
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
    cap.release()
    pbar.close()
    log.info("Converting to protobuf")
    processed_video = tfposes_to_ProcessedVideo(poses, timestamps, width, height, input_file, model_name)
    return processed_video
    #output_filename = "{}__ProcessedVideo.pb".format(input_file)
    #log.info("Writing to file: {}".format(output_filename))
    #with open(output_filename, "wb") as f:
    #    f.write(processed_video.SerializeToString())



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
                output_filename = "{}__ProcessedVideo-{}.pb".format(f, model)
                with open(output_filename, 'wb') as out:
                    out.write(processed_video.SerializeToString())

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
