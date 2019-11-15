from concurrent import futures
import time
import math
import numpy as np
import io
import grpc
from suppose.pose2d import humans_to_Frame
import suppose_pb2
import suppose_pb2_grpc
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from logbook import Logger, StreamHandler
import sys
from datetime import datetime, timedelta
import cv2


StreamHandler(sys.stdout).push_application()
log = Logger("suppose-server")


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def extract_poses(input_file, e, model_name, datetime_start=None):
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
    # import ipdb;ipdb.set_trace()
    if cap.isOpened() is False:
        raise IOError("Error reading file {}".format(input_file))
    # import ipdb;ipdb.set_trace()
    timestamps = []
    frame_numbers = []
    # pbar = tqdm(total=video_length)
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
        frame = suppose_pb2.Frame()
        humans_to_Frame(frame, humans, width, height)
        yield frame

        # pbar.update(1)
    time1 = time.time()
    log.info("{} fps".format((count / (time1 - time0))))
    log.info("{} s".format(time1 - time0))
    cap.release()
    # pbar.close()
    log.info("Done processing file: {}".format(input_file))


class PoseExtractorServicer(suppose_pb2_grpc.PoseExtractorServicer):
    def __init__(self, model, inference_width=432, inference_height=368):
        self.model = model
        self.estimator = TfPoseEstimator(
            get_graph_path(model), target_size=(inference_width, inference_height)
        )
        log.info("Estimator loaded!")

    def GetPose(self, image, context):
        imbuf = image.data
        imbytes = np.fromstring(imbuf, np.uint8)
        img = cv2.imdecode(imbytes, cv2.IMREAD_COLOR)
        humans = self.estimator.inference(img, resize_to_default=True, upsample_size=4.0)
        frame = suppose_pb2.Frame()
        humans_to_Frame(frame, humans, img.shape[1], img.shape[0])
        return frame

    def StreamPoses(self, request_iterator, context):
        for image in request_iterator:
            imbuf = image.data
            imbytes = np.fromstring(imbuf, np.uint8)
            img = cv2.imdecode(imbytes, cv2.IMREAD_COLOR)
            humans = self.estimator.inference(img, resize_to_default=True, upsample_size=4.0)
            frame = suppose_pb2.Frame()
            humans_to_Frame(frame, humans, img.shape[1], img.shape[0])
            yield frame

    def StreamPosesFromVideo(self, file, context):
        log.info("Here!")
        # import ipdb;ipdb.set_trace()
        for it in extract_poses(file.path, self.estimator, self.model):
            yield it


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    suppose_pb2_grpc.add_PoseExtractorServicer_to_server(PoseExtractorServicer("mobilenet_thin"), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
