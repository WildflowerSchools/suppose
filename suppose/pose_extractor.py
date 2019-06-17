from suppose.pose2d import human_to_pose2d
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from suppose.proto import *
import datetime as datetime

class PoseExtractor:
    def __init__(self, model='cmu', width=432, height=368):
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(width, height))

    def extract(self, image, timestamp=0):
        height, width, _ = image.shape
        humans = self.e.inference(image, resize_to_default=True, upsample_size=4.0)
        frame = Frame(timestamp=timestamp)
        for human in humans:
            frame.poses.append(human_to_pose2d(human, width, height))
        return frame
