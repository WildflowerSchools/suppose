import numpy as np
from suppose.proto import *
from suppose import suppose_pb2
from math import fabs
import tempfile


def assert_almost_equals(a, b, eps=1e-6):
    assert fabs(a - b) <= eps


def test_pose2d_to_array():
    pose = suppose_pb2.Pose2D()
    for i in range(3):
        kp = pose.keypoints.add()
        kp.point.x = i
        kp.point.y = i*2
        kp.score = i*3
    expected_a = np.array([
        [0,0,0],
        [1,2,3],
        [2,4,6]
    ], dtype=np.float32)
    a = pose2d_to_array(pose)
    assert np.all(a == expected_a)


def test_array_to_pose2d():
    a = np.array([
        [0,0,0],
        [1,2,3],
        [2,4,6]
    ], dtype=np.float32)
    expected_pose = suppose_pb2.Pose2D()
    for i in range(3):
        kp = expected_pose.keypoints.add()
        kp.point.x = i
        kp.point.y = i*2
        kp.score = i*3
    pose = array_to_pose2d(a)
    assert pose == expected_pose

def test_protonic_from_proto():
    pv_pb = suppose_pb2.ProcessedVideo(camera="testcam")
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    for j in range(1, n_frames+1):
        frame = pv_pb.frames.add()
        frame.timestamp = 123.1*j
        for i in range(1, n_poses+1):
            pose = frame.poses.add()
            for k in range(1, n_keypoints+1):
                keypoint = pose.keypoints.add()
                id = k * i * j + .1
                keypoint.point.x = id
                keypoint.point.y = id + 1
                keypoint.score = id + 2


    pv = ProcessedVideo.from_proto(pv_pb)
    assert pv.camera == "testcam"
    assert len(pv.frames) == n_frames
    frame = pv.frames[0]
    assert frame.timestamp == 123.1
    assert len(frame.poses) == n_poses
    pose = frame.poses[1]
    assert len(pose.keypoints) == n_keypoints
    keypoint = pose.keypoints[3]
    assert_almost_equals(keypoint.point.x, 8.1)
    assert_almost_equals(keypoint.point.y, 9.1)
    assert_almost_equals(keypoint.score, 10.1)

    # santity check
    pv_pb2 = pv.to_proto()
    assert pv_pb == pv_pb2

def test_protonic_to_proto():
    pv = ProcessedVideo(camera="testcam")
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    for j in range(1, n_frames+1):
        frame = Frame()
        pv.frames.append(frame)
        frame.timestamp = 123.1*j
        for i in range(1, n_poses+1):
            pose = Pose2D()
            frame.poses.append(pose)
            for k in range(1, n_keypoints+1):
                keypoint = Keypoint2D()
                pose.keypoints.append(keypoint)
                id = k * i * j + .1
                keypoint.point.x = id
                keypoint.point.y = id + 1
                keypoint.score = id + 2

    pv_pb = suppose_pb2.ProcessedVideo(camera="testcam")
    for j in range(1, n_frames+1):
        frame = pv_pb.frames.add()
        frame.timestamp = 123.1*j
        for i in range(1, n_poses+1):
            pose = frame.poses.add()
            for k in range(1, n_keypoints+1):
                keypoint = pose.keypoints.add()
                id = k * i * j + .1
                keypoint.point.x = id
                keypoint.point.y = id + 1
                keypoint.score = id + 2

    pv_pb2 = pv.to_proto()
    assert pv_pb2 == pv_pb

def test_protonic_to_file_from_file():
    pv = ProcessedVideo(camera="testcam")
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    for j in range(1, n_frames+1):
        frame = Frame()
        pv.frames.append(frame)
        frame.timestamp = 123.1*j
        for i in range(1, n_poses+1):
            pose = Pose2D()
            frame.poses.append(pose)
            for k in range(1, n_keypoints+1):
                keypoint = Keypoint2D()
                pose.keypoints.append(keypoint)
                id = k * i * j + .1
                keypoint.point.x = id
                keypoint.point.y = id + 1
                keypoint.score = id + 2

    with tempfile.TemporaryFile() as fp:
        pv.to_proto_file(fp)
        fp.seek(0)
        pv2 = ProcessedVideo.from_proto_file(fp)

        # floating point precision issue with native python float vs
        # protobuf float, so do the following comparision on the protobuf forms
        assert pv.to_proto() == pv2.to_proto()
