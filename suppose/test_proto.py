import os
import numpy as np
from suppose.pose_extractor import PoseExtractor
from suppose.proto import *
from suppose import suppose_pb2
from suppose.camera import load_calibration
from math import fabs
import tempfile
import attr
import cattr
from hypothesis import given, infer
#from hypothesis.strategies import from_type, builds, floats, lists, composite, assume
import hypothesis.strategies as st
from google.protobuf import json_format
import networkx as nx
import cv2
from datetime import datetime
import suppose.pose2d


def assert_almost_equals(a, b, eps=1e-6):
    assert fabs(a - b) <= eps


def make_processed_video(n_frames=3, n_poses=2, n_keypoints=4):
    pv = ProcessedVideo(camera="testcam")
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
    return pv

def make_processed_video_pb(n_frames=3, n_poses=2, n_keypoints=4):
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
    return pv_pb

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
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    pv_pb = make_processed_video_pb(n_frames, n_poses, n_keypoints)

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
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    pv = make_processed_video(n_frames, n_poses, n_keypoints)
    pv_pb = make_processed_video_pb(n_frames, n_poses, n_keypoints)
    pv_pb2 = pv.to_proto()
    assert pv_pb2 == pv_pb

def test_protonic_to_file_from_file():
    n_frames = 3
    n_poses = 2
    n_keypoints = 4
    pv = make_processed_video(n_frames, n_poses, n_keypoints)

    with tempfile.TemporaryFile() as fp:
        pv.to_proto_file(fp)
        fp.seek(0)
        pv2 = ProcessedVideo.from_proto_file(fp)

        # floating point precision issue with native python float vs
        # protobuf float, so do the following comparision on the protobuf forms
        assert pv.to_proto() == pv2.to_proto()


def test_protonic_to_from_dict():
    pv = make_processed_video()
    d = pv.to_dict()
    pv2 = ProcessedVideo.from_dict(d)
    assert pv == pv2


@st.composite
def make_ProcessedVideo(draw):
    vector2f_build = st.builds(Vector2f, x=st.floats(max_value=50000, min_value=0, allow_nan=False), y=st.floats(max_value=50000, min_value=0, allow_nan=False))
    keypoint2D_build = st.builds(Keypoint2D, point=vector2f_build, score=st.floats(min_value=0, max_value=1))
    keypoint2Ds_build = st.lists(keypoint2D_build, min_size=1, max_size=5)
    pose_build = st.builds(Pose2D, keypoints=keypoint2Ds_build)
    poses_build = st.lists(pose_build, min_size=0, max_size=5)
    frame_build = st.builds(Frame, timestamp=st.floats(min_value=0, allow_nan=False), poses=poses_build)
    frames_build = st.lists(frame_build, min_size=0, max_size=5)

    pv_build = st.builds(ProcessedVideo, camera=st.text(max_size=100), width=st.integers(max_value=100000, min_value=0), height=st.integers(max_value=100000, min_value=0), file=st.text(max_size=100), model=st.text(max_size=100))
    pv = draw(pv_build)
    pv.frames = draw(frames_build)
    return pv


@given(make_ProcessedVideo())
def test_ProcessedVideo_to_from_dict(pv):
    d = pv.to_dict()
    pv2 = ProcessedVideo.from_dict(d)
    assert pv == pv2

@given(make_ProcessedVideo())
def test_ProcessedVideo_to_from_proto_file(pv):
    with tempfile.TemporaryFile() as fp:
        pv.to_proto_file(fp)
        fp.seek(0)
        pv2 = ProcessedVideo.from_proto_file(fp)

        # floating point precision issue with native python float vs
        # protobuf float, so do the following comparision on the protobuf forms
        pb = pv.to_proto()
        pb2 = pv2.to_proto()
        assert pb == pb2

def test_nx():
    def f32(a):
        return np.float32(a).item()
    G = nx.Graph()

    G.add_edge('a', 'b', weight=f32(0.6))
    G.add_edge('a', 'c', weight=f32(0.2))
    G.add_edge('c', 'd', weight=f32(0.1))
    G.add_edge('c', 'e', weight=f32(0.7))
    G.add_edge('c', 'f', weight=f32(0.9))
    G.add_edge('a', 'd', weight=f32(0.3))

    pnxg = NXGraph(graph=G)
    pb = pnxg.to_proto()

    node_ids = []
    for node in pb.nodes:
        node_ids.append(node.id)

    node_ids.sort()
    assert node_ids == ['a', 'b', 'c', 'd', 'e', 'f']

    pb_dict = json_format.MessageToDict(pb, including_default_value_fields=True)
    G_dict = nx.json_graph.node_link_data(G)
    assert pb_dict == G_dict


def test_pose3dgraph_reconstruct():
    cwd = os.path.dirname(os.path.realpath(__file__))
    camera_names = ["camera01", "camera02", "camera03", "camera04"]
    cameras = []
    frames = []
    extractor = PoseExtractor()
    for name in camera_names:
        camera_file = "test/data/{}_cal.json".format(name)
        image_file = "test/data/still_2018-07-04-18-23-00_{}.jpg".format(name)
        file = os.path.join(cwd, camera_file)
        camera = load_calibration(file)
        cameras.append(camera)
        file = os.path.join(cwd, image_file)
        im = cv2.imread(file)
        frame = extractor.extract(im)
        frames.append(frame)

    graph = Pose3DGraph.reconstruct(frames, cameras)
    frame3d = Frame3D.from_graph(graph)
    import ipdb;ipdb.set_trace()