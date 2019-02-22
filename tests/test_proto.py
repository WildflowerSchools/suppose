import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from suppose.pose_extractor import PoseExtractor
from suppose.proto import *
from suppose import suppose_pb2
from suppose.camera import load_calibration
from math import fabs
import tempfile
from hypothesis import given
#from hypothesis.strategies import from_type, builds, floats, lists, composite, assume
import hypothesis.strategies as st
from google.protobuf import json_format
import networkx as nx
import cv2
from . import test_fixtures


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

def test_pose2d_pb_to_array():
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

def test_pose2d_to_numpy():
    pose = suppose_pb2.Pose2D()
    for i in range(3):
        kp = pose.keypoints.add()
        kp.point.x = i
        kp.point.y = i*2
        kp.score = i*3
    p = Pose2D.from_proto(pose)
    expected_a = np.array([
        [0,0,0],
        [1,2,3],
        [2,4,6]
    ], dtype=np.float32)
    a = pose2d_to_array(pose)
    assert np.all(a == expected_a)
    assert np.all(p.to_numpy() == expected_a)

def test_array_to_pose2d_pb():
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

def test_numpy_to_pose2d_pb():
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
    expected_p = Pose2D.from_proto(expected_pose)
    pose = array_to_pose2d(a)
    assert pose == expected_pose
    p = Pose2D.from_numpy(a)
    assert p == expected_p

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
    assert G_dict == pnxg.to_dict()
    assert pnxg == NXGraph.from_dict(pnxg.to_dict())


def test_pose3dgraph_reconstruct():
    cwd = os.path.dirname(os.path.realpath(__file__))
    camera_names = ["camera01", "camera02", "camera03", "camera04"]
    cameras = []
    frames = []
    extractor = PoseExtractor()
    for name in camera_names:
        camera_file = "data/{}_cal.json".format(name)
        image_file = "data/still_2018-07-04-18-23-00_{}.jpg".format(name)
        file = os.path.join(cwd, camera_file)
        camera = ImmutableCamera.from_legacy_json_file(file)
        cameras.append(camera)
        file = os.path.join(cwd, image_file)
        im = cv2.imread(file)
        frame = extractor.extract(im)
        frames.append(frame)

    graph = Pose3DGraph.reconstruct(frames, cameras)
    frame3d = Frame3D.from_graph(graph)
    assert len(frame3d.poses) == 4


#def test_pose3dgraph_reconstruct_2():
#    cwd = os.path.dirname(os.path.realpath(__file__))
#    camera_names = ["camera01", "camera03", "camera05"]
#    cameras = []
#    frames = [Frame.from_dict(f) for f in test_fixtures.POSE3DGRAPH_RECONSTRUCT_2_FRAMES]
#    for name in camera_names:
#        camera_file = "data/feb14_{}.json".format(name)
#        file = os.path.join(cwd, camera_file)
#        camera = load_calibration(file)
#        cameras.append(camera)
#
#    graph = Pose3DGraph.reconstruct(frames, cameras)
#    frame3d = Frame3D.from_graph(graph)
#    assert len(frame3d.poses) == 3

def test_pose3dgraph_reconstruct_2():
    cwd = os.path.dirname(os.path.realpath(__file__))
    camera_names = ["camera01", "camera03", "camera05"]
    cameras = []
    frames = [Frame.from_dict(f) for f in test_fixtures.POSE3DGRAPH_RECONSTRUCT_2_FRAMES]
    for name in camera_names:
        camera_file = "data/feb14_{}.json".format(name)
        file = os.path.join(cwd, camera_file)
        #camera = load_calibration(file)
        camera = ImmutableCamera.from_legacy_json_file(file)
        cameras.append(camera)

    graph = Pose3DGraph.reconstruct(frames, cameras)
    frame3d = Frame3D.from_graph(graph)
    assert len(frame3d.poses) == 3


def test_immutablecamera_from_legacy_dict():
    cwd = os.path.dirname(os.path.realpath(__file__))
    name = "camera01"
    camera_file = "data/feb14_{}.json".format(name)
    file = os.path.join(cwd, camera_file)
    camera = ImmutableCamera.from_legacy_json_file(file, name)
    matrix = camera.matrix.to_numpy()
    assert matrix.shape == (3, 3)
    matrix2 = camera.matrix.to_numpy()
    # numpy form should be cached
    assert id(matrix) == id(matrix2)
    #try:
    #    camera.matrix.data[0] = 0
    #except:
    #    pass
    #else:
    #    raise AssertionError("should not be able to modify immutable matrix")

    expected_matrix = np.array([[937.9434138713071, 0.0, 725.3442435932047],
                       [0.0, 958.7864143830515, 505.9760884028349],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    diff = np.fabs(matrix - expected_matrix)
    assert np.all(diff == 0)

    pb = camera.to_proto()
    camera2 = ImmutableCamera.from_proto(pb)
    assert camera == camera2

    assert camera.projection.shape == (3, 4)

def test_batch_from_search():
    cwd = os.path.dirname(os.path.realpath(__file__))
    cameras = []
    camera_names = ("camera01", "camera03", "camera05")
    for name in camera_names:
        camera_file = "data/feb14_{}.json".format(name)
        file = os.path.join(cwd, camera_file)
        camera = ImmutableCamera.from_legacy_json_file(file, name)
        cameras.append(camera)

    video_glob = "/Users/lue/wildflower/test_feb14_2019/camera*/*.mp4"
    batch = Batch.from_search(cameras, video_glob)

class NullPoseExtractor:
    def extract(self, image, timestamp=0):
        frame = Frame(timestamp=timestamp)
        return frame

def test_processed_video_from_video():
    home = os.environ['HOME']
    _file = "wildflower/test_feb14_2019/camera01/video_2019-02-14-18-10-10.mp4"
    file = os.path.join(home, _file)
    pb_file = file + "__proto-ProcessedVideo-cmu.pb"
    extractor = NullPoseExtractor()
    pv = ProcessedVideo.from_video(file, extractor=extractor)
    pv_expected = ProcessedVideo.from_proto_file(pb_file)
    timestamps = [f.timestamp for f in pv.frames]
    expected_timestamps = [f.timestamp for f in pv_expected.frames]
    assert timestamps == expected_timestamps
