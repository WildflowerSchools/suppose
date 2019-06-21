"""
Protobuf-related utilities for suppose.proto
"""
import json
import typing
import itertools
from collections import defaultdict
import attr
import cattr
import numpy as np
from suppose import suppose_pb2
import pandas as pd
import networkx as nx
import cv2
from matplotlib.axes import Axes
import glob
import os
import datetime
import re
from tqdm import tqdm
from logbook import Logger

from suppose.common import rmse, LIMB_COLORS, NECK_INDEX, SHOULDER_INDICES, HEAD_AND_TORSO_INDICES, TIMESTAMP_REGEX
from tf_pose.common import CocoPairsRender

import matplotlib.pyplot as plt

from google.protobuf import json_format
from networkx import json_graph

import typing
from typing import TypeVar, Type

T = TypeVar('T', bound='Parent')

def load_protobuf(file, pb_cls):
    """
    Instantiates protobuf object and loads data from file into object.
    :param file: file containing serialized protobuf contents
    :param pb_cls: class from generated client
    :return: instance of pb_cls
    """
    obj = pb_cls()
    if hasattr(file, 'read'):
        contents = file.read()
    else:
        with open(file, 'rb') as f:
            contents = f.read()
    obj.ParseFromString(contents)
    return obj


def write_protobuf(file, pb):
    contents = pb.SerializeToString()
    if hasattr(file, 'write'):
        file.write(contents)
    else:
        with open(file, 'wb') as f:
            f.write(contents)


def pose2d_to_array(pb):
        a = []
        for kp in pb.keypoints:
            a.append([kp.point.x, kp.point.y, kp.score])
        return np.array(a, dtype=np.float32)


def array_to_pose2d(a):
    pose = suppose_pb2.Pose2D()
    for row in a:
        kp = pose.keypoints.add()
        kp.point.x = row[0]
        kp.point.y = row[1]
        kp.score = row[2]
    return pose


def protonic(protobuf_cls):
    """ Class decorator to support marshing/unmarshaling of attr classes to protobuf equivalents """
    def protonic_mix(cls):

        def from_proto(pb):
            d = json_format.MessageToDict(pb, including_default_value_fields=True)
            return cls.from_dict(d)

        def __from_proto_OLD(pb):
            # TODO: explore using google.protobuf.json_format.ParseDict
            obj = cls()
            for a in attr.fields(cls):
                value = getattr(pb, a.name)  # allow AttributeError for now
                # TODO: support for mapping type
                if isinstance(a.default, attr.Factory):
                    atype = a.metadata['type']
                    if a.default.factory == list:
                        things = []
                        for thing in value:
                            if any(isinstance(thing, t) for t in (int, float, str, bytes)):
                                things.append(thing)
                            else:
                                things.append(atype.from_proto(thing))
                        value = things
                    else:
                        value = atype.from_proto(value)
                setattr(obj, a.name, value)
            return obj

        def from_proto_file(file):
            pb = load_protobuf(file, protobuf_cls)
            return cls.from_proto(pb)

        def to_proto(self):
            pb = protobuf_cls()
            json_format.ParseDict(self.to_dict(), pb)
            return pb

        def __to_proto_OLD(self):
            # TODO: explore using google.protobuf.json_format.MessageToDict
            pb = protobuf_cls()
            for a in attr.fields(cls):
                value = getattr(self, a.name)
                if any(isinstance(value, t) for t in (int, float, str, bytes)):
                    # serialize primitive types directly
                    setattr(pb, a.name, value)
                else:
                    # TODO: support for mapping type
                    if isinstance(value, list):
                        items = getattr(pb, a.name)
                        for v in value:
                            item = items.add()
                            item.CopyFrom(v.to_proto())
                    else:
                        item = getattr(pb, a.name)
                        item.CopyFrom(value.to_proto())
            return pb

        def to_proto_file(self, file):
            write_protobuf(file, self.to_proto())

        def from_dict(d):
            return cattr.structure(d, cls)

        def to_dict(self):
            return cattr.unstructure(self)

        cls.from_proto = staticmethod(from_proto)
        cls.from_proto_file = staticmethod(from_proto_file)
        cls.to_proto = to_proto
        cls.to_proto_file = to_proto_file
        cls.from_dict = staticmethod(from_dict)
        cls.to_dict = to_dict
        return cls

    return protonic_mix


@protonic(suppose_pb2.Matrix)
@attr.s
class Matrix:
    rows: int = attr.ib(default=0)
    columns: int = attr.ib(default=0)
    data: typing.List[float] = attr.ib(default=attr.Factory(list), metadata={"type": float})

    def to_numpy(self):
        a = np.array(self.data, dtype=np.float32)
        a = a.reshape((self.rows, self.columns))
        return a


@protonic(suppose_pb2.Matrix)
@attr.s(frozen=True)
class ImmutableMatrix:
    rows: int = attr.ib(default=0)
    columns: int = attr.ib(default=0)
    data: typing.List[float] = attr.ib(default=attr.Factory(list), metadata={"type": float})

    def __attrs_post_init__(self):
        a = np.array(self.data, dtype=np.float64)
        a = a.reshape((self.rows, self.columns))
        object.__setattr__(self, "_array", a)

    def to_numpy(self):
        return self._array

    @classmethod
    def from_legacy_dict(cls, m) -> 'ImmutableMatrix':
        rows = len(m)
        columns = None
        data = []
        for row in m:
            if columns is None:
                columns = len(row)
            else:
                columns_check = len(row)
                if columns_check != columns:
                    raise ValueError("Legacy matrix dict is malformed. Columns are not same size")
            for column in row:
                data.append(column)
        return cls(rows=rows, columns=columns, data=data)

    @classmethod
    def from_numpy(cls, a):
        rows, columns = a.shape
        b = a.flatten()
        return cls(rows=rows, columns=columns, data=b)


@protonic(suppose_pb2.Camera)
@attr.s
class Camera:
    name: str = attr.ib(default="")
    matrix: Matrix = attr.ib(default=attr.Factory(Matrix), metadata={"type": Matrix})
    distortion: Matrix = attr.ib(default=attr.Factory(Matrix), metadata={"type": Matrix})
    rotation: Matrix = attr.ib(default=attr.Factory(Matrix), metadata={"type": Matrix})
    translation: Matrix = attr.ib(default=attr.Factory(Matrix), metadata={"type": Matrix})




@protonic(suppose_pb2.Camera)
@attr.s(frozen=True)
class ImmutableCamera:
    name: str = attr.ib(default="")
    matrix: ImmutableMatrix = attr.ib(default=attr.Factory(ImmutableMatrix), metadata={"type": ImmutableMatrix})
    distortion: ImmutableMatrix = attr.ib(default=attr.Factory(ImmutableMatrix), metadata={"type": ImmutableMatrix})
    rotation: ImmutableMatrix = attr.ib(default=None, metadata={"type": ImmutableMatrix})
    translation: ImmutableMatrix = attr.ib(default=None, metadata={"type": ImmutableMatrix})

    def __attrs_post_init__(self):
        projection = self.__get_projection_matrix()
        object.__setattr__(self, "_projection", projection)

    @property
    def projection(self):
        return self._projection

    def __get_projection_matrix(self):
        """
        Calculates projection matrix from camera calibration.
        :return: projection matrix
        """
        R = cv2.Rodrigues(self.rotation.to_numpy())[0]
        t = self.translation.to_numpy()
        Rt = np.concatenate([R, t], axis=1)
        C = self.matrix.to_numpy()
        P = np.matmul(C, Rt)
        return P

    @classmethod
    def from_legacy_json_file(cls, file, name=""):
        with open(file, 'rb') as f:
            d = json.load(f)
        return cls.from_legacy_dict(d, name=name)

    @classmethod
    def from_legacy_dict(cls, d, name="") -> 'ImmutableCamera':
        matrix = ImmutableMatrix.from_legacy_dict(d['cameraMatrix'])
        if matrix.rows != 3 or matrix.columns != 3:
            raise ValueError("Camera matrix not 3x3")
        distortion = ImmutableMatrix.from_legacy_dict(d['distortionCoefficients'])
        if distortion.rows != 1 or distortion.columns != 5:
            raise ValueError("Distortion coefficients not 1x5")
        rotation = ImmutableMatrix.from_legacy_dict(d['rotationVector'])
        if rotation.rows != 3 or rotation.columns != 1:
            raise ValueError("Rotation vector not 3x1")
        translation = ImmutableMatrix.from_legacy_dict(d['translationVector'])
        if translation.rows != 3 or translation.columns != 1:
            raise ValueError("Translation vector not 3x1")
        return cls(name=name, matrix=matrix, distortion=distortion, rotation=rotation, translation=translation)

    @classmethod
    def from_charuco_diamond(cls, image, camera_matix, distortion_coefficients, aruco_dict, target_ids):
        """
        :param image:
        :param aruco_dict:
        :param target_ids:
        :return:
        """
        pass

    def undistort_points(self, pts):
        if pts.size == 0:
            return pts
        camera_matrix = self.matrix.to_numpy()
        distortion_coefficients = self.distortion.to_numpy()
        original_shape = pts.shape
        pts2 = np.ascontiguousarray(pts).reshape(-1, 1, 2)
        undistorted_points = cv2.undistortPoints(pts2, camera_matrix, distortion_coefficients, P=camera_matrix)
        undistorted_points = undistorted_points.reshape(original_shape)
        return undistorted_points

    @classmethod
    def triangulate(cls, p1, p2, camera1, camera2):
        if p1.size == 0 or p2.size == 0:
            return np.zeros((0, 3))
        object_points_homogeneous = cv2.triangulatePoints(camera1.projection, camera2.projection, p1.T, p2.T)
        object_points = cv2.convertPointsFromHomogeneous(object_points_homogeneous.T)
        object_points = object_points.reshape(-1, 3)
        return object_points

    def project(self, pts3d):
        if pts3d.size == 0:
            return np.zeros((0, 2))
        return cv2.projectPoints(pts3d, self.rotation.to_numpy(), self.translation.to_numpy(), self.matrix.to_numpy(), self.distortion.to_numpy())[0].squeeze()

@protonic(suppose_pb2.Vector2f)
@attr.s
class Vector2f:
    x: float = attr.ib(default=0)
    y: float = attr.ib(default=0)

    def to_pandas(self):
        return pd.Series(self.to_dict())

    def to_numpy(self):
        return np.array([self.x, self.y], dtype=np.float32)


@protonic(suppose_pb2.Pose2D.Keypoint2D)
@attr.s
class Keypoint2D:
    point: Vector2f = attr.ib(default=attr.Factory(Vector2f), metadata={"type": Vector2f})
    score: float = attr.ib(default=0)

    def to_pandas(self):
        d = {}
        ps = self.point.to_pandas()
        d['point'] = ps
        d['score'] = pd.Series(self.score, index=['value'])
        return pd.concat(d)

    def to_numpy(self):
        point = self.point.to_numpy()
        a = np.append(point, self.score).astype(np.float32)     # flatten
        return a

    @property
    def is_valid(self):
        return (self.score != 0) and (not np.isnan(self.score))


@protonic(suppose_pb2.Pose2D)
@attr.s
class Pose2D:
    keypoints: typing.List[Keypoint2D] = attr.ib(default=attr.Factory(list), metadata={"type": Keypoint2D})

    def to_pandas(self):
        return pd.DataFrame(kp.to_pandas() for kp in self.keypoints)

    def to_numpy(self):
        return np.array([kp.to_numpy() for kp in self.keypoints])

    @classmethod
    def from_numpy(cls, pts):
        obj = cls()
        for pt in pts:
            keypoint = Keypoint2D(point=Vector2f(x=pt[0], y=pt[1]))
            if len(pt) >= 3:
                keypoint.score = pt[2]
            elif np.isnan(pt[0]):
                # TODO: not liking all this finagling to handle 2d poses from dnn versus reprojection
                keypoint.score = np.nan
            else:
                keypoint.score = 1
            obj.keypoints.append(keypoint)
        return obj

    @property
    def points_array(self):
        return np.array([kp.to_numpy()[:-1] for kp in self.keypoints])

    @property
    def valid_keypoints(self):
        return np.array([kp.to_numpy()[:-1] for kp in self.keypoints if kp.is_valid])

    @property
    def valid_keypoints_mask(self):
        return np.array([kp.is_valid for kp in self.keypoints])

    def draw(self, ax=None) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        all_points = self.to_numpy()[:, :2]
        #camera_utilities.draw_2d_image_points(self.valid_keypoints)
        ax.plot(self.valid_keypoints[:, 0], self.valid_keypoints[:,1], '.', color='green', markersize=10)

        valid_keypoints_mask = self.valid_keypoints_mask

        for body_part_connector in CocoPairsRender:
            body_part_from_index = body_part_connector[0]
            body_part_to_index = body_part_connector[1]
            if valid_keypoints_mask[body_part_from_index] and valid_keypoints_mask[body_part_to_index]:
                pt1 = [all_points[body_part_from_index, 0], all_points[body_part_to_index, 0]]
                pt2 = [all_points[body_part_from_index, 1], all_points[body_part_to_index, 1]]
                color = LIMB_COLORS[(body_part_from_index, body_part_to_index)]
                ax.plot(pt1, pt2, 'k-', linewidth=3, markersize=10, color=[c/255 for c in color], alpha=0.8)
        return ax

    # Plot a pose onto a chart with the coordinate system of the origin image.
    # Calls the drawing function above, adds formating, and shows the plot
    def plot(self, ax=None, image_size=(1296, 972)) -> Axes:
        ax = self.draw(ax)
        self.format_plot(ax, image_size=image_size)
        return ax

    @staticmethod
    def format_plot(ax: Axes, image_size):
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(0, image_size[1])
        ax.set_xlabel(r'$u$')
        ax.set_ylabel('$v$')
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.set_aspect('equal')

    def draw_on_image(self, canvas, draw_limbs=True, copy=False, lineType=cv2.LINE_AA):
        if copy:
            canvas = np.copy(canvas)

        #color = (255, 100, 255)
        all_points = self.points_array
        valid_keypoints_mask = self.valid_keypoints_mask
        plottable_points = self.valid_keypoints
        #import ipdb;ipdb.set_trace()

        joint_color = (0, 255, 0)

        height = canvas.shape[0]
        width = canvas.shape[1]

        for x, y in plottable_points:
            # draw keypoint on canvas
            if (x > 0 and y > 0) and (x <= width and y <= height):
                cv2.circle(canvas, (x, y), 4, joint_color, -1)

        if draw_limbs:
            for body_part_connector in CocoPairsRender:
                body_part_from_index = body_part_connector[0]
                body_part_to_index = body_part_connector[1]
                if valid_keypoints_mask[body_part_from_index] and valid_keypoints_mask[body_part_to_index]:
                    #draw line
                    pt1 = (int(all_points[body_part_from_index,0]), int(all_points[body_part_from_index, 1]))
                    pt2 = (int(all_points[body_part_to_index,0]), int(all_points[body_part_to_index, 1]))
                    #print(pt1, pt2)
                    if (0 <= pt1[0] < 99999) and (0 <= pt1[1] < 99999) and ( 0 <= pt2[0] < 99999) and (0 <= pt2[1] < 99999):
                        color = LIMB_COLORS[(body_part_from_index, body_part_to_index)]
                        cv2.line(canvas, pt1, pt2, color, thickness=2, lineType=lineType)
        return canvas

    def points(self):
        return [kp.point for kp in self.keypoints]

    def confidence_scores(self):
        return [kp.score for kp in self.keypoints]


@protonic(suppose_pb2.Frame)
@attr.s
class Frame:
    timestamp: float = attr.ib(default=0)
    poses: typing.List[Pose2D] = attr.ib(default=attr.Factory(list), metadata={"type": Pose2D})

    def to_pandas(self):
        ps = [p.to_pandas() for p in self.poses]
        return pd.concat(ps, keys=range(len(ps)), names=['pose', 'keypoint'])

    def to_numpy(self):
        return np.array([p.to_numpy() for p in self.poses])

    # Plot the poses onto a set of charts, one for each source camera view.
    def plot(self, ax=None, image_size=(1296, 972)) -> Axes:
        if ax is None:
            fig, ax = plt.subplots()
        for pose in self.poses:
            pose.draw(ax=ax)
        Pose2D.format_plot(ax, image_size)
        return ax

    def draw_on_image(self, canvas, copy=False, lineType=cv2.LINE_AA):
        for pose in self.poses:
            canvas = pose.draw_on_image(canvas, copy=copy, lineType=lineType)
        return canvas

    # cvutilities
    def num_poses(self):
        return len(self.poses)

    # cvutilities
    def keypoints(self):
        return [p.points() for p in self.poses]

    def confidence_scores(self):
        return [p.confidence_scores() for p in self.poses]


@protonic(suppose_pb2.ProcessedVideo)
@attr.s
class ProcessedVideo:
    camera: str = attr.ib(default="")
    width: int = attr.ib(default=0)
    height: int = attr.ib(default=0)
    file: str = attr.ib(default="")
    model: str = attr.ib(default="")
    frames: typing.List[Frame] = attr.ib(default=attr.Factory(list), metadata={"type": Frame})

    def to_pandas(self):
        frame_dfs = []
        timestamps = []
        for frame in self.frames:
            frame_dfs.append(frame.to_pandas())
            timestamps.append(frame.timestamp)
        df = pd.concat(frame_dfs, keys=timestamps, names=['timestamp'])
        metadata = {}
        for a in attr.fields(self.__class__):
            value = getattr(self, a.name)
            if any(isinstance(value, t) for t in (int, float, str, bytes)):
                metadata[a.name] = value
        df._metadata = metadata
        return df

    def to_numpy(self):
        return [f.to_numpy() for f in self.frames]

    @property
    def start_time(self):
        if self.frames:
            return self.frames[0].timestamp
        else:
            return None

    @property
    def end_time(self):
        if self.frames:
            return self.frames[-1].timestamp
        else:
            return None

    @classmethod
    def from_video(cls, file, extractor, read_from_cache=True, write_to_cache=True, cache_dir=None, cache_suffix="__proto-ProcessedVideo-cmu.pb", draw_viz=False, viz_suffix="__proto-ProcessedVideo-cmu_viz.mp4", timestamp_start=datetime.datetime.fromtimestamp(0)):
        file = os.path.abspath(file)
        if cache_dir is None:
            cache_dir = os.path.abspath(os.path.dirname(file))
        cached_result_file = os.path.join(cache_dir, os.path.basename(file) + cache_suffix)
        viz_result_file = os.path.join(cache_dir, os.path.basename(file) + viz_suffix)

        if read_from_cache:
            o = cls.from_proto_file(cached_result_file)
            return o

        # try:
        #     # legacy format
        #     timestamp_start = cls._get_timestamp_from_file(file, "video_%Y-%m-%d-%H-%M-%S.mp4")
        # except ValueError:
        #     # using passed in regex
        #     m = re.search(timestamp_regex, file)
        #     if m is None:
        #         raise ValueError("Could not decode timestamp from file name")
        #     d = m.groupdict()
        #     timestamp_str = "{year}-{month}-{day}-{hour}-{minute}-{second}".format(**d)
        #     # lazy way to get validation
        #     timestamp_start = cls._get_timestamp_from_file(timestamp_str, "%Y-%m-%d-%H-%M-%S")

        cap = cv2.VideoCapture(file)
        out = None
        o = cls(file=file)
        #DEBUG_COUNTER = 0
        while True:
            #DEBUG_COUNTER += 1
            #if DEBUG_COUNTER >= 3:
            #    break
            time_offset = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, image = cap.read()
            if not ret:
                break
            timestamp = timestamp_start + datetime.timedelta(milliseconds=time_offset)
            frame = extractor.extract(image, timestamp=timestamp.timestamp())
            o.frames.append(frame)
            if draw_viz:
                viz = frame.draw_on_image(image, copy=True)
                if out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(viz_result_file, fourcc, 10,
                                         (viz.shape[1], viz.shape[0]))
                out.write(viz)

        if write_to_cache:
            if os.path.exists(cached_result_file):
                print("Warning: overriding cached result '{}'".format(cached_result_file))
            o.to_proto_file(cached_result_file)
        return o

    @classmethod
    def _get_timestamp_from_file(cls, file, file_datetime_format):
        timestamp = datetime.datetime.strptime(os.path.basename(file), file_datetime_format)
        return timestamp




@protonic(suppose_pb2.Vector3f)
@attr.s
class Vector3f:
    x: float = attr.ib(default=0.0)
    y: float = attr.ib(default=0.0)
    z: float = attr.ib(default=0.0)

    def to_pandas(self):
        return pd.Series(self.to_dict())

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def project_2d(self, camera: ImmutableCamera) -> Vector2f:
        pts = camera.project(np.array([self.to_numpy()]))
        return Vector2f(x=pts[0], y=pts[1])


@protonic(suppose_pb2.Pose3D.Keypoint3D)
@attr.s
class Keypoint3D:
    point: Vector3f = attr.ib(default=attr.Factory(Vector3f), metadata={"type": Vector3f})
    std: Vector3f = attr.ib(default=attr.Factory(Vector3f), metadata={"type": Vector3f})

    def to_pandas(self):
        d = {}
        d['point'] = self.point.to_pandas()
        d['score'] = self.std.to_pandas()
        return pd.concat(d)

    def to_numpy(self):
        point = self.point.to_numpy()
        #a = np.append(point, self.score).astype(np.float32)     # flatten
        return point

    @property
    def is_valid(self):
        return not np.isnan(self.point.x)


@protonic(suppose_pb2.Pose3D)
@attr.s
class Pose3D:
    type: str = attr.ib(default=suppose_pb2.Pose3D.Type.keys()[0], validator=attr.validators.in_(suppose_pb2.Pose3D.Type.keys()))
    error: float = attr.ib(default=0)
    keypoints: typing.List[Keypoint3D] = attr.ib(default=attr.Factory(list), metadata={"type": Keypoint3D})

    def to_pandas(self):
        return pd.DataFrame(kp.to_pandas() for kp in self.keypoints)

    def to_numpy(self):
        return np.array([kp.to_numpy() for kp in self.keypoints])

    @property
    def valid_keypoints(self):
        return np.array([kp.to_numpy() for kp in self.keypoints if kp.is_valid])

    @property
    def valid_keypoints_mask(self):
        return np.array([kp.is_valid for kp in self.keypoints])

    @classmethod
    def from_2d(cls: Type[T], a: Pose2D, b: Pose2D, camera_calibration1: ImmutableCamera, camera_calibration2: ImmutableCamera) -> T:
        valid_a = a.valid_keypoints_mask
        valid_b = b.valid_keypoints_mask
        p1_orig = a.points_array
        p2_orig = b.points_array
        pts_present = np.logical_and(valid_a, valid_b)  # keypoints exist in both p1 and p2
        if not np.any(pts_present):
            # no commont points
            return cls(error=np.nan)
        p1_orig_shared = p1_orig[pts_present]
        p2_orig_shared = p2_orig[pts_present]

        p1 = camera_calibration1.undistort_points(p1_orig_shared)
        p2 = camera_calibration2.undistort_points(p2_orig_shared)
        pts3d = ImmutableCamera.triangulate(p1, p2, camera_calibration1, camera_calibration2)

        p1_reprojected = camera_calibration1.project(pts3d)
        p2_reprojected = camera_calibration2.project(pts3d)
        p1_rmse = rmse(p1_orig_shared, p1_reprojected)
        p2_rmse = rmse(p2_orig_shared, p2_reprojected)
        if np.isnan(p1_rmse) or np.isnan(p2_rmse):
            # should just continue the loop and ignore this edge
            reprojection_error = np.nan
        else:
            reprojection_error = max(p1_rmse, p2_rmse)
            # should just reject 3d poses errors > threshold

        keypoints_3d = np.full([len(pts_present)]+list(pts3d.shape[1:]), np.nan)
        keypoints_3d[pts_present] = pts3d

        obj = cls(error=reprojection_error)
        for x, y ,z in keypoints_3d:
            obj.keypoints.append(Keypoint3D(point=Vector3f(x=x, y=y, z=z)))
        return obj

    def project_2d(self, camera: ImmutableCamera) -> Pose2D:
        pts_reprojected = camera.project(self.to_numpy())
        return Pose2D.from_numpy(pts_reprojected)

    def anchor_points(self):
        keypoints = self.to_numpy()
        if self.valid_keypoints_mask[NECK_INDEX]:
            return keypoints[NECK_INDEX]
        if np.all(self.valid_keypoints_mask[SHOULDER_INDICES]):
            return np.mean(keypoints[SHOULDER_INDICES], axis = 0)
        if np.any(self.valid_keypoints_mask[HEAD_AND_TORSO_INDICES]):
            head_and_torso_keypoints = np.full(18, False)
            head_and_torso_keypoints[HEAD_AND_TORSO_INDICES] = True
            valid_head_and_torso_keypoints = np.logical_and(
                head_and_torso_keypoints,
                self.valid_keypoints_mask)
            return np.mean(keypoints[valid_head_and_torso_keypoints], axis=0)
        return np.mean(keypoints[self.valid_keypoints_mask], axis=0)


@protonic(suppose_pb2.Frame3D)
@attr.s
class Frame3D:
    timestamp: float = attr.ib(default=0)
    poses: typing.List[Pose3D] = attr.ib(default=attr.Factory(list), metadata={"type": Pose3D})

    def to_pandas(self):
        ps = [p.to_pandas() for p in self.poses]
        return pd.concat(ps, keys=range(len(ps)), names=['pose', 'keypoint'])

    def to_numpy(self):
        return np.array([p.to_numpy() for p in self.poses])

    def num_poses(self):
        return len(self.poses)

    def keypoints(self):
        return [p.to_numpy() for p in self.poses]

    @classmethod
    def from_frames(cls, frames: typing.Iterable[Frame], cameras: typing.Iterable[ImmutableCamera]) -> 'Frame3D':
        graph = Pose3DGraph.reconstruct(frames, cameras)
        return cls.from_graph(graph, timestamp=frames[0].timestamp)

    @classmethod
    def from_graph(cls, graph, timestamp=0):
        poses = []
        for src, tgt, data in graph.graph.edges(data=True):
            poses.append(data['pose'])
        return cls(poses=poses, timestamp=timestamp)

    def project_2d(self, camera: ImmutableCamera) -> Frame:
        frame = Frame(timestamp=self.timestamp)
        for pose3d in self.poses:
            pose2d = pose3d.project_2d(camera)
            frame.poses.append(pose2d)
        return frame




    # Plot the poses onto a set of charts, one for each source camera view.
    #def plot(self, image_size=(1296, 972)):
    #    for pose in self.poses:
    #        pose.draw()
    #    camera_utilities.format_2d_image_plot(image_size)
    #    plt.show()

@protonic(suppose_pb2.ProcessedVideo3D)
@attr.s
class ProcessedVideo3D:
    #camera: str = attr.ib(default="")
    #width: int = attr.ib(default=0)
    #height: int = attr.ib(default=0)
    #file: str = attr.ib(default="")
    #model: str = attr.ib(default="")
    room: str = attr.ib(default="")
    cameras: typing.List[ImmutableCamera] = attr.ib(default=attr.Factory(list), metadata={"type": ImmutableCamera})
    frames: typing.List[Frame3D] = attr.ib(default=attr.Factory(list), metadata={"type": Frame3D})

    def to_pandas(self):
        frame_dfs = []
        timestamps = []
        for frame in self.frames:
            frame_dfs.append(frame.to_pandas())
            timestamps.append(frame.timestamp)
        df = pd.concat(frame_dfs, keys=timestamps, names=['timestamp'])
        metadata = {}
        for a in attr.fields(self.__class__):
            value = getattr(self, a.name)
            if any(isinstance(value, t) for t in (int, float, str, bytes)):
                metadata[a.name] = value
        df._metadata = metadata
        return df

    def to_numpy(self):
        return [f.to_numpy() for f in self.frames]

    def project_2d(self, camera: ImmutableCamera) -> ProcessedVideo:
        # todo should have camera calibration member variables
        pv = ProcessedVideo()
        for frame3d in self.frames:
            frame = frame3d.project_2d(camera)
            pv.frames.append(frame)
        return pv

    @property
    def start_time(self):
        if self.frames:
            return self.frames[0].timestamp
        else:
            return None

    @property
    def end_time(self):
        if self.frames:
            return self.frames[-1].timestamp
        else:
            return None

    @classmethod
    def from_processed_video_2d(cls, pv2ds: typing.List[ProcessedVideo], cameras: typing.Iterable[ImmutableCamera], room="") -> 'ProcessedVideo3D':
        start_times = set(pv.start_time for pv in pv2ds)
        if len(start_times) > 1:
            raise ValueError("Passed in ProcessedVideos do not have the same start times")
        if len(cameras) != len(pv2ds):
            raise ValueError("len(cameras) != len(pv2ds)")
        o = cls(cameras=list(cameras), room=room)
        for frames in zip(*(p.frames for p in pv2ds)):
            frame3d = Frame3D.from_frames(frames, cameras)
            o.frames.append(frame3d)
        return o

    def project_2d(self, camera: ImmutableCamera) -> ProcessedVideo:
        pv = ProcessedVideo()
        for frame3d in self.frames:
            frame2d = frame3d.project_2d(camera)
            pv.frames.append(frame2d)
        return pv


#@attr.s
#@protonic(suppose_pb2.NXGraph.Link)
#class NXGraphLink:
#    source = attr.ib()
#    target = attr.ib()
#    weight = attr.ib()

@attr.s(cmp=False)
class NXGraph:
    graph = attr.ib()

    def to_proto(self):
        data = json_graph.node_link_data(self.graph)
        msg = suppose_pb2.NXGraph()
        pb = json_format.ParseDict(data, msg)
        return pb

    def to_proto_file(self, file):
        pb = self.to_proto()
        write_protobuf(file, pb)

    @classmethod
    def from_proto_file(cls, file):
        pb = load_protobuf(file, suppose_pb2.NXGraph)
        return cls.from_proto(pb)

    @classmethod
    def from_proto(cls, pb):
        d = json_format.MessageToDict(pb, including_default_value_fields=True)
        G = json_graph.node_link_graph(d)
        return cls(graph=G)

    @classmethod
    def from_dict(cls, d):
        json_graph.node_link_graph(d)
        G = json_graph.node_link_graph(d)
        return cls(graph=G)

    def to_dict(self):
        return json_graph.node_link_data(self.graph)

    def __eq__(self, other):
        # quick hack, should use nx.is_isomorphic
        return self.to_dict() == other.to_dict()

@attr.s
class Pose3DGraph:
    # TODO: move to Frame3D?
    #num_cameras_source_images = attr.ib()
    #num_2d_poses_source_images = attr.ib()
    #source_cameras = attr.ib()
    #source_images = attr.ib()
    graph: nx.Graph = attr.ib(metadata={"type": NXGraph})
    cameras:typing.List[ImmutableCamera] = attr.ib(default=attr.Factory(list), metadata={"type": ImmutableCamera})
    #state: typing.Any = attr.ib(default="")
    frames: typing.List[Frame] = attr.ib(default=attr.Factory(list), metadata={"type": Frame})

    @classmethod
    def reconstruct(cls, frames, cameras):
        graph = cls.graph_from_frames(frames, cameras)
        graph = cls.get_likely_matches(graph)
        graph = cls.get_best_matches(graph)
        return cls(graph, cameras, frames)


    @staticmethod
    def graph_from_frames(frames, cameras):
        # TODO: make camera object
        # TODO: make cameras and poses hashable
        camera_index = dict(zip((id(c) for c in cameras), itertools.count()))
        graph = nx.Graph()
        for (frame_a, camera_a), (frame_b, camera_b) in itertools.product(zip(frames, cameras), repeat=2):
            if frame_a == frame_b and camera_a == camera_b:
                continue
            pose_a_indices = dict(zip((id(p) for p in frame_a.poses), itertools.count()))
            pose_b_indices = dict(zip((id(p) for p in frame_b.poses), itertools.count()))
            for pose_a, pose_b in itertools.product(frame_a.poses, frame_b.poses):
                pose3d = Pose3D.from_2d(pose_a, pose_b, camera_a, camera_b)
                camera_a_index = camera_index[id(camera_a)]
                camera_b_index = camera_index[id(camera_b)]
                pose_a_index = pose_a_indices[id(pose_a)]
                pose_b_index = pose_b_indices[id(pose_b)]
                graph.add_edge((camera_a_index, pose_a_index), (camera_b_index, pose_b_index), pose=pose3d)
        return graph

    @staticmethod
    def get_likely_matches(graph, max_error=30):
        gg = nx.DiGraph()
        for node in graph.nodes:
            best_edges = {}    # target_camera -> best_edge
            for edge in graph.edges(node, data=True):
                _, tgt, data = edge
                weight = data['pose'].error
                # reject when no reprojection found or beyond max error threshold
                if np.isnan(weight) or weight > max_error:
                    continue
                # tgt[0] is camera the paired keypoints is located in
                # alternatively, access data['pose']['camera2']
                if tgt[0] in best_edges:
                    _, _, data2 = best_edges[tgt[0]]
                    # compare against np.nan should return false
                    if weight < data2['pose'].error:
                        best_edges[tgt[0]] = edge
                else:
                    best_edges[tgt[0]] = edge
            gg.add_edges_from(best_edges.values())
        return gg.to_undirected(reciprocal=True)

    @staticmethod
    def get_best_matches(graph, min_edges=1):
        graph_best = nx.Graph()
        best_edges = []
        #for subgraph in nx.connected_component_subgraphs(graph):
        for subgraph in (graph.subgraph(c).copy() for c in nx.connected_components(graph)):
            if subgraph.number_of_edges() >= min_edges:
                best_edge = sorted(subgraph.edges(data=True), key=lambda node: node[2]['pose'].error)[0]
                best_edges.append(best_edge)
        graph_best.add_edges_from(best_edges)
        return graph_best


@attr.s
class VideoFile:
    file: str = attr.ib()
    timestamp: str = attr.ib()
    camera: str = attr.ib()


@attr.s
class VideoIndex:
    video_files: typing.List[VideoFile] = attr.ib(default=attr.Factory(list), metadata={"type": VideoFile})

    def __attrs_post_init__(self):
        d = [attr.asdict(vf) for vf in self.video_files]
        df = pd.DataFrame.from_dict(d)
        object.__setattr__(self, "_df", df)

    def groupby_iter(self, column):
        for gr, items in sorted(self._df.groupby(column)):
            yield gr, [VideoFile(*args) for args in items[['file','timestamp','camera']].values]

    @classmethod
    def create_from_search(cls, glob_pattern='*.mp4', timestamp_regex=TIMESTAMP_REGEX):

        files = glob.glob(glob_pattern)
        vfs = []
        for file in files:
            m = re.search(timestamp_regex, file)
            if m is None:
                raise ValueError("Could not decode timestamp from file name")
            d = m.groupdict()
            timestamp_str = "{year}-{month}-{day}-{hour}-{minute}-{second}".format(**d)
            # lazy way to get validation
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
            camera = d['camera_id']
            vf = VideoFile(file=file, timestamp=timestamp, camera=camera)
            vfs.append(vf)
        o = cls(video_files=vfs)
        return o

@attr.s
class App:
    def process_2d(self, extractor, glob_pattern='*.mp4', timestamp_regex=TIMESTAMP_REGEX, display_progress=True, draw_viz=True):
        log = Logger('App')
        vi = VideoIndex.create_from_search(glob_pattern=glob_pattern, timestamp_regex=timestamp_regex)
        log.info("Files to process: {}".format(vi.video_files))
        length = len(vi.video_files)
        processed_videos = []
        files_by_timestamp = list(vi.groupby_iter('timestamp'))
        disable_tqdm = not display_progress
        with tqdm(files_by_timestamp, disable=disable_tqdm) as tqdm_files:
            for timestamp, video_files in tqdm_files:
                pvs = []
                display_timestamp = timestamp.isoformat()
                tqdm_files.set_description(display_timestamp)
                with tqdm(video_files, disable=disable_tqdm) as tqdm_video_file:
                    for video_file in tqdm_video_file:
                        display_filename = "{} : {}".format(video_file.camera, video_file.file)
                        tqdm_video_file.set_description(display_filename)
                        pv = ProcessedVideo.from_video(video_file.file,
                                                       extractor,
                                                       read_from_cache=False,
                                                       timestamp_start=timestamp,
                                                       draw_viz=draw_viz)
                        pvs.append(pv)
                processed_videos.append((timestamp, pvs))
        return processed_videos

    def process_3d(self, cameras: typing.Mapping[str, ImmutableCamera], glob_pattern='*.mp4', timestamp_regex=TIMESTAMP_REGEX, display_progress=True, output_file_pattern="ProcessedVideo3D_{}.pb"):
        log = Logger('App')
        vi = VideoIndex.create_from_search(glob_pattern=glob_pattern, timestamp_regex=TIMESTAMP_REGEX)
        log.info("Files to process: {}".format(vi.video_files))
        length = len(vi.video_files)
        processed_videos = []
        files_by_timestamp = list(vi.groupby_iter('timestamp'))

        DATETIME_FORMAT="%Y-%m-%d-%H-%M-%S"

        disable_tqdm = not display_progress
        with tqdm(files_by_timestamp, disable=disable_tqdm) as tqdm_files:
            for timestamp, video_files in tqdm_files:
                pvs = []
                display_timestamp = timestamp.isoformat()
                tqdm_files.set_description(display_timestamp)
                with tqdm(video_files, disable=disable_tqdm) as tqdm_video_file:
                    cams = []
                    for video_file in tqdm_video_file:
                        display_filename = "{} : {}".format(video_file.camera, video_file.file)
                        tqdm_video_file.set_description(display_filename)
                        pv = ProcessedVideo.from_video(video_file.file,
                                                       None,
                                                       read_from_cache=True,
                                                       write_to_cache=False,
                                                       timestamp_start=timestamp,
                                                       draw_viz=False)
                        pvs.append(pv)
                        cams.append(cameras[video_file.camera])
                    pv3d = ProcessedVideo3D.from_processed_video_2d(pv2ds=pvs, cameras=cams)

                    date_text = timestamp.strftime(DATETIME_FORMAT)
                    output_filename = output_file_pattern.format(date_text)
                    pv3d.to_proto_file(output_filename)
                    processed_videos.append((timestamp, pv3d))
        return processed_videos

@attr.s
class VideoListing:
    camera: ImmutableCamera = attr.ib()
    files: typing.List[str] = attr.ib(default=attr.Factory(list), metadata={"type": str})

    def sort_files(self):
        self.files.sort()

    @property
    def first_video_start_time(self):
        pass

    @property
    def last_video_start_time(self):
        pass

    @property
    def size(self):
        return len(self.files)


@attr.s
class Batch:
    # cameras?
    listings: typing.Mapping[str, VideoListing] = attr.ib(default=attr.Factory(dict), metadata={"type": VideoListing})

    @classmethod
    def from_search(cls: Type[T], cameras: typing.List[ImmutableCamera], glob_pattern: str) -> T:
        """ Finds videos for camera views and creates a Batch matching cameras to video files """
        files = glob.glob(glob_pattern)
        listings = {camera.name: VideoListing(camera=camera) for camera in cameras}
        for _file in files:
            file = os.path.abspath(_file)
            camera = cls._parse_file_path(file)
            listings[camera].files.append(file)
        for listing in listings.values():
            listing.sort_files()
        o = cls(listings=listings)
        return o

    @classmethod
    def _parse_file_path(cls, file) -> str:
        parts = file.split(os.sep)
        camera_name = parts[-2]
        return camera_name

    def process(self, extractor, read_from_cache=True, write_to_cache=True, cache_dir=None, cache_suffix="__proto-ProcessedVideo-cmu.pb"):
        """
        Processes this batch of videos.

        :param extractor:
        :param read_from_cache:
        :param write_to_cache:
        :param cache_dir:
        :param cache_suffix:
        :return: iterator of ProcessedVideo3D objects
        """
        """ Run pipeline to extract poses and 3d poses from videos """
        if len(set(len(l.files) for l in self.listings.values())) != 1:
            raise ValueError("Length of video files in listing is not equal")
        cameras = [l.camera for l in self.listings.values()]
        for files in zip(*(l.files for l in self.listings.values())):
            pv2ds = []
            for file in files:
                if file.endswith(".pb"):
                    pv2d = ProcessedVideo.from_proto_file(file)
                else:
                    pv2d = ProcessedVideo.from_video(file, extractor, read_from_cache=read_from_cache, write_to_cache=write_to_cache, cache_dir=cache_dir, cache_suffix=cache_suffix)
                pv2ds.append(pv2d)
            pv3d = ProcessedVideo3D.from_processed_video_2d(pv2ds=pv2ds, cameras=cameras)
            yield pv3d



@attr.s
class KeypointModel:
    reference_delta_t = attr.ib()
    reference_position_transition_error = attr.ib()
    reference_velocity_transition_error = attr.ib()
    position_observation_error = attr.ib()

@attr.s
class PoseInitializationModel:
    initial_keypoint_position_means = attr.ib()
    initial_keypoint_velocity_means = attr.ib()
    initial_keypoint_position_error = attr.ib()
    initial_keypoint_velocity_error = attr.ib()

@attr.s
class Pose3DDistribution:
    keypoint_distributions = attr.ib()
    tag = attr.ib()
    timestamp = attr.ib()

@attr.s
class Pose3DTrack:
    keypoint_model = attr.ib()
    pose_3d_distributions = attr.ib()
    track_id = attr.ib()
    num_missing_observations = attr.ib()

@attr.s
class PoseTrackingModel:
    pass

@attr.s
class Pose3DTracks:
    pass
