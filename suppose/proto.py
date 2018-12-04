"""
Protobuf-related utilities for suppose.proto
"""
import typing
import attr
import cattr
import numpy as np
from suppose import suppose_pb2
import pandas as pd

from suppose.pose3d import undistort_points, undistort_2d_poses, triangulate, project_3d_to_2d, rmse

from cvutilities import camera_utilities
from cvutilities.common import BODY_PART_CONNECTORS
import matplotlib.pyplot as plt

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
            obj = cls()
            for a in attr.fields(cls):
                value = getattr(pb, a.name)  # allow AttributeError for now
                # TODO: support for mapping type
                if isinstance(a.default, attr.Factory):
                    atype = a.metadata['type']
                    if a.default.factory == list:
                        things = []
                        for thing in value:
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

    def draw(self):
        all_points = self.to_numpy()[:, :2]
        camera_utilities.draw_2d_image_points(self.valid_keypoints)
        valid_keypoints_mask = self.valid_keypoints_mask

        for body_part_connector in BODY_PART_CONNECTORS:
            body_part_from_index = body_part_connector[0]
            body_part_to_index = body_part_connector[1]
            if valid_keypoints_mask[body_part_from_index] and valid_keypoints_mask[body_part_to_index]:
                pt1 = [all_points[body_part_from_index, 0], all_points[body_part_to_index, 0]]
                pt2 = [all_points[body_part_from_index, 1], all_points[body_part_to_index, 1]]
                plt.plot(pt1, pt2, 'k-', alpha=0.2)

    # Plot a pose onto a chart with the coordinate system of the origin image.
    # Calls the drawing function above, adds formating, and shows the plot
    def plot(self, image_size=(1296, 972)):
        self.draw()
        camera_utilities.format_2d_image_plot(image_size)
        plt.show()



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
    def plot(self, image_size=(1296, 972)):
        for pose in self.poses:
            pose.draw()
        camera_utilities.format_2d_image_plot(image_size)
        plt.show()



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

@protonic(suppose_pb2.Vector3f)
@attr.s
class Vector3f:
    x: float = attr.ib(default=0)
    y: float = attr.ib(default=0)
    z: float = attr.ib(default=0)

    def to_pandas(self):
        return pd.Series(self.to_dict())

    def to_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)


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
    type: int = attr.ib(default=0)
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
    def from_2d(cls: Type[T], a: Pose2D, b: Pose2D, camera_calibration1: typing.Mapping, camera_calibration2: typing.Mapping) -> T:
        valid_a = a.valid_keypoints_mask
        valid_b = b.valid_keypoints_mask
        p1_orig = a.points_array
        p2_orig = b.points_array
        pts_present = np.logical_and(valid_a, valid_b)  # keypoints exist in both p1 and p2
        p1_orig_shared = p1_orig[pts_present]
        p2_orig_shared = p2_orig[pts_present]

        p1 = undistort_points(p1_orig_shared, camera_calibration1)
        p2 = undistort_points(p2_orig_shared, camera_calibration2)

        projection_matrix1 = camera_calibration1['projection']
        projection_matrix2 = camera_calibration2['projection']
        pts3d = triangulate(p1, p2, projection_matrix1, projection_matrix2)

        p1_reprojected = project_3d_to_2d(pts3d, camera_calibration1)
        p2_reprojected = project_3d_to_2d(pts3d, camera_calibration2)
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

    def project_2d(self, camera_calibration: typing.Mapping) -> Pose2D:
        pts_reprojected = project_3d_to_2d(self.to_numpy(), camera_calibration)
        return Pose2D.from_numpy(pts_reprojected)
