"""
Protobuf-related utilities for suppose.proto
"""
import typing
import attr
import cattr
import numpy as np
from suppose import suppose_pb2


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


@protonic(suppose_pb2.Pose2D.Keypoint2D)
@attr.s
class Keypoint2D:
    point: Vector2f = attr.ib(default=attr.Factory(Vector2f), metadata={"type": Vector2f})
    score: float = attr.ib(default=0)


@protonic(suppose_pb2.Pose2D)
@attr.s
class Pose2D:
    keypoints: typing.List[Keypoint2D] = attr.ib(default=attr.Factory(list), metadata={"type": Keypoint2D})


@protonic(suppose_pb2.Frame)
@attr.s
class Frame:
    timestamp: float = attr.ib(default=0)
    poses: typing.List[Pose2D] = attr.ib(default=attr.Factory(list), metadata={"type": Pose2D})


@protonic(suppose_pb2.ProcessedVideo)
@attr.s
class ProcessedVideo:
    camera: str = attr.ib(default="")
    width: int = attr.ib(default=0)
    height: int = attr.ib(default=0)
    file: str = attr.ib(default="")
    model: str = attr.ib(default="")
    frames: typing.List[Frame] = attr.ib(default=attr.Factory(list), metadata={"type": Frame})

