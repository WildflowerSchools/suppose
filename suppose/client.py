from __future__ import print_function

from concurrent import futures

import numpy as np
import cv2
import io
import time
import grpc
import suppose_pb2
import suppose_pb2_grpc

time_start = 0
time_end = 0

def process_response(call_future):
    global time_end
    global time_start
    time_end = time.time()
    elapsed_time = time_end - time_start
    print("elapsed: {}".format(elapsed_time))
    print(type(call_future.result()))

def get_pose(stub, image):
    #buf = io.BytesIO()
    #np.savez_compressed(buf, image=image)
    #np.savez(buf, image=image)
    width = 432
    height = 368
    img = cv2.resize(image, (height, width))
    ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    image_pb = suppose_pb2.Image(data=buf.tostring())
    frame = stub.GetPose(image_pb)
    #import ipdb;ipdb.set_trace()
    return frame

def stream_poses(stub, image, N):
    def gen():
        for _ in range(N):
            width = 432
            height = 368
            img = cv2.resize(image, (height, width))
            ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            image_pb = suppose_pb2.Image(data=buf.tostring())
            yield image_pb
    responses = stub.StreamPoses(gen())
    return responses

#def stream_poses_async(stub, image, N):
#    def gen():
#        for _ in range(N):
#            buf = io.BytesIO()
#            #np.savez_compressed(buf, image=image)
#            np.savez(buf, image=image)
#            image_pb = suppose_pb2.Image(value=[buf.getvalue()])
#            yield image_pb
#    global time_start
#    time_start = time.time()
#    call_future = stub.StreamPoses.future(gen())
#    call_future.add_done_callback(process_response)



def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50052') as channel:
        stub = suppose_pb2_grpc.PoseExtractorStub(channel)
        if False:
            image = cv2.imread("/Users/lue/wildflower/still_2018-07-04-18-23-00_camera01.jpg")
            N = 10
            t0 = time.time()
            for _ in range(N):
                frame = get_pose(stub, image)
            t1 = time.time()
            elapsed_time = t1 - t0
            fps = N/elapsed_time
            print("Elapsed time: {}".format(elapsed_time))
            print("FPS: {}".format(fps))

            N = 150
            t0 = time.time()
            responses = stream_poses(stub, image, N)
            fs = []
            for r in responses:
                fs.append(r)
            t1 = time.time()
            elapsed_time = t1 - t0
            fps = N/elapsed_time
            print("Elapsed time: {}".format(elapsed_time))
            print("FPS: {}".format(fps))
            #import ipdb;ipdb.set_trace()
        else:
            file = suppose_pb2.File(path="/home/lue/data/wf-classroom-data/camera-sandbox/camera/2018-07-04/camera01/video_2018-07-04-18-20-00.mp4")
            N = 0
            t0 = time.time()
            responses = stub.StreamPosesFromVideo(file)
            #import ipdb;ipdb.set_trace()
            for r in responses:
                N += 1
            t1 = time.time()
            elapsed_time = t1 - t0
            fps = N/elapsed_time
            print("Elapsed time: {}".format(elapsed_time))
            print("FPS: {}".format(fps))

if __name__ == '__main__':
    run()
