import json
import numpy as np
from suppose import proto
#import matplotlib.pyplot as plt
import os
import glob
import re
from tqdm import tqdm
from collections import OrderedDict, defaultdict
import datetime

ALPHAPOSE_ORDER_DEFAULT = OrderedDict([
    (0,  "Nose"),
    (1,  "LEye"),
    (2,  "REye"),
    (3,  "LEar"),
    (4,  "REar"),
    (5,  "LShoulder"),
    (6,  "RShoulder"),
    (7,  "LElbow"),
    (8,  "RElbow"),
    (9,  "LWrist"),
    (10, "RWrist"),
    (11, "LHip"),
    (12, "RHip"),
    (13, "LKnee"),
    (14, "RKnee"),
    (15, "LAnkle"),
    (16, "RAnkle"),
])

ALPHAPOSE_ORDER_CMU = OrderedDict([
    (0,  "Nose"),
    (1,  "Neck"),
    (2,  "RShoulder"),
    (3,  "RElbow"),
    (4,  "RWrist"),
    (5,  "LShoulder"),
    (6,  "LElbow"),
    (7,  "LWrist"),
    (8,  "RHip"),
    (9,  "RKnee"),
    (10, "RAnkle"),
    (11, "LHip"),
    (12, "LKnee"),
    (13, "LAnkle"),
    (14, "REye"),
    (15, "LEye"),
    (16, "REar"),
    (17, "LEar"),
])
ALPHAPOSE_ORDER_CMU2 = OrderedDict([(value, key) for key, value in ALPHAPOSE_ORDER_CMU.items()])

alphapose_order_default_to_cmu = [
    (index, ALPHAPOSE_ORDER_CMU2[joint]) for index, joint in ALPHAPOSE_ORDER_DEFAULT.items()
]

def load_alphapose_keypoints(keypoints):
    kps = np.array(keypoints).reshape((17, 3))
    new_kps = np.zeros((18, 3))
    for index, new_index in alphapose_order_default_to_cmu:
        if kps[index][2] <= 0.05:
            continue
        new_kps[new_index] = kps[index]
    
    neck = kps[5:7].mean(axis=0)
    new_kps[ALPHAPOSE_ORDER_CMU2['Neck']] = neck
    
    return new_kps


def convert_alphapose_to_suppose(data, camera_id, filename, start_time=datetime.datetime(2019, 6, 6, hour=15, minute=40, second=40)):
    frames = []
    poses = []
    current_image_id = None
    counter = 0
    #start_time = datetime.datetime(2019, 6, 6, hour=15, minute=40, second=40)
    pv = proto.ProcessedVideo(camera=camera_id, width=1296, height=972, model="alphapose", file=os.path.abspath(filename))
    for d in data:
        image_id = d['image_id']
        if current_image_id is None:
            current_image_id = image_id        
        elif image_id != current_image_id:
            current_image_id = image_id
            dt = datetime.timedelta(milliseconds=counter*100)
            ts = start_time + dt
            frame = proto.Frame(timestamp=ts.timestamp(), poses=poses)
            pv.frames.append(frame)


            #if poses is not None and counter % 10 == 0:
            #    frame.plot()

                #canvas = np.zeros((972, 1296, 3), dtype=np.uint8)+255
                #im = frame.draw_on_image(canvas)
                #plt.figure()
                #plt.imshow(im)
                #print(counter)
                #print(frame.timestamp)
                #break  # DEBUG
            counter += 1
            frames.append(frame)
            poses = []

        k = load_alphapose_keypoints(d['keypoints'])
        pose = proto.Pose2D.from_numpy(k)
        poses.append(pose)
    
    if pv.start_time is None:
        # no data, create dummy empty frame
        frame = proto.Frame(timestamp=start_time.timestamp())
        pv.frames.append(frame)
        
    ts = datetime.datetime.fromtimestamp(pv.start_time)
    
    #print("ts is:", ts)
    out_filename = "ProcessedVideo_{camera_id}_{year}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}.pb".format(camera_id=camera_id,
                                                                                      year=ts.year,
                                                                                      month=ts.month,
                                                                                      day=ts.day,
                                                                                      hour=ts.hour,
                                                                                      minute=ts.minute,
                                                                                      second=ts.second)
    out_path = os.path.dirname(filename)
    out_filepath = os.path.join(out_path, out_filename)
    pv.to_proto_file(out_filepath)
    #print("Wrote to file: {}".format(out_filepath))


def main():
    #FILE_SEARCH = "/Users/lue/wildflower/data/results-alphapose/raw-video/wework/2019-08-09/*-*-*-*-*/2019/08/09/19/??-??.mp4/alphapose-results.json"
    FILE_SEARCH = "/home/lue/src/AlphaPose/results-alphapose/raw-video/capucine/videos/*-*-*-*-*/2019/06/06/15/??-??.mp4/alphapose-results.json"
    files = glob.glob(FILE_SEARCH)
    files.sort()

    regex = r".*\/(?P<camera_id>\w{8}-\w{4}-\w{4}-\w{4}-\w{12})\/(?P<date>(?P<year>\d{4})\/(?P<month>\d{2})\/(?P<day>\d{2}))\/(?P<hour>\d{2})\/(?P<filename>(?P<minute>\d{2})-(?P<second>\d{2}).mp4)\/alphapose-results.json$"

    pbar = tqdm(files)

    for file in pbar:
        #print(file)
        pbar.set_description(file)
        m = re.search(regex, file)
        d = m.groupdict()
        #print(d)
        
        with open(file, 'r') as f:
            data = json.load(f)
            
        start_time = datetime.datetime(int(d['year']), int(d['month']), int(d['day']), hour=int(d['hour']), minute=int(d['minute']), second=int(d['second']))
        #print("start time:")
        #print(start_time)
        convert_alphapose_to_suppose(data, d['camera_id'], file, start_time=start_time)
        #break
    

if __name__ == "__main__":
    main()