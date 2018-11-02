import argparse
import time
import pickle
import glob
from collections import defaultdict, OrderedDict
import os
from math import *
from itertools import cycle, product
from logbook import Logger
from cvutilities.openpose_utilities import Pose2D, Pose3D, Poses2D, Poses3D
from cvutilities.openpose_utilities import num_body_parts, body_part_connectors, body_part_long_names
from cvutilities.camera_utilities import fetch_camera_calibration_data_from_local_drive_multiple_cameras
import cv2
import numpy as np
import networkx as nx
from palettable.cartocolors.qualitative import Pastel_10 as COLORS
from .common import timing
from .camera import load_calibration


#from tf_pose.estimator import TfPoseEstimator, Human, BodyPart
#from tf_pose.networks import get_graph_path, model_wh

import pandas as pd


log = Logger('pose3d')


def get_projection_matrix(calibration):
    """
    Calculates projection matrix from camera calibration.

    :param calibration: camera calibration dictionary
    :return: projection matrix
    """
    R = cv2.Rodrigues(calibration['rotationVector'])[0]
    t = calibration['translationVector']
    Rt = np.concatenate([R, t], axis=1)
    C = calibration['cameraMatrix']
    P = np.matmul(C, Rt)
    return P


def undistort_points(pts, calibration):
    camera_matrix = calibration['cameraMatrix']
    distortion_coefficients = calibration['distortionCoefficients']
    original_shape = pts.shape
    pts2 = np.ascontiguousarray(pts).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(pts2, camera_matrix, distortion_coefficients, P=camera_matrix)
    undistorted_points = undistorted_points.reshape(original_shape)
    return undistorted_points

def undistort_2d_poses(poses, camera_calibration):
    poses1 = poses[:, :, :2]
    scores1 = poses[:, :, 2]
    poses1_undistorted = undistort_points(poses1, camera_calibration)
    p1 = np.dstack((poses1_undistorted, scores1))
    return p1

def triangulate(p1, p2, projection_matrix1, projection_matrix2):
    object_points_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2, p1.T, p2.T)
    object_points = cv2.convertPointsFromHomogeneous(object_points_homogeneous.T)
    object_points = np.squeeze(object_points)
    return object_points

def project_3d_to_2d(pts3d, camera_calibration):
    return cv2.projectPoints(pts3d, camera_calibration['rotationVector'], camera_calibration['translationVector'], camera_calibration['cameraMatrix'], camera_calibration['distortionCoefficients'])[0].squeeze()

def rmse(p1, p2):
    return np.linalg.norm(p1 - p2)/np.sqrt(p1.shape[0])

@timing
def reconstruct3d(file, camera_calibration):
    log.info("3D Reconstruction from poses in multiple views")
    log.info("file: {}".format(file))
    log.info("calibration: {}".format(camera_calibration))
    cameras = defaultdict(dict)
    for c in camera_calibration:
        name, camera_file = c.split(',')
        cameras[name]['calibration'] = load_calibration(camera_file)
        cameras[name]['projection'] = get_projection_matrix(cameras[name]['calibration'])

    df = pd.read_pickle(file)
    camera_names = sorted(list(cameras.keys()))

    graph = nx.Graph()

    count = 0
    for index, row in df.iterrows():
        count += 1
        if count > 10:
            break
        for camera1_idx, camera_name1 in enumerate(camera_names):
            camera_calibration1 = cameras[camera_name1]['calibration']
            pts1 = undistort_2d_poses(row[camera_name1].poses, camera_calibration1)
            projection_matrix1 = cameras[camera_name1]['projection']
            for camera2_idx, camera_name2 in enumerate(camera_names):
                if camera_name1 == camera_name2:
                    continue
                camera_calibration2 = cameras[camera_name2]['calibration']
                pts2 = undistort_2d_poses(row[camera_name2].poses, camera_calibration2)
                projection_matrix2 = cameras[camera_name2]['projection']
                for (idx1, p1), (idx2, p2) in product(enumerate(pts1), enumerate(pts2)):
                    pts_present = np.logical_and(p1[:, 2], p2[:, 2])  # keypoints exist in both p1 and p2 if score != 0
                    pp1 = p1[pts_present][:, :2]    # drop score dimension
                    pp2 = p2[pts_present][:, :2]
                    pts3d = triangulate(pp1, pp2, projection_matrix1, projection_matrix2)
                    pp1_reprojected = project_3d_to_2d(pts3d, camera_calibration1)
                    pp2_reprojected = project_3d_to_2d(pts3d, camera_calibration2)
                    pp1_rmse = rmse(pp1, pp1_reprojected)
                    pp2_rmse = rmse(pp2, pp2_reprojected)
                    if np.isnan(pp1_rmse) or np.isnan(pp2_rmse):
                        reprojection_error = np.nan
                    else:
                        reprojection_error = max(pp1_rmse, pp2_rmse)
                    keypoints_3d = np.full([len(pts_present)]+list(pts3d.shape[1:]), np.nan)
                    keypoints_3d[pts_present] = pts3d
                    pose3d = {
                        "keypoints": keypoints_3d,
                        "reprojection_error": reprojection_error,
                        "camera1": camera1_idx,
                        "camera2": camera2_idx,
                        "pose1": idx1,
                        "pose2": idx2,
                    }
                    graph.add_edge(
                        (camera1_idx, idx1),
                        (camera2_idx, idx2),
                        pose=pose3d
                    )

    import ipdb;ipdb.set_trace()
    #import matplotlib.pyplot as plt
    #nx.draw(graph, with_labels=True)
    #plt.show()

    if False:
        all_3d_poses = Poses3D.from_poses_2d_timestep(
            poses_2d,
            camera_calibration_data_all_cameras)

        matched_3d_poses = all_3d_poses.extract_matched_poses(projection_error_threshold=10)
        #print('\nMatched pose projection errors:')
        #print(matched_3d_poses.projection_errors())

        reprojected_poses_2d = matched_3d_poses.to_poses_2d()
        #import ipdb;ipdb.set_trace()
        rvis = []
        for rposes, oimage, camera_name in zip(reprojected_poses_2d.poses, orig_images, camera_names):
            ivis = draw_pose2d(rposes, oimage.copy())
            #cv2.imshow("{}-reproject".format(camera_name), ivis)
            rvis.append(ivis)

        vis1 = stack_images(drawn_images)
        vis2 = stack_images(rvis)
        #vis3 = stack_images([vis1, vis2], canvas_width=480*3)

        # pack data
        kps_2d = poses_2d.keypoints()
        conf_2d = poses_2d.confidence_scores()
        for camera_idx, (k, c) in enumerate(zip(kps_2d, conf_2d)):
            #print(camera_idx, k, c)
            pose_list = []
            for pose, confidence in zip(k, c):
                pose_conf = np.append(pose, confidence.reshape(-1, 1), axis=1)
                pose_list.append(pose_conf)
            keypoints_2d[camera_names[camera_idx]].append(np.array(pose_list))

        # pack data
        kps_2d = reprojected_poses_2d.keypoints()
        conf_2d = reprojected_poses_2d.confidence_scores()
        for camera_idx, (k, c) in enumerate(zip(kps_2d, conf_2d)):
            #print(camera_idx, k, c)
            pose_list = []
            for pose, confidence in zip(k, c):
                pose_conf = np.append(pose, confidence.reshape(-1, 1), axis=1)
                pose_list.append(pose_conf)
            keypoints_2d_reprojected[camera_names[camera_idx]].append(np.array(pose_list))

        keypoints_3d.append(matched_3d_poses.keypoints())

