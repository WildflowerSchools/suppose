from collections import defaultdict
from math import *
from itertools import  product
from logbook import Logger
import cv2
import numpy as np
import networkx as nx
import math
from tqdm import tqdm
#from palettable.cartocolors.qualitative import Pastel_10 as COLORS
from .common import timing
from .camera import load_calibration


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
    if poses.size == 0:
        return poses
    poses1 = poses[:, :, :2]
    scores1 = poses[:, :, 2]
    poses1_undistorted = undistort_points(poses1, camera_calibration)
    p1 = np.dstack((poses1_undistorted, scores1))
    return p1

def triangulate(p1, p2, projection_matrix1, projection_matrix2):
    if p1.size == 0 or p2.size == 0:
        return np.zeros((0, 3))
    object_points_homogeneous = cv2.triangulatePoints(projection_matrix1, projection_matrix2, p1.T, p2.T)
    object_points = cv2.convertPointsFromHomogeneous(object_points_homogeneous.T)
    object_points = object_points.reshape(-1, 3)
    return object_points

def project_3d_to_2d(pts3d, camera_calibration):
    if pts3d.size == 0:
        return np.zeros((0, 2))
    return cv2.projectPoints(pts3d, camera_calibration['rotationVector'], camera_calibration['translationVector'], camera_calibration['cameraMatrix'], camera_calibration['distortionCoefficients'])[0].squeeze()

def rmse(p1, p2):
    return np.linalg.norm(p1 - p2)/np.sqrt(p1.shape[0])

def get_likely_matches(g, max_error):
    gg = nx.DiGraph()

    for node in g.nodes:
        best_edges = {}    # target_camera -> best_edge
        for edge in g.edges(node, data=True):
            _, tgt, data = edge
            if data['weight'] > max_error:
                continue
            # tgt[0] is camera the paired keypoints is located in
            # alternatively, access data['pose']['camera2']
            if tgt[0] in best_edges:
                _, _, data2 = best_edges[tgt[0]]
                if data['weight'] < data2['weight']:
                    best_edges[tgt[0]] = edge
            else:
                best_edges[tgt[0]] = edge
        gg.add_edges_from(best_edges.values())
        #for best_edge in best_edges.values():
            #gg.add_edges_from([best_edge])


    return gg.to_undirected(reciprocal=True)

def get_best_matches(graph_likely, min_edges=1):
    graph_best = nx.Graph()
    best_edges = []
    for subgraph in nx.connected_component_subgraphs(graph_likely):
        if subgraph.number_of_edges() >= min_edges:
            best_edge = sorted(subgraph.edges(data=True), key=lambda node: node[2]['weight'])[0]
            best_edges.append(best_edge)
    graph_best.add_edges_from(best_edges)
    return graph_best

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
    all_poses3d = []
    pbar = tqdm(total=len(df))
    for frame_number, (index, row) in enumerate(df.iterrows()):
        #if frame_number < 1800:
        #    continue
        #if frame_number > 1900:
        #    break
        #if frame_number % 600 == 0:
        #    print(frame_number)
        graph = nx.Graph()
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
                        "reprojection1": pp1_reprojected,
                        "reprojection2": pp2_reprojected,
                        "pts1": pp1,
                        "pts2": pp2
                    }

                    graph.add_edge(
                        (camera1_idx, idx1),
                        (camera2_idx, idx2),
                        pose=pose3d,
                        weight_inverse=-math.log(reprojection_error),
                        weight=reprojection_error
                    )
        graph_likely = get_likely_matches(graph, 15)
        graph_best = get_best_matches(graph_likely, min_edges=1)
        poses3d = []
        for src, tgt, data in graph_best.edges(data=True):
            poses3d.append(data['pose']['keypoints'])
        all_poses3d.append(poses3d)
        pbar.update(1)
    pbar.close()

    all_poses = {idx: {"poses": poses} for idx, poses in enumerate(all_poses3d)}
    df = pd.DataFrame.from_dict(all_poses, orient="index")
    df.poses.to_json("poses3d-2-edge-min.json")
    print("done!")
    #import ipdb;ipdb.set_trace()
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

