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
            weight = data['weight']
            # reject when no reprojection found or beyond max error threshold
            if np.isnan(weight) or weight > max_error:
                continue
            # tgt[0] is camera the paired keypoints is located in
            # alternatively, access data['pose']['camera2']
            if tgt[0] in best_edges:
                _, _, data2 = best_edges[tgt[0]]
                # compare against np.nan should return false
                if weight < data2['weight']:
                    best_edges[tgt[0]] = edge
            else:
                best_edges[tgt[0]] = edge
        gg.add_edges_from(best_edges.values())

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
def reconstruct3d(file, camera_calibration, output, debug_output):
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
        graph = nx.Graph()
        for camera1_idx, camera_name1 in enumerate(camera_names):
            camera_calibration1 = cameras[camera_name1]['calibration']
            poses1 = row[camera_name1].poses
            pts1 = undistort_2d_poses(poses1, camera_calibration1)
            projection_matrix1 = cameras[camera_name1]['projection']
            for camera2_idx, camera_name2 in enumerate(camera_names):
                if camera_name1 == camera_name2:
                    continue
                camera_calibration2 = cameras[camera_name2]['calibration']
                poses2 = row[camera_name2].poses
                pts2 = undistort_2d_poses(poses2, camera_calibration2)
                projection_matrix2 = cameras[camera_name2]['projection']
                for (idx1, (p1, p1_orig)), (idx2, (p2, p2_orig)) in product(enumerate(zip(pts1, poses1)), enumerate(zip(pts2, poses2))):
                    pts_present = np.logical_and(p1[:, 2], p2[:, 2])  # keypoints exist in both p1 and p2 if score != 0
                    pp1 = p1[pts_present][:, :2]    # drop score dimension
                    pp2 = p2[pts_present][:, :2]
                    p1_orig_shared = p1_orig[pts_present][:, :2]
                    p2_orig_shared = p2_orig[pts_present][:, :2]
                    pts3d = triangulate(pp1, pp2, projection_matrix1, projection_matrix2)
                    pp1_reprojected = project_3d_to_2d(pts3d, camera_calibration1)
                    pp2_reprojected = project_3d_to_2d(pts3d, camera_calibration2)
                    pp1_rmse = rmse(p1_orig_shared, pp1_reprojected)
                    pp2_rmse = rmse(p2_orig_shared, pp2_reprojected)
                    if np.isnan(pp1_rmse) or np.isnan(pp2_rmse):
                        # should just continue the loop and ignore this edge
                        reprojection_error = np.nan
                    else:
                        reprojection_error = max(pp1_rmse, pp2_rmse)
                        # should just reject 3d poses errors > threshold

                    keypoints_3d = np.full([len(pts_present)]+list(pts3d.shape[1:]), np.nan)
                    keypoints_3d[pts_present] = pts3d
                    if debug_output:
                        pose3d = {
                            "keypoints_3d": keypoints_3d,
                            "reprojection_error": reprojection_error,
                            "camera1": camera1_idx,
                            "camera2": camera2_idx,
                            "pose1": idx1,
                            "pose2": idx2,
                            "pp1_reprojected": pp1_reprojected,
                            "pp2_reprojected": pp2_reprojected,
                            "pp1": pp1,
                            "pp2": pp2,
                            "pts1": pts1,
                            "pts2": pts2,
                            "p1": p1,
                            "p2": p2,
                        }
                    else:
                        pose3d = {
                            "keypoints_3d": keypoints_3d,
                            "reprojection_error": reprojection_error,
                            "camera1": camera1_idx,
                            "camera2": camera2_idx,
                            "pose1": idx1,
                            "pose2": idx2,
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
            poses3d.append(data['pose'])
        all_poses3d.append(poses3d)
        pbar.update(1)

    pbar.close()

    log.info("Creating dataframe for serialization")
    if debug_output:
        all_poses = {idx: {"poses": np.array([p['keypoints_3d'] for p in poses]), "debug": [{k: p[k] for k in p.keys() if k != "keypoints_3d"} for p in poses]} for idx, poses in enumerate(all_poses3d)}
    else:
        all_poses = {idx: {"poses": np.array([p['keypoints_3d'] for p in poses])} for idx, poses in enumerate(all_poses3d)}

    df = pd.DataFrame.from_dict(all_poses, orient="index")
    log.info("Writing output to ")
    df.poses.to_json("{}.json".format(output))
    df.to_pickle("{}.pickle.xz".format(output), compression='xz')
