import numpy as np
import open3d as od
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def ego_to_world_sensor(points,lidar_mdy_dict):
    # points_world = np.concatenate([points,np.random.rand(3,100)],axis=1)
    # Lidar Point
    points_ego = points.copy()
    points_ego[2, :] = -points[2, :]
    ego_translation = lidar_mdy_dict['translation']
    ego_rotation = lidar_mdy_dict['rotation']
    rot_mat = rot_dict_to_rotation(ego_rotation,yaw_bias=90)
    loc_mat = get_location_array(ego_translation)
    points_world = np.matmul(rot_mat,points_ego)
    points_world = points_world + loc_mat
    return np.asarray(points_world)
def world_to_ego_sensor(points,lidar_mdy_dict):
    # points_world = np.concatenate([points,np.random.rand(3,100)],axis=1)
    # Lidar Point

    ego_translation = lidar_mdy_dict['translation']
    ego_rotation = lidar_mdy_dict['rotation']
    rot_mat = rot_dict_to_rotation(ego_rotation,yaw_bias=90)
    loc_mat = get_location_array(ego_translation)
    points_ego = points - loc_mat
    points_ego = np.matmul(np.linalg.inv(rot_mat),points_ego)
    points_ego[2, :] = -points_ego[2, :]
    return np.asarray(points_ego)

def ego_to_world_v(points,v_mdy_dict):
    points_ego = points.copy()
    # points_ego[2, :] = -points[2, :]
    ego_translation = v_mdy_dict['translation']
    ego_rotation = v_mdy_dict['rotation']
    rot_mat = rot_dict_to_rotation(ego_rotation, yaw_bias=0)
    loc_mat = get_location_array(ego_translation)
    points_world = np.matmul(rot_mat, points_ego)
    points_world = points_world + loc_mat
    return np.asarray(points_world)
def world_to_ego(points,target_mdy_dict):
    ego_translation = target_mdy_dict['translation']
    ego_rotation = target_mdy_dict['rotation']
    rot_mat = rot_dict_to_rotation(ego_rotation, yaw_bias=0)
    loc_mat = get_location_array(ego_translation)
    points_ego = points - loc_mat
    points_ego = np.matmul(np.linalg.inv(rot_mat), points_ego)
    points_ego[2, :] = -points_ego[2, :]
    return np.asarray(points_ego)

def get_location_array(location_dict):
    v_loc_x = float(location_dict['x'])
    v_loc_y = float(location_dict['y'])
    v_loc_z = float(location_dict['z'])
    return np.array([[v_loc_x],[v_loc_y],[v_loc_z]])
# def in_v_bounding_box(lidar_mdy_v_dict,v_mdy_dict,v_mst_dict,lidar_points):
#     points_world = ego_to_world_sensor(lidar_points, lidar_mdy_v_dict)
#
#     extend = get_location_array(v_mst_dict['bounding_box']['extent'])
#     bb_loc = get_location_array(v_mst_dict['bounding_box']['loc'])
#     vehicle_pts_relative = bb_loc + (extend * (np.round((2* np.random.rand(3, 1000))-1)))
#     vehicle_pts_world = ego_to_world_v(vehicle_pts_relative,v_mdy_dict)
#     # points_world = np.concatenate([points_world,vehicle_pts_world],axis=1)
#     points_relative_target = world_to_ego(points_world,v_mdy_dict)
#     points_flag = np.all(np.abs(points_relative_target)<=(extend+1),axis = 0)
#     res_filtered = points_world[:, points_flag]
#     res_filtered[2,:] = res_filtered[2,:]
#
#     res = points_world
#     return res_filtered,vehicle_pts_world,points_world

    NotImplemented


def in_v_bounding_box(lidar_mdy_v_dict,v_mdy_dict,v_mst_dict,lidar_points):

    extend = get_location_array(v_mst_dict['bounding_box']['extent'])
    bb_loc = get_location_array(v_mst_dict['bounding_box']['loc'])
    bb_loc[2,0]=bb_loc[2,0]
    # vehicle_pts_relative = bb_loc + (extend * (np.round((2* np.random.rand(3, 1000))-1)))
    # vehicle_pts_world = ego_to_world_v(vehicle_pts_relative,v_mdy_dict)
    points_world = ego_to_world_sensor(lidar_points, lidar_mdy_v_dict)
    # points_world = np.concatenate([points_world,vehicle_pts_world],axis=1)
    points_relative_target = world_to_ego(points_world,v_mdy_dict)
    points_flag = np.all(np.abs(points_relative_target+bb_loc)<=(extend+.5),axis = 0)
    res_filtered = points_world[:, points_flag]
    res_filtered[2,:] = res_filtered[2,:]
    # vehicle_pts_world[2,:]-=2
    res = points_world
    return res_filtered
def bb_points_to_img_coord(pts,img_size,boundry_condition):
    pc= pts.transpose().copy()
    pc[:, 0] = np.int_(np.floor(pc[:, 0] * img_size/(2*boundry_condition)) + img_size/ 2)
    # PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    pc[:, 1] = np.int_(np.floor(pc[:, 1] *img_size/ (2*boundry_condition)) + img_size / 2)
    return pc.transpose()
def lidar_transform_bb_points(v_mst_dict,v_mdy_dict,lidar_mdy_dict):
    pts_world = get_bounding_box_points(v_mst_dict['bounding_box'],v_mdy_dict['translation'],v_mdy_dict['rotation'])
    pts_world = pts_world[0:3,:]
    points_lidar = world_to_ego_sensor(pts_world,lidar_mdy_dict)
    return points_lidar
def get_rect_coordinate(points):
    offset =0
    max_val = np.max(points,axis=0)
    min_val = np.min(points,axis=0)
    return np.array([min_val[0]-offset,min_val[1]-offset,max_val[0]+offset,max_val[1]+offset])
def is_bounding_box_in_window(x_min,y_min,x_max,y_max,x_max_bound,y_max_bound,x_min_bound=0,y_min_bound=0):
    f1 = x_min<x_max_bound
    f2 = x_max>x_min_bound
    f3 = y_max>y_min_bound
    f4 = y_min<y_max_bound
    return (f1 and f2 and f3 and f4)
def linear_interp(x_min,y_min,x_max,y_max,x_max_bound,y_max_bound):
    # finds the bounding box that is inside the observation window. this is for the case that bounding box is bigger than observation window
    res_y_min = y_min
    res_x_min = x_min
    res_y_max = y_max
    res_x_max = x_max
    slope = float(y_max-y_min)/float(x_max-x_min)
    bias = y_max -(slope*x_max)
    x_max_clip = min(x_max,x_max_bound)
    y_max_clip = min(y_max,y_max_bound)
    x_min_clip = max(x_min,0)
    y_min_clip = max(y_min,0)
    y_xmin = (slope*x_min_clip)+bias
    y_xmax = (slope*x_max_clip)+bias
    x_ymin = (y_min_clip-bias)/slope
    x_ymax=  (y_max_clip-bias)/slope
    if is_inbound(y_xmin,0,y_max_bound):
        res_y_min = y_xmin
        res_x_min = x_min_clip
    if is_inbound(x_ymin, 0, x_max_bound):
        res_y_min = y_min_clip
        res_x_min = x_ymin
    if is_inbound(y_xmax,0,y_max_bound):
        res_y_max = y_xmax
        res_x_max = x_max_clip
    if is_inbound(x_ymax, 0, x_max_bound):
        res_y_max = y_max_clip
        res_x_max = x_ymax
    return res_x_min,res_y_min,res_x_max,res_y_max
def is_inbound(val,min_bound,max_bound):
    if val>=min_bound and val<=max_bound:
        return True
    else:
        return False
def get_bounding_box_points(bounding_box_dict,location_dict,rotation_dict):
    bb_loc = bounding_box_dict['loc']
    bb_extent = bounding_box_dict['extent']
    bb_loc_x = float(bb_loc['x'])
    bb_loc_y = float(bb_loc['y'])
    bb_loc_z = float(bb_loc['z'])
    len_x = float(bb_extent['x'])
    len_y = float(bb_extent['y'])
    len_z = float(bb_extent['z'])
    v_loc_x = location_dict['x']
    v_loc_y = location_dict['y']
    v_loc_z = location_dict['z']
    cen_x = bb_loc_x + v_loc_x
    cen_y = bb_loc_y + v_loc_y
    cen_z = bb_loc_z + v_loc_z

    perm = np.transpose(np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3))
    perm = np.concatenate([perm, np.zeros([3, 1])], axis=1)
    center = np.repeat([[cen_x], [cen_y], [cen_z]], 9, axis=1)
    extent = np.repeat([[len_x], [len_y], [len_z]], 9, axis=1)
    extent = np.multiply(extent, perm)
    extent_rotated = rotate(extent, rotation_dict)
    points = center + extent_rotated
    points_conc = np.concatenate([points, np.ones([1, 9])], axis=0)
    return points_conc
def rot_dict_to_rotation(rotation_dict,yaw_bias = 0):
    roll = np.radians(rotation_dict['r'])
    yaw = np.radians(rotation_dict['y']+yaw_bias)
    pitch = np.radians(rotation_dict['p'])

    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_y = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    return R
def rotate(points3d,rotation_dict):
    roll = np.radians(rotation_dict['r'])
    yaw = np.radians(rotation_dict['y'])
    pitch = np.radians(rotation_dict['p'])

    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_y = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    return np.matmul(R,points3d)
def proj3x3(extrinsic,loc_vector):
    pos3d_rel = np.matmul(extrinsic, loc_vector)
    return pos3d_rel[0:3]
def proj2x3(K,points3d):
    pos2d_scaled = np.matmul(K, points3d)
    pos2d = np.array([
        pos2d_scaled[0] / pos2d_scaled[2],
        pos2d_scaled[1] / pos2d_scaled[2],
        pos2d_scaled[2]
    ])
    pos2d = np.squeeze(pos2d)
    return pos2d
def camera_project(extrinsic,K,points3d):
    pos3d = proj3x3(extrinsic,points3d)
    p2d = proj2x3(K, pos3d)
    return p2d
def camera_project_bounding_box(ego_mst_s_dict,ego_mdy_s_dict,target_dy_dict,target_mst_dict):
    fov = int(ego_mst_s_dict['attr']['fov'])
    width = int(ego_mst_s_dict['attr']['image_size_x'])
    height = int(ego_mst_s_dict['attr']['image_size_y'])
    ego_translation = ego_mdy_s_dict['translation']
    ego_rotation = ego_mdy_s_dict['rotation']
    points3d = get_bounding_box_points(target_mst_dict['bounding_box'],target_dy_dict['translation'],target_dy_dict['rotation'])
    ext = get_extrinsic(ego_translation,ego_rotation)
    K = get_intrinsic(width,height,fov)
    points2d = camera_project(ext,K,points3d)
    x_2d = width - points2d[0]
    y_2d = height - points2d[1]
    front_flag =points2d[2]<0
    return x_2d,y_2d,front_flag


def get_intrinsic(window_width,window_height,fov):
    K = np.identity(3)
    K[0, 2] = window_width/ 2.0
    K[1, 2] = window_height / 2.0
    K[0, 0] = K[1, 1] = window_width/ (2.0 * np.tan(fov * np.pi / 360.0))
    return K
def get_extrinsic(translation_dict,rotation_dict):
    roll = np.radians(rotation_dict['r'])
    yaw = np.radians(rotation_dict['y']+90)
    pitch = np.radians(rotation_dict['p']+90)
    x = translation_dict['x']
    y = translation_dict['y']
    z = translation_dict['z']
    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_y = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    a = np.concatenate((R,[[x],[y],[z]]),axis=1)
    b = np.concatenate((a,[[0,0,0,1]]),axis=0)
    return np.linalg.inv(b)