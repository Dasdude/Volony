import numpy as np
import carla
import cPickle as pickle
import open3d as od
import os
import utill.raw_dataset_util as rdu
import utill.geom_util as gu
from PIL import Image as PImage
from PIL import ImageDraw
import open3d as od
import cv2
import shutil
import utill.lidar_util as lu
import matplotlib.pyplot as plt
DATA_FOLDER_PATH = '../Data/Output'
PROCESSED_FOLDER_NAME = 'Processed'
META_FOLDER_NAME = 'Meta'
META_DYNAMIC_FILE_NAME = 'mdy'
META_STATIC_FILE_NAME = 'mst'
META_STATIC_SENSOR_FILE_NAME = META_STATIC_FILE_NAME+'_s'
META_STATIC_VEHICLE_FILE_NAME = META_STATIC_FILE_NAME+'_v'
META_EXT = '.p'
STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH,META_FOLDER_NAME,'meta_dataset'+META_EXT)
FRAME_CAPTURE_INTERVAL =10


SEN_RGB_LABEL = 'sensor.camera.rgb'
SEN_SS_LABEL = 'sensor.camera.semantic_segmentation'
SEN_LIDAR_LABEL = 'sensor.lidar.ray_cast'
#TODO ADD GPS DATA
SEN_GPS_LABLE = ''
IMG_EXT = '.png'
LIDAR_EXT='.ply'
PROCESSED_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH,PROCESSED_FOLDER_NAME)
MAX_FRAME =2000
def pid2vid_list(pid_list,pid2vid):
    vid_list = []
    for pid in pid_list:
        vid_list+=[pid2vid[pid]]
    return vid_list
if __name__ == '__main__':
    try:
        if os.path.exists(PROCESSED_FOLDER_PATH):shutil.rmtree(PROCESSED_FOLDER_PATH)
        os.makedirs(PROCESSED_FOLDER_PATH)
        mdy_file_path = os.path.join(DATA_FOLDER_PATH,META_FOLDER_NAME,META_DYNAMIC_FILE_NAME+META_EXT)
        mst_vehicle_file_path = os.path.join(DATA_FOLDER_PATH,META_FOLDER_NAME,META_STATIC_VEHICLE_FILE_NAME+META_EXT)
        mst_sensor_file_path = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, META_STATIC_SENSOR_FILE_NAME + META_EXT)
        mdy_dict = rdu.get_meta_dict(mdy_file_path)
        mst_s_dict = rdu.get_meta_dict(mst_sensor_file_path)
        mst_v_dict = rdu.get_meta_dict(mst_vehicle_file_path)
        recorded_frames = np.sort(mdy_dict.keys())
        sensor_types = rdu.get_available_sensor_types(mst_s_dict)
        id_vehicle_with_sensor = mst_s_dict.keys()
        id_vehicles = mst_v_dict.keys()
        distance_matrix = np.zeros([MAX_FRAME,len(id_vehicles),len(id_vehicles)])
        observed_2d_points_matrix = np.zeros([MAX_FRAME, len(id_vehicles), len(id_vehicles)])
        observed_lidar_points_matrix = np.zeros([MAX_FRAME, len(id_vehicles), len(id_vehicles)])
        # TODO find a way to test neighbor_matrix validity
        #construct_neighbor_graph
        vid_to_idx_map = {}

        for idx,v in enumerate(id_vehicles):
            vid_to_idx_map[v] = idx
        for idx,f in enumerate(recorded_frames):
            if idx>MAX_FRAME-1:
                break
            frame_index = int((f - recorded_frames[0])/FRAME_CAPTURE_INTERVAL)
            print frame_index
            for ego_vid in id_vehicle_with_sensor:
                print('Frame %d Ego ID: %d'%(f,ego_vid))
                PROCESSED_FOLDER_PATH_VEHICLE =os.path.join(PROCESSED_FOLDER_PATH,str(ego_vid))
                processed_folder_path_veh = os.path.join(PROCESSED_FOLDER_PATH, str(ego_vid))
                if not os.path.exists(PROCESSED_FOLDER_PATH_VEHICLE):
                    os.makedirs(PROCESSED_FOLDER_PATH_VEHICLE)
                ego_pid = vid_to_idx_map[ego_vid]
                ego_location = rdu.mdy_get_vehicle_translation(mdy_dict,f,ego_vid)
                ego_rotation = rdu.mdy_get_vehicle_rotation(mdy_dict, f, ego_vid)
                ego_s_mst = mst_s_dict[ego_vid][SEN_LIDAR_LABEL]
                ego_sen_id = ego_s_mst['id']
                ego_mdy_dict = mdy_dict[f][ego_sen_id]
                ## LIDAR
                lidar_data = od.read_point_cloud(
                    rdu.get_sensor_data_file_path(f, SEN_LIDAR_LABEL, ego_vid, DATA_FOLDER_PATH, LIDAR_EXT))
                filtered_lidar =[]
                lidar_np = np.asarray(lidar_data.points)
                resolution_scale = 4
                lidar_cy = lu.lidar_to_cy(lidar_np, resolution_scale)
                lidar_colored = lidar_cy
                # lidar_colored = np.ones([lidar_cy.shape[0], lidar_cy.shape[1], 3]) * lidar_cy
                lidarp_cy = PImage.fromarray(np.uint8(lidar_colored), 'RGB')
                draw_lidar = ImageDraw.Draw(lidarp_cy)
                all_points = []
                for target_vid in vid_to_idx_map.keys():
                    target_pid = vid_to_idx_map[target_vid]
                    if (target_pid == ego_pid):
                        continue

                    target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, f)
                    target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
                    lidar_np = np.transpose(np.asarray(lidar_data.points))

                    [points_vtarget,vehicle_points,all_points_pc] = gu.in_v_bounding_box(ego_mdy_dict, target_mdy,target_mst, lidar_np)
                    observed_lidar_points_matrix[frame_index,ego_pid,target_pid] = points_vtarget.shape[1]
                    c = (0, 255, 0)
                    if observed_lidar_points_matrix[frame_index,ego_pid,target_pid]<1:
                        c = (255,0,0)
                    if target_vid in id_vehicle_with_sensor:
                        c = (0, 0, 255)
                    target_pid = vid_to_idx_map[target_vid]
                    target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, f)
                    target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
                    lidar_np = np.transpose(np.asarray(lidar_data.points))
                    distance = rdu.calc_dist(ego_location, target_mdy['translation'])
                    distance_matrix[frame_index, ego_pid, target_pid] = distance
                    distance_matrix[frame_index, target_pid, ego_pid] = distance
                    bb_lidar_points =gu.lidar_transform_bb_points(target_mst,target_mdy,ego_mdy_dict)
                    bb_points_cy = lu.lidar_points_on_cy_image(np.transpose(bb_lidar_points),resolution_scale)
                    rec_vals = gu.get_rect_coordinate(bb_points_cy)
                    # if distance_matrix[frame_index, ego_pid, target_pid]>200:
                    #     continue


                    captured_points = observed_lidar_points_matrix[frame_index, ego_pid, target_pid]
                    draw_lidar.text(xy=[rec_vals[2], rec_vals[3]], text='%d' % captured_points,
                                    fill=c)
                    if captured_points>0:
                        draw_lidar.text(xy=[(rec_vals[0]), rec_vals[1]],
                                        text='%d' % target_vid, fill=c)
                        draw_lidar.text(xy=[(rec_vals[0]), rec_vals[3]],
                                        text='%d' % distance, fill=c)
                    if rec_vals[2]-rec_vals[0]>200 or rec_vals[1]-rec_vals[3]>200:
                        continue
                    draw_lidar.rectangle(list(rec_vals),
                                         width=1, outline=c)
                    # lidar_cy = lidar_cy+vehicle_bb_cy
                    all_points = [points_vtarget]+[vehicle_points]+all_points
                all_points = all_points+[all_points_pc]
                print(f,ego_vid)
                file_name =os.path.join(processed_folder_path_veh, '%d_%d.png' % (f, ego_vid))
                lidarp_cy.save(file_name)
                if ego_vid == 550 and f==278260:
                    all_points = np.concatenate(all_points, axis=1)
                # all_points = np.concatenate([points, all_points], axis=1)
                    pcd = od.PointCloud()
                    pcd.points = od.Vector3dVector(np.transpose(all_points))
                    od.draw_geometries([pcd])

                print(file_name)
                # img_od_cyl = od.Image(lidar_cy.astype(np.uint8))











    finally:
        dict_to_store = {'2d_observed':observed_2d_points_matrix,'lidar_observed':observed_lidar_points_matrix,'idx_map':vid_to_idx_map,'distance_graph':distance_matrix}
        rdu.store(dict_to_store,STORE_META_FILE_PATH)
            # LIDAR

def cy_proj(xyz,res_scale):
    xyz_load = xyz
    norm = np.sqrt(np.sum(xyz_load ** 2, axis=1))
    depth_val = 255 * (norm) / 120
    depth_val = 32 * np.log2(depth_val + 1)
    thetaz = np.rad2deg(np.arcsin(xyz_load[:, 2] / norm))
    thetax = (180 * (np.int64(xyz_load[:, 0] < 0))) + np.rad2deg(np.arctan(xyz_load[:, 1] / xyz_load[:, 0]))
    theta_x = thetax + 90 - 180 % 360
    theta_z = thetaz + 90
    img_cy = np.zeros((180 * res_scale, 360 * res_scale, 1))
    img_cy[np.int64(theta_z * res_scale), np.int64(theta_x * res_scale)] = np.expand_dims(255 - depth_val, 1)
    return img_cy















