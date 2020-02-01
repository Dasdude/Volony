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
        distance_matrix = np.zeros([len(recorded_frames),len(id_vehicles),len(id_vehicles)])
        fov_matrix = np.zeros([len(recorded_frames), len(id_vehicles), len(id_vehicles)])
        observed_2d_points_matrix = np.zeros([len(recorded_frames), len(id_vehicles), len(id_vehicles)])
        observed_lidar_points_matrix = np.zeros([len(recorded_frames), len(id_vehicles), len(id_vehicles)])
        # TODO find a way to test neighbor_matrix validity
        #construct_neighbor_graph
        vid_to_idx_map = {}
        for idx,v in enumerate(id_vehicles):
            vid_to_idx_map[v] = idx
        for idx,f in enumerate(recorded_frames):
            frame_index = int((f - recorded_frames[0])/FRAME_CAPTURE_INTERVAL)
            print frame_index
            for ego_v_id in id_vehicle_with_sensor:
                PROCESSED_FOLDER_PATH_VEHICLE =os.path.join(PROCESSED_FOLDER_PATH,str(ego_v_id))
                if not os.path.exists(PROCESSED_FOLDER_PATH_VEHICLE):
                    os.makedirs(PROCESSED_FOLDER_PATH_VEHICLE)
                ego_pid = vid_to_idx_map[ego_v_id]
                rgb_data = rdu.get_image_from_rawdb(f,SEN_RGB_LABEL,ego_v_id,DATA_FOLDER_PATH,IMG_EXT)
                ss_data = rdu.get_image_from_rawdb(f, SEN_SS_LABEL, ego_v_id, DATA_FOLDER_PATH, IMG_EXT)

                ego_location = rdu.mdy_get_vehicle_translation(mdy_dict,f,ego_v_id)
                ego_rotation = rdu.mdy_get_vehicle_rotation(mdy_dict, f, ego_v_id)
                ego_s_mst = mst_s_dict[ego_v_id][SEN_RGB_LABEL]
                ego_sen_id = ego_s_mst['id']
                ego_mdy_dict = mdy_dict[f][ego_sen_id]
                width = int(ego_s_mst['attr']['image_size_x'])
                height = int(ego_s_mst['attr']['image_size_y'])

                draw_rgb = ImageDraw.Draw(rgb_data)
                draw_ss = ImageDraw.Draw(ss_data)
                x_min = -np.zeros([len(id_vehicles),1])*np.nan
                x_max = -np.zeros([len(id_vehicles),1])*np.nan
                y_min = -np.zeros([len(id_vehicles),1])*np.nan
                y_max = -np.zeros([len(id_vehicles),1])*np.nan
                observed_draw = np.zeros([width,height])
                for target_v_id in id_vehicles:
                    target_pid = vid_to_idx_map[target_v_id]
                    if target_v_id == ego_v_id:
                        continue
                    target_mdy =rdu.get_actor_dy_dict(target_v_id,mdy_dict,f)
                    target_mst = rdu.get_actor_mst_dict(target_v_id,mst_v_dict)
                    target_rotation = rdu.mdy_get_vehicle_rotation(mdy_dict,f,target_v_id)
                    target_location = rdu.mdy_get_vehicle_translation(mdy_dict,f,target_v_id)
                    distance = rdu.calc_dist(ego_location,target_location)
                    distance_matrix[frame_index,ego_pid,target_pid]=distance
                    distance_matrix[frame_index, target_pid, ego_pid] = distance
                    x_2d,y_2d,front_flag = gu.camera_project_bounding_box(ego_s_mst,ego_mdy_dict,target_mdy,target_mst)
                    if np.all(front_flag):
                        x_max[target_pid] = np.int(np.clip(np.max(x_2d),0,width))
                        y_max[target_pid] = np.int(np.clip(np.max(y_2d),0,height))
                        x_min[target_pid] = np.int(np.clip(np.min(x_2d),0,width))
                        y_min[target_pid] = np.int(np.clip(np.min(y_2d),0,height))

                ### LIDAR
                # lidar_data = od.read_point_cloud(
                #     rdu.get_sensor_data_file_path(f, SEN_LIDAR_LABEL, ego_v_id, DATA_FOLDER_PATH, LIDAR_EXT))
                # filtered_lidar =[]
                # for target_vid in vid_to_idx_map.keys():
                #     target_pid = vid_to_idx_map[target_vid]
                #     if distance_matrix[frame_index, ego_pid, target_pid]>200:
                #         continue
                #     target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, f)
                #     target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
                #     lidar_np = np.transpose(np.asarray(lidar_data.points))
                #     if (target_pid == ego_pid):
                #         continue
                #     points = gu.in_v_bounding_box(ego_mdy_dict, target_mdy,target_mst, lidar_np)
                #     filtered_lidar = filtered_lidar+[np.transpose(points)]
                #     observed_lidar_points_matrix[frame_index,ego_pid,target_pid] = points.shape[1]
                #
                #
                # res = np.concatenate(filtered_lidar,axis=0)
                # pcd = od.PointCloud()
                # pcd.points = od.Vector3dVector(res)
                # od.draw_geometries([pcd])
                for target_vid in vid_to_idx_map.keys():
                    target_pid = vid_to_idx_map[target_vid]
                    if (target_pid == ego_pid):
                        continue
                    observed_draw = np.zeros([height, width])
                    imshow_draw = np.zeros([height, width,3])
                    if np.isnan(x_max[target_pid]+y_max[target_pid]+x_min[target_pid]+y_min[target_pid]):
                        continue
                    target_x_max = np.int(x_max[target_pid])
                    target_y_max = np.int(y_max[target_pid])
                    target_x_min = np.int(x_min[target_pid])
                    target_y_min = np.int(y_min[target_pid])
                    all_area = np.abs(target_x_max-target_x_min)*(target_y_max-target_y_min)
                    if all_area==0:
                        continue

                    observed_draw[target_y_min:target_y_max,target_x_min:target_x_max]=1.
                    ss_array = np.array(ss_data)
                    ss_array = (ss_array[:,:,0]==10)
                    ss_array =1- ss_array
                    imshow_draw[:,:,1] = observed_draw
                    imshow_draw[:,:,0]= 1-ss_array

                    # since 10 is the label for car and 0 is for unlabled
                    observed_draw = np.clip(observed_draw - ss_array,0,1)

                    for other_vid in vid_to_idx_map.keys():
                        other_pid = vid_to_idx_map[other_vid]
                        if (other_pid == ego_pid) or (other_pid ==target_pid):
                            continue

                        if np.isnan(x_max[other_pid] + y_max[other_pid] + x_min[other_pid] + y_min[other_pid]):
                            continue
                        other_x_max = np.int(x_max[other_pid])
                        other_y_max = np.int(y_max[other_pid])
                        other_x_min = np.int(x_min[other_pid])
                        other_y_min = np.int(y_min[other_pid])
                        imshow_draw[other_y_min:other_y_max, other_x_min:other_x_max,2] =1
                        if distance_matrix[frame_index,ego_pid,other_pid]<distance_matrix[frame_index,ego_pid,target_pid]:
                            observed_draw[other_y_min:other_y_max, other_x_min:other_x_max] = 0
                    vis_sum = np.sum(observed_draw)
                    observed_2d_points_matrix[frame_index, ego_pid, target_pid] = vis_sum
                    c= (0,255,0)
                    if target_vid in id_vehicle_with_sensor:
                        c = (0,0,255)
                    if vis_sum == 0:
                        c = (255, 0, 0)
                    if distance_matrix[frame_index, target_pid, ego_pid]>80:
                        continue
                        c = (255, 255, 255)
                    distance = distance_matrix[frame_index, target_pid, ego_pid]
                    draw_rgb.rectangle([x_min[target_pid], y_min[target_pid], x_max[target_pid], y_max[target_pid]],
                                   width=2, outline=c)
                    draw_rgb.text(xy=[(target_x_max), target_y_max], text='%d' % distance,
                                  fill=c)
                    draw_rgb.text(xy=[(target_x_min), target_y_min],
                                  text='%d' % target_vid, fill=c)

                    if int(100*vis_sum/all_area)>100:
                        assert 'value greater than 100'

                rgb_data.save(os.path.join(PROCESSED_FOLDER_PATH_VEHICLE, '%d_%d.png' % (f, ego_v_id)))
        pid2vid = dict((v, k) for k, v in vid_to_idx_map.iteritems())
        pid_list = np.sort(pid2vid.keys())
        for f_idx, f in enumerate(recorded_frames):
            for ego_v_id in id_vehicle_with_sensor:
                ego_pid = vid_to_idx_map[ego_v_id]
                observed_v_pid_arr = pid_list[observed_2d_points_matrix[f_idx, ego_pid, :] > 0]
                # print np.any(observed[:,:,:]>0)

                coop_pid_list = []
                for o_pid in observed_v_pid_arr:
                    coop_pid_list = coop_pid_list + [pid_list[observed_2d_points_matrix[f_idx, :, o_pid] > 0]]

                if len(coop_pid_list) == 0: continue
                coop_pid_list = np.unique(np.concatenate(coop_pid_list))
                print pid2vid_list(observed_v_pid_arr, pid2vid)
                print('Frame %d: ego: %d coop : %d ' % (f, ego_v_id, coop_pid_list))
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















