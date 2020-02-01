import numpy as np
import os
import shutil
import utill.raw_dataset_util as rdu
import open3d as od
import utill.lidar_util as lu
import PIL.Image as PImage
from PIL import ImageDraw
from utill import geom_util as gu
import cPickle as pickle
from utill import graphics as g
SEN_RGB_LABEL = 'sensor.camera.rgb'
SEN_SS_LABEL = 'sensor.camera.semantic_segmentation'
SEN_LIDAR_LABEL = 'sensor.lidar.ray_cast'

# META PATH CONSTANTS
META_FOLDER_NAME = 'Meta'
META_DYNAMIC_FILE_NAME = 'mdy'
META_STATIC_FILE_NAME = 'mst'
META_STATIC_SENSOR_FILE_NAME = META_STATIC_FILE_NAME+'_s'
META_STATIC_VEHICLE_FILE_NAME = META_STATIC_FILE_NAME+'_v'
GPMETA_FILE_NAME = 'gpmeta'
META_EXT = '.p'

# EXPERIMENT PATH CONSTANTS
ALL_EXPERIMENTS_ROOT = '../../Dataset/Volony'
# SENSOR PATH CONSTANTS
SENSOR_FOLDER_NAME = 'Sensor'
# PROCESSED PATH CONSTANTS
PROCESSED_FOLDER_NAME = 'Dataset/Data'
# SENSORS SPECIFIC
SEN_RGB_LABEL = 'sensor.camera.rgb'
SEN_SS_LABEL = 'sensor.camera.semantic_segmentation'
SEN_LIDAR_LABEL = 'sensor.lidar.ray_cast'
SEN_DEPTH_LABEL ='sensor.camera.depth'
LIDAR_EXT = '.ply'
IMG_EXT = '.png'
def ask_which_folder(path):
    folders = os.listdir(path)
    for idx,fd in enumerate(folders):
        print('[%d] %s'%(idx,fd))
    folder_idx= raw_input('Type Folder index')
    target_folder = folders[int(folder_idx)]
    print(target_folder)
    return target_folder
def get_available_sensors(sensor_path):
    folders = os.listdir(sensor_path)
    return folders
def get_vid_idx_map(id_vehicles):
    vid_to_id = {}
    for idx, v in enumerate(id_vehicles):
        vid_to_id[v] = idx
    return vid_to_id
def create_processed_folder_structure(processed_folder_path,sensor_names,vehicles_with_sensor_vid):
    if os.path.exists(processed_folder_path):
        shutil.rmtree(processed_folder_path)
    # os.makedirs(processed_folder_path)
    # for sn in sensor_names:
    #     for ego_id in vehicles_with_sensor_vid:
    #         os.makedirs(os.path.join(processed_folder_path,sn,str(ego_id)))
def copy_no_rotate(mdy_dict):
    translation = mdy_dict['translation']
    a = {'rotation': {'p': 0.0,
  'r': 0.0,
  'y': 0.0},
 'translation': {'x': translation['x'], 'y': translation['y'], 'z': translation['z']},
 'type_id': mdy_dict['type_id']}
    return a
def get_distance(one_mdy_dict,two_mdy_dict):
    dx2 = (one_mdy_dict['translation']['x']-two_mdy_dict['translation']['x'])**2
    dy2 = (one_mdy_dict['translation']['y'] - two_mdy_dict['translation']['y']) ** 2
    dz2 = (one_mdy_dict['translation']['z'] - two_mdy_dict['translation']['z']) ** 2
    return np.sqrt(dx2+dy2+dz2)
def process_lidar(data_path,store_path_general,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map,res_scale = 4):
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    observation_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])*np.nan

    bc = {}
    bc['minX'] = -41.6
    bc['maxX'] = 41.6
    bc['minY'] = -41.6
    bc['maxY'] = 41.6
    bc['minZ'] = -3
    bc['maxZ'] = 4
    for ego_vid in id_vehicles_with_sensor:
        ego_lidar_path = os.path.join(data_path, str(ego_vid), str(frame_val) + '_' + str(ego_vid) + LIDAR_EXT)
        if not os.path.exists(ego_lidar_path):
            print(ego_lidar_path,'Does not exist')
            continue
        store_path = os.path.join(store_path_general,'V%sF%s'%(str(ego_vid),str(frame_val)))
        ego_s_mst = mst_s_dict[ego_vid][f_sensor_string]
        ego_sen_id = ego_s_mst['id']
        ego_mdy_dict = mdy_dict_frame[ego_sen_id]
        ego_mdy_dict_no_rotate = copy_no_rotate(ego_mdy_dict)
        for coop_vid in id_vehicles_with_sensor:
            coop_pid = vid_to_id_map[coop_vid]
            coop_s_mst = mst_s_dict[coop_vid][f_sensor_string]
            coop_sen_id = coop_s_mst['id']
            coop_mdy_dict = mdy_dict_frame[coop_sen_id]
            if get_distance(ego_mdy_dict,coop_mdy_dict)>np.sqrt(2)*bc['maxX']:
                continue
            coop_mdy_dict_no_rotate = copy_no_rotate(coop_mdy_dict)
            input_data_path = os.path.join(data_path, str(coop_vid), str(frame_val) + '_' + str(coop_vid) + LIDAR_EXT)
            lidar_data = od.read_point_cloud(input_data_path)
            lp_coop_coop_cord = np.asarray(lidar_data.points)


            lp_coop_world = gu.ego_to_world_sensor(lp_coop_coop_cord.transpose(), coop_mdy_dict)
            lp_coop_coop_cord_no_rotation = gu.world_to_ego_sensor(lp_coop_world,coop_mdy_dict_no_rotate)
            lp_coop_coop_cord_no_rotation_crop = lu.removePoints(lp_coop_coop_cord_no_rotation.transpose(), bc)
            lp_coop_ego_coord_no_rotation = gu.world_to_ego_sensor(lp_coop_world, ego_mdy_dict_no_rotate)
            lp_coop_ego_coord_crop = lu.removePoints(lp_coop_ego_coord_no_rotation.transpose(), bc)
            all_target_points=[];bb_list = [];bb_dict={}
            for target_vid in vid_to_id_map.keys():
                target_pid = vid_to_id_map[target_vid]
                target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
                target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
                points_in_target = gu.in_v_bounding_box(coop_mdy_dict, target_mdy,
                                                                                   target_mst,
                                                                                   lp_coop_coop_cord.transpose())
                observation_matrix[coop_pid, target_pid] = points_in_target.shape[1]
                bb_lidar_points_coop = gu.lidar_transform_bb_points(target_mst, target_mdy, coop_mdy_dict)
                bb_lidar_points_world= gu.ego_to_world_sensor(bb_lidar_points_coop,coop_mdy_dict)
                bb_lidar_points_ego_coord = gu.world_to_ego_sensor(bb_lidar_points_world,ego_mdy_dict_no_rotate)
                bev_bb_box_ego_coord = gu.bb_points_to_img_coord(bb_lidar_points_ego_coord, 1024.0, bc['maxX'])
                min_bb = np.int64(np.min(bev_bb_box_ego_coord, axis=1))
                max_bb = np.int64(np.max(bev_bb_box_ego_coord, axis=1))
                # height = max_bb[0]-min_bb[0]
                # width = max_bb[1]-min_bb[1]
                # vert_center = min_bb[0]+np.int((width/2))
                # horiz_center = min_bb[1]+np.int((height/2))
                if any(min_bb[:2]>1024) or any(max_bb[:2]<0):
                    continue
                # bb_list += [np.array([0,min_bb[1], min_bb[0], max_bb[1], max_bb[0]],points_in_target.shape[1])]
                bb_list += [np.array([0, min_bb[1], min_bb[0], max_bb[1], max_bb[0], points_in_target.shape[1],target_vid])]
                bb_dict[target_vid] = [np.array([0, min_bb[1], min_bb[0], max_bb[1], max_bb[0], points_in_target.shape[1],target_vid])]
                bb_dict['desc'] = 'CLS, Min Horizontal , Min Vertical, Max Horizontal, Max Vertical, Samples In BoundingBOX, VID'
                all_target_points +=[points_in_target,bb_lidar_points_world]

            # all_target_points = np.concatenate(all_target_points,axis=1).transpose()
            # pcd = od.PointCloud()
            # pcd.points = od.Vector3dVector(all_target_points)
            # od.draw_geometries([pcd])
            bb_box_target = np.stack(bb_list, axis=0)
            if len(bb_list)==0 or len(lp_coop_ego_coord_crop)<2000 or np.sum(bb_box_target[:,-2])<10:
                continue
            img_coop_ego_coord = lu.makeBVFeature_multi_channel(lp_coop_ego_coord_crop, bc, Discretization=2*bc['maxX']/ 1024.0)*255
            img_coop_coop_coord = lu.makeBVFeature_multi_channel(lp_coop_coop_cord_no_rotation_crop, bc, Discretization=2*bc['maxX'] / 1024.0)*255
            # img_coop_ego_coord2 = lu.BV2BV(img_coop_coop_coord,1024,bc,coop_mdy_dict_no_rotate,ego_mdy_dict_no_rotate)
            pimg_annt = g.draw_bb_numpy_img(img_coop_ego_coord,bb_box_target[:,1:],None,False,color=(255,255,0))
            # pimg_anntBV2BV = g.draw_bb_numpy_img(img_coop_ego_coord2, bb_box_target[:,1:], None, False, color=(255, 0, 0))
            coop_img = PImage.fromarray(img_coop_coop_coord.astype('uint8'),'RGB')
            coop_img_ego_coord = PImage.fromarray(img_coop_ego_coord.astype('uint8'),'RGB')
            if coop_vid!=ego_vid:
                vid_string = str(coop_vid)
            else:
                vid_string = 'ego'
            store_vid_path = os.path.join(store_path, vid_string)
            os.makedirs(store_vid_path)
            if ego_vid == coop_vid:
                coop_img.save(os.path.join(store_vid_path,'img.png'))
                # pimg_annt.save(os.path.join(store_vid_path,'img_annotated_ego_coord.png'))
                # pimg_anntBV2BV.save(os.path.join(store_vid_path, 'img_annotated_ego_coordbv2bv.png'))
                # coop_img_ego_coord.save(os.path.join(store_vid_path, 'img_ego_coord.png'))

            ann_file_path = os.path.join(store_vid_path,'targets.pickle')
            meta_file_path = os.path.join(store_vid_path, 'locrot.pickle')
            ann_header_txt = 'CLS, Min Horizontal , Min Vertical, Max Horizontal, Max Vertical, Samples In BoundingBOX, VID'
            coop_mdy_dict['boundry_condition'] = bc
            with open(ann_file_path,'wb') as ann_file:
                pickle.dump(bb_box_target, ann_file, 0)
            with open(meta_file_path,'wb') as meta_file:
                pickle.dump(coop_mdy_dict,meta_file,0)

if __name__ == '__main__':
    process_handle_dict = {SEN_RGB_LABEL: None, SEN_SS_LABEL: None, SEN_LIDAR_LABEL: process_lidar,SEN_DEPTH_LABEL:None}
    experiment_name = ask_which_folder(ALL_EXPERIMENTS_ROOT)
    # experiment_name = 'MAP2v2'
    experiment_path = os.path.join(ALL_EXPERIMENTS_ROOT,experiment_name)

    # General Folder Path
    meta_folder_path = os.path.join(experiment_path,META_FOLDER_NAME)
    general_sensor_data_folder_path = os.path.join(experiment_path,SENSOR_FOLDER_NAME)
    processed_folder_path = os.path.join(experiment_path,PROCESSED_FOLDER_NAME)
    available_sensors = get_available_sensors(general_sensor_data_folder_path)

    # META Data File Path
    mdy_file_path = os.path.join(meta_folder_path,META_DYNAMIC_FILE_NAME+META_EXT)
    mst_s_file_path = os.path.join(meta_folder_path, META_STATIC_SENSOR_FILE_NAME + META_EXT)
    mst_v_file_path = os.path.join(meta_folder_path, META_STATIC_VEHICLE_FILE_NAME + META_EXT)

    # Gather META Data
    mdy_dict = rdu.get_meta_dict(mdy_file_path)
    mst_s_dict = rdu.get_meta_dict(mst_s_file_path)
    mst_v_dict = rdu.get_meta_dict(mst_v_file_path)

    # General Variables
    recorded_frames = np.sort(mdy_dict.keys())
    sensor_types = rdu.get_available_sensor_types(mst_s_dict)
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    vid_to_id_map = get_vid_idx_map(id_vehicles)

    create_processed_folder_structure(processed_folder_path, available_sensors,id_vehicles_with_sensor)
    for f_sensor_string in available_sensors:
        sensor_pmeta_path = os.path.join(meta_folder_path, f_sensor_string + META_EXT)
        if os.path.exists(sensor_pmeta_path):
            os.remove(sensor_pmeta_path)
    gpmeta_path = os.path.join(meta_folder_path, GPMETA_FILE_NAME + META_EXT)
    if os.path.exists(gpmeta_path):
        os.remove(gpmeta_path)
    with open(gpmeta_path, 'wb') as dy_met_file:
        print('writing to %s' % (gpmeta_path))
        pickle.dump(vid_to_id_map, dy_met_file, 0)
    f_count = 0
    for frame_index,frame_val in enumerate(recorded_frames):
        frame_processed_dict = {'f_val':frame_val}
        mdy_dict_frame = mdy_dict[frame_val]
        print frame_val
        # Iterate Through Sensors
        for f_sensor_string in available_sensors:

            sensor_processed_folder_path = os.path.join(processed_folder_path)
            sensor_data_folder_path = os.path.join(general_sensor_data_folder_path,f_sensor_string)
            sensor_pmeta_path = os.path.join(meta_folder_path,f_sensor_string+META_EXT)
            if process_handle_dict[f_sensor_string] is None:
                continue
            f_handle = process_handle_dict[f_sensor_string]
            p_dict = f_handle(sensor_data_folder_path,sensor_processed_folder_path,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map)
            with open(sensor_pmeta_path, 'ab') as dy_met_file:
                pickle.dump(p_dict, dy_met_file, 0)








    # STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, 'meta_dataset' + META_EXT)

    print