import numpy as np
import os
import shutil
import utill.raw_dataset_util as rdu
from utill import graphics as g
import csv
import cPickle as pickle

# META PATH CONSTANTS
META_FOLDER_NAME = 'Meta'
META_DYNAMIC_FILE_NAME = 'mdy'
META_STATIC_FILE_NAME = 'mst'
META_STATIC_SENSOR_FILE_NAME = META_STATIC_FILE_NAME+'_s'
META_STATIC_VEHICLE_FILE_NAME = META_STATIC_FILE_NAME+'_v'
GPMETA_FILE_NAME = 'gpmeta'
META_EXT = '.p'


# EXPERIMENT PATH CONSTANTS
ALL_EXPERIMENTS_ROOT = '../Data'
# SENSOR PATH CONSTANTS
SENSOR_FOLDER_NAME = 'Sensor'
# PROCESSED PATH CONSTANTS
PROCESSED_FOLDER_NAME = 'Processed'
# DATASET PATH CONSTANTS
DATASET_FOLDER_NAME = 'Dataset'
RAW_FOLDER_NAME = 'Raw'
DATASET_META_FILE_NAME = 'Val'
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
def pid2vid_list(pid_list,pid2vid):
    vid_list = []
    for pid in pid_list:
        vid_list+=[pid2vid[pid]]
    return vid_list
def get_rule_matrix(observation_matrix,distance_matrix,ego_pid,observation_threshold,dataset_dict):
    distance_threshold = 100
    is_in_field_of_view = (1 - np.isnan(observation_matrix[ego_pid, :]))
    task = np.float32(((distance_matrix[ego_pid, :] < distance_threshold) * (is_in_field_of_view)) + (
                observation_matrix[ego_pid, :] > observation_threshold))
    # rule[rule == 0] = np.nan
    is_observed = observation_matrix > observation_threshold
    observed_tasks = task*is_observed
    rules = np.sum(observed_tasks,axis=0)
    bb_mask = observed_tasks
    bb_mask[ego_pid,:]=rules
    bb_mask = np.expand_dims(bb_mask,-1)
    bb_mask = np.float16(bb_mask>0)
    bb_mask[bb_mask==0] = np.nan
    dataset_dict['bb_2d'] = dataset_dict['bb_2d']*bb_mask
    observers_flag = np.sum(observed_tasks,axis=1)>0
    pid_list = np.arange(task.shape[0])
    observers_pid = pid_list[observers_flag]
    observers_pid = list(set(list(observers_pid)+[ego_pid]))
    rules_mask = list(rules>0)
    return rules_mask,dataset_dict,observers_pid


if __name__ == '__main__':
    experiment_name = ask_which_folder(ALL_EXPERIMENTS_ROOT)
    experiment_path = os.path.join(ALL_EXPERIMENTS_ROOT,experiment_name)

    # General Folder Path
    meta_folder_path = os.path.join(experiment_path,META_FOLDER_NAME)
    general_sensor_data_folder_path = os.path.join(experiment_path,SENSOR_FOLDER_NAME)
    processed_folder_path = os.path.join(experiment_path,PROCESSED_FOLDER_NAME)
    dataset_folder_path = os.path.join(experiment_path,DATASET_FOLDER_NAME)
    available_sensors = get_available_sensors(general_sensor_data_folder_path)
    available_sensors = [SEN_RGB_LABEL,SEN_DEPTH_LABEL]
    if os.path.exists(dataset_folder_path): shutil.rmtree(dataset_folder_path)

    # META Data File Path
    mdy_file_path = os.path.join(meta_folder_path,META_DYNAMIC_FILE_NAME+META_EXT)
    mst_s_file_path = os.path.join(meta_folder_path, META_STATIC_SENSOR_FILE_NAME + META_EXT)
    mst_v_file_path = os.path.join(meta_folder_path, META_STATIC_VEHICLE_FILE_NAME + META_EXT)

    # Gather META Data
    mdy_dict = rdu.get_meta_dict(mdy_file_path)
    mst_s_dict = rdu.get_meta_dict(mst_s_file_path)
    mst_v_dict = rdu.get_meta_dict(mst_v_file_path)
    gpmeta_path = os.path.join(meta_folder_path, GPMETA_FILE_NAME + META_EXT)

    # General Variables
    recorded_frames = np.sort(mdy_dict.keys())
    sensor_types = rdu.get_available_sensor_types(mst_s_dict)
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()

    dataset_samples_dict = {}
    with open(gpmeta_path, 'rb') as gp_met_file:
        print('reading %s' % (gpmeta_path))
        vid2pid = pickle.load(gp_met_file)
    pid_list = np.sort(vid2pid.values())
    pid2vid = dict((v, k) for k, v in vid2pid.iteritems())
    for f_sensor_string in available_sensors:
        if not f_sensor_string in dataset_samples_dict.keys():
            dataset_samples_dict[f_sensor_string] = {}
        # Meta File Init
        store_meta_path_folder = os.path.join(dataset_folder_path,f_sensor_string, META_FOLDER_NAME)
        if not os.path.exists(store_meta_path_folder): os.makedirs(store_meta_path_folder)
        store_meta_path_file = os.path.join(store_meta_path_folder, DATASET_META_FILE_NAME+ META_EXT)
        if os.path.exists(store_meta_path_file): os.remove(store_meta_path_file)
        # End Meta File
        sensor_pmeta_path = os.path.join(meta_folder_path, f_sensor_string + META_EXT)
        if not os.path.exists(sensor_pmeta_path): continue
        with open(sensor_pmeta_path, 'rb') as meta_file:
            while 1:
                try:
                    dataset_dict_ref = pickle.load(meta_file)
                    frame_val = dataset_dict_ref['frame']
                    # if not frame_val in dataset_samples_dict[f_sensor_string].keys():
                    #     dataset_samples_dict[f_sensor_string][frame_val] = {}
                    for ego_v_id in id_vehicles_with_sensor:

                        dataset_dict = dataset_dict_ref.copy()
                        observation_matrix = dataset_dict['observation']
                        distance_matrix = dataset_dict['distance']
                        ego_pid = vid2pid[ego_v_id]
                        observation_threshold = 10
                        rule,dataset_dict,coop_pid_list = get_rule_matrix(observation_matrix,distance_matrix,ego_pid,observation_threshold,dataset_dict)
                        target_pid_list = pid_list[rule]
                        coop_vid_list = np.unique(pid2vid_list(coop_pid_list, pid2vid) + [ego_v_id])
                        dataset_samples_dict={'sensor_type':f_sensor_string,'frame':frame_val,'ego_vid':ego_v_id, 'coop': {}}
                        if len(coop_pid_list) > 1:

                            for idx,store_vid in enumerate(coop_vid_list):

                                store_pid = vid2pid[store_vid]
                                #
                                # store_processed_path_folder = os.path.join(dataset_folder_path, f_sensor_string,PROCESSED_FOLDER_NAME,
                                #                                            str(frame_val), str(ego_v_id))
                                #
                                store_raw_path_folder = os.path.join(dataset_folder_path, f_sensor_string, RAW_FOLDER_NAME,str(frame_val))
                                #
                                if not os.path.exists(store_raw_path_folder): os.makedirs(store_raw_path_folder)
                                store_raw_path_file = os.path.join(store_raw_path_folder, '%s%s' % (str(store_vid), IMG_EXT))
                                # if not os.path.exists(store_processed_path_folder): os.makedirs(store_processed_path_folder)
                                # store_processed_path_file = os.path.join(store_processed_path_folder, '%s%s' % (str(store_vid), IMG_EXT))
                                img = rdu.get_image_from_rawdb(frame_val, f_sensor_string, store_vid,
                                                               general_sensor_data_folder_path,
                                                               IMG_EXT)
                                bb_array = dataset_dict['bb_2d'][store_pid,target_pid_list]
                                bb_array = bb_array[np.bool8(1 - np.any(np.isnan(bb_array), axis=1)), :]
                                if not os.path.exists(store_raw_path_file):
                                    img.save(store_raw_path_file)
                                # img_p = g.draw_bb(img,bb_array,pid2vid,False)
                                # img_p.save(store_processed_path_file)
                                dataset_samples_dict['coop'][store_vid] = {'pid': store_pid,'bb':bb_array}
                            # dataset_dict['ego_pid'] = ego_pid
                            # dataset_dict['ego_vid'] = ego_v_id
                            # dataset_dict['coop_vid'] =coop_vid_list
                            # dataset_dict['coop_pid'] = coop_pid_list

                            with open(store_meta_path_file, 'ab') as ds_meta_file:
                                pickle.dump(dataset_samples_dict, ds_meta_file, 0)
                            print('Frame %d: ego: %d coop : ' % (frame_val, ego_v_id), pid2vid_list(coop_pid_list, pid2vid))
                except EOFError:
                    break