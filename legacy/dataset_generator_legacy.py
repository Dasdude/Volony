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
if __name__ == '__main__':
    experiment_name = ask_which_folder(ALL_EXPERIMENTS_ROOT)
    experiment_path = os.path.join(ALL_EXPERIMENTS_ROOT,experiment_name)

    # General Folder Path
    meta_folder_path = os.path.join(experiment_path,META_FOLDER_NAME)
    general_sensor_data_folder_path = os.path.join(experiment_path,SENSOR_FOLDER_NAME)
    processed_folder_path = os.path.join(experiment_path,PROCESSED_FOLDER_NAME)
    dataset_folder_path = os.path.join(experiment_path,DATASET_FOLDER_NAME)
    available_sensors = get_available_sensors(general_sensor_data_folder_path)
    available_sensors = [SEN_RGB_LABEL]
    if os.path.exists(dataset_folder_path): shutil.rmtree(dataset_folder_path)
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

    gpmeta_path = os.path.join(meta_folder_path, GPMETA_FILE_NAME + META_EXT)
    with open(gpmeta_path, 'rb') as gp_met_file:
        print('reading %s' % (gpmeta_path))
        vid2pid = pickle.load(gp_met_file)
    pid_list = np.sort(vid2pid.values())
    pid2vid = dict((v, k) for k, v in vid2pid.iteritems())
    for f_sensor_string in available_sensors:
        # Meta File Init
        store_meta_path_folder = os.path.join(dataset_folder_path,f_sensor_string, META_FOLDER_NAME)
        if not os.path.exists(store_meta_path_folder): os.makedirs(store_meta_path_folder)
        store_meta_path_file = os.path.join(store_meta_path_folder, DATASET_META_FILE_NAME+ META_EXT)
        if os.path.exists(store_meta_path_file): os.remove(store_meta_path_file)
        # End Meta File
        sensor_pmeta_path = os.path.join(meta_folder_path, f_sensor_string + META_EXT)
        if not os.path.exists(sensor_pmeta_path):
            continue
        with open(sensor_pmeta_path, 'rb') as meta_file:
            while 1:
                try:

                    dataset_dict_ref = pickle.load(meta_file)

                    for ego_v_id in id_vehicles_with_sensor:
                        print ego_v_id
                        dataset_dict = dataset_dict_ref.copy()
                        frame_val = dataset_dict['frame']
                        observation_matrix = dataset_dict['observation']
                        distance_matrix = dataset_dict['distance']
                        ego_pid = vid2pid[ego_v_id]
                        observation_threshold = 10
                        is_in_field_of_view = (1-np.isnan(observation_matrix[ego_pid,:]))
                        rule = ((distance_matrix[ ego_pid, :] < 300)*(is_in_field_of_view))+(observation_matrix[ego_pid,:]>observation_threshold)
                        rule = rule
                        rule = np.float32(rule)
                        rule[rule==0]=np.nan
                        # make ruled out pid bounding box to nan
                        dataset_dict['bb_2d'][ego_pid] = dataset_dict['bb_2d'][ego_pid]*np.expand_dims(rule,-1)
                        # select ruled pid
                        ego_rules_pid = pid_list[(rule > 0)]
                        coop_pid_list = []
                        # check if all the rules pid are observable
                        for r_pid in ego_rules_pid:
                            is_observed_r_pid = any(observation_matrix[:,r_pid]>observation_threshold)
                            if not is_observed_r_pid:
                                dataset_dict['bb_2d'][ego_pid,r_pid] = dataset_dict['bb_2d'][ego_pid,r_pid] * np.expand_dims(np.nan,
                                                                                                                             -1)
                                ego_rules_pid = ego_rules_pid[np.invert(ego_rules_pid==r_pid)]
                        for o_pid in ego_rules_pid:
                            coop_flag = (observation_matrix[ :, o_pid] > observation_threshold)
                            coop_pid_list = list(coop_pid_list) + list(pid_list[coop_flag])


                        for coop in coop_pid_list:
                            if coop == ego_pid:
                                continue
                            flag = np.float32(observation_matrix[coop] > observation_threshold)
                            flag[flag==0]= np.nan
                            dataset_dict['bb_2d'][coop]=dataset_dict['bb_2d'][coop]*np.expand_dims(flag,-1)
                        if len(coop_pid_list) == 0: continue
                        coop_pid_list = np.unique(coop_pid_list)
                        coop_vid_list = np.unique(pid2vid_list(coop_pid_list, pid2vid) + [ego_v_id])
                        if len(coop_pid_list) > 1:

                            for idx,store_vid in enumerate(coop_vid_list):
                                store_pid = vid2pid[store_vid]
                                store_processed_path_folder = os.path.join(dataset_folder_path, f_sensor_string,PROCESSED_FOLDER_NAME,
                                                                           str(frame_val), str(ego_v_id))

                                store_raw_path_folder = os.path.join(dataset_folder_path, f_sensor_string, RAW_FOLDER_NAME,str(frame_val), str(ego_v_id))

                                if not os.path.exists(store_raw_path_folder): os.makedirs(store_raw_path_folder)
                                store_raw_path_file = os.path.join(store_raw_path_folder, '%s%s' % (str(store_vid), IMG_EXT))
                                if not os.path.exists(store_processed_path_folder): os.makedirs(store_processed_path_folder)
                                store_processed_path_file = os.path.join(store_processed_path_folder, '%s%s' % (str(store_vid), IMG_EXT))
                                img = rdu.get_image_from_rawdb(frame_val, f_sensor_string, store_vid,
                                                               general_sensor_data_folder_path,
                                                               IMG_EXT)

                                img.save(store_raw_path_file)
                                bb_array = dataset_dict['bb_2d'][store_pid,ego_rules_pid]
                                dataset_dict['ego_pid'] = ego_pid
                                dataset_dict['ego_vid'] = ego_v_id
                                img_p = g.draw_bb(img,bb_array,pid2vid,False)
                                # img = rdu.get_image_from_processed(frame_val, f_sensor_string, store_vid,
                                #                                    processed_folder_path,
                                #                                    IMG_EXT)
                                img_p.save(store_processed_path_file)
                            with open(store_meta_path_file, 'ab') as ds_meta_file:
                                pickle.dump(dataset_dict, ds_meta_file, 0)
                                # with open('dict.csv', 'w') as csv_file:
                                #     writer = csv.writer(csv_file)
                                #     for key, value in dataset_dict.items():
                                #         writer.writerow([key, value])

                            # print pid2vid_list(observed_v_pid_arr, pid2vid)
                            print('Frame %d: ego: %d coop : ' % (frame_val, ego_v_id), pid2vid_list(coop_pid_list, pid2vid))
                except EOFError:
                    break











    # STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, 'meta_dataset' + META_EXT)

    print