import numpy as np
import carla
import open3d as od
import PIL as p
import utill.raw_dataset_util as rdu
import os
import shutil
from PIL import ImageDraw
DATA_FOLDER_PATH = '../Data/Output'
PROCESSED_FOLDER_NAME = 'Processed'
META_FOLDER_NAME = 'Meta'
META_DYNAMIC_FILE_NAME = 'mdy'
META_STATIC_FILE_NAME = 'mst'
META_STATIC_SENSOR_FILE_NAME = META_STATIC_FILE_NAME+'_s'
META_STATIC_VEHICLE_FILE_NAME = META_STATIC_FILE_NAME+'_v'
META_EXT = '.p'
STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH,META_FOLDER_NAME,'meta_dataset'+META_EXT)
SEN_RGB_LABEL = 'sensor.camera.rgb'
SEN_SS_LABEL = 'sensor.camera.semantic_segmentation'
SEN_LIDAR_LABEL = 'sensor.lidar.ray_cast'
#TODO ADD GPS DATA
SEN_GPS_LABLE = ''
IMG_EXT = '.png'
LIDAR_EXT='.ply'
STORE_DATASET_PATH = os.path.join(DATA_FOLDER_PATH,'DATASET')
PROCESSED_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH,PROCESSED_FOLDER_NAME)
FRAME_CAPTURE_INTERVAL =10
def pid2vid_list(pid_list,pid2vid):
    vid_list = []
    for pid in pid_list:
        vid_list+=[pid2vid[pid]]
    return vid_list
if __name__ == '__main__':
    if os.path.exists(STORE_DATASET_PATH):
        shutil.rmtree(STORE_DATASET_PATH)
    os.makedirs(STORE_DATASET_PATH)
    mdy_file_path = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, META_DYNAMIC_FILE_NAME + META_EXT)
    mst_vehicle_file_path = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, META_STATIC_VEHICLE_FILE_NAME + META_EXT)
    mst_sensor_file_path = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, META_STATIC_SENSOR_FILE_NAME + META_EXT)
    mdy_dict = rdu.get_meta_dict(mdy_file_path)
    mst_s_dict = rdu.get_meta_dict(mst_sensor_file_path)
    mst_v_dict = rdu.get_meta_dict(mst_vehicle_file_path)
    recorded_frames = np.sort(mdy_dict.keys())
    sensor_types = rdu.get_available_sensor_types(mst_s_dict)
    id_vehicle_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    mdst = rdu.load_dataset_meta(STORE_META_FILE_PATH)
    vid2pid = mdst['idx_map']
    pid2vid = dict((v, k) for k, v in vid2pid.iteritems())
    observed = mdst['lidar_observed']
    dist_graph = mdst['distance_graph']
    pid_list = np.sort(pid2vid.keys())
    for f_idx,f in enumerate(recorded_frames):
        for ego_v_id in id_vehicle_with_sensor:
            ego_pid = vid2pid[ego_v_id]
            rule = dist_graph[f_idx,ego_pid,:]<50
            observable = observed[f_idx,ego_pid,:]>10
            observed_v_pid_arr = pid_list[(rule>0)*(observable>0)]

            coop_pid_list = []
            for o_pid in observed_v_pid_arr:
                coop_flag = (observed[f_idx, :, o_pid] > 10) * (dist_graph[f_idx,:,o_pid]<100)
                coop_pid_list = coop_pid_list+[pid_list[coop_flag]]

            if len(coop_pid_list)==0: print ('no coop');continue
            coop_pid_list = np.unique(np.concatenate(coop_pid_list))
            coop_vid_list = pid2vid_list(coop_pid_list, pid2vid)+[ego_v_id]
            if len(coop_pid_list)>1:
                for store_pid in coop_vid_list:
                    store_path_folder = os.path.join(STORE_DATASET_PATH,str(ego_v_id),str(f))
                    if not os.path.exists(store_path_folder):os.makedirs(store_path_folder)
                    store_path_file = os.path.join(store_path_folder,'%s%s'%(str(store_pid),IMG_EXT))
                    img = rdu.get_image_from_processed(f,SEN_RGB_LABEL,store_pid,PROCESSED_FOLDER_PATH,IMG_EXT)
                    img.save(store_path_file)

                print pid2vid_list(observed_v_pid_arr, pid2vid)
                print('Frame %d: ego: %d coop : ' % (f, ego_v_id), pid2vid_list(coop_pid_list,pid2vid))

