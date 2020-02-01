import os
import numpy as np
import pickle as pickle
from matplotlib import pyplot as plt
from PIL import Image as PImage




def get_image_from_rawdb(frame_number,sensor_type,vehicle_id,data_folder_path,file_extension):
    file_path = os.path.join(data_folder_path,sensor_type,str(vehicle_id),str(frame_number)+'_'+str(vehicle_id)+file_extension)
    img = PImage.open(file_path)
    return img
def get_image_from_processed(frame_number,sensor_type,vehicle_id,data_folder_path,file_extension):
    file_path = os.path.join(data_folder_path,sensor_type,str(vehicle_id),str(frame_number)+'_'+str(vehicle_id)+file_extension)
    img = PImage.open(file_path)
    return img
def get_sensor_data_file_path(frame_number,sensor_type,vehicle_id,data_folder_path,file_extension):
    file_path = os.path.join(data_folder_path, sensor_type, str(vehicle_id),
                             str(frame_number) + '_' + str(vehicle_id) + file_extension)
    return file_path
def get_lidar_data_from_rawdb(frame_number,sensor_type,vehicle_id,data_folder_path,file_extension):
    file_path = os.path.join(data_folder_path,sensor_type,str(vehicle_id),str(frame_number)+'_'+str(vehicle_id)+file_extension)
    img_np = plt.imread(file_path)
    return img_np
def get_meta_dict(meta_file_path):
    # type: (str) -> dict
    res_dict = {}
    with open(meta_file_path,'rb') as meta_file:
        while 1:
            try:
                mdy_dict = pickle.load(meta_file)
                res_dict.update(mdy_dict)
            except EOFError:
                break
    return res_dict
def get_one_sensor_mst_dict(vehicle_id,mst_dict,sensor_type_id):
    return mst_dict[vehicle_id][sensor_type_id]
def get_one_sensor_dy_dict_by_vehicle_id(vehicle_id,sensor_type_id,mst_s_dict,mdy_dict,frame_number):
    sensor_mst_dict = get_one_sensor_mst_dict(vehicle_id,mst_s_dict,sensor_type_id)
    sensor_id = sensor_mst_dict.id
    return mdy_dict[frame_number][vehicle_id][sensor_type_id]
def get_actor_dy_dict(vehicle_id,mdy_dict,frame_number):
    return mdy_dict[frame_number][vehicle_id]
def get_actor_mst_dict(id,mst_dict):
    return mst_dict[id]
def get_vehicle_sensor_id(mst_dict,vehicle_id,sensor_type_id):
    return mst_dict[vehicle_id][sensor_type_id].id
def get_one_vehicle_mst_dict(vehicle_id,mst_v_dict):
    return mst_v_dict[vehicle_id]
def get_available_sensor_types(mst_s_dict):
    return mst_s_dict[mst_s_dict.keys()[0]].keys()
def mdy_get_vehicle_translation(mdy_dict,frame,vid):
    return mdy_dict[frame][vid]['translation']
def mdy_get_vehicle_rotation(mdy_dict,frame,vid):
    return mdy_dict[frame][vid]['rotation']
def mdy_get_vehicle_type_id(mdy_dict,frame,vid):
    return mdy_dict[frame][vid]['type_id']
def mst_v_get_vehicle_bounding_box(mst_v,vid):
    # type: (dict, int) -> dict
    return mst_v[vid]['bounding_box']
def calc_dist(loc_v1,loc_v2):
    # type: (dict, dict) -> float
    return np.sqrt((loc_v1['x']-loc_v2['x'])**2+(loc_v1['y']-loc_v2['y'])**2+(loc_v1['z']-loc_v2['z'])**2)
def store(graph_dict,file_path):
    with open(file_path,'wb') as graph_dict_file:
        pickle.dump(graph_dict,graph_dict_file,0)
def load_dataset_meta(file_path):
    with open(file_path, 'rb') as outfile:
        try:
            return pickle.load(outfile)
        except EOFError:
            assert('File Empty')