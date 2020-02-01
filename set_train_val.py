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
ALL_EXPERIMENTS_ROOT = '../Data'
# SENSOR PATH CONSTANTS
SENSOR_FOLDER_NAME = 'Sensor'
# PROCESSED PATH CONSTANTS
PROCESSED_FOLDER_NAME = 'Processed'
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

if __name__ == '__main__':
    datasets_repo_path = '../../Dataset/Volony/'

    dataset_name = 'MAP5Ehsanv4'
    dataset_path =os.path.join(datasets_repo_path,dataset_name,'Dataset')
    data_path = os.path.join(dataset_path,'Data')
    train_text_file = os.path.join(dataset_path,'train.txt')
    val_text_file = os.path.join(dataset_path, 'val.txt')
    if os.path.exists(train_text_file):
        os.remove(train_text_file)
    if os.path.exists(val_text_file):
        os.remove(val_text_file)
    dirs = os.listdir(data_path)
    train_ratio = .9

    # os.makedirs(os.path.join(data_path,'train'))
    # os.makedirs(os.path.join(data_path, 'val'))
    all_folders_dict = {}
    training = []
    validation = []
    frames = []
    for dir_idx, sample_dir in enumerate(dirs):
        f= int(sample_dir.split('F')[1])
        frames+=[f]
    frames = np.unique(frames)
    frames.sort()
    train_val_cut = int(len(frames) * train_ratio)
    train_frames = frames[int(len(frames)*train_ratio)-50]
    val_frames = frames[int(len(frames)*train_ratio)]
    for dir_idx, sample_dir in enumerate(dirs):
        f= int(sample_dir.split('F')[1])
        if f<train_frames:
            training += [sample_dir]
        if f>val_frames:
            validation += [sample_dir]
    print len(training)
    print(len(validation))
    print  float(len(training))/float(len(validation)+len(training))
    # for i in range(len(dirs)):
    #     d = dirs[i]
    #     print i
    #     # ego_path = os.path.join(data_path,d,'ego')
    #     a = os.listdir(os.path.join(data_path,d))
    #     total_coop = len(a)
    #     if not total_coop in all_folders_dict.keys():
    #         all_folders_dict[total_coop] = []
    #     all_folders_dict[total_coop]+=[d]
    # for k in all_folders_dict.keys():
    #     path_list  = all_folders_dict[k]
    #     samples = len(path_list)
    #     train_idx = int(train_ratio*samples)
    #     training+=path_list[:train_idx]
    #     validation+=path_list[train_idx:]
    for d in training:
        with open(train_text_file, "a") as train_text:
            train_text.write(d + '\n')
    for d in validation:
        with open(val_text_file, "a") as val_text:
            val_text.write(d + '\n')

        # if len(a)==1:
        #     folder_to_remove = os.path.join(data_path, d)
        #     shutil.rmtree(folder_to_remove)
        #     continue
        # if not os.path.exists(ego_path):
        #     folder_to_remove = os.path.join(data_path, d)
        #     shutil.rmtree(folder_to_remove)
        #     print('%s removed***************************************'%folder_to_remove)







    # STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, 'meta_dataset' + META_EXT)

    print('finished')