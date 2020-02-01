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
    proc_path = '../../Dataset/Volony/MAP5Ehsanv3/Dataset/Data'
    dirs = os.listdir(proc_path)
    for i in range(len(dirs)):
        d = dirs[i]
        print i
        ego_path = os.path.join(proc_path,d,'ego')
        a = os.listdir(os.path.join(proc_path,d))
        # if len(a)==1:
        #     folder_to_remove = os.path.join(proc_path, d)
        #     shutil.rmtree(folder_to_remove)
        #     continue
        for vid in a:
            coop_dir = os.path.join(proc_path,d,vid)
            # os.remove(os.path.join(coop_dir, 'img_annotated_ego_coord.png'))
            if vid=='ego':
                if os.path.exists(os.path.join(coop_dir, 'img_ego_coord.png')):os.remove(os.path.join(coop_dir, 'img_ego_coord.png'))
                continue
            for dx in os.listdir(coop_dir):
                if dx.split('.')[1]=='png':
                    os.remove(os.path.join(coop_dir, dx))
            if os.path.exists(os.path.join(coop_dir, 'locrot.pickle')):os.remove(os.path.join(coop_dir,'locrot.pickle'))
            # os.remove(os.path.join(coop_dir, 'img_ego_coord.png'))
            # os.remove(os.path.join(coop_dir, 'img_annotated_ego_coord.png'))

        if not os.path.exists(ego_path):
            folder_to_remove = os.path.join(proc_path, d)
            shutil.rmtree(folder_to_remove)
            print('%s removed***************************************'%folder_to_remove)







    # STORE_META_FILE_PATH = os.path.join(DATA_FOLDER_PATH, META_FOLDER_NAME, 'meta_dataset' + META_EXT)

    print('finished')