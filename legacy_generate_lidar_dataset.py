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

def process_lidar_all(data_path,store_path_general,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map,res_scale = 4):
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    observation_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])*np.nan
    distance_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])
    bounding_box_cy = np.zeros([len(id_vehicles),len(id_vehicles),4])
    bounding_box_3d = np.zeros([len(id_vehicles),len(id_vehicles),9,3])

    for ego_vid in id_vehicles_with_sensor:
        world_lp = []
        store_path = os.path.join(store_path_general, str(ego_vid))
        ego_s_mst = mst_s_dict[ego_vid][f_sensor_string]
        ego_sen_id = ego_s_mst['id']
        ego_mdy_dict = mdy_dict_frame[ego_sen_id]
        input_data_path = os.path.join(data_path, str(ego_vid), str(frame_val) + '_' + str(ego_vid) + LIDAR_EXT)
        lidar_data = od.read_point_cloud(input_data_path)
        lidar_np = np.asarray(lidar_data.points)

        lp_w = gu.ego_to_world_sensor(lidar_np.transpose(), ego_mdy_dict)
        file_name = os.path.join(store_path, '%d_%d.png' % (frame_val, ego_vid))
        lidarp_cy.save(file_name)
        world_lp += [lp_w]
        for coop_vid in id_vehicles_with_sensor:
            pass
        store_path = os.path.join(store_path_general,str(ego_vid))
        ego_pid = vid_to_id_map[ego_vid]
        ego_s_mst = mst_s_dict[ego_vid][f_sensor_string]
        ego_sen_id = ego_s_mst['id']
        ego_mdy_dict = mdy_dict_frame[ego_sen_id]
        ego_location = ego_mdy_dict['translation']
        # rdu.get_sensor_data_file_path(f, SEN_LIDAR_LABEL, ego_vid, data_path, LIDAR_EXT))
        input_data_path = os.path.join(data_path,str(ego_vid),str(frame_val)+'_'+str(ego_vid)+LIDAR_EXT)
        lidar_data = od.read_point_cloud(input_data_path)
        lidar_np = np.asarray(lidar_data.points)

        lp_w = gu.ego_to_world_sensor(lidar_np.transpose(),ego_mdy_dict)
        lidar_cy = lu.lidar_to_cy(lidar_np, res_scale)
        lidar_colored = lidar_cy
        lidarp_cy = PImage.fromarray(np.uint8(lidar_colored), 'RGB')
        print(frame_val, ego_vid)
        file_name = os.path.join(store_path, '%d_%d.png' % (frame_val, ego_vid))
        lidarp_cy.save(file_name)
        world_lp +=[lp_w]
    world_lp_all = np.concatenate(world_lp,axis=1)
    # axis_point = np.zeros([3,300])
    # axis_point[0,0:10] = np.arange(10)
    # axis_point[1, 10:20] = np.arange(10)
    # axis_point[2, 20:30] = np.arange(10)
    # x_axis = np.arange(-10.0,10.0,.2)
    # xy = np.meshgrid(x_axis,x_axis)
    # xyz_plane = np.stack(xy+[np.zeros_like(xy[0])],axis=0)
    # xyz_plane = xyz_plane[:].reshape([3, -1, 1]).squeeze()
    # world_lp_all = np.concatenate([world_lp_all,axis_point],axis=1)


    world_lp_all_ego = gu.world_to_ego_sensor(world_lp_all,ego_mdy_dict)
    # world_lp_all_ego = world_lp_all_ego.transpose()
    # world_lp_all_ego = np.concatenate([world_lp_all_ego,axis_point,xyz_plane],axis=1)
    world_lp_all_ego = world_lp_all_ego.transpose()
    bc = {}
    bc['minX'] = -80;
    bc['maxX'] = 80;
    bc['minY'] = -80;
    bc['maxY'] = 80
    bc['minZ'] = -3;
    bc['maxZ'] = 3
    world_clean = world_lp_all_ego
    world_clean = lu.removePoints(world_lp_all_ego,bc)
    img = lu.makeBVFeature(world_clean,None,Discretization=160.0/1024.0)
    img = img*255
    pimg = PImage.fromarray(img.astype('uint8'), 'RGB')
    # b = np.asarray(pimg)
    pimg.show()
    pcd = od.PointCloud()
    pcd.points = od.Vector3dVector(world_clean)
    od.draw_geometries([pcd])
    # if ego_vid == 550 and frame_val == 278260:
    #     all_points = np.concatenate(all_points, axis=1)
    #     # all_points = np.concatenate([points, all_points], axis=1)
    #     pcd = od.PointCloud()
    #     pcd.points = od.Vector3dVector(np.transpose(all_points))
    #     od.draw_geometries([pcd])
    # img_od_cyl = od.Image(lidar_cy.astype(np.uint8))
    return {'frame':frame_val,'observation':observation_matrix,'distance':distance_matrix,'bb_2d':bounding_box_cy,'bb_3d':bounding_box_3d}
def process_lidar(data_path,store_path_general,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map,res_scale = 4):
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    observation_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])*np.nan
    distance_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])
    bounding_box_cy = np.zeros([len(id_vehicles),len(id_vehicles),4])
    bounding_box_3d = np.zeros([len(id_vehicles),len(id_vehicles),9,3])
    bc = {}
    bc['minX'] = -40
    bc['maxX'] = 40
    bc['minY'] = -40
    bc['maxY'] = 40
    bc['minZ'] = 0
    bc['maxZ'] = 3
    for ego_vid in id_vehicles_with_sensor:
        # concat_points = []
        store_path = os.path.join(store_path_general,'V%sF%s'%(str(ego_vid),str(frame_val)))
        # ego_pid = vid_to_id_map[ego_vid]
        ego_s_mst = mst_s_dict[ego_vid][f_sensor_string]
        ego_sen_id = ego_s_mst['id']
        ego_mdy_dict = mdy_dict_frame[ego_sen_id]
        # ego_location = ego_mdy_dict['translation']
        # rdu.get_sensor_data_file_path(f, SEN_LIDAR_LABEL, ego_vid, data_path, LIDAR_EXT))
        # input_data_path = os.path.join(data_path,str(ego_vid),str(frame_val)+'_'+str(ego_vid)+LIDAR_EXT)
        # lidar_data = od.read_point_cloud(input_data_path)
        # lp_ego_ego_coord = np.asarray(lidar_data.points)
        # lp_ego_ego_coord= lu.removePoints(lp_ego_ego_coord, bc)
        # lp_ego_vehicle_world_coord = gu.ego_to_world_sensor(lp_ego_ego_coord.transpose(), ego_mdy_dict)

        all_points = []
        for coop_vid in id_vehicles_with_sensor:
            coop_pid = vid_to_id_map[coop_vid]
            coop_s_mst = mst_s_dict[coop_vid][f_sensor_string]
            coop_sen_id = coop_s_mst['id']
            coop_mdy_dict = mdy_dict_frame[coop_sen_id]
            input_data_path = os.path.join(data_path, str(coop_vid), str(frame_val) + '_' + str(coop_vid) + LIDAR_EXT)
            lidar_data = od.read_point_cloud(input_data_path)
            lp_coop_coop_cord = np.asarray(lidar_data.points)
            lp_coop_coop_cord_crop = lu.removePoints(lp_coop_coop_cord, bc)
            lp_coop_world = gu.ego_to_world_sensor(lp_coop_coop_cord.transpose(), coop_mdy_dict)
            lp_coop_ego_coord = gu.world_to_ego_sensor(lp_coop_world, ego_mdy_dict)
            lp_coop_ego_coord_crop = lu.removePoints(lp_coop_ego_coord.transpose(), bc)

                # lidar_np = np.transpose(np.asarray(lidar_data.points))


            # lp_concat_ego_coord = np.concatenate([lp_coop_ego_coord,lp_ego_ego_coord],axis=0)
            all_target_points=[]
            bb_list = []
            bb_dict={}
            for target_vid in vid_to_id_map.keys():
                target_pid = vid_to_id_map[target_vid]
                target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
                target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
                points_in_target = gu.in_v_bounding_box(coop_mdy_dict, target_mdy,
                                                                                   target_mst,
                                                                                   lp_coop_coop_cord.transpose())
                observation_matrix[coop_pid, target_pid] = points_in_target.shape[1]
                # if observation_matrix[coop_pid, target_pid]<40:
                #     continue
                bb_lidar_points_coop = gu.lidar_transform_bb_points(target_mst, target_mdy, coop_mdy_dict)

                bb_lidar_points_world= gu.ego_to_world_sensor(bb_lidar_points_coop,coop_mdy_dict)
                bb_lidar_points_ego_coord = gu.world_to_ego_sensor(bb_lidar_points_world,ego_mdy_dict)
                # bb_lidar_points_ego_coord_crop = lu.removePoints(bb_lidar_points_ego_coord.transpose(),bc).transpose()
                bev_bb_box_ego_coord = gu.bb_points_to_img_coord(bb_lidar_points_ego_coord, 1024.0, bc['maxX'])
                # if len(bev_bb_box_ego_coord) <5 :
                #     continue
                min_bb = np.int64(np.min(bev_bb_box_ego_coord, axis=1))
                max_bb = np.int64(np.max(bev_bb_box_ego_coord, axis=1))
                height = max_bb[0]-min_bb[0]
                width = max_bb[1]-min_bb[1]
                vert_center = min_bb[0]+np.int((width/2))
                horiz_center = min_bb[1]+np.int((height/2))
                if any(min_bb[:2]>1024) or any(max_bb[:2]<0):
                    continue
                # bb_list += [np.array([0,min_bb[1], min_bb[0], max_bb[1], max_bb[0]],points_in_target.shape[1])]
                bb_list += [np.array([0, min_bb[1], min_bb[0], max_bb[1], max_bb[0], points_in_target.shape[1],target_vid])]
                bb_dict[target_vid] = [np.array([0, min_bb[1], min_bb[0], max_bb[1], max_bb[0], points_in_target.shape[1],target_vid])]
                bb_dict['desc'] = 'CLS, Min Horizontal , Min Vertical, Max Horizontal, Max Vertical, Samples In BoundingBOX, VID'
                # bb_list += [np.array([0, min_bb[1], min_bb[0], max_bb[1]-min_bb[1], max_bb[0]-min_bb[0]], points_in_target.shape[1])]

                # bb_lidar_points_coop_crop = lu.removePoints(bb_lidar_points_coop.transpose(),bc).transpose()
                # bev_bb_box_coop_coord = gu.bb_points_to_img_coord(bb_lidar_points_coop_crop, 1024.0, bc['maxX'])
                # if len(bev_bb_box_coop_coord)==0:
                #     continue
                # min_bb = np.min(bev_bb_box_coop_coord,axis=1)
                # max_bb = np.max(bev_bb_box_coop_coord, axis=1)
                # bb_list_coop += [np.array([min_bb[1],min_bb[0],max_bb[1],max_bb[0]])]
                # # bb = np.reshape(np.stack(np.meshgrid(np.array([min_bb[0],max_bb[0]]),np.array([min_bb[1],max_bb[1]])),axis=-1),[4,2])
                # np.max(bev_bb_box_coop_coord, axis=1)
                all_target_points +=[points_in_target,bb_lidar_points_world]

            # all_target_points = np.concatenate(all_target_points,axis=1).transpose()
            # pcd = od.PointCloud()
            # pcd.points = od.Vector3dVector(all_target_points)
            # od.draw_geometries([pcd])
            bb_box_target = np.stack(bb_list, axis=0)
            if len(bb_list)==0 or len(lp_coop_ego_coord_crop)<5000 or np.sum(bb_box_target[:,-2])<10:
                continue
            img_coop_ego_coord = lu.makeBVFeature_multi_channel(lp_coop_ego_coord_crop, bc, Discretization=2*bc['maxX']/ 1024.0)*255
            img_coop_coop_coord = lu.makeBVFeature_multi_channel(lp_coop_coop_cord_crop, bc, Discretization=2*bc['maxX'] / 1024.0)*255
            # img_coop_ego_coord2 = lu.BV2BV(img_coop_coop_coord,1024,bc,coop_mdy_dict,ego_mdy_dict)
            pimg_annt = g.draw_bb_numpy_img(img_coop_ego_coord,bb_box_target[:,1:],None,False,color=(255,255,0))
            # pimg_annt = g.draw_bb_numpy_img(img_coop_ego_coord2, bb_box_target[:,1:], None, True, color=(255, 0, 0))
            coop_img = PImage.fromarray(img_coop_coop_coord.astype('uint8'),'RGB')
            coop_img_ego_coord = PImage.fromarray(img_coop_ego_coord.astype('uint8'),'RGB')
            if coop_vid!=ego_vid:
                vid_string = str(coop_vid)
            else:
                vid_string = 'ego'
            store_vid_path = os.path.join(store_path, vid_string)
            os.makedirs(store_vid_path)
            coop_img.save(os.path.join(store_vid_path,'img.png'))
            pimg_annt.save(os.path.join(store_vid_path,'img_annotated_coop_coord.png'))
            coop_img_ego_coord.save(os.path.join(store_vid_path, 'img_ego_coord.png'))

            ann_file_path = os.path.join(store_vid_path,'targets.pickle')
            meta_file_path = os.path.join(store_vid_path, 'locrot.pickle')
            ann_header_txt = 'CLS, Min Horizontal , Min Vertical, Max Horizontal, Max Vertical, Samples In BoundingBOX, VID'
            coop_mdy_dict['boundry_condition'] = bc
            with open(ann_file_path,'wb') as ann_file:
                pickle.dump(bb_box_target, ann_file, 0)
            with open(meta_file_path,'wb') as meta_file:
                pickle.dump(coop_mdy_dict,meta_file,0)

def np_array_to_str(val):

    for i in range(val):
        str_line = val[i]

        for j in range(val[i]):
            str
            # pimg = PImage.fromarray(img_coop.astype('uint8'), 'RGB')
            # pimg.show()
            # world_clean = world_lp_all_ego

        # for target_vid in vid_to_id_map.keys():
        #     target_pid = vid_to_id_map[target_vid]
        #     if (target_pid == ego_pid):
        #         continue
        #
        #     target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
        #     target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
        #     lp_coop_coop_cord = np.transpose(np.asarray(lidar_data.points))
        #
        #     [points_in_target, vehicle_points, all_points_pc] = gu.in_v_bounding_box(ego_mdy_dict, target_mdy, target_mst,
        #                                                                            lp_coop_coop_cord)
        #     observation_matrix[ego_pid, target_pid] = points_in_target.shape[1]
        #     c = (0, 255, 0)
        #     if observation_matrix[ego_pid, target_pid] < 1:
        #         c = (255, 0, 0)
        #     if target_vid in id_vehicles_with_sensor:
        #         c = (0, 0, 255)
        #     target_pid = vid_to_id_map[target_vid]
        #     target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
        #     target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
        #     distance = rdu.calc_dist(ego_location, target_mdy['translation'])
        #     distance_matrix[ ego_pid, target_pid] = distance
        #     distance_matrix[target_pid, ego_pid] = distance
        #     bb_lidar_points = gu.lidar_transform_bb_points(target_mst, target_mdy, ego_mdy_dict)
        #     bb_points_cy = lu.lidar_points_on_cy_image(np.transpose(bb_lidar_points), res_scale)
        #     rec_vals = gu.get_rect_coordinate(bb_points_cy)
        #     bounding_box_3d[ego_pid,target_pid] = np.transpose(bb_lidar_points)
        #     bounding_box_cy[ego_pid,target_pid] = rec_vals
        #     # draw_lidar.text(xy=[rec_vals[2], rec_vals[3]], text='%d' % observation_matrix[ego_pid, target_pid],
        #     #                 fill=c)
        #     # if observation_matrix[ego_pid, target_pid] > 0:
        #     #     draw_lidar.text(xy=[(rec_vals[0]), rec_vals[1]],
        #     #                     text='%d' % target_vid, fill=c)
        #     #     draw_lidar.text(xy=[(rec_vals[0]), rec_vals[3]],
        #     #                     text='%d' % distance, fill=c)
        #     # if rec_vals[2] - rec_vals[0] > 200 or rec_vals[1] - rec_vals[3] > 200:
        #     #     continue
        #     # draw_lidar.rectangle(list(rec_vals),
        #     #                      width=1, outline=c)
        #     # lidar_cy = lidar_cy+vehicle_bb_cy
        #     all_points = [points_in_target] + [vehicle_points] + all_points
        # all_points = all_points + [all_points_pc]
        print(frame_val, ego_vid)
        file_name = os.path.join(store_path, '%d_%d.png' % (frame_val, ego_vid))
        # lidarp_cy.save(file_name)
    # if ego_vid == 550 and frame_val == 278260:
    #     all_points = np.concatenate(all_points, axis=1)
    #     # all_points = np.concatenate([points, all_points], axis=1)
    #     pcd = od.PointCloud()
    #     pcd.points = od.Vector3dVector(np.transpose(all_points))
    #     od.draw_geometries([pcd])
    # img_od_cyl = od.Image(lidar_cy.astype(np.uint8))
    return {'frame':frame_val,'observation':observation_matrix,'distance':distance_matrix,'bb_2d':bounding_box_cy,'bb_3d':bounding_box_3d}
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