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
    for sn in sensor_names:
        for ego_id in vehicles_with_sensor_vid:
            os.makedirs(os.path.join(processed_folder_path,sn,str(ego_id)))
def process_image2d(data_path,store_path_general,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map):
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    observation_matrix = np.zeros([len(id_vehicles), len(id_vehicles)])*np.nan
    distance_matrix = np.zeros([len(id_vehicles), len(id_vehicles)])
    bounding_box = np.zeros([len(id_vehicles), len(id_vehicles), 4])*np.nan
    ss_folder_path = os.path.join(os.path.split(data_path)[0],SEN_SS_LABEL)
    for ego_vid in id_vehicles_with_sensor:
        store_path = os.path.join(store_path_general, str(ego_vid))
        ego_pid = vid_to_id_map[ego_vid]
        ego_s_mst = mst_s_dict[ego_vid][f_sensor_string]
        ego_sen_id = ego_s_mst['id']
        ego_mdy_dict = mdy_dict_frame[ego_sen_id]
        ego_location = ego_mdy_dict['translation']
        input_data_path = os.path.join(data_path, str(ego_vid), str(frame_val) + '_' + str(ego_vid) + IMG_EXT)
        im = PImage.open(input_data_path)
        ss_data_path = os.path.join(ss_folder_path, str(ego_vid), str(frame_val) + '_' + str(ego_vid) + IMG_EXT)
        im_ss = PImage.open(ss_data_path)
        im_draw = ImageDraw.Draw(im)
        x_min = -np.zeros([len(id_vehicles), 1]) * np.nan
        x_max = -np.zeros([len(id_vehicles), 1]) * np.nan
        y_min = -np.zeros([len(id_vehicles), 1]) * np.nan
        y_max = -np.zeros([len(id_vehicles), 1]) * np.nan
        width = int(ego_s_mst['attr']['image_size_x'])
        height = int(ego_s_mst['attr']['image_size_y'])
        for target_vid in vid_to_id_map.keys():
            target_pid = vid_to_id_map[target_vid]
            if (target_pid == ego_pid):
                continue
            target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
            target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
            distance = rdu.calc_dist(ego_location, target_mdy['translation'])
            distance_matrix[ego_pid, target_pid] = distance
            # distance_matrix[target_pid, ego_pid] = distance
            x_2d, y_2d, front_flag = gu.camera_project_bounding_box(ego_s_mst, ego_mdy_dict, target_mdy, target_mst)
            tmp_min_y = np.min(y_2d)
            tmp_min_x = np.min(x_2d)
            tmp_max_y = np.max(y_2d)
            tmp_max_x = np.max(x_2d)
            if np.all(front_flag) and gu.is_bounding_box_in_window(tmp_min_x,tmp_min_y,tmp_max_x,tmp_max_y,width,height):
                # tmp_x_min , tmp_y_min , tmp_x_max,tmp_y_max = gu.linear_interp(tmp_min_x,tmp_min_y,tmp_max_x,tmp_max_y,width,height)
                x_max[target_pid] = np.int(np.clip(np.max(x_2d), 0, width))
                y_max[target_pid] = np.int(np.clip(np.max(y_2d), 0, height))
                x_min[target_pid] = np.int(np.clip(np.min(x_2d), 0, width))
                y_min[target_pid] = np.int(np.clip(np.min(y_2d), 0, height))
                # if not(gu.is_inbound(tmp_max_x,0,width) and gu.is_inbound(tmp_min_y,0,height) and gu.is_inbound(tmp_min_x,0,width) and gu.is_inbound(tmp_max_y,0,height)):
                #     tmp_x_min, tmp_y_min, tmp_x_max, tmp_y_max = gu.linear_interp(tmp_min_x, tmp_min_y, tmp_max_x,
                #                                                                   tmp_max_y, width, height)
                # x_max[target_pid] = tmp_x_max
                # y_max[target_pid] = tmp_y_max
                # x_min[target_pid] = tmp_x_min
                # y_min[target_pid] = tmp_y_min
                bounding_box[ego_pid,target_pid] = [x_min[target_pid],y_min[target_pid] ,x_max[target_pid] ,y_max[target_pid]  ]
        for target_vid in vid_to_id_map.keys():
            target_pid = vid_to_id_map[target_vid]
            if (target_pid == ego_pid):
                continue
            observed_draw = np.zeros([height, width])
            imshow_draw = np.zeros([height, width, 3])
            if np.isnan(x_max[target_pid] + y_max[target_pid] + x_min[target_pid] + y_min[target_pid]):
                continue
            target_x_max = np.int(x_max[target_pid])
            target_y_max = np.int(y_max[target_pid])
            target_x_min = np.int(x_min[target_pid])
            target_y_min = np.int(y_min[target_pid])
            all_area = np.abs(target_x_max - target_x_min) * (target_y_max - target_y_min)
            if all_area == 0:
                continue

            observed_draw[target_y_min:target_y_max, target_x_min:target_x_max] = 1.
            ss_array = np.array(im_ss)
            ss_array = (ss_array[:, :, 0] == 10)
            ss_array = 1 - ss_array
            imshow_draw[:, :, 1] = observed_draw
            imshow_draw[:, :, 0] = 1 - ss_array

            # since 10 is the label for car and 0 is for unlabled
            observed_draw = np.clip(observed_draw - ss_array, 0, 1)

            for other_vid in vid_to_id_map.keys():
                other_pid = vid_to_id_map[other_vid]
                if (other_pid == ego_pid) or (other_pid == target_pid):
                    continue

                if np.isnan(x_max[other_pid] + y_max[other_pid] + x_min[other_pid] + y_min[other_pid]):
                    continue
                other_x_max = np.int(x_max[other_pid])
                other_y_max = np.int(y_max[other_pid])
                other_x_min = np.int(x_min[other_pid])
                other_y_min = np.int(y_min[other_pid])
                imshow_draw[other_y_min:other_y_max, other_x_min:other_x_max, 2] = 1
                if distance_matrix[ ego_pid, other_pid] < distance_matrix[ ego_pid, target_pid]:
                    observed_draw[other_y_min:other_y_max, other_x_min:other_x_max] = 0
            vis_sum = np.sum(observed_draw)
            observation_matrix[ ego_pid, target_pid] = vis_sum
            c = (0, 255, 0)
            if target_vid in id_vehicles_with_sensor:
                c = (0, 0, 255)
            if vis_sum == 0:
                c = (255, 0, 0)
            if distance_matrix[target_pid, ego_pid] > 200:
                c = (0, 0, 0)
            else:
                # im_draw.text(xy=[(target_x_max), target_y_min], text='%s%d' % ('d',distance_matrix[ego_pid, target_pid]),
                #              fill=c)
                # im_draw.text(xy=[(target_x_max), target_y_min+10],
                #              text='%s%d' % ('v',vis_sum), fill=c)
                im_draw.text(xy=[(target_x_max), target_y_max],
                             text='%s%d' % ('',target_vid), fill=c)

            # distance = distance_matrix[ target_pid, ego_pid]
            im_draw.rectangle([x_min[target_pid], y_min[target_pid], x_max[target_pid], y_max[target_pid]],
                               width=2, outline=c)
            # TODO observation matrix sometimes is greater than 0 meanwhile the bounding box area is zero
            if int(100 * vis_sum / all_area) > 100:
                assert 'value greater than 100'

        im.save(os.path.join(store_path, '%d_%d.png' % (frame_val, ego_vid)))
    return {'frame':frame_val,'observation':observation_matrix,'distance':distance_matrix,'bb_2d':bounding_box,'bb_3d':None}

def process_lidar_all(data_path,store_path_general,frame_val,mdy_dict,mst_s_dict,mst_v_dict,vid_to_id_map,res_scale = 4):
    id_vehicles_with_sensor = mst_s_dict.keys()
    id_vehicles = mst_v_dict.keys()
    observation_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])*np.nan
    distance_matrix = np.zeros([len(id_vehicles),len(id_vehicles)])
    bounding_box_cy = np.zeros([len(id_vehicles),len(id_vehicles),4])
    bounding_box_3d = np.zeros([len(id_vehicles),len(id_vehicles),9,3])
    world_lp = []
    for ego_vid in id_vehicles_with_sensor:
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
    bc['minZ'] = -10;
    bc['maxZ'] = 10
    world_clean = world_lp_all_ego
    world_clean = lu.removePoints(world_lp_all_ego,bc)
    img = lu.makeBVFeature(world_clean,None,Discretization=160.0/1024.0)
    img = img*255
    # a = np.int8(img)
    # a = np.array(np.zeros([500,500,3]))
    # # a[400:512,400:512,1]=1
    # a[400:500, 400:500, 1] = 255
    # a = np.int8(a)
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

    for ego_vid in id_vehicles_with_sensor:
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
        lidar_cy = lu.lidar_to_cy(lidar_np, res_scale)
        lidar_colored = lidar_cy
        lidarp_cy = PImage.fromarray(np.uint8(lidar_colored), 'RGB')
        draw_lidar = ImageDraw.Draw(lidarp_cy)
        all_points = []
        for target_vid in vid_to_id_map.keys():
            target_pid = vid_to_id_map[target_vid]
            if (target_pid == ego_pid):
                continue

            target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
            target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
            lidar_np = np.transpose(np.asarray(lidar_data.points))

            [points_vtarget, vehicle_points, all_points_pc] = gu.in_v_bounding_box(ego_mdy_dict, target_mdy, target_mst,
                                                                                   lidar_np)
            observation_matrix[ego_pid, target_pid] = points_vtarget.shape[1]
            c = (0, 255, 0)
            if observation_matrix[ego_pid, target_pid] < 1:
                c = (255, 0, 0)
            if target_vid in id_vehicles_with_sensor:
                c = (0, 0, 255)
            target_pid = vid_to_id_map[target_vid]
            target_mdy = rdu.get_actor_dy_dict(target_vid, mdy_dict, frame_val)
            target_mst = rdu.get_actor_mst_dict(target_vid, mst_v_dict)
            distance = rdu.calc_dist(ego_location, target_mdy['translation'])
            distance_matrix[ ego_pid, target_pid] = distance
            distance_matrix[target_pid, ego_pid] = distance
            bb_lidar_points = gu.lidar_transform_bb_points(target_mst, target_mdy, ego_mdy_dict)
            bb_points_cy = lu.lidar_points_on_cy_image(np.transpose(bb_lidar_points), res_scale)
            rec_vals = gu.get_rect_coordinate(bb_points_cy)
            bounding_box_3d[ego_pid,target_pid] = np.transpose(bb_lidar_points)
            bounding_box_cy[ego_pid,target_pid] = rec_vals
            draw_lidar.text(xy=[rec_vals[2], rec_vals[3]], text='%d' % observation_matrix[ego_pid, target_pid],
                            fill=c)
            if observation_matrix[ego_pid, target_pid] > 0:
                draw_lidar.text(xy=[(rec_vals[0]), rec_vals[1]],
                                text='%d' % target_vid, fill=c)
                draw_lidar.text(xy=[(rec_vals[0]), rec_vals[3]],
                                text='%d' % distance, fill=c)
            if rec_vals[2] - rec_vals[0] > 200 or rec_vals[1] - rec_vals[3] > 200:
                continue
            draw_lidar.rectangle(list(rec_vals),
                                 width=1, outline=c)
            # lidar_cy = lidar_cy+vehicle_bb_cy
            all_points = [points_vtarget] + [vehicle_points] + all_points
        all_points = all_points + [all_points_pc]
        print(frame_val, ego_vid)
        file_name = os.path.join(store_path, '%d_%d.png' % (frame_val, ego_vid))
        lidarp_cy.save(file_name)
    # if ego_vid == 550 and frame_val == 278260:
    #     all_points = np.concatenate(all_points, axis=1)
    #     # all_points = np.concatenate([points, all_points], axis=1)
    #     pcd = od.PointCloud()
    #     pcd.points = od.Vector3dVector(np.transpose(all_points))
    #     od.draw_geometries([pcd])
    # img_od_cyl = od.Image(lidar_cy.astype(np.uint8))
    return {'frame':frame_val,'observation':observation_matrix,'distance':distance_matrix,'bb_2d':bounding_box_cy,'bb_3d':bounding_box_3d}
if __name__ == '__main__':
    process_handle_dict = {SEN_RGB_LABEL: process_image2d, SEN_SS_LABEL: None, SEN_LIDAR_LABEL: process_lidar_all,SEN_DEPTH_LABEL:process_image2d}
    # experiment_name = ask_which_folder(ALL_EXPERIMENTS_ROOT)
    experiment_name = 'MAP2'
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

            sensor_processed_folder_path = os.path.join(processed_folder_path,f_sensor_string)
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