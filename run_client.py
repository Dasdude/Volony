#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import os

# try:
#     sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
import shutil
import carla

from carla import ColorConverter as cc
# from carla import sensor
from carla.libcarla import Client

import utill.sensor_utills as seu
import utill.pickle_util as pu
import argparse
import random
import time
import numpy as np
import colorsys
from PIL import Image as PImage
from PIL import ImageDraw
import utill.general_utills as gu
TOTAL_VEHICLES = 80
TOTAL_VEHICLES_WITH_SENSOR = 10
DATA_FOLDER_PATH = '../Data/MAP5EhsanRGBLidar'
SENSOR_DATA_FOLDER = os.path.join(DATA_FOLDER_PATH,'Sensor')
META_FOLDER_NAME = 'Meta'
META_DYNAMIC_FILE_NAME = 'mdy'
META_STATIC_FILE_NAME = 'mst'
META_STATIC_SENSOR_FILE_NAME = META_STATIC_FILE_NAME+'_s'
META_STATIC_VEHICLE_FILE_NAME = META_STATIC_FILE_NAME+'_v'
META_EXT = '.p'
WINDOW_HEIGHT =512
WINDOW_WIDTH = 512
SENSOR_TICK = 5
FOV = 110
FRAME_CAPTURE_INTERVAL=40
RESPAWN_INTERVAL =1000
WEATHER_CHANGE_INTERVAL =1000000
MAX_FRAMES = 8000*FRAME_CAPTURE_INTERVAL
sensors = {
            'rgb':['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            'depth_r':['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            'depth_d':['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            'depth_l':['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            'semseg_r':['sensor.camera.semantic_segmentation', cc.Raw, 'SSRaw'],
            'semseg_cs':['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'SSPallette'],
            'lidar':['sensor.lidar.ray_cast', None, 'lidar']}

def main():
    sensor_list = []
    general_sensors =[]
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=2,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=1.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    args = argparser.parse_args()

    vehicle_list =[]
    try:
        if os.path.exists(DATA_FOLDER_PATH):
            shutil.rmtree(DATA_FOLDER_PATH)
            print('folder deleted')
        client = carla.Client(args.host, args.port)  # type: Client
        client.set_timeout(6.0)
        client.start_record = False
        client.total_frames_captured = 0
        client.terminate = False


        world = client.get_world()  # type: carla.World
        sensor_bp_lib = world.get_blueprint_library()
        rgb_bp = sensor_bp_lib.find('sensor.camera.rgb');
        depth_bp = sensor_bp_lib.find('sensor.camera.depth');
        seg_bp = sensor_bp_lib.find('sensor.camera.semantic_segmentation');
        lidar_bp = sensor_bp_lib.find('sensor.lidar.ray_cast')
        collision_bp= sensor_bp_lib.find('sensor.other.collision')
        lane_dector_bp = sensor_bp_lib.find('sensor.other.lane_detector')
        rgb_bp.set_attribute('image_size_x', str(WINDOW_WIDTH));rgb_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT));rgb_bp.set_attribute('fov', str(FOV))
        seg_bp.set_attribute('image_size_x', str(WINDOW_WIDTH));
        seg_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT));
        seg_bp.set_attribute('fov', str(FOV))
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('upper_fov', '45')
        lidar_bp.set_attribute('lower_fov', '-45')
        lidar_bp.set_attribute('range', '12000')
        # lidar_bp.set_attribute('rotation_frequency', '100')
        depth_bp.set_attribute('image_size_x', str(WINDOW_WIDTH));depth_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT));depth_bp.set_attribute('fov', str(FOV))
        camera_transform = carla.Transform(carla.Location(x=2, z=1.7))
        lidar_char = gu.SensorCharacteristic(camera_transform,lidar_bp,'Lidar',None,'.ply')
        rgb_char = gu.SensorCharacteristic(camera_transform,rgb_bp,'RGB',None,'.png')
        seg_char = gu.SensorCharacteristic(camera_transform,seg_bp,'Segmentation',cc.CityScapesPalette,'.png')
        seg_raw_char = gu.SensorCharacteristic(camera_transform, seg_bp, 'Segmentation', None, '.png')
        depth_char = gu.SensorCharacteristic(camera_transform, depth_bp, 'Depth', None, '.png')
        ########### Sensor List Selection ##########
        sensor_char_list = [lidar_char,rgb_char,seg_char]
        ###########
        gu.destroy_veh_and_sensors_in_world(client)
        [vehicle_list,general_sensors] = gu.spawn_set_random_vehicle(client,n=TOTAL_VEHICLES,shuffle_points=True)
        ## Deploy Vehicles and Sensors

        gu.set_autopilot_on_list(vehicle_list)
        sensor_list = gu.attach_sensors_to_vehicle(client,vehicle_list=vehicle_list,n=TOTAL_VEHICLES_WITH_SENSOR,save_path=SENSOR_DATA_FOLDER,observation_sensor_list=sensor_char_list,frame_interval=FRAME_CAPTURE_INTERVAL)
        ## Deploy Meta Data Observer
        meta_data_rel_path = os.path.join(DATA_FOLDER_PATH,META_FOLDER_NAME)
        os.makedirs(meta_data_rel_path)
        meta_dy_path = os.path.join(meta_data_rel_path,META_DYNAMIC_FILE_NAME+META_EXT)
        meta_sensor_static_path = os.path.join(meta_data_rel_path,META_STATIC_SENSOR_FILE_NAME+META_EXT)
        meta_vehicle_static_path = os.path.join(meta_data_rel_path, META_STATIC_VEHICLE_FILE_NAME+ META_EXT)
        seu.write_sensor_static_data(sensor_list,meta_sensor_static_path)
        seu.write_vehicle_static_data(client,meta_vehicle_static_path)

        world.on_tick(lambda f: seu.on_tick_write_dynamic(f,client,meta_dy_path,FRAME_CAPTURE_INTERVAL))
        world.on_tick(lambda f: seu.on_tick_respawn(f, client, RESPAWN_INTERVAL))
        world.on_tick(lambda f: seu.on_tick_end(f, client,MAX_FRAMES))
        # world.on_tick(lambda f: seu.on_tick_change_weather(f, client, WEATHER_CHANGE_INTERVAL))
        # pu.get_sensor_static_meta_data(sensor_list[0],{})

        ## End of Code
        while not client.terminate:
            s = raw_input('type t for terminate \n')

            print s
            if s=='t':
                print('Terminated')
                client.terminate=True
            pass
    finally:
        for sen_idx, sensor in enumerate(sensor_list):  # type: (int, carla.Sensor)
            sensor.stop()
            print('\n Sensor %s Stopped ' % len(sensor.type_id))
        gu.destroy_veh_and_sensors_in_world(client)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')




