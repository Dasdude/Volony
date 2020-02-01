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
import argparse
import random
import time
import numpy as np
import colorsys
from PIL import Image as PImage
from PIL import ImageDraw

DATA_FOLDER_PATH = '../Data/Output'
WINDOW_HEIGHT =200
WINDOW_WIDTH = 400
SENSOR_TICK = 5
FOV = 110
FRAME_CAPTURE_INTERVAL=10
sensors = {
            'rgb':['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            'depth_r':['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            'depth_d':['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            'depth_l':['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            'semseg_r':['sensor.camera.semantic_segmentation', cc.Raw, 'SSRaw'],
            'semseg_cs':['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'SSPallette'],
            'lidar':['sensor.lidar.ray_cast', None, 'lidar']}

def main():

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

    actor_list = []
    vehicle_list =[]
    sensor_bp_list =[]
    try:
        if os.path.exists(DATA_FOLDER_PATH):
            shutil.rmtree(DATA_FOLDER_PATH)
            print('folder deleted')
        client = carla.Client(args.host, args.port)
        client.set_timeout(6.0)
        world = client.get_world()

        sensor_bp_lib = world.get_blueprint_library().filter('sensor*')
        rgb_bp = sensor_bp_lib.find('sensor.camera.rgb')
        depth_bp = sensor_bp_lib.find('sensor.camera.depth')
        seg_bp = sensor_bp_lib.find('sensor.camera.semantic_segmentation')
        blueprints = world.get_blueprint_library().filter('vehicle.*')

        sensor_bp_list = sensor_bp_list+[sensors['rgb']]
        sensor_bp_list = sensor_bp_list + [sensors['depth_l']]
        sensor_bp_list = sensor_bp_list + [sensors['lidar']]
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        def try_spawn_random_vehicle_at_with_sensor(transform,sensor_bp_list,vehicle_index):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.spawn_actor(blueprint, transform)
            if vehicle is not None:
                # sensor = sensor_bp_lib.find(sensor_bp_list[0][0])
                # sensor.set_attribute('image_size_x', str(WINDOW_WIDTH))
                # sensor.set_attribute('image_size_y', str(WINDOW_HEIGHT))
                # sensor.set_attribute('fov', str(FOV))
                # sensor.set_attribute('enable_postprocess_effects', str(True))
                # # sensor.set_attribute('sensor_tick',str(SENSOR_TICK))
                # camera_front_transform = carla.Transform(carla.Location(x=0 + np.random.rand(), z=3))
                # sensor_inst = vehicle.get_world().spawn_actor(sensor, camera_front_transform, attach_to=vehicle)
                # sensor_inst.listen(
                #     lambda image: parse_image_classic(image, vehicle_list, vehicle_index, sensor_bp_list[0], ''))
                # actor_list.append(sensor_inst)
                #
                #
                # sensor = sensor_bp_lib.find(sensor_bp_list[1][0])
                #
                # sensor.set_attribute('image_size_x', str(WINDOW_WIDTH))
                # sensor.set_attribute('image_size_y', str(WINDOW_HEIGHT))
                # sensor.set_attribute('fov', str(FOV))
                # # sensor.set_attribute('sensor_tick', str(SENSOR_TICK))
                # camera_front_transform = carla.Transform(carla.Location(x=0 + np.random.rand(), z=3))
                # sensor_inst = vehicle.get_world().spawn_actor(sensor, camera_front_transform, attach_to=vehicle)
                # sensor_inst.listen(
                #     lambda image: parse_image_classicdepth(image, vehicle_list, vehicle_index, sensor_bp_list[1], ''))
                # actor_list.append(sensor_inst)

                sensor = sensor_bp_lib.find(sensor_bp_list[2][0])
                sensor.set_attribute('channels', '64')
                sensor.set_attribute('points_per_second', '2000000')
                sensor.set_attribute('upper_fov', '32')
                sensor.set_attribute('lower_fov', '-32')
                sensor.set_attribute('range', '12000')## Assuming meters: 120 meters
                # sensor.set_attribute('sensor_tick', str(SENSOR_TICK))
                camera_front_transform = carla.Transform(carla.Location(x=0 + np.random.rand(), z=2))
                sensor_inst = vehicle.get_world().spawn_actor(sensor, camera_front_transform, attach_to=vehicle)
                sensor_inst.listen(
                    lambda image: parse_image_classiclidar(image, vehicle_list, vehicle_index, sensor_bp_list[2], ''))
                actor_list.append(sensor_inst)
                    # for sensor_idx, sensor_opt_list in enumerate(sensor_bp_list):
                    #
                    #     sensor = sensor_bp_lib.find(sensor_opt_list[0])
                    #     sensor.set_attribute('image_size_x', str(WINDOW_WIDTH))
                    #     sensor.set_attribute('image_size_y', str(WINDOW_HEIGHT))
                    #     sensor.set_attribute('fov', str(FOV))
                    #     camera_front_transform = carla.Transform(carla.Location(x=0 + np.random.rand(), z=3))
                    #     camera_back_transform = carla.Transform(carla.Location(x=0 + np.random.rand(), z=3),
                    #                                             carla.Rotation(yaw=-180))
                    #     sensor_inst = vehicle.get_world().spawn_actor(sensor, camera_front_transform, attach_to=vehicle)
                    #     sensor_inst.listen(
                    #         lambda image: parse_image_classic(image,vehicle_list,vehicle_index,sensor_opt_list,'Back'))
                    #     actor_list.append(sensor_inst)
                    #     sensor_inst = vehicle.get_world().spawn_actor(sensor, camera_back_transform, attach_to=vehicle)
                    #     sensor_inst.listen(
                    #         lambda image: parse_image_classicdepth(image, vehicle_list, vehicle_index,sensor_opt_list,'Front'))
                    #     print('spawned %s'%sensor_opt_list[2])
                    #     actor_list.append(sensor_inst)
                actor_list.append(vehicle)
                vehicle_list.append(vehicle)
                vehicle.set_autopilot()
                print('sensor spawned %r at %s' % (vehicle.type_id, transform.location))
                return True
            return False
        def try_spawn_random_vehicle_at(transform):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:

                actor_list.append(vehicle)
                vehicle_list.append(vehicle)
                vehicle.set_autopilot()
                print('spawned with no sensor %r at %s' % (vehicle.type_id, transform.location))
                return True
            else:
                print('no sensor vehicle not spawned')
            return False,sensor_bp_list,sp_idx


        # @todo Needs to be converted to list to be shuffled.
        spawn_points = list(world.get_map().get_spawn_points())
        # random.shuffle(spawn_points)


        print('found %d spawn points.' % len(spawn_points))
        total_ns_actor = count = 0;


        for sp_idx, spawn_point in enumerate(spawn_points):
            if count <= 0:
                break
            if try_spawn_random_vehicle_at(spawn_point):
                count -= 1
        for idx in range(args.number_of_vehicles):
            # print idx
            try_spawn_random_vehicle_at_with_sensor(spawn_points[len(vehicle_list)], sensor_bp_list, len(vehicle_list))
        print('spawned %d vehicles, press Ctrl+C to exit.' % count)

        sleep_time = 10
        time_s = 1
        time_m = 0
        while time_m<15000:
            time.sleep(sleep_time)
            time_s +=sleep_time
            print(time_m,time_s)
            if time_s%60==0:
                time_m+=1



    finally:

        print('\ndestroying %d actors' % len(actor_list))
        for actor in actor_list:
            actor.destroy()

def get_translation_matrix(transform):
    x = transform.location.x
    y = transform.location.y
    z = transform.location.z
    T = [[x], [y], [z]]
    return T
def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False
def draw_points(array, pos, size, color=(255, 0, 255)):
    """Draws a rect"""
    len_points = 9
    result = np.array(array)
    for point_idx in range(len_points):
        point_0 = (pos[0][point_idx] - size / 2, pos[1][point_idx] - size / 2)
        point_1 = (pos[0][point_idx] + size / 2, pos[1][point_idx] + size / 2)
        # point_0 = (pos[0]-size/2, pos[1]-size/2)
        # point_1 = (pos[0]+size/2, pos[1]+size/2)
        if point_in_canvas(point_0) and point_in_canvas(point_1):
            for i in range(size):
                for j in range(size):
                    result[int(point_0[0]+i), int(point_0[1]+j)] =color
    return result
def draw_box(array, pos, size, color=(255, 0, 255)):
    """Draws a rect"""

    len_points = 8
    result = np.array(array)
    for point_idx in range(len_points):
        point_0 = (pos[0][point_idx] - size / 2, pos[1][point_idx] - size / 2)
        point_1 = (pos[0][point_idx] + size / 2, pos[1][point_idx] + size / 2)
        # point_0 = (pos[0]-size/2, pos[1]-size/2)
        # point_1 = (pos[0]+size/2, pos[1]+size/2)
        if point_in_canvas(point_0) and point_in_canvas(point_1):
            for i in range(size):
                for j in range(size):
                    result[int(point_0[0]+i), int(point_0[1]+j)] =color
    return result
def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array
def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    # if not isinstance(image, sensor.Image):
    #     raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array
def rand_color(seed):
    """Return random color based on a seed"""
    random.seed(seed)
    col = colorsys.hls_to_rgb(random.random(), random.uniform(.2, .8), 1.0)
    return (int(col[0]*255), int(col[1]*255), int(col[2]*255))


def get_extrinsic(transform):
    roll = np.radians(transform.rotation.roll)
    yaw = np.radians(transform.rotation.yaw+90)
    pitch = np.radians(transform.rotation.pitch+90)
    x=transform.location.x
    y=transform.location.y
    z= transform.location.z
    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_y = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    a = np.concatenate((R,[[x],[y],[z]]),axis=1)
    b = np.concatenate((a,[[0,0,0,1]]),axis=0)
    return np.linalg.inv(b)
def rotate(points3d,rotation):
    roll = np.radians(rotation.roll)
    yaw = np.radians(rotation.yaw)
    pitch = np.radians(rotation.pitch )

    R_r = np.matrix([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R_p = np.matrix([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)],
    ])
    R_y = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = R_y * R_p * R_r
    return np.matmul(R,points3d)

def get_bounding_box_points(vehicle,vehicle_location):
    # returns 4x8 point vector
    cen_x=vehicle.bounding_box.location.x+vehicle_location.x
    cen_y = vehicle.bounding_box.location.y+vehicle_location.y
    cen_z = vehicle.bounding_box.location.z+vehicle_location.z
    len_x = vehicle.bounding_box.extent.x
    len_y = vehicle.bounding_box.extent.y
    len_z = vehicle.bounding_box.extent.z
    perm = np.transpose(np.array(np.meshgrid([1,-1],[1,-1],[1,-1])).T.reshape(-1,3))
    perm = np.concatenate([perm,np.zeros([3,1])],axis=1)
    center = np.repeat([[cen_x],[cen_y],[cen_z]],9,axis=1)
    extent = np.repeat([[len_x],[len_y],[len_z]],9,axis=1)
    extent = np.multiply(extent, perm)
    extent_rotated = rotate(extent,vehicle.get_transform().rotation)
    points = center + extent_rotated
    points_conc = np.concatenate([points, np.ones([1, 9])], axis=0)
    return points_conc
def parse_image_classic(image, vehicle_list, vehicle_idx, sensor_opt_list,name=''):

    if not image.frame_number%FRAME_CAPTURE_INTERVAL==0:
        return
    sensor_name = name + sensor_opt_list[2]
    print (image.frame_number, sensor_name)
    image_transform = image.transform
    flag = 0
    K = np.identity(3)
    K[0, 2] = WINDOW_WIDTH / 2.0
    K[1, 2] = WINDOW_HEIGHT / 2.0
    K[0, 0] = K[1, 1] = WINDOW_WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))
    ext = get_extrinsic(image_transform)
    image.convert(sensor_opt_list[1])
    image_array_imm = to_rgb_array(image)
    image_array = np.array(image_array_imm)
    image_array = image_array
    imageP = PImage.frombytes(
        mode='RGB',
        size=(WINDOW_WIDTH, WINDOW_HEIGHT),
        data=image_array,
        decoder_name='raw')
    draw = ImageDraw.Draw(imageP)

    for idx, v in enumerate(vehicle_list):
        dist = v.get_location().distance(image.transform.location)
        if dist > 200 or vehicle_idx==idx :
            continue
        loc = v.get_transform().location
        loc_vector = get_bounding_box_points(v,loc)
        pos3d_rel = np.matmul(ext,loc_vector)
        pos3d_rel_subspace = pos3d_rel[0:3]
        pos2d_scaled = np.matmul(K,pos3d_rel_subspace)
        pos2d = np.array([
            pos2d_scaled[0] / pos2d_scaled[2],
            pos2d_scaled[1] / pos2d_scaled[2],
            pos2d_scaled[2]
        ])
        pos2d = np.squeeze(pos2d)
        if np.any(pos2d[2] < 1e-4):
            x_2d = WINDOW_WIDTH-pos2d[0]
            y_2d = WINDOW_HEIGHT-pos2d[1]
            x_max = np.max(x_2d)
            y_max = np.max(y_2d)
            x_min = np.min(x_2d)
            y_min = np.min(y_2d)
            draw.rectangle([x_min,y_min,x_max,y_max],width=2,outline=(0,255,0))
            draw.text(xy=[(x_min+x_max)/2,y_min],text=str(idx),fill=(255,0,0))
            flag = 1
    if flag:

        file_name = os.path.join(DATA_FOLDER_PATH,'out%s/%d/%d_%d.png' % (
        sensor_name, vehicle_idx, image.frame_number, vehicle_idx))
        folder = os.path.dirname(file_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        imageP.save(file_name)

def parse_image_classicdepth(image, vehicle_list, vehicle_idx, sensor_opt_list,name=''):
    if not image.frame_number%FRAME_CAPTURE_INTERVAL==0:
        return
    sensor_name = name + sensor_opt_list[2]
    image_transform = image.transform
    flag = 0
    K = np.identity(3)
    K[0, 2] = WINDOW_WIDTH / 2.0
    K[1, 2] = WINDOW_HEIGHT / 2.0
    K[0, 0] = K[1, 1] = WINDOW_WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))
    ext = get_extrinsic(image_transform)
    image.convert(sensor_opt_list[1])
    image_array_imm = to_rgb_array(image)
    image_array = np.array(image_array_imm)
    image_array = image_array
    imageP = PImage.frombytes(
        mode='RGB',
        size=(WINDOW_WIDTH, WINDOW_HEIGHT),
        data=image_array,
        decoder_name='raw')
    draw = ImageDraw.Draw(imageP)

    for idx, v in enumerate(vehicle_list):
        dist = v.get_location().distance(image.transform.location)
        if dist > 200 or vehicle_idx==idx :
            continue
        loc = v.get_transform().location
        loc_vector = np.array([[loc.x], [loc.y], [loc.z],[1]])
        loc_vector = get_bounding_box_points(v,loc)

        # loc_vector =
        pos3d_rel = np.matmul(ext,loc_vector)
        pos3d_rel_subspace = pos3d_rel[0:3]
        pos2d_scaled = np.matmul(K,pos3d_rel_subspace)
        pos2d = np.array([
            pos2d_scaled[0] / pos2d_scaled[2],
            pos2d_scaled[1] / pos2d_scaled[2],
            pos2d_scaled[2]
        ])
        pos2d = np.squeeze(pos2d)
        if np.any(pos2d[2] < 1e-4):
            x_2d = WINDOW_WIDTH-pos2d[0]
            y_2d = WINDOW_HEIGHT-pos2d[1]
            x_max = np.max(x_2d)
            y_max = np.max(y_2d)
            x_min = np.min(x_2d)
            y_min = np.min(y_2d)
            draw.rectangle([x_min,y_min,x_max,y_max],width=2,outline=(0,255,0))
            draw.text(xy=[(x_min+x_max)/2,y_min],text=str(idx),fill=(255,0,0))
            flag = 1
    if flag:

        file_name = 'out%s/%d/%d_%d.png' % (
        sensor_name, vehicle_idx, image.frame_number, vehicle_idx)
        file_name= os.path.join(DATA_FOLDER_PATH,file_name)
        folder = os.path.dirname(file_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        imageP.save(file_name)
def parse_image_classiclidar(image, vehicle_list, vehicle_idx, sensor_opt_list,name=''):

    if not image.frame_number%FRAME_CAPTURE_INTERVAL==1:
        # print(image.frame_number)
        return
    else:
        print(image.frame_number)
    vehicle_inrange = 0
    for idx, v in enumerate(vehicle_list):
        if idx==vehicle_idx:
            continue
        dist = v.get_location().distance(image.transform.location)
        if dist<150:
            vehicle_inrange+=1

    if vehicle_inrange<1:
        return
    sensor_name = name + sensor_opt_list[2]
    print (image.frame_number, sensor_name)
    file_name = 'out%s/%d/%d_%d.ply' % (
        sensor_name, vehicle_idx, image.frame_number, vehicle_idx)
    file_name = os.path.join(DATA_FOLDER_PATH, file_name)
    folder = os.path.dirname(file_name)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    image.save_to_disk(file_name)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False


