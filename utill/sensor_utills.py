import os
import carla
import cPickle as pickle
from carla.libcarla import World
from carla.libcarla import WeatherParameters as wp
import pickle_util as pu
import numpy as np
import random
WEATHER_LIST = [wp.ClearNoon,wp.ClearSunset,wp.CloudyNoon,wp.CloudySunset,wp.WetCloudyNoon,wp.WetCloudySunset,wp.WetNoon,wp.WetSunset]
def on_tick_write_dynamic(timestamp,client,dynamic_meta_file_path,capture_interval):
    """
    :type timestamp: carla.Timestamp
    :type dynamic_meta_file: str
    """
    if not client.start_record:
        return
    frame_num = timestamp.frame_count

    if not frame_num % capture_interval== 0:
        return
    else:
        print('Frame %d captured'%frame_num)
    world = client.get_world()  # type: World
    if frame_num % 100 == 0:
        w_idx = np.random.randint(0,len(WEATHER_LIST)-1)
        world.set_weather(WEATHER_LIST[w_idx])
    actors = world.get_actors()  # type: carla.ActorList #
    vehicle_actor = actors.filter('vehicle.*')
    sensor_actor = actors.filter('sensor.*')
    res_dict = {}
    for v in vehicle_actor:  # type: carla.Vehicle
        res_dict = pu.get_actor_dynamic_meta_data(v, res_dict)
    for v in sensor_actor:  # type: carla.Vehicle
        res_dict = pu.get_actor_dynamic_meta_data(v, res_dict)
    frame_dict = {frame_num:res_dict}
    with open(dynamic_meta_file_path,'ab') as dy_met_file:
        pickle.dump(frame_dict,dy_met_file,0)
def on_tick_respawn(timestamp,client,respawn_frame_interval):
    """
    :type timestamp: carla.Timestamp
    :type dynamic_meta_file: str
    """
    # client.start_record = False
    frame_num = timestamp.frame_count
    if not frame_num % respawn_frame_interval== 0:
        client.start_record = True
        return
    print('Frame %d Vehicles Respawned'%frame_num)
    world = client.get_world()  # type: World
    actors = world.get_actors()  # type: carla.ActorList #
    vehicle_actor = actors.filter('vehicle.*')
    respawn_set_random_vehicle(world,vehicle_actor)


def on_tick_end(timestamp, client, max_frame):
    """
    :type timestamp: carla.Timestamp
    :type dynamic_meta_file: str
    """
    # frame_num = timestamp.frame_count
    client.total_frames_captured +=1
    if client.total_frames_captured >= max_frame:
        client.terminate = True
        client.start_record = False
        print('client terminated')
def destroy_veh_and_sensors_in_world(client):
    """

    :type client: carla.Client
    """
    world = client.get_world()  # type: carla.World
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    sensors = actors.filter('sensor.*')
    print('%d Vehicles existed from before'%len(vehicles))
    print('%d Sensors existed from before'%len(sensors))
    for s in sensors:  # type: carla.Actor
        print('%s REMOVED' % s.type_id)
        s.destroy()
    for s in vehicles:  # type: carla.Actor
        print('%s REMOVED' % s.type_id)
        s.destroy()
    print('%d Vehicles removed' % len(vehicles))
    print('%d Sensors removed' % len(sensors))

def on_tick_change_weather(timestamp,client,change_frame_interval):
    """
    :type timestamp: carla.Timestamp
    """
    frame_num = timestamp.frame_count
    if not frame_num % change_frame_interval== 0:
        return
    timestamp.frame_count
    world = client.get_world()  # type: World
    wp.CloudySunset
    w_idx = np.random.randint(0,len(WEATHER_LIST)-1)
    world.set_weather(WEATHER_LIST[w_idx])
    print('Frame %d Weather Changed to %s' % (frame_num,str(WEATHER_LIST[w_idx])))
def write_sensor_static_data(sensor_actors,static_meta_file_path):
    res_dict = {}
    for s in sensor_actors:  # type: carla.Sensor
        res_dict = pu.get_sensor_static_meta_data(s, res_dict)
    with open(static_meta_file_path,'ab') as dy_met_file:
        pickle.dump(res_dict,dy_met_file,0)
def write_vehicle_static_data(client,static_meta_file_path):
    world = client.get_world()  # type: World
    actors = world.get_actors()  # type: carla.ActorList #
    vehicle_actors = actors.filter('vehicle.*')
    res_dict = {}
    for v_idx,v in enumerate(vehicle_actors):  # type: carla.Vehicle
        res_dict = pu.get_vehicle_static_meta_data(v,v_idx,res_dict)
    with open(static_meta_file_path,'ab') as dy_met_file:
        pickle.dump(res_dict,dy_met_file,0)

def parse_sensor_data(client,image,data_folder_path, ego_vehicle, sensor_charct,frame_capture_interval=10):
    if not client.start_record:
        return
    if not image.frame_number%frame_capture_interval==0:
        return

    sensor_name = sensor_charct.sensor_name
    file_name = '%s/%d/%d_%d%s' % (
        sensor_name, ego_vehicle.id, image.frame_number, ego_vehicle.id,sensor_charct.store_file_extension)
    file_name = os.path.join(data_folder_path, file_name)
    folder = os.path.dirname(file_name)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    image.save_to_disk(file_name)
    # print('%d %s Captured'%(image.frame_number,sensor_name))
def respawn_on_collision_listener(collision,parent_vehicle,client):
    """

    :type collision: carla.CollisionEvent
    :type parent_vehicle: carla.Vehicle
    """
    ni = collision.normal_impulse  # type: carla.Vector3D
    norm = ni.x**2 + ni.y**2 +ni.z**2
    if norm>100:
        world = client.get_world()  # type: World
        respawn_set_random_vehicle(world, [parent_vehicle])
        print ('Frame %d Vehicle %s respawned cause by collision with %s'%(collision.frame_number,parent_vehicle.type_id,collision.other_actor.type_id))
def respawn_set_random_vehicle(world,vehicle_list,shuffle_points=True):
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    for v in vehicle_list:  # type: carla.Vehicle
        v.set_autopilot(False)
    if shuffle_points:
        random.shuffle(spawn_points)
    for idx,v in enumerate(vehicle_list):  # type: carla.Vehicle
        v.set_velocity(carla.Vector3D(0,0,0))
        v.set_transform(spawn_points[idx])

    for idx,v in enumerate(vehicle_list):  # type: carla.Vehicle
        v.set_autopilot(True)
class SensorCharacteristic:
    def __init__(self,relative_transform,sensor_bp,sensor_name,data_parser,file_extension):
    # Listener Function gets an image object as input
        self.transform = relative_transform
        self.sensor_bp = sensor_bp
        self.sensor_name = sensor_bp.id
        self.data_parser = data_parser
        self.store_file_extension = file_extension
    def get_relative_transform(self):
        return self.transform

