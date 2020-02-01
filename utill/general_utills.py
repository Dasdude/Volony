import random
from utill import sensor_utills
from utill.sensor_utills import SensorCharacteristic
import carla.libcarla
import numpy as np
def get_listener_function(client,output_path,v,frame_interval,sensor_obs):
    # return lambda image: sensor_utills.parse_image_classiclidar(image, output_path, vehicle_list, ego_idx, sensor_obs.sensor_name, frame_interval)
    return lambda image: sensor_utills.parse_sensor_data(client,image, output_path, v, sensor_obs, frame_interval)
def get_collison_listener_function(v,client):
    return lambda event: sensor_utills.respawn_on_collision_listener(event, v,client)

def get_vehicles_bp(world,n_of_wheels=-1):
    bp_list = world.get_blueprint_library()
    bp_vehicles = bp_list.filter('vehicle.*')
    if not n_of_wheels ==-1:
        blueprints = [x for x in bp_vehicles if int(x.get_attribute('number_of_wheels')) == n_of_wheels]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    return blueprints

def get_vehicle_actor_list(world):
    pass
def spawn_set_random_vehicle_with_sensor(world,n,save_path,observation_sensor_list=None,actor_list=[],sensor_list=[],vehicle_list=[],frame_interval=10,shuffle_points=True):
    vehicle_sensor_list =[]
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    bp_list_veh = get_vehicles_bp(world,n_of_wheels=4)
    if shuffle_points:
        random.shuffle(spawn_points)
    for idx in range(n):
        veh_bp = random.choice(bp_list_veh)
        if veh_bp.has_attribute('color'):
            color = random.choice(veh_bp.get_attribute('color').recommended_values)
            veh_bp.set_attribute('color', color)
        veh_bp.set_attribute('role_name', 'autopilot')
        print('spawning vehicle with sensors at %s' % ( spawn_points[idx]))
        vehicle =world.spawn_actor(veh_bp,spawn_points[idx])
        print('vehicle: %r spawned vehicle with sensors at %s' % (vehicle.type_id, spawn_points[idx]))
        vehicle_sensor_list+=[vehicle]
    for v_idx,v in enumerate(vehicle_sensor_list):
        for idx_sen, sen_ob in enumerate(observation_sensor_list):
            sensor_inst =v.get_world().spawn_actor(sen_ob.sensor_bp,sen_ob.get_relative_transform(),attach_to=v)
            sensor_inst.listen(get_listener_function(save_path,v_idx,vehicle_sensor_list,frame_interval,sen_ob))
            sensor_list+=[sensor_inst]
            v.set_autopilot(False)

    vehicle_list +=vehicle_sensor_list
    actor_list+=vehicle_sensor_list
    actor_list+=sensor_list
    return vehicle_list,sensor_list
def attach_sensors_to_vehicle(client,vehicle_list,n,save_path,observation_sensor_list=None,frame_interval=10):
    # attach sensors to n vehicles from vehicle_list
    sensor_list = []
    vehicle_list_shuffled = np.random.permutation(vehicle_list)
    selected_vehicles = vehicle_list_shuffled[:n]
    for v_idx,v in enumerate(selected_vehicles):
        for idx_sen, sen_ob in enumerate(observation_sensor_list):
            sensor_inst =v.get_world().spawn_actor(sen_ob.sensor_bp,sen_ob.get_relative_transform(),attach_to=v)
            print('%s attached to %s'%(sen_ob.sensor_name,v.type_id) )
            sensor_inst.listen(get_listener_function(client,save_path,v,frame_interval,sen_ob))
            # sensor_inst.stop()
            sensor_list+=[sensor_inst]
    client.start_record=True
    return sensor_list
def start_sensor_listening(save_path,observation_sensor_list,frame_interval=10):
    # attach sensors to n vehicles from vehicle_list
        for idx_sen, sen_ob in enumerate(observation_sensor_list):
            sen_ob.listen()
def spawn_set_random_vehicle(client,n,shuffle_points=True):
    world = client.get_world()
    sensor_bp_lib = world.get_blueprint_library()
    collision_bp = sensor_bp_lib.find('sensor.other.collision')
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    bp_list_veh = get_vehicles_bp(world,n_of_wheels=4)
    spawned_vehicle_list = []
    actor_list = []
    sensor_list = []
    if shuffle_points:
        random.shuffle(spawn_points)
    for idx in range(min(n,len(spawn_points))):
        veh_bp = random.choice(bp_list_veh)
        if veh_bp.has_attribute('color'):
            color = random.choice(veh_bp.get_attribute('color').recommended_values)
            veh_bp.set_attribute('color', color)
        veh_bp.set_attribute('role_name', 'autopilot')
        print('Vehicle spawning at %s' % spawn_points[idx])
        vehicle = None
        while vehicle is None:

            vehicle = world.try_spawn_actor(veh_bp,spawn_points[idx])
            if vehicle is None:
                spawn_points = map.get_spawn_points()
                random.shuffle(spawn_points)
        # collision_inst = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        # collision_inst.listen(get_collison_listener_function(vehicle,client))
        print('Spawned vehicle: %r at %s' % (vehicle.type_id, vehicle.get_location()))
        actor_list+=[vehicle]
        spawned_vehicle_list+= [vehicle]
        # sensor_list+=[collision_inst]

    return actor_list,sensor_list


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
def set_autopilot_on_list(vehicle_list):
    for v in vehicle_list:
        v.set_autopilot()





