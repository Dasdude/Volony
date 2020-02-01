import cPickle as pickle
import carla
def get_actor_dynamic_meta_data(vehicle, dict_obj):
    """

    :type dict_obj: dict
    :type vehicle: carla.Vehicle
    """
    v_t = vehicle.get_transform()  # type: carla.Transform
    v_r = v_t.rotation  # type: carla.Rotation
    v_l = v_t.location  # type: carla.Location
    id = vehicle.id
    ret_dict = {'type_id':vehicle.type_id,'rotation':{'p':v_r.pitch,'r':v_r.roll,'y':v_r.yaw},'translation':{'x':v_l.x,'y':v_l.y,'z':v_l.z}}
    assert not dict_obj.has_key(id),'Error: Duplicate Vehicle ID in meta data dictionary'
    dict_obj[id] = ret_dict
    return dict_obj
def get_vehicle_static_meta_data(vehicle,v_idx,dict_obj):
    """
        returns dict having all the vehicle static data. v_idx is a zero based index in order to have a consistant graph. e.g vehicle id is 1523 bu v_idx is 0 since there is only one vehicle
    :param v_idx:
    :param dict_obj:
    :return:
    :type vehicle: carla.Vehicle
    """

    bb = vehicle.bounding_box  # type: carla.BoundingBox
    bb_l = bb.location  # type: carla.Location
    bb_e = bb.extent

    ret_dict = {'g_index':v_idx,'type':'v','type_id':str(vehicle.type_id),'bounding_box':{'loc':{'x':str(bb_l.x),'y':str(bb_l.y),'z':str(bb_l.z)},'extent':{'x':str(bb_e.x),'y':str(bb_e.y),'z':str(bb_e.z)}}}
    dict_obj[vehicle.id] = ret_dict
    return dict_obj
def get_sensor_static_meta_data(sensor,dict_obj):
    """

    :type dict_obj: dict
    :type sensor: carla.Sensor
    """
    p_id = sensor.parent.id
    ret_dict = {'type': 's', 'type_id': sensor.type_id,'parent_id':p_id,'parent_type':sensor.parent.type_id,
                'attr':sensor.attributes,'id':sensor.id}
    #TODO sensor parent id returns none but in future it might get fixed. fix it later
    if dict_obj.has_key(p_id):
        dict_obj[sensor.parent.id][sensor.type_id] = ret_dict
        dict_obj[sensor.parent.id]['sensor_ids'].append(sensor.id)
    else:
        dict_obj[sensor.parent.id] = {}
        dict_obj[sensor.parent.id]['sensor_ids'] =[]
        dict_obj[sensor.parent.id][sensor.type_id] = ret_dict
        dict_obj[sensor.parent.id]['sensor_ids'].append(sensor.id)
    return dict_obj
