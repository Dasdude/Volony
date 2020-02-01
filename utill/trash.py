class ActorReplay:
    def __init__(self,actor):
        # type: (Actor) -> None
        if not actor.is_alive():
            raise('Actor %s is not alive'%actor.type_id)
        self.__current_frame = -1
        self.__history = {}
        self.__static_build = False
    def __static_snapshot(self,actor):
        """
        :type actor: Actor
        """
        pass
    def set_current_frame(self,f):
        self.__current_frame = f
    def get_current_frame(self):
        return self.__current_frame
    def store_dynamic_info_snapshot(self,f):
        pass
class VehicleReplay(ActorReplay):
    def __init__(self,vehicle):
        """

        :type vehicle: Vehicle
        """
        super(VehicleReplay).__init__(vehicle)
        self.__static_snapshot(vehicle)
        pass
    def __static_snapshot(self,vehicle):
        """
        :type vehicle: Vehicle
        """
        self.id = vehicle.id
        self.type_id = vehicle.type_id
        self.attr = vehicle.attributes
        self.__bounding_box = vehicle.bounding_box
        self.__static_build = True

class SensorReplay(ActorReplay):
    def __init__(self, sensor,experiment_name,image_extension,output_path='../Output',sensor_meta_folder = 'META',static_file_name = 'mst.p',dynamic_file_name = 'mdy.p'):
        """
        :type file_extension: str
        :type sensor: ClientSideSensor
        """
        super(SensorReplay).__init__(sensor)
        self.__sensor__ = sensor
        self.__static_snapshot(sensor)
        self.__static_build = True
        self.__file_extension = image_extension
        self.__output_path = os.path.join(output_path,experiment_name)
        self.__sensor_meta_file_path = os.path.join(self.__output_path,sensor_meta_folder,sensor.type_id,sensor.id)
        self.__sensor_raw_data_path = os.path.join(self.__output_path, self.type_id, self.id, 'RAW')
        os.makedirs(self.__sensor_raw_data_path)
        os.makedirs(self.__sensor_meta_file_path)




    def __static_snapshot(self,sensor):
        """

        :type sensor: ClientSideSensor
        """
        self.attr = sensor.attributes
        self.type_id = sensor.type_id
        self.id = sensor.id

    def store_dynamic(self):

        pass
class SensorPlaneReplay(SensorReplay):
    def __init__(self,sensor):
        super(SensorPlaneReplay).__init__(sensor)

    def project(self):
        pass
class SensorLidarReplay(SensorReplay):
    def __init__(self,sensor):
        super(SensorLidarReplay).__init__(sensor)
    def get_static_lower_fov(self):

    def project(self):
        pass