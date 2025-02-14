from .client import Client
from .dataset import Dataset
import traceback
import carla
import time
from datetime import datetime

class Generator:
    def __init__(self,config):
        self.config = config        #cofig中包含了yaml中所有的数据
        self.collect_client = Client(self.config["client"])     #将config中的client数据提取到collect_client中，host post time_out

    def generate_dataset(self,load=False):
        self.dataset = Dataset(**self.config["dataset"],load=load) #创建文件夹（maps\sweeps\samples\v1.14），生成或读取json文件，load为false时创建,并打印json文件位置名，load为true时读取，并打印json文件位置名
        print('---------------------------------------------------')

        print(self.dataset.data["progress"])        #打印current_world_index、current_capture_index、current_scene_index、current_scene_count参数
        print('---------------------------------------------------')

        for sensor in self.config["sensors"]:       
            self.dataset.update_sensor(sensor["name"],sensor["modality"])#读取sensors.yaml中的sensor设置，生成各自对应的token、channel、modality，生成samples及sweeps下对应的传感器文件夹.特定的传感器类型
        for category in self.config["categories"]:
            self.dataset.update_category(category["name"],category["description"])#读取categories.yaml中的的类别设置，生成对应token，包含token、name、description.对象类别的分类（例如车辆、人类）。子类别由句号划分（例如human.pedestrian.adult）
        for attribute in self.config["attributes"]:
            self.dataset.update_attribute(attribute["name"],category["description"])#读取attributes.yaml中的的属性设置，生成对应token，包含token、name、description。属性是实例的属性，在类别保持不变的情况下可以更改。示例：正在停放/停止/移动的车辆，以及自行车是否有骑手。
        for visibility in self.config["visibility"]:
            self.dataset.update_visibility(visibility["description"],visibility["level"])#读取visibility.yaml中的的属性设置，生成对应token，透明度的token为数字1-4，包含token、description、level.实例的可见性是在所有 6 个图像中可见的注释部分
        
        current_world_index = self.dataset.data["progress"]["current_world_index"]  #world_index参数


        for world_config in self.config["worlds"][current_world_index:]: # world参数，在哪个城市
            try:
                print('the current world index is '+str(world_config['map_name']))
                print('---------------------------------------------------')
                self.collect_client.generate_world(world_config)        #连接carla世界，carla中的城市
                map_token = self.dataset.update_map(world_config["map_name"],world_config["map_category"])      # 生成map_token并加入到self.dataset.date.map中，放到这里是因为不同城市的map_token不同

                current_capture_index = self.dataset.data["progress"]["current_capture_index"]  # capture_index参数，当前采集时间、地点、数据集数据名前缀


                for capture_config in world_config["captures"][current_capture_index:]: # capture参数,收集时间地点，数据集数据名前缀
                    print('the current captrue index is '+str(capture_config['capture_vehicle']))
                    print('---------------------------------------------------')

                    # capture_current_timestamp = time.time()
                    # capture_local_time_str = datetime.fromtimestamp(capture_current_timestamp)
                    # date_part = capture_local_time_str.date()  # 仅日期部分
                    # time_part = capture_local_time_str.time()  # 仅时间部分

                    # capture_config["date"] = date_part.strftime('%Y-%m-%d')
                    # capture_config["time"] = time_part.strftime('%H-%M-%S')


                    # log_token = self.dataset.update_log(map_token,capture_config["date"],capture_config["time"],capture_config["timezone"],capture_config["capture_vehicle"],capture_config["location"])
                    # 生成token值 更新数据日志，储存当前时间、采集地点、主车辆 map_item["log_tokens"]更新添加token值，取决于log和当前城市及采集时间地点

                    current_scene_index = self.dataset.data["progress"]["current_scene_index"]


                    for scene_config in capture_config["scenes"][current_scene_index:]:#scene参数
                        all_count = scene_config["count"]
                        print('the all scene count is '+str(all_count)) #总共要采集的场景次数
                        print('---------------------------------------------------')

                        print('the current scene index is '+str(self.dataset.data["progress"]["current_scene_index"]))   #scene_index参数 具体采集场景，涉及到采集场景次数、总的采集时间和指定场景天气情况等
                        print('---------------------------------------------------')

                        current_scene_count = self.dataset.data["progress"]["current_scene_count"]

                        for scene_count in range(current_scene_count,all_count):#count参数，同一场景下天气、交通等种类数
                            print('the current scene count is '+str(self.dataset.data["progress"]["current_scene_count"]))   #当前场景采集
                            print('---------------------------------------------------')

                            capture_current_timestamp = time.time()  + 86400*4.0
                            capture_local_time_str = datetime.fromtimestamp(capture_current_timestamp)
                            date_part = capture_local_time_str.date()  # 仅日期部分
                            time_part = capture_local_time_str.time()  # 仅时间部分

                            capture_config["date"] = date_part.strftime('%Y-%m-%d')
                            capture_config["time"] = time_part.strftime('%H-%M-%S')

                            log_token = self.dataset.update_log(map_token,capture_config["date"],capture_config["time"],capture_config["timezone"],capture_config["capture_vehicle"],capture_config["location"])


                            self.dataset.update_scene_count()#scene_count参数+1
                            self.add_one_scene(log_token,scene_config,capture_current_timestamp)#主要函数
                            self.dataset.save()#json文件储存
                        self.dataset.update_scene_index()#scene_index参数+1
                    self.dataset.update_capture_index()#capture_index参数+1
                self.dataset.update_world_index()#world_index参数+1
            except:
                traceback.print_exc()
            finally:
                self.collect_client.destroy_world()
                
    def add_one_scene(self,log_token,scene_config,capture_current_timestamp):
        try:
            calibrated_sensors_token = {}   #传感器位置、角度及自身参数
            samples_data_token = {}         #传感器
            instances_token = {}            #行人和汽车
            samples_annotation_token = {}   #传感器对应标签
            # current_timestamp = time.time()

            self.collect_client.generate_scene(scene_config) # 关键程序，    更新carla世界场景，custom为true时指定当前场景天气、交通等参数，布置传感器，custom为false时指定天气参数，随机生成汽车行人，布置传感器
            scene_token = self.dataset.update_scene(log_token,scene_config["description"])#更新场景参数，description描述当前场景，如：test1

            # instance的token，添加行人和汽车token，物体实例数量，表示数据集中不同的物体实例。我们观察到的所有对象实例的枚举
            for instance in self.collect_client.walkers+self.collect_client.vehicles:
                up_instance_token = self.collect_client.get_instance(scene_token,instance)  #category_token
                instance_token = self.dataset.update_instance(*up_instance_token)
                instance_id = instance.get_actor().id
                instances_token[instance_id] = instance_token
                samples_annotation_token[instance.get_actor().id] = ""

            # sensor和calibrated_sensor的token
            for sensor in self.collect_client.sensors:
                up_scene_token = self.collect_client.get_calibrated_sensor(sensor)  #关键
                calibrated_sensor_token = self.dataset.update_calibrated_sensor(scene_token,*up_scene_token)    # 特定车辆上校准的特定传感器（激光雷达/雷达/相机）的定义
                calibrated_sensors_token[sensor.name] = calibrated_sensor_token
                samples_data_token[sensor.name] = ""

            sample_token = ""
            collect_num = int(scene_config["collect_time"]/self.collect_client.settings.fixed_delta_seconds) # config_fuyan中的collect_time，fixed_delta_seconds，1/0.01=100
            for frame_count in range(collect_num):
                # print("frame count:",frame_count)
                self.collect_client.tick()  # 更新世界

                self.world = self.collect_client.client.get_world()#获取世界
                self.spectator = self.world.get_spectator()
                ego_transform = self.collect_client.ego_vehicle.get_actor().get_transform()
                self.spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(x=0,y=0,z=5),ego_transform.rotation))

                keyframe_num = int(scene_config["keyframe_time"]/self.collect_client.settings.fixed_delta_seconds)#0.5/0.01=50，每50次（0.5s）进行一次保存
                frame_num = (frame_count+1)%keyframe_num
                # if (frame_count+1)%int(scene_config["keyframe_time"]/self.collect_client.settings.fixed_delta_seconds) == 0:
                if  frame_num == 0:     #50次进入一次采集
                    print("frame count:",frame_count)

                    sample_token = self.dataset.update_sample(sample_token,scene_token,*self.collect_client.get_sample(capture_current_timestamp))
                    for sensor in self.collect_client.sensors:      #单个传感器0.5s内采集到的数据
                        if sensor.bp_name in ['sensor.camera.rgb','sensor.other.radar','sensor.lidar.ray_cast']:  #判断是否是camera、radar或lidar数据，这里没有保存GNSS和IMU的数据
                            sensor_data = sensor.get_data_list()        #获取当前传感器0.5s内的所有数据
                            for idx,sample_data in enumerate(sensor_data):  #同时列出数据和数据下标
                                ego_pose_token = self.dataset.update_ego_pose(scene_token,calibrated_sensors_token[sensor.name],*self.collect_client.get_ego_pose(capture_current_timestamp,sample_data)) #自我车辆在特定时间戳的姿势,timestamp,translation,rotation
                                is_key_frame = False        #sweeps数据
                                if idx == len(sensor_data)-1:
                                    is_key_frame = True     #samples数据，储存最后一帧
                                samples_data_token[sensor.name] = self.dataset.update_sample_data(samples_data_token[sensor.name],calibrated_sensors_token[sensor.name],sample_token,ego_pose_token,is_key_frame,*self.collect_client.get_sample_data(sample_data))#保存sample_data，is_key_frame为true时保存到samples中，为false时保存到sweeps中
                    npc_view_all = 0
                    for instance in self.collect_client.walkers + self.collect_client.vehicles:   #遍历所有车辆和行人，放到instance中
                        visible_cam, visible_cam_num = self.collect_client.get_visibility(instance)
                        if visible_cam and visible_cam_num :
                            npc_view_all += 1
                            samples_annotation_token[instance.get_actor().id]  = self.dataset.update_sample_annotation(samples_annotation_token[instance.get_actor().id],sample_token,*self.collect_client.get_sample_annotation(scene_token,instance)) #更新sample_annotation标签，保存到samples_annotation中
                    print("npc_view_all:",npc_view_all)

                    for sensor in self.collect_client.sensors:
                        sensor.get_data_list().clear()  #清空传感器缓存

            self.dataset.data['instance'] = list(filter(lambda ins_data: ins_data['nbr_annotations'] != 0, self.dataset.data['instance']))
        except:
            traceback.print_exc()
        finally:
            self.collect_client.destroy_scene()
            print('------------------------------------------------------------------------------------------------------')
