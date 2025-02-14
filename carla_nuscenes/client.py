import carla
from .sensor import *
from .vehicle import Vehicle
from .walker import Walker
from .actor import Actor
import math
from .utils import generate_token,get_nuscenes_rt,get_intrinsic,transform_timestamp,clamp
import random

class Client:
    def __init__(self,client_config):
        self.client = carla.Client(client_config["host"],client_config["port"])
        self.client.set_timeout(client_config["time_out"])
        # self.transform.K = Client.build_projection_matrix()

    def generate_world(self,world_config):
        print("generate world start!")
        self.client.load_world(world_config["map_name"])#加载地图
        self.world = self.client.get_world()#获取世界
        self.original_settings = self.world.get_settings()#保存当前世界的原始设置
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)#卸载地图中的静态停放车辆层
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None

        get_category = lambda bp: "vehicle.car" if bp.id.split(".")[0] == "vehicle" else "human.pedestrian.adult" if bp.id.split(".")[0] == "walker" else None
        self.category_dict = {bp.id: get_category(bp) for bp in self.world.get_blueprint_library()}#创建另一个字典，将蓝图 ID 映射到属性列表
        get_attribute = lambda bp: ["vehicle.moving"] if bp.id.split(".")[0] == "vehicle" else ["pedestrian.moving"] if bp.id.split(".")[0] == "walker" else None
        self.attribute_dict = {bp.id: get_attribute(bp) for bp in self.world.get_blueprint_library()}

        self.trafficmanager = self.client.get_trafficmanager(6000)#获取当前的交通管理器实例。
        self.trafficmanager.set_synchronous_mode(True)#设置交通管理器为同步模式
        self.trafficmanager.set_global_distance_to_leading_vehicle(5.0)
        self.trafficmanager.set_respawn_dormant_vehicles(True)#设置交通管理器在车辆变为休眠状态时重新生成这些车辆。

        self.settings = carla.WorldSettings(**world_config["settings"])#根据 world_config["settings"] 设置新的世界参数
        self.settings.synchronous_mode = True# 设置为 True 表示模拟将在同步模式下运行，即每次调用 tick() 时都会更新一次模拟状态
        self.settings.no_rendering_mode = False#设置为 False 表示启用渲染
        self.world.apply_settings(self.settings)

        self.world.set_pedestrians_cross_factor(0.8)#设置行人的交叉因素为1，这会影响行人在模拟中的行为，例如他们穿过马路的概率
        print("generate world success!")

    def generate_scene(self,scene_config):
        print("generate scene start!")
        if scene_config["custom"]:
            self.generate_custom_scene(scene_config)
        else:
            self.generate_random_scene(scene_config)
        print("generate scene success!")

    def generate_custom_scene(self,scene_config):
        
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"]) # 预设天气状况类型
        
        self.world.set_weather(self.weather)
        SpawnActor = carla.command.SpawnActor       #用于创建新的演员（例如车辆）。
        SetAutopilot = carla.command.SetAutopilot   # 用于设置演员的自动驾驶状态。
        FutureActor = carla.command.FutureActor     # 用于引用尚未完全创建的演员。

        self.ego_vehicle = Vehicle(world=self.world,**scene_config["ego_vehicle"])#自我车辆位置参数
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')# 将role_name设置为可查找的值hero
        self.ego_vehicle.spawn_actor()#生成演员
        self.ego_vehicle.get_actor().set_autopilot(True,6000)#设置自动驾驶

        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)#告诉交通管理器让主角车辆启用自动变道功能。这意味着主角车辆可以根据需要自动变换车道。
        self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),8.0)#告诉交通管理器让主角车辆与前方车辆保持0米的距离

        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)#完全忽略红绿灯（百分比为100%）。这意味着主角车辆在遇到红灯时不会停车。
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)#完全忽略交通标志（百分比为100%）。这意味着主角车辆在遇到停车标志等交通标志时不会采取相应的行动。
        self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),20)#完全忽略其他车辆的存在（百分比为100%）。这意味着主角车辆在模拟中不会考虑与其他车辆之间的碰撞风险。

        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)#告诉交通管理器让主角车辆的速度相对于其他车辆慢20%。这意味着主角车辆会以比周围车辆平均速度慢20%的速度行驶。

        self.vehicles = [Vehicle(world=self.world,**vehicle_config) for vehicle_config in scene_config["vehicles"]]#读取vehicles.yaml中的数据
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform)
                            .then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) 
                            for vehicle in self.vehicles]#对于列表中的每一个 Vehicle 对象，都会创建一个 SpawnActor 命令来生成车辆，并紧接着创建一个 SetAutopilot 命令来开启自动驾驶。
        
        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id)
            else:
                print(f"Failed to spawn ehicles: {response.error}")
        #使用 apply_batch_sync 方法来同步地执行之前创建的批量命令 vehicles_batch。它会依次处理每个响应，如果响应没有错误，则设置车辆的 actor_id；如果有错误，则输出错误信息。
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles))
        #使用 filter 函数来移除那些没有成功生成的车辆。lambda vehicle: vehicle.get_actor() 是一个匿名函数，用于检查每个 Vehicle 对象是否已经成功生成。

        for vehicle in self.vehicles:
            self.trafficmanager.set_path(vehicle.get_actor(),vehicle.path)
        #这段代码遍历所有成功生成的车辆，并为每个车辆设置路径。vehicle.path 是一个路径对象或路径数据

        self.walkers = [Walker(world=self.world,**walker_config) for walker_config in scene_config["walkers"]]
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_batch)):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            else:
                print(f"Failed to spawn walker: {response.error}")
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
                    if not response.error:
                        self.walkers[i].set_controller(response.actor_id)
                    else:
                        print(response.error)
        self.world.tick()
        for walker in self.walkers:
            walker.start()

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                 self.sensors[i].set_actor(response.actor_id)
            else:
                print(f"Failed to spawn sensors: {response.error}")
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))

    def tick(self):
        self.world.tick()

    def generate_random_scene(self,scene_config):
        print("generate setting scene start!")
        # self.weather = carla.WeatherParameters(**self.get_random_weather())#随机天气
        # self.world.set_weather(self.weather)
        if scene_config["weather_mode"] == "custom":
            self.weather = carla.WeatherParameters(**scene_config["weather"])
        else:
            self.weather = getattr(carla.WeatherParameters, scene_config["weather_mode"]) # 预设天气状况类型
        self.world.set_weather(self.weather)


        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        spawn_points = self.world.get_map().get_spawn_points()      #获取所有可用的生成点。
        for spawn_point in spawn_points:
            if spawn_point.location.z <= 0:  # 确保生成点不在地下
                spawn_points.remove(spawn_point)
        random.shuffle(spawn_points)        #随机打乱生成点列表的顺序
        
        
        ego_bp_name=scene_config["ego_bp_name"]
        ego_location={attr:getattr(spawn_points[0].location,attr) for attr in ["x","y","z"]}#从第一个生成点中提取位置和旋转信息，并将它们分别存储在字典ego_location和ego_rotation中。
        ego_rotation={attr:getattr(spawn_points[0].rotation,attr) for attr in ["yaw","pitch","roll"]}
        self.ego_vehicle = Vehicle(world=self.world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)#创建了一个Vehicle对象，该对象代表了ego车辆。这里假设Vehicle是一个自定义的类，它接受世界对象、车辆蓝图名称、位置和旋转作为参数。
        self.ego_vehicle.blueprint.set_attribute('role_name', 'hero')#设置了ego车辆的蓝图属性role_name为'hero'
        self.ego_vehicle.spawn_actor()#使用Vehicle对象的方法spawn_actor来实际生成车辆
        self.ego_vehicle.get_actor().set_autopilot(True,6000)#启用了ego车辆的自动驾驶模式

        self.trafficmanager.ignore_lights_percentage(self.ego_vehicle.get_actor(),100)
        self.trafficmanager.ignore_signs_percentage(self.ego_vehicle.get_actor(),100)
        # self.trafficmanager.ignore_vehicles_percentage(self.ego_vehicle.get_actor(),100)
        # self.trafficmanager.ignore_walkers_percentage(self.ego_vehicle.get_actor(),100)

        # self.trafficmanager.distance_to_leading_vehicle(self.ego_vehicle.get_actor(),5.0)
        self.trafficmanager.set_global_distance_to_leading_vehicle(4.0)

        self.trafficmanager.vehicle_percentage_speed_difference(self.ego_vehicle.get_actor(),-20)       # 通过将perc设置为负值，可以超过速度限制
        self.trafficmanager.auto_lane_change(self.ego_vehicle.get_actor(), True)        #允许自动变道。

        vehicle_bp_list = self.world.get_blueprint_library().filter('*vehicle*')#获取所有可用的车辆蓝图。
        vehicle_bp_list_2 = []
        for vehicle_bp in vehicle_bp_list:
            # print(vehicle_bp.id)
            type = vehicle_bp.get_attribute('base_type')
            car_type = type.as_str()
            # print(type)
            if car_type == 'car':
                vehicle_bp_list_2.append(vehicle_bp)
        # print("the all vehicle of the base_type is car: "+str(len(vehicle_bp_list_2)))

        self.vehicles = []#遍历一个随机数量的生成点（从第二个开始），创建一个Vehicle对象，并将其添加到self.vehicles列表中
        len_spawn_points = len(spawn_points)  # len_spawn_points定义为可用生成点的一半的值
        len_spawn_points_vehicle = random.randint(int(len_spawn_points/2),len_spawn_points-20)   # len_spawn_points_vehicle定义为从10到len_spawn_points间的随机整数，用于随机生成车辆的数量，随机车辆最小有20个，最多为len_spawn_points个。
        for spawn_point in spawn_points[0:len_spawn_points_vehicle]:
            location = {attr:getattr(spawn_point.location,attr) for attr in ["x","y","z"]}  #从当前生成点中获取位置信息，将其存储在字典location中
            rotation = {attr:getattr(spawn_point.rotation,attr) for attr in ["yaw","pitch","roll"]} #从当前生成点中获取旋转信息，将其存储在字典rotation中
            # bp_name = random.choice(vehicle_bp_list).id # 从车辆蓝图列表中随机选择一个

            blueprint_vehicle = random.choice(vehicle_bp_list_2) # 从车辆蓝图列表中随机选择一个
            if blueprint_vehicle.has_attribute('color'):
                color = random.choice(blueprint_vehicle.get_attribute('color').recommended_values)
                blueprint_vehicle.set_attribute('color', color)

            bp_name = blueprint_vehicle.id # 从车辆蓝图列表中随机选择一个
            self.vehicles.append(Vehicle(world=self.world,bp_name=bp_name,location=location,rotation=rotation)) # 创建一个Vehicle对象，并将其添加到self.vehicles列表中
        vehicles_batch = [SpawnActor(vehicle.blueprint,vehicle.transform).then(SetAutopilot(FutureActor, True, self.trafficmanager.get_port())) for vehicle in self.vehicles] 

        fail_num_vehicle = 0
        for i,response in enumerate(self.client.apply_batch_sync(vehicles_batch)):  #遍历vehicles_batch列表，使用enumerate函数获取当前索引和响应对象。apply_batch_sync同步模式下应用批量命令。i是当前索引，response是响应对象。
            if not response.error:
                self.vehicles[i].set_actor(response.actor_id) # 将生成的actor_id设置到Vehicle对象中
            elif response.error == "Spawn failed because of collision at spawn position":
                fail_num_vehicle += 1
                vehicle_error = response.error # 如果失败，打印错误信息
        if fail_num_vehicle > 0:
            print(f"Failed to spawn vehicle: {vehicle_error}"+"   "+str(fail_num_vehicle))
        self.vehicles = list(filter(lambda vehicle:vehicle.get_actor(),self.vehicles)) # 过滤掉没有actor_id的Vehicle对象，并重新赋值给self.vehicles列表。 lambda 函数检查 vehicle 对象的 actor 属性是否为 None。filter 函数会遍历 self.vehicles 列表中的每一个 Vehicle 对象，返回 True，则该 Vehicle 对象会被保留在结果中；如果返回 False，则该 Vehicle 对象会被过滤掉。

        walker_bp_list = self.world.get_blueprint_library().filter('*pedestrian*')#批量创建行人
        self.walkers = []
        len_spawn_points_pedestrian = random.randint(100,200) # 随机生成行人的数量，随机行人最小有len_spawn_points_half个

        walker_position = []
        for j in range(len_spawn_points_pedestrian+1):
            spawn_1 = self.world.get_random_location_from_navigation()
            spawn_location = {attr:getattr(spawn_1,attr) for attr in ["x","y","z"]}
            if spawn_location not in walker_position:
                walker_position.append(spawn_location)

        for i in range(len_spawn_points_pedestrian):
            # spawn = world.get_random_location_from_navigation()
            if walker_position[i] != None:
                walker_bp = random.choice(walker_bp_list)
                bp_name=walker_bp.id    #从 walker_bp_list 中随机选择一个蓝图对象， 获取该蓝图对象的 ID，存储在 bp_name 变量中
                spawn_location = walker_position[i]

                spawn_rotation = {"yaw":0.0,"pitch":0.0,"roll":0.0}
                destination_location=walker_position[i+1] #提取随机目的地的位置信息

                self.walkers.append(Walker(world=self.world,location=spawn_location,rotation=spawn_rotation,destination=destination_location,bp_name=bp_name))
            else:
                print("walker generate fail")
        walkers_batch = [SpawnActor(walker.blueprint,walker.transform) for walker in self.walkers]


        walker_xxx = self.client.apply_batch_sync(walkers_batch)
        fail_num_walkers = 0
        for i,response in enumerate(walker_xxx):
            if not response.error:
                self.walkers[i].set_actor(response.actor_id)
            elif response.error == "Spawn failed because of collision at spawn position":
                fail_num_walkers += 1
                walker_error = response.error
        if fail_num_walkers > 0:
            print(f"Failed to spawn walker: {walker_error}"+"   "+str(fail_num_walkers))
        self.walkers = list(filter(lambda walker:walker.get_actor(),self.walkers))

        # 在CARLA模拟器中为已经生成的行人创建AI控制器，并将这些控制器与相应的行人关联起来。这样可以使得行人在模拟环境中按照一定的行为模式移动
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')  #从蓝图库中找到名为 controller.ai.walker 的蓝图，这是用于控制行人的 AI 控制器蓝图。
        walkers_controller_batch = [SpawnActor(walker_controller_bp,carla.Transform(),walker.get_actor()) for walker in self.walkers]   #创建批量命令来生成行人控制器
        for i,response in enumerate(self.client.apply_batch_sync(walkers_controller_batch)):
            if not response.error:
                self.walkers[i].set_controller(response.actor_id)
            else:
                print(f"Failed to add walker: {response.error}")
        self.world.tick()#执行世界的一个模拟步骤，并启动所有的行人AI控制器
        for walker in self.walkers:
            walker.start()

        print('need %d vehicles and %d walkers' % (len_spawn_points_vehicle, len_spawn_points_pedestrian))
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles), len(self.walkers)))
        if (len(self.walkers)+fail_num_walkers) != len_spawn_points_pedestrian:
            print("some else errors while spawning pedestrians")
        if (len(self.vehicles)+fail_num_vehicle) != len_spawn_points_vehicle:
            print("some else errors while spawning pedestrians")

        self.sensors = [Sensor(world=self.world, attach_to=self.ego_vehicle.get_actor(), **sensor_config) for sensor_config in scene_config["calibrated_sensors"]["sensors"]]
        sensors_batch = [SpawnActor(sensor.blueprint,sensor.transform,sensor.attach_to) for sensor in self.sensors]
        for i,response in enumerate(self.client.apply_batch_sync(sensors_batch)):
            if not response.error:
                self.sensors[i].set_actor(response.actor_id)
            else:
                print(f"Failed to spawn sensor: {response.error}")
        self.sensors = list(filter(lambda sensor:sensor.get_actor(),self.sensors))
        print("generate random scene success!")

    def destroy_scene(self):
        if self.walkers is not None:
            for walker in self.walkers:
                walker.controller.stop()
                walker.destroy()
        if self.vehicles is not None:
            for vehicle in self.vehicles:
                vehicle.destroy()
        if self.sensors is not None:
            for sensor in self.sensors:
                sensor.destroy()
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()


    def destroy_world(self):
        self.trafficmanager.set_synchronous_mode(False)
        self.ego_vehicle = None
        self.sensors = None
        self.vehicles = None
        self.walkers = None
        self.world.apply_settings(self.original_settings)

    def get_calibrated_sensor(self,sensor):
        sensor_token = generate_token("sensor",sensor.name) # 生成传感器的token
        channel = sensor.name
        if sensor.bp_name == "sensor.camera.rgb":
            # 校准矩阵
            intrinsic = get_intrinsic(float(sensor.get_actor().attributes["fov"]),
                            float(sensor.get_actor().attributes["image_size_x"]),
                            float(sensor.get_actor().attributes["image_size_y"])).tolist()
            
            rotation,translation = get_nuscenes_rt(sensor.transform,"zxy")  #关键
        else:
            intrinsic = []
            rotation,translation = get_nuscenes_rt(sensor.transform)
        return sensor_token,channel,translation,rotation,intrinsic
        
    def get_ego_pose(self,capture_current_timestamp,sample_data):
        timestamp = transform_timestamp(capture_current_timestamp+sample_data[1].timestamp)
        rotation,translation = get_nuscenes_rt(sample_data[0])
        return timestamp,translation,rotation
    
    def  get_sample_data(self,sample_data):
        height = 0
        width = 0
        if isinstance(sample_data[1],carla.Image):
            height = sample_data[1].height
            width = sample_data[1].width
        return sample_data,height,width

    def get_sample(self,current_timestamp):
        world_current_timestamp = current_timestamp + self.world.get_snapshot().timestamp.elapsed_seconds
        return (transform_timestamp(world_current_timestamp),)#世界快照,world.get_snapshot() 方法可以用来获取当前仿真世界的一个“快照”,包含了当前时刻世界的状态信息，包括但不限于时间信息、天气条件、交通状况等.timestamp 属性提供了当前时刻的时间信息，通常包括：frame: 当前帧,elapsed_seconds: 自仿真开始以来经过的时间（以秒为单位）。

    def get_instance(self,scene_token,instance):
        category_token = generate_token("category",self.category_dict[instance.blueprint.id])

        # instance_actor = instance.get_actor()
        # instance_id = instance_actor.id
        # print(instance_id)
        
        id = hash((scene_token,instance.get_actor().id))
        return category_token,id

    def get_sample_annotation(self,scene_token,instance):
        instance_token = generate_token("instance",hash((scene_token,instance.get_actor().id)))
        visibility_token,visibility_cam_token = self.get_visibility(instance)
        visibility_token = str(visibility_token)
        visibility_cam_token = str(visibility_cam_token)
        
        attribute_tokens = [generate_token("attribute",attribute) for attribute in self.get_attributes(instance)]
        rotation,translation = get_nuscenes_rt(instance.get_transform())
        size = [instance.get_size().y,instance.get_size().x,instance.get_size().z]#xyz to whl
        num_lidar_pts = 0
        num_radar_pts = 0
        for sensor in self.sensors:
            if sensor.bp_name == 'sensor.lidar.ray_cast':
                num_lidar_pts += self.get_num_lidar_pts(instance,sensor.get_last_data(),sensor.get_transform())
            elif sensor.bp_name == 'sensor.other.radar':
                num_radar_pts += self.get_num_radar_pts(instance,sensor.get_last_data(),sensor.get_transform())
        return instance_token, visibility_token, visibility_cam_token, attribute_tokens, translation, rotation, size, num_lidar_pts, num_radar_pts
    

    def get_image_point(loc, K, w2c):
        point = np.array([loc.x, loc.y, loc.z, 1])
        point_camera = np.dot(w2c, point)
        point_camera = np.array(
            [point_camera[1], -point_camera[2], point_camera[0]]).T
        point_img = np.dot(K, point_camera)

        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]
    
    def build_projection_matrix(w, h, fov, is_behind_camera=False):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)

        if is_behind_camera:
            K[0, 0] = K[1, 1] = -focal
        else:
            K[0, 0] = K[1, 1] = focal

        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    def point_in_canvas(pos, img_w, img_h):
        """Return true if point is in canvas"""
        if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
            return True
        return False
    def get_visibility(self,instance):
        world_2_npc = np.array(instance.get_actor().get_transform().get_inverse_matrix())
        visible_cam = 0
        visible_cam_num = 0

        instance_type = instance.bp_name
        if "vehicle" in instance_type:
            points_check = [[0,1,1], [0,-1,1], [0,0,0], [0,0,2],
                            [1,1,1], [1,-1,1], [-1,1,1], [-1,-1,1], [1,0,1], [-1,0,1], 
                            [0.5,1,1], [-0.5,1,1], [0.5,-1,1], [-0.5,-1,1]]
            points_check = np.array(points_check)*0.9
            visibility_near = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:3, 13:4, 14:4}
            visibility_10 = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:3, 13:4, 14:4}
            visibility = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:2, 8:2, 9:3, 10:3, 11:3, 12:3, 13:4, 14:4}
            dist_range = 100
            error_bbox = 0.25
            mul_z=2
            
        elif "pedestrian" in instance_type:
            points_check = [[0,0,1], [0,0,0.5], [0,0,-0.5], [0,0,-1], [0,1,0], [0,-1,0], [1,0,0], [-1,0,0]]
            points_check = np.array(points_check)*0.9
            visibility_near = {0:0, 1:0, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4}
            visibility_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4}
            visibility = {0:0, 1:0, 2:0, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4}
            dist_range = 50
            error_bbox = 0.25
            mul_z=1

        instance_actor = instance.get_actor()
        npc_transform = instance.get_transform()
        vehicle_location = self.ego_vehicle.get_actor().get_transform().location
        dist = npc_transform.location.distance(vehicle_location)
        if instance_actor.is_alive and self.ego_vehicle.get_actor().id != instance_actor.id and dist < dist_range:
            for sensor in self.sensors:
                if sensor.bp_name == 'sensor.camera.rgb' and sensor.name != "BEV_CAM_RGB" and dist < dist_range:
                    if dist < dist_range/2:
                        visibility = visibility_near
                        if dist < 10:
                            visibility = visibility_10
                    fov = float(sensor.get_actor().attributes["fov"])
                    image_h = float(sensor.get_actor().attributes["image_size_x"])
                    image_w = float(sensor.get_actor().attributes["image_size_y"])
                    K = get_intrinsic(fov = fov, image_size_x = image_w, image_size_y = image_h)

                    camera_forwaard_vec = sensor.get_transform().get_forward_vector()
                    world_2_camera = np.array(sensor.get_transform().get_inverse_matrix())
                    camera_location = sensor.get_transform().location

                    instance_transform = instance.get_transform()
                    instance_actor = instance.get_actor()
                    ray = instance_transform.location - camera_location

                    if camera_forwaard_vec.dot(ray) > 0:
                        bb = instance.get_actor().bounding_box
                        visible_camera = 0

                        verts = [v for v in bb.get_world_vertices(instance.get_transform())]
                        points_image = []

                        for vert in verts:
                            p = Client.get_image_point(vert, K, world_2_camera)
                            points_image.append(p)
                        for points_check_2D in points_image:
                            p1_in_canvas = Client.point_in_canvas(points_check_2D, img_w = image_w, img_h = image_h)
                            if p1_in_canvas:
                                visible_camera += 1
                        if visible_camera > 0:
                            visible_point_count = 0
                                
                            size = instance.get_actor().bounding_box.extent
                            K_instance = np.identity(3)
                            K_instance[0, 0] = size.x
                            K_instance[1, 1] = size.y
                            K_instance[2, 2] = size.z
                            for point in points_check:
                                # print("point:",point)
                                points_check_instance = np.dot(point,K_instance)
                                # print(f"points_check_instance: {points_check_instance}")

                                world_2_npc_inv = np.linalg.inv(world_2_npc)
                                points_check_instance = np.append(points_check_instance,1)
                                # print(f"points_check_instance: {points_check_instance}")

                                points_check_world = np.dot(world_2_npc_inv,points_check_instance)
                                check_point = carla.Location(points_check_world[0], points_check_world[1], points_check_world[2])
                                
                                points_check_2d = Client.get_image_point(check_point, K, world_2_camera) 
                                p_in_canvas = Client.point_in_canvas(points_check_2d, image_h, image_w)

                                camera_location = sensor.get_transform().location
                                ray_points =  self.world.cast_ray(camera_location,check_point)
                                point_over = 0

                                for point in ray_points:
                                    if point.label is not carla.libcarla.CityObjectLabel.NONE:
                                        instance_sign = instance.get_actor().bounding_box.contains\
                                            (point.location,instance.get_actor().get_transform())

                                        if not instance_sign:
                                            ray_point = np.array([point.location.x, point.location.y, point.location.z, 1])
                                            ray_point_npc = np.dot(world_2_npc,ray_point)

                                            if abs(ray_point_npc[0]) <= (size.x+error_bbox) \
                                                and abs(ray_point_npc[1]) <= (size.y+error_bbox) \
                                                and abs(ray_point_npc[2]) <= (size.z*mul_z+error_bbox) \
                                                and abs(ray_point_npc[2]) >= -0.1:
                                                point_over = point_over
                                            else:
                                                point_over += 1
                                                break
                                
                                if point_over == 0 and p_in_canvas:
                                    visible_point_count+=1

                            visibility_one = visibility[visible_point_count]

                            if visibility_one > 0:
                                visible_cam_num += 1
                                visible_cam = max(visible_cam, visibility_one)
        # print(f"npc_view_all: {npc_view_all}, visible_cam: {visible_cam}")
        return visible_cam,visible_cam_num

    def get_attributes(self,instance):
        return self.attribute_dict[instance.bp_name]

    def get_num_lidar_pts(self,instance,lidar_data,lidar_transform):
        num_lidar_pts = 0
        if lidar_data is not None:
            for data in lidar_data[1]:
                point = lidar_transform.transform(data.point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_lidar_pts+=1
        return num_lidar_pts

    def get_num_radar_pts(self,instance,radar_data,radar_transform):
        num_radar_pts = 0
        if radar_data is not None:
            for data in radar_data[1]:
                point = carla.Location(data.depth*math.cos(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.altitude)*math.cos(data.azimuth),
                        data.depth*math.sin(data.azimuth)
                        )
                point = radar_transform.transform(point)
                if instance.get_actor().bounding_box.contains(point,instance.get_actor().get_transform()):
                    num_radar_pts+=1
        return num_radar_pts

    def get_random_weather(self):
        weather_param = {
            "cloudiness":clamp(random.gauss(0,30)),
            "sun_azimuth_angle":random.random()*360,
            "sun_altitude_angle":random.random()*120-30,
            "precipitation":clamp(random.gauss(0,30)),
            "precipitation_deposits":clamp(random.gauss(0,30)),
            "wind_intensity":random.random()*100,
            "fog_density":clamp(random.gauss(0,30)),
            "fog_distance":random.random()*100,
            "wetness":clamp(random.gauss(0,30)),
            "fog_falloff":random.random()*5,
            "scattering_intensity":max(random.random()*2-1,0),
            "mie_scattering_scale":max(random.random()*2-1,0),
            "rayleigh_scattering_scale":max(random.random()*2-1,0),
            "dust_storm":clamp(random.gauss(0,30))
        }
        return weather_param

    