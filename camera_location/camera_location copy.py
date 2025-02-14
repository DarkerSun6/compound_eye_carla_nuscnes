import time
import carla
import random
import pandas as pd
from pynput.keyboard import Key, Listener
global pressed
pressed = 'aaa'
camera_points_path = "/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan/points/points_transform.csv"

 
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
world = client.get_world()
world = client.load_world('Town03')
SpawnActor = carla.command.SpawnActor

class Actor:
    def __init__(self,world,bp_name,location,rotation,options=None,attach_to=None):
        self.bp_name = bp_name
        self.world = world
        self.blueprint = world.get_blueprint_library().find(bp_name)
        if options is not None:
            for key in options:
                self.blueprint.set_attribute(key, options[key])
        self.transform = carla.Transform(carla.Location(**location),carla.Rotation(**rotation))
        self.attach_to = attach_to
        self.actor = None
    def set_actor(self,id):
        self.actor = self.world.get_actor(id)
    def spawn_actor(self):
        self.actor = self.world.spawn_actor(self.blueprint,self.transform,self.attach_to)
    def get_actor(self):
        return self.actor
    def destroy(self):
        self.actor.destroy()
class Vehicle(Actor):
    def __init__(self,path=[],**args):
        super().__init__(**args)
        self.path=[carla.Location(**location) for location in path]
    def get_transform(self):
        location = self.actor.get_transform().transform(self.actor.bounding_box.location)
        rotation = self.actor.get_transform().rotation
        return carla.Transform(location,rotation)
    def get_bbox(self):
        return self.actor.bounding_box.get_world_vertices(self.actor.get_transform())
    def get_size(self):
        return self.actor.bounding_box.extent*2
class Sensor(Actor):
    def __init__(self, name, **args):
        super().__init__(**args)
        self.name = name
        self.data_list = []
    def get_data_list(self):
        return self.data_list
    def set_actor(self, id):
        super().set_actor(id)
        self.actor.listen(self.add_data)
    def spawn_actor(self):
        super().spawn_actor()
        self.actor.listen(self.add_data)
    def get_last_data(self):
        if self.data_list:
            return self.data_list[-1]
        else:
            return None
    def get_transform(self):
        return self.actor.get_transform()
def draw_point_with_color(world, location, str_1='S', life_time=10.0):
    aaa = str_1
    point = world.get_map().get_waypoint(location)
    point_loca = point.transform.location
    point_loca.x = location.x
    point_loca.y = location.y
    point_loca.z = location.z

    world.debug.draw_string(point_loca, str(aaa), draw_shadow=True,
                              color=carla.Color(r=255, g=0, b=0), life_time=life_time)
def on_press(key):
    global pressed
    if key == Key.tab:
        pressed = 'tab'
    if key == Key.esc:
        pressed = 'esc'
def check_tab_pressed():
    global pressed  # 明确声明 tab_pressed 是全局变量
    if pressed == 'tab':
        pressed = 'aaa'  # 重置状态
        return 'tab'
    elif pressed == 'esc':
        pressed = 'aaa'  # 重置状态
        return 'esc'
    return 'aaa'

def get_points(out_dir_name):
    '''
    功能：读取文件中的坐标点并储存为列表
    out_dir_name:摄像机点位储存位置
    返回：摄像机点位列表
    '''
    df = pd.read_csv(out_dir_name) 
    points_lo = []
    points = df[['X', 'Y', 'Z']].values.tolist()
    for i in range(0,len(points)):
        points_lo.append((float(format(points[i][0],'.5f')),float(format(points[i][1],'.5f')),float(format(points[i][2],'.5f'))))
    pointa_ro = []
    points = df[['pitch','yaw','roll']].values.tolist()
    for i in range(0,len(points)):
        pointa_ro.append((float(format(points[i][0],'.5f')),float(format(points[i][1],'.5f')),float(format(points[i][2],'.5f'))))
    return points_lo,pointa_ro


try:
    # 设置相机相对位置

    camera_points_lo,camera_points_ro = get_points(camera_points_path)

    # camera_z = 1.8
    # camera_x = 1.2
    life_time = 20
    # 获取可用点位
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # 创建ego车辆
    ego_bp_name="vehicle.tesla.model3"                      
    ego_location={attr:getattr(spawn_points[0].location,attr) for attr in ["x","y","z"]}
    ego_rotation={attr:getattr(spawn_points[0].rotation,attr) for attr in ["yaw","pitch","roll"]}
    ego_vehicle = Vehicle(world=world,bp_name=ego_bp_name,location=ego_location,rotation=ego_rotation)
    ego_vehicle.spawn_actor()
    time.sleep(0.1)             

    print("ego_vehicle.transform:")
    vehicle_transform = ego_vehicle.get_actor().get_transform()
    print(vehicle_transform)
    vehicle_location = ego_vehicle.get_actor().get_location()
    draw_point_with_color(world, location=vehicle_location, str_1='e', life_time=life_time)

    # 创建相机
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_list = []
    for i in range(len(camera_points_lo)):
        camera_name = 'camera_'+str(i)
        # print(camera_name)
        camera_init_trans = carla.Transform(carla.Location(x=camera_points_lo[i][0],y=camera_points_lo[i][1],z=camera_points_lo[i][2]),
                                            carla.Rotation(pitch = camera_points_ro[i][0],yaw = camera_points_ro[i][1],roll = camera_points_ro[i][2]))
        camera_name = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle.get_actor())
        camera_list.append(camera_name)

    for j in range(len(camera_list)):
        camera_location = camera_list[j].get_location()
        draw_point_with_color(world, location=camera_location, str_1='o', life_time=life_time)
        time.sleep(0.02) 

   
    print("camera_transform:")
    camera_transform = camera_list[2].get_transform()  
    print(camera_list[2].get_transform())
    camera_location = camera_list[2].get_location()  
    time.sleep(0.1)             
    # draw_point_with_color(world, location=camera_location, str_1='c', life_time=life_time)


    relative_location = camera_location - vehicle_location
    print("relative_location:")
    print(relative_location)
    print("----------------------------------------------------------------")

    # 获取ego_vehicle的尺寸
    # bounding_box = ego_vehicle.get_size()
    # print(bounding_box)
    # 观察角度列表
    relative_view = 2
    Camera_transform_list = [
            [carla.Location(x=0, y=relative_view,  z=camera_points_lo[2][2]),
            carla.Rotation(pitch=0, yaw=-90, roll=0)],

            [carla.Location(x=-relative_view, y=0, z=camera_points_lo[2][2]),
            carla.Rotation(pitch=0, yaw=0, roll=0)],

            [carla.Location(x=0, y=-relative_view, z=camera_points_lo[2][2]),
            carla.Rotation(pitch=0, yaw=90, roll=0)],

            [carla.Location(x=relative_view, y=0,  z=camera_points_lo[2][2]),
            carla.Rotation(pitch=0, yaw=180, roll=0)],
                        
            [carla.Location(x=0, y=0, z=6),
            carla.Rotation(pitch=-90, yaw=0, roll=0)],       
            
            [carla.Location(x=-4, y=0, z=4),
            carla.Rotation(pitch=-15, yaw=0, roll=0)]                                  
        ]
    spectator = world.get_spectator()

    # 监听键盘事件
    i = 0 
    with Listener(on_press=on_press) as listener:
        while True: 
            x = check_tab_pressed()
            if x == 'tab':
                print("Tab pressed")
                print("camera_transform:")
                print(camera_transform)

                camera_transform_add = carla.Transform(Camera_transform_list[i][0],Camera_transform_list[i][1])
                print("camera_transform_add:")
                print(camera_transform_add)

                camera_2 = world.spawn_actor(camera_bp, camera_transform_add, attach_to=ego_vehicle.get_actor())
                time.sleep(0.1)
                
                spectator_transform = camera_2.get_transform()
                print("spectator_transform:")
                print(spectator_transform) 
                print("----------------------------------------------------------------")

                spectator.set_transform(spectator_transform)
                i+=1
                if i>=len(Camera_transform_list):
                    i=0
            if x == 'esc':           
                print(" Esc pressed")
                break
            time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    # 销毁所有车辆
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.destroy()
    # 销毁所有行人
    for walker in world.get_actors().filter('*walker*'):
        walker.destroy()
    # 停止并销毁所有controller
    for controller in world.get_actors().filter('*controller*'):
        controller.stop()
    print("All actors destroyed")
    