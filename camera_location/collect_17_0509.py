import glob
import os
import sys
import random
import math
import cv2
import numpy as np
import time
import queue
import csv
import pandas as pd

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# 连接Carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 设置起始点和终点
# start_point = world.get_map().get_waypoint(RU)
start_point = None
stop_point = world.get_map().get_waypoint(carla.Location(20,-60,0))    
# 获取模拟环境中位置 (-25, -60, 0) 处的路点

# 拍摄车辆实例颜色位置
SEG_LOCATION = carla.Location(120,105,5)

# 文件路径
out_dir = '/home/darker/CARLA_0.9.14/fuyan_collect/fuyan_17'

data_output_dir = os.path.join(out_dir, 'data') # csv数据输出路径
pic_rgb_output_dir = os.path.join(out_dir, 'pic_rgb') # rgb图片输出路径
pic_seg_output_dir = os.path.join(out_dir, 'pic_seg ') # seg图片输出路径
pic_seg_info_dir = os.path.join(out_dir, 'info', 'pic_seg_info') # 车辆seg信息输出路径
points_dir = os.path.join(out_dir, 'points/points_17.csv')  # 点位位置csv文件

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(pic_rgb_output_dir):
    os.makedirs(pic_rgb_output_dir)
if not os.path.exists(pic_seg_output_dir):
    os.makedirs(pic_seg_output_dir)
if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)
if not os.path.exists(pic_seg_info_dir):
    os.makedirs(pic_seg_info_dir)

time_step = 0.1 # 仿真时间步长

# 主摄像头参数
IM_WIDTH = 1280
IM_HEIGHT = 720
IM_FOV = 120
IM_FRAME = 1/time_step

MAIN_LOCATION = carla.Location(1, 0, 1.8)

# 多摄像头参数
MULTI_WIDTH = 120
MULTI_HEIGHT = 120
MULTI_FOV = 60
MULTI_FRAME = 1/time_step

# 路径点
RD = carla.Location(-55,140,0) # 右下(-55,140,0)
LD = carla.Location(-113,-20,0) # 左下(-113,-20,0)
RU = carla.Location(108,90,0) # 右上(108,90,0)
LU = carla.Location(60,-68,0) # 左上(60,-68,0)


def polar2cart(r, theta):
    # 极坐标转换为笛卡尔坐标
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

def get_points_self():
    r = 0.2 
    theta = 30
    pitch = 20
    h = r * math.sin(math.radians(pitch))
    _r = math.sqrt(r**2 - h**2)

    points = []
    for i in range(-theta*2, theta*2+1, theta):
        x, y = polar2cart(r, math.radians(i))
        points.append((float(format(x, '.5f')), float(format(y, '.5f')), float(format(0,'.5f'))))

    for i in range(int(-theta*2.5), int(theta*2.5+1), theta):
        x, y = polar2cart(_r, math.radians(i))
        points.append((float(format(x, '.5f')), float(format(y, '.5f')), float(format(h, '.5f'))))

    for i in range(int(-theta*2.5), int(theta*2.5+1), theta):
        x, y = polar2cart(_r, math.radians(i))
        points.append((float(format(x, '.5f')), float(format(y, '.5f')), float(format(-h, '.5f'))))
    # print(points)  

    column_names = ['X', 'Y', 'Z']

    df = pd.DataFrame(points, columns=column_names)
    df.to_csv(os.path.join(out_dir, 'points/points_2.csv'), index=False)

    return points

def set_sync_mode(mode = True):
    '''
    # 设置同步模式
    '''
    if mode == True:
        # set synchorinized mode
        settings = world.get_settings()
        settings.fixed_delta_seconds = time_step
        settings.max_substep_delta_time = time_step/10
        settings.max_substeps = 10
        settings.synchronous_mode = True
        world.apply_settings(settings)

        settings = world.get_settings()
        # print(settings.fixed_delta_seconds)
        # print(settings.max_substep_delta_time)
        # print(settings.max_substeps)

        # 设置交通管理器为同步模式
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
    elif mode == False:
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # 设置交通管理器为同步模式
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(False)

def set_traffic_light(state = carla.TrafficLightState.Off):
    '''
    # 设置交通灯
    '''
    # 获取所有交通信号灯
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    # 设置交通信号灯状态
    for traffic_light in traffic_lights:
        traffic_light.set_state(state)

def destroy_actor(type = 'all'):
    '''
    # 销毁actor
    type = 'all', 'hero_vehicle', 'other_vehicle', 'vehicle', 'walker', 'sensor'
    '''
    # 获取所有的actor
    actor_list = world.get_actors()

    if type == 'all': # 销毁地图中所有的actor
        for actor in actor_list:
            while True:
                destroyed_sucessfully = actor.destroy()
                if destroyed_sucessfully == True:
                    break

    elif type == 'hero_vehicle': # 销毁地图中的hero车辆
        # 获取所有车辆的列表
        all_vehicles = actor_list.filter('vehicle.*')
        # 根据role_name为hero来选择特定的车辆
        hero_vehicle = [vehicle for vehicle in all_vehicles if vehicle.attributes['role_name'] == 'hero']
        if len(hero_vehicle) == 1:
            while True:
                destroyed_sucessfully = actor.destroy()
                if destroyed_sucessfully == True:
                    break
        else:
            print(f'{len(hero_vehicle)} hero_vehicles in the world!')

    elif type == 'other_vehicle': # 销毁地图中的除了主车辆以外的vehicle
        for actor in actor_list:
            if 'vehicle' in actor.type_id and actor.attributes['role_name'] != 'hero':
                while True:
                    destroyed_sucessfully = actor.destroy()
                    if destroyed_sucessfully == True:
                        break

    else:
        for actor in actor_list:
            if type in actor.type_id:
                while True:
                    destroyed_sucessfully = actor.destroy()
                    if destroyed_sucessfully == True:
                        break
                

    # # 获取所有的actor
    # actor_list = world.get_actors()   
    # for actor in actor_list:
    #         print(actor)

def create_vehicle(type = 'vehicle.audi.a2', transform = None, color = '0, 255, 255', auto = False, hero=False):
    '''
    # 生成车辆
    '''
    blueprint_library = world.get_blueprint_library() # 拿到蓝图库
    vehicle_bp = blueprint_library.find(type) # 从蓝图库中选择车辆蓝图
    vehicle_bp.set_attribute('color', color) # 设置车辆颜色

    if hero == True:
        vehicle_bp.set_attribute('role_name', 'hero') # 将该车设置为模拟主车辆

    if transform == None:
        transform = random.choice(world.get_map().get_spawn_points()) # 若未指定位置则随机选择一个可以作为初始点的位置

    vehicle = world.spawn_actor(vehicle_bp, transform) # 生成车辆
    vehicle.set_autopilot(auto) # 设置车辆是否自动驾驶

    return vehicle

def set_view(center_transform, relative_location, type = 'forward'):
    '''
    # 设置中心视角
    '''
    # 获取中心参照的位置和朝向
    center_location = center_transform.location
    center_rotation = center_transform.rotation
    # print(center_transform)
    # print(center_rotation)
    # print(center_location)

    # 计算视野的位置和朝向
    view_location = carla.Location()
    view_location = center_location + \
        relative_location.x * center_transform.get_forward_vector() + \
        relative_location.y * center_transform.get_right_vector() + \
        relative_location.z * center_transform.get_up_vector()
    # print(view_location)
    # 与center同向
    view_rotation = center_rotation

    if type != 'same':
        # 指向center坐标中心
        view_rotation.pitch += -math.degrees(math.atan2(relative_location.z, math.sqrt(relative_location.x**2 + relative_location.y**2)))
        # math.degrees(x) 方法将角度 x 从弧度转换为度数
        # math.atan2(y,x) 返回给定的 y 及 x 坐标值的反正切值 atan(y / x)，以弧度为单位，结果是在 -pi 和 pi 之间。
        view_rotation.yaw +=  math.degrees(math.atan2(relative_location.y, relative_location.x)) + 180

        if type == 'backward':
            # 背向center坐标中心
            view_rotation.pitch = -view_rotation.pitch
            view_rotation.yaw += 180


    # print(view_rotation)
    # 设置位置和朝向
    view_transform = carla.Transform(view_location, view_rotation)
    # print(view_transform)
    return view_transform

def set_spectator(center_transform, relative_location, mode = 'center'):
    '''
    # 设置观察者视角
    '''
    if mode == 'center':
        view_transform = set_view(center_transform, relative_location)

    elif mode == 'bev':
        view_transform = set_view(center_transform, relative_location)
        view_transform.rotation = center_transform.rotation
        view_transform.rotation.pitch = -90
        view_transform.rotation.yaw += 180

    spectator = client.get_world().get_spectator()
    spectator.set_transform(view_transform)

def creat_main_camera(vehicle, relative_transform, kind = 'instance_segmentation'):
    # 创建RGB摄像头蓝图
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.' + kind)
    camera_blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_blueprint.set_attribute('fov', f'{IM_FOV}')
    camera_blueprint.set_attribute('sensor_tick', f'{1/IM_FRAME}')

    # 设置摄像头的位置和朝向
    camera_transform = relative_transform

    # 将RGB摄像头实例绑定到车辆
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)

    return camera

def creat_bev_camera(vehicle, hight = 20, kind = 'instance_segmentation'):
    # 创建RGB摄像头蓝图
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.' + kind)
    camera_blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    camera_blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    camera_blueprint.set_attribute('fov', f'{IM_FOV}')
    camera_blueprint.set_attribute('sensor_tick', f'{1/IM_FRAME}')

    # 设置摄像头的位置和朝向
    bev_camera_transform  = carla.Transform()
    bev_camera_transform.location.x = hight
    bev_camera_transform.location.y = 0
    bev_camera_transform.location.z = hight
    bev_camera_transform.rotation.pitch = -90
    # print(bev_camera_transform)

    if vehicle != None:
        # 将RGB摄像头实例绑定到车辆
        camera = world.spawn_actor(camera_blueprint, bev_camera_transform, attach_to=vehicle)
    else:
        # 将RGB摄像头实例绑定到世界
        camera = world.spawn_actor(camera_blueprint, bev_camera_transform)

    return camera

def creat_multi_camera(vehicle, center_transform, center_relative_location, kind = 'instance_segmentation'):
    '''
    'rgb'
    'depth'
    'semantic_segmentation'
    'instance_segmentation'
    '''
    # 创建RGB摄像头蓝图
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.' + kind)
    camera_blueprint.set_attribute('image_size_x', f'{MULTI_WIDTH}')
    camera_blueprint.set_attribute('image_size_y', f'{MULTI_HEIGHT}')
    camera_blueprint.set_attribute('fov', f'{MULTI_FOV}')
    camera_blueprint.set_attribute('sensor_tick', f'{1/MULTI_FRAME}')

    # 设置RGB摄像头位置和朝向    
    # print(relative_location)
    camera_transform = set_view(center_transform, center_relative_location, 'backward')
    print(camera_transform)

    print(format(camera_transform.location.x, '.5f'))
    
    
    # 将RGB摄像头实例绑定到车辆
    camera = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)

    return camera

def creat_inseg_camera(location, kind = 'instance_segmentation'):
    # 创建RGB摄像头蓝图
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.' + kind)
    camera_blueprint.set_attribute('image_size_x', f'{IM_WIDTH//8}')
    camera_blueprint.set_attribute('image_size_y', f'{IM_HEIGHT//8}')
    camera_blueprint.set_attribute('fov', f'{IM_FOV}')
    camera_blueprint.set_attribute('sensor_tick', f'{1/IM_FRAME}')

    # 设置摄像头位置和朝向
    camera_transform = carla.Transform(location)
    camera_transform.rotation.pitch = -90
    # print(camera_transform)

    # 将摄像头实例绑定到世界
    camera = world.spawn_actor(camera_blueprint, camera_transform)

    return camera

def save_img_callback(name, image, path, queue):

    frame = image.frame - start_frame
    # 保存图片
    image.save_to_disk(os.path.join(path, name, '%06d.png' % frame))
    # image.save_to_disk(os.path.join('out', name, '%06d.png' % timestamp))
    queue.put((name, frame))

    # # 将CARLA传感器获得的图像数据转换成numpy数组
    # image_array = np.array(image.raw_data)
    # image_array = image_array.reshape((IM_HEIGHT, IM_WIDTH, 4))
    # image_array = image_array[:, :, :3]

def save_img_callback_2(image, path ,queue):
    
    frame = image.frame - now_frame
    pic_path = os.path.join(path, vehicle_name)
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    # 保存图片
    image.save_to_disk(os.path.join(pic_path, '%06d.png' % frame))
    queue.put((vehicle_name, frame))

def save_pose_data(path, actor, frame):

    # 获取车辆的位置和朝向
    transform = actor.get_transform()
    location = transform.location
    rotation = transform.rotation
    # 获得速度
    velocity = actor.get_velocity()  # 获取线速度
    angular_velocity = actor.get_angular_velocity()  # 获取角速度

    n = 3
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
        frame,
        round(location.x, n),
        round(location.y, n),
        round(location.z, n),
        round(rotation.roll, n),
        round(rotation.pitch, n),
        round(rotation.yaw, n),
        round(velocity.x, n),
        round(velocity.y, n),
        round(velocity.z, n),
        round(angular_velocity.x, n),
        round(angular_velocity.y, n),
        round(angular_velocity.z, n)
    ])

def calculate_spherical_coordinates(r, phase, bias=[0, 0, 0]):
    # phase 的取值范围应为[0,60)
    R = 0.2 # 复眼球半径
    angles = range(phase, 360, 60)
    points = []

    x = math.sqrt(R**2 - r**2) + bias[0]
    for angle in angles:
        angle = math.radians(angle)  # 角度转弧度
        y = r * math.cos(angle) + bias[1]
        z = r * math.sin(angle) + bias[2]
        points.append((float(format(x, '.5f')), float(format(y, '.5f')), float(format(z, '.5f'))))
    # print(points)
    return points

def points():

    df = pd.read_csv(points_dir) 
    points2 = []
    points = df[['X', 'Y', 'Z']].values.tolist()
    for i in range(0,len(points)):
        points2.append((float(format(points[i][0],'.5f')),float(format(points[i][1],'.5f')),float(format(points[i][2],'.5f'))))
    return points2

def save_point_transform(points_transform, center_transform,relative_location,type = 'forward'):
    # 获取中心参照的位置和朝向
    center_location = center_transform.location
    center_rotation = center_transform.rotation
    # print(center_transform)
    # print(center_rotation)
    # print(center_location)

    # 计算视野的位置和朝向
    view_location = carla.Location()
    view_location = center_location + \
        relative_location.x * center_transform.get_forward_vector() + \
        relative_location.y * center_transform.get_right_vector() + \
        relative_location.z * center_transform.get_up_vector()
    # print(view_location)
    # 与center同向
    view_rotation = center_rotation

    if type != 'same':
        # 指向center坐标中心
        view_rotation.pitch += -math.degrees(math.atan2(relative_location.z, math.sqrt(relative_location.x**2 + relative_location.y**2)))
        # math.degrees(x) 方法将角度 x 从弧度转换为度数
        # math.atan2(y,x) 返回给定的 y 及 x 坐标值的反正切值 atan(y / x)，以弧度为单位，结果是在 -pi 和 pi 之间。
        view_rotation.yaw +=  math.degrees(math.atan2(relative_location.y, relative_location.x)) + 180

        if type == 'backward':
            # 背向center坐标中心
            view_rotation.pitch = -view_rotation.pitch
            view_rotation.yaw += 180

    # print(view_rotation)
    # 设置位置和朝向
    view_transform = carla.Transform(view_location, view_rotation)
    # print(view_transform)

    points_transform.append((
        float(format(view_transform.location.x,'.5f')),float(format(view_transform.location.y,'.5f')),float(format(view_transform.location.z,'.5f')),float(format(view_transform.rotation.pitch,'.5f')),float(format(view_transform.rotation.yaw,'.5f')),float(format(view_transform.rotation.roll,'.5f'))
                            ))
    return points_transform


# ---------------------------------------------用于调试---------------------------------------------
def reset_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    set_sync_mode(False)
    destroy_actor('vehicle')

def draw_point(location=carla.Location(0,0,0), string='0'):
    point = world.get_map().get_waypoint(location)
    world.debug.draw_string(point.transform.location, str('f'), draw_shadow=True, color=carla.Color(r=255, g=0, b=0), life_time=60.0, persistent_lines=True)
    print(point)

def draw_points(dataset_path):
    df = pd.read_csv(os.path.join(dataset_path,'info/vehicle.csv'))

    type_list = df['Vehicle_Type'].tolist()
    color_list = df[['Color_R', 'Color_G', 'Color_B']].values.tolist()
    transform_list = df[['Transform_X', 'Transform_Y', 'Transform_Z']].values.tolist()

    color_list = [",".join(map(str, _)) for _ in color_list]
    transform_list = [tuple(_) for _ in transform_list]

    print(type_list, '\n', color_list, '\n', transform_list)

    for i in range(len(type_list)):
        spawn_transform = world.get_map().get_waypoint(carla.Location(*transform_list[i])).transform
        print(spawn_transform)
        world.debug.draw_string(spawn_transform.location, str(i), draw_shadow=True, color=carla.Color(r=255, g=0, b=0), life_time=60.0, persistent_lines=True)

    if start_point != None:
        world.debug.draw_string(start_point.transform.location, 'b', draw_shadow=True, color=carla.Color(r=255, g=0, b=0), life_time=60.0, persistent_lines=True)
        print(f'start_point: {start_point}')
    if stop_point != None:
        world.debug.draw_string(stop_point.transform.location, 's', draw_shadow=True, color=carla.Color(r=255, g=0, b=0), life_time=60.0, persistent_lines=True)
        print(f'stop_point: {stop_point}')

def simulate():
    points = []
    points.insert(0, (0, 0, 0))
    # points += calculate_spherical_coordinates(0.015, 30)
    # points += calculate_spherical_coordinates(0.03, 0)
    # points += calculate_spherical_coordinates(0.045, 30)
    # points += calculate_spherical_coordinates(0.06, 0)
    points += calculate_spherical_coordinates(0.025, 30)
    points += calculate_spherical_coordinates(0.05, 0)
    points += calculate_spherical_coordinates(0.075, 30)
    points += calculate_spherical_coordinates(0.1, 0)
    print(f'number of multi :{len(points)}')
    print(points)

    # 保存复眼坐标 
    column=['X','Y','Z'] # 列表对应每列的列名
    test=pd.DataFrame(columns=column,data=points)
    test.to_csv(os.path.join(out_dir, 'info/fuyan_coordinate.csv'),index=False) # 如果生成excel，可以用to_excel
 
    # 模拟车辆运行状况
    vehicle_dict = {}

    # 获取车辆信息
    df = pd.read_csv(os.path.join(out_dir, 'info/vehicle.csv'))
    type_list = df['Vehicle_Type'].tolist() # 获取车辆类型
    color_list = df[['Color_R', 'Color_G', 'Color_B']].values.tolist() # 获取车辆颜色
    color_list = [",".join(map(str, _)) for _ in color_list]   #将每个 RGB 值转换为字符串并用逗号连接起来，形成一个字符串列表
    transform_list = df[['Transform_X', 'Transform_Y', 'Transform_Z']].values.tolist() # 获取车辆位置
    transform_list = [tuple(_) for _ in transform_list] # 转换为元组形式
    #print(type_list, '\n', color_list, '\n', transform_list)

    # 创建车辆
    for i in range(len(type_list)):
        if i == 0:
            # 创建主车辆
            if start_point != None:
                spawn_transform = start_point.transform # 使用自定义的起始点位置和朝向
            else:
                spawn_transform = world.get_map().get_waypoint(carla.Location(*transform_list[i])).transform # 使用csv中的起始点位置和朝向
            spawn_transform.location.z=0.6
            vehicle_dict[type_list[i]] = create_vehicle(type_list[i], spawn_transform, color_list[i], auto=False, hero=True)
            ego_vehicle = vehicle_dict[type_list[i]]
            # time.sleep(0.1)
            # 设置观察者视角
            spectator = world.get_spectator()
            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30), carla.Rotation(pitch=-90)))
            # ego_vehicle_transform = ego_vehicle.get_transform()
            # set_spectator(ego_vehicle_transform, carla.Location(-0.0001, 0, 20)) 

        else:
            spawn_transform = world.get_map().get_waypoint(carla.Location(*transform_list[i])).transform
            spawn_transform.location.z=0.6
            vehicle_dict[type_list[i]] = create_vehicle(type_list[i], spawn_transform, color_list[i], False)
    print(vehicle_dict)
            
    for v in vehicle_dict.values():
        v.set_autopilot(True)

    while True:
        time.sleep(0.02)
        #设置观察者视角
        spectator = world.get_spectator()
        transform = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),carla.Rotation(pitch=-90)))
        
        set_traffic_light(state = carla.TrafficLightState.Off)
        if ego_vehicle.get_transform().location.distance(stop_point.transform.location) < 2.0:
            break

    control = carla.VehicleControl()
    control.throttle = 0.0  # 不加速
    control.brake = 5.0     # 使用刹车
    # 停止车辆运动
    for v in vehicle_dict.values():
        print(v)
        v.apply_control(control)
        v.set_autopilot(False)

    time.sleep(2)
    destroy_actor('vehicle')

# -----------------------调试时务必注释主函数------------------------------
# draw_point(carla.Location(0,0,0), '0')
set_spectator(carla.Transform(carla.Location(0, 0, 0)), carla.Location(0, 0, 20), 'bev')  #设置观察者视角
world.unload_map_layer(carla.MapLayer.ParkedVehicles)  #卸载停车车辆
draw_points(out_dir)   #车辆初始位置及停止位置
# get_points_self()
# simulate()
#reset_carla()

# ---------------------------------------------------------------------------主函数--------------------------------------------------------------------------------

if __name__ == '__main__':
    # 计算复眼摄像头位置
    # points = points()
    points = get_points_self()
    # points.insert(0, (0, 0, 0))
    # points += calculate_spherical_coordinates(0.015, 30)
    # points += calculate_spherical_coordinates(0.03, 0)
    # points += calculate_spherical_coordinates(0.045, 30)
    # points += calculate_spherical_coordinates(0.06, 0)

    # points += calculate_spherical_coordinates(0.025, 30)
    # points += calculate_spherical_coordinates(0.05, 0)
    # points += calculate_spherical/home/darker/ARLA_0.9.14/fuyan_collect/fuyan_17/info/vehicle.csv_coordinates(0.075, 30)
    # points += calculate_spherical_coordinates(0.1, 0)

    print(f'number of multi :{len(points)}')
    print(points)

    points_transform = []
    for j in range(len(points)):
        points_transform = save_point_transform(points_transform, carla.Transform(MAIN_LOCATION),carla.Location(*points[j]), 'backward')
    
    column_names = ['X', 'Y', 'Z', 'pitch', 'yaw', 'roll']
    df = pd.DataFrame(points_transform, columns=column_names)
    df.to_csv(os.path.join(out_dir, 'points/points_transform.csv'), index=False)
    
    print(points_transform)

    camera_rgb_list = [] # 创建RGB摄像头列表
    camera_seg_list = [] # 创建seg摄像头列表
    camera_queue = queue.Queue() # 创建摄像头queue队列
    seg_queue = queue.Queue() # 创建seg信息queue队列
    vehicle_dict = {} # 创建车辆字典

    try:
        set_sync_mode(True) # 设置同步模式
        # world.unload_map_layer(carla.MapLayer.ParkedVehicles) # 卸载地图中的停车车辆

        # 获取车辆信息
        df = pd.read_csv(os.path.join(out_dir, 'info/vehicle.csv')) 
        type_list = df['Vehicle_Type'].tolist() # 获取车辆类型
        color_list = df[['Color_R', 'Color_G', 'Color_B']].values.tolist() # 获取车辆颜色
        color_list = [",".join(map(str, _)) for _ in color_list]
        transform_list = df[['Transform_X', 'Transform_Y', 'Transform_Z']].values.tolist() # 获取车辆位置
        transform_list = [tuple(_) for _ in transform_list]
        # print(type_list, '\n', color_list, '\n', transform_list)

        # 创建车辆
        for i in range(len(type_list)):
            if i == 0:
                # 创建主车辆
                if start_point != None:
                    spawn_transform = start_point.transform # 使用自定义的起始点位置和朝向
                else:
                    spawn_transform = world.get_map().get_waypoint(carla.Location(*transform_list[i])).transform # 使用csv中的起始点位置和朝向
                spawn_transform.location.z=0.6
                ego_vehicle = create_vehicle(type_list[i], spawn_transform, color_list[i], auto=False, hero=True)
                vehicle_dict[type_list[i]] = ego_vehicle

                # time.sleep(0.1)
                # ego_vehicle_transform = ego_vehicle.get_transform()
                # set_spectator(ego_vehicle_transform, carla.Location(-0.0001, 0, 20)) 
                # 设置观察者视角
            else:
                spawn_transform = world.get_map().get_waypoint(carla.Location(*transform_list[i])).transform
                spawn_transform.location.z=0.6
                vehicle_dict[type_list[i]] = create_vehicle(type_list[i], spawn_transform, color_list[i], False)   
        # print("vehicle_dict: ")  
        print(vehicle_dict)

        # 同步模式
        for i in range(5):
            world.tick()
        
        # 获取并保存车辆3D信息
        df = pd.DataFrame(columns=['Vehicle_Type', 'length', 'width', 'height'])
        for i in range(len(vehicle_dict)):
            vehicle = vehicle_dict[type_list[i]]
            dimensions = vehicle.bounding_box.extent
            df.loc[i] = [type_list[i], round(2*dimensions.x, 3), round(2*dimensions.y, 3), round(2*dimensions.z, 3)]
        
        df.to_csv(os.path.join(out_dir, 'info/vehicle_dimension.csv'), index=False)



        # RGB摄像头
        # 生成主摄像头
        main_camera_transform = carla.Transform(MAIN_LOCATION)
        main_camera_rgb = creat_main_camera(ego_vehicle, main_camera_transform, 'rgb')
        main_camera_rgb.listen(lambda image: save_img_callback('main_camera_rgb', image, pic_rgb_output_dir, camera_queue)) # 将回调函数绑定到RGB摄像头
        camera_rgb_list.append(main_camera_rgb)

        # 生成BEV摄像头
        bev_camera_rgb = creat_bev_camera(ego_vehicle, 20, 'rgb')
        bev_camera_rgb.listen(lambda image: save_img_callback('bev_camera_rgb', image, pic_rgb_output_dir, camera_queue)) # 将回调函数绑定到RGB摄像头
        camera_rgb_list.append(bev_camera_rgb)


        # 生成复眼摄像头
        multi_camera_rgb_00 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[0]), 'rgb')
        multi_camera_rgb_01 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[1]), 'rgb')
        multi_camera_rgb_02 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[2]), 'rgb')
        multi_camera_rgb_03 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[3]), 'rgb')
        multi_camera_rgb_04 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[4]), 'rgb')
        multi_camera_rgb_05 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[5]), 'rgb')
        multi_camera_rgb_06 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[6]), 'rgb')
        multi_camera_rgb_07 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[7]), 'rgb')
        multi_camera_rgb_08 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[8]), 'rgb')
        multi_camera_rgb_09 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[9]), 'rgb')
        multi_camera_rgb_10 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[10]), 'rgb')
        multi_camera_rgb_11 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[11]), 'rgb')
        multi_camera_rgb_12 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[12]), 'rgb')
        multi_camera_rgb_13 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[13]), 'rgb')
        multi_camera_rgb_14 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[14]), 'rgb')
        multi_camera_rgb_15 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[15]), 'rgb')
        multi_camera_rgb_16 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[16]), 'rgb')
        # multi_camera_rgb_17 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[17]), 'rgb')
        # multi_camera_rgb_18 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[18]), 'rgb')
        # multi_camera_rgb_19 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[19]), 'rgb')
        # multi_camera_rgb_20 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[20]), 'rgb')
        # multi_camera_rgb_21 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[21]), 'rgb')
        # multi_camera_rgb_22 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[22]), 'rgb')
        # multi_camera_rgb_23 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[23]), 'rgb')
        # multi_camera_rgb_24 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[24]), 'rgb')

        camera_rgb_list.append(multi_camera_rgb_00)
        camera_rgb_list.append(multi_camera_rgb_01)
        camera_rgb_list.append(multi_camera_rgb_02)
        camera_rgb_list.append(multi_camera_rgb_03)
        camera_rgb_list.append(multi_camera_rgb_04)
        camera_rgb_list.append(multi_camera_rgb_05)
        camera_rgb_list.append(multi_camera_rgb_06)
        camera_rgb_list.append(multi_camera_rgb_07)
        camera_rgb_list.append(multi_camera_rgb_08)
        camera_rgb_list.append(multi_camera_rgb_09)
        camera_rgb_list.append(multi_camera_rgb_10)
        camera_rgb_list.append(multi_camera_rgb_11)
        camera_rgb_list.append(multi_camera_rgb_12)
        camera_rgb_list.append(multi_camera_rgb_13)
        camera_rgb_list.append(multi_camera_rgb_14)
        camera_rgb_list.append(multi_camera_rgb_15)
        camera_rgb_list.append(multi_camera_rgb_16)
        # camera_rgb_list.append(multi_camera_rgb_17)
        # camera_rgb_list.append(multi_camera_rgb_18)
        # camera_rgb_list.append(multi_camera_rgb_19)
        # camera_rgb_list.append(multi_camera_rgb_20)
        # camera_rgb_list.append(multi_camera_rgb_21)
        # camera_rgb_list.append(multi_camera_rgb_22)
        # camera_rgb_list.append(multi_camera_rgb_23)
        # camera_rgb_list.append(multi_camera_rgb_24)

        multi_camera_rgb_00.listen(lambda image: save_img_callback('multi_camera_rgb_00', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_01.listen(lambda image: save_img_callback('multi_camera_rgb_01', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_02.listen(lambda image: save_img_callback('multi_camera_rgb_02', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_03.listen(lambda image: save_img_callback('multi_camera_rgb_03', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_04.listen(lambda image: save_img_callback('multi_camera_rgb_04', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_05.listen(lambda image: save_img_callback('multi_camera_rgb_05', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_06.listen(lambda image: save_img_callback('multi_camera_rgb_06', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_07.listen(lambda image: save_img_callback('multi_camera_rgb_07', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_08.listen(lambda image: save_img_callback('multi_camera_rgb_08', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_09.listen(lambda image: save_img_callback('multi_camera_rgb_09', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_10.listen(lambda image: save_img_callback('multi_camera_rgb_10', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_11.listen(lambda image: save_img_callback('multi_camera_rgb_11', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_12.listen(lambda image: save_img_callback('multi_camera_rgb_12', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_13.listen(lambda image: save_img_callback('multi_camera_rgb_13', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_14.listen(lambda image: save_img_callback('multi_camera_rgb_14', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_15.listen(lambda image: save_img_callback('multi_camera_rgb_15', image, pic_rgb_output_dir, camera_queue))
        multi_camera_rgb_16.listen(lambda image: save_img_callback('multi_camera_rgb_16', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_17.listen(lambda image: save_img_callback('multi_camera_rgb_17', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_18.listen(lambda image: save_img_callback('multi_camera_rgb_18', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_19.listen(lambda image: save_img_callback('multi_camera_rgb_19', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_20.listen(lambda image: save_img_callback('multi_camera_rgb_20', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_21.listen(lambda image: save_img_callback('multi_camera_rgb_21', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_22.listen(lambda image: save_img_callback('multi_camera_rgb_22', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_23.listen(lambda image: save_img_callback('multi_camera_rgb_23', image, pic_rgb_output_dir, camera_queue))
        # multi_camera_rgb_24.listen(lambda image: save_img_callback('multi_camera_rgb_24', image, pic_rgb_output_dir, camera_queue))

        # 实例分割摄像头
        # 生成主摄像头
        main_camera_transform = carla.Transform(MAIN_LOCATION)
        main_camera_seg = creat_main_camera(ego_vehicle, main_camera_transform, 'instance_segmentation')
        main_camera_seg.listen(lambda image: save_img_callback('main_camera_seg', image, pic_seg_output_dir, camera_queue)) # 将回调函数绑定到seg摄像头
        camera_seg_list.append(main_camera_seg)

        # 生成BEV摄像头
        bev_camera_seg = creat_bev_camera(ego_vehicle, 20, 'instance_segmentation')
        bev_camera_seg.listen(lambda image: save_img_callback('bev_camera_seg', image, pic_seg_output_dir, camera_queue)) # 将回调函数绑定到seg摄像头
        camera_seg_list.append(bev_camera_seg)

        # 生成复眼摄像头
        multi_camera_seg_00 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[0]), 'instance_segmentation')
        multi_camera_seg_01 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[1]), 'instance_segmentation')
        multi_camera_seg_02 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[2]), 'instance_segmentation')
        multi_camera_seg_03 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[3]), 'instance_segmentation')
        multi_camera_seg_04 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[4]), 'instance_segmentation')
        multi_camera_seg_05 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[5]), 'instance_segmentation')
        multi_camera_seg_06 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[6]), 'instance_segmentation')
        multi_camera_seg_07 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[7]), 'instance_segmentation')
        multi_camera_seg_08 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[8]), 'instance_segmentation')
        multi_camera_seg_09 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[9]), 'instance_segmentation')
        multi_camera_seg_10 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[10]), 'instance_segmentation')
        multi_camera_seg_11 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[11]), 'instance_segmentation')
        multi_camera_seg_12 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[12]), 'instance_segmentation')
        multi_camera_seg_13 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[13]), 'instance_segmentation')
        multi_camera_seg_14 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[14]), 'instance_segmentation')
        multi_camera_seg_15 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[15]), 'instance_segmentation')
        multi_camera_seg_16 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[16]), 'instance_segmentation')
        # multi_camera_seg_17 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[17]), 'instance_segmentation')
        # multi_camera_seg_18 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[18]), 'instance_segmentation')
        # multi_camera_seg_19 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[19]), 'instance_segmentation')
        # multi_camera_seg_20 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[20]), 'instance_segmentation')
        # multi_camera_seg_21 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[21]), 'instance_segmentation')
        # multi_camera_seg_22 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[22]), 'instance_segmentation')
        # multi_camera_seg_23 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[23]), 'instance_segmentation')
        # multi_camera_seg_24 = creat_multi_camera(ego_vehicle, carla.Transform(MAIN_LOCATION), carla.Location(*points[24]), 'instance_segmentation')

        camera_seg_list.append(multi_camera_seg_00)
        camera_seg_list.append(multi_camera_seg_01)
        camera_seg_list.append(multi_camera_seg_02)
        camera_seg_list.append(multi_camera_seg_03)
        camera_seg_list.append(multi_camera_seg_04)
        camera_seg_list.append(multi_camera_seg_05)
        camera_seg_list.append(multi_camera_seg_06)
        camera_seg_list.append(multi_camera_seg_07)
        camera_seg_list.append(multi_camera_seg_08)
        camera_seg_list.append(multi_camera_seg_09)
        camera_seg_list.append(multi_camera_seg_10)
        camera_seg_list.append(multi_camera_seg_11)
        camera_seg_list.append(multi_camera_seg_12)
        camera_seg_list.append(multi_camera_seg_13)
        camera_seg_list.append(multi_camera_seg_14)
        camera_seg_list.append(multi_camera_seg_15)
        camera_seg_list.append(multi_camera_seg_16)
        # camera_seg_list.append(multi_camera_seg_17)
        # camera_seg_list.append(multi_camera_seg_18)
        # camera_seg_list.append(multi_camera_seg_19)
        # camera_seg_list.append(multi_camera_seg_20)
        # camera_seg_list.append(multi_camera_seg_21)
        # camera_seg_list.append(multi_camera_seg_22)
        # camera_seg_list.append(multi_camera_seg_23)
        # camera_seg_list.append(multi_camera_seg_24)

        multi_camera_seg_00.listen(lambda image: save_img_callback('multi_camera_seg_00', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_01.listen(lambda image: save_img_callback('multi_camera_seg_01', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_02.listen(lambda image: save_img_callback('multi_camera_seg_02', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_03.listen(lambda image: save_img_callback('multi_camera_seg_03', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_04.listen(lambda image: save_img_callback('multi_camera_seg_04', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_05.listen(lambda image: save_img_callback('multi_camera_seg_05', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_06.listen(lambda image: save_img_callback('multi_camera_seg_06', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_07.listen(lambda image: save_img_callback('multi_camera_seg_07', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_08.listen(lambda image: save_img_callback('multi_camera_seg_08', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_09.listen(lambda image: save_img_callback('multi_camera_seg_09', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_10.listen(lambda image: save_img_callback('multi_camera_seg_10', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_11.listen(lambda image: save_img_callback('multi_camera_seg_11', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_12.listen(lambda image: save_img_callback('multi_camera_seg_12', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_13.listen(lambda image: save_img_callback('multi_camera_seg_13', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_14.listen(lambda image: save_img_callback('multi_camera_seg_14', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_15.listen(lambda image: save_img_callback('multi_camera_seg_15', image, pic_seg_output_dir, camera_queue))
        multi_camera_seg_16.listen(lambda image: save_img_callback('multi_camera_seg_16', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_17.listen(lambda image: save_img_callback('multi_camera_seg_17', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_18.listen(lambda image: save_img_callback('multi_camera_seg_18', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_19.listen(lambda image: save_img_callback('multi_camera_seg_19', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_20.listen(lambda image: save_img_callback('multi_camera_seg_20', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_21.listen(lambda image: save_img_callback('multi_camera_seg_21', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_22.listen(lambda image: save_img_callback('multi_camera_seg_22', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_23.listen(lambda image: save_img_callback('multi_camera_seg_23', image, pic_seg_output_dir, camera_queue))
        # multi_camera_seg_24.listen(lambda image: save_img_callback('multi_camera_seg_24', image, pic_seg_output_dir, camera_queue))
        
        world.tick()
        # print(camera_rgb_list)
        # print(camera_seg_list)
        
        # 设置为自动驾驶
        for v in vehicle_dict.values():
            v.set_autopilot(True)

        # 创建文件用于保存车辆位姿速度信息
        for k in vehicle_dict.keys():
            with open(os.path.join(data_output_dir, f'{k}.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Frame', 'X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Linear_X', 'Linear_Y', 'Linear_Z', 'Angular_X', 'Angular_Y', 'Angular_Z'])

        start_frame = world.get_snapshot().frame # 获取当前仿真帧号
        print(f'start_frame: {start_frame}')

        a=1

        while True:
            set_traffic_light(state = carla.TrafficLightState.Off) # 关闭红绿灯

            world.tick() # 更新世界
            time.sleep(0.05)
            
            # 数据集保存数量
            print("collected "+str(a))
            a+=1

            # 观察者视角跟随主车辆
            ego_vehicle_transform = ego_vehicle.get_transform()
            set_spectator(ego_vehicle_transform, carla.Location(20, 0, 40), 'bev')

            # 获取摄像头数据
            try:
                for camera in camera_rgb_list:
                    frame = camera_queue.get(True, 1.0)
                for camera in camera_seg_list:
                    frame = camera_queue.get(True, 1.0)
            except queue.Empty:
                print("!!Some of the Cameras information is missed!!")

            # 获取当前仿真帧号
            current_frame = world.get_snapshot().frame 
            frame = current_frame - start_frame

            # 将数据写入CSV文件
            for k, v in vehicle_dict.items():
                save_pose_data(os.path.join(data_output_dir, f'{k}.csv'), v, frame)

            # 到达终点停止仿真
            if ego_vehicle.get_transform().location.distance(stop_point.transform.location) < 2.0:
                break
        
        # 停止监听并销毁摄像头
        for _ in camera_rgb_list:
            # print(_)
            _.stop()
            _.destroy()

        for _ in camera_seg_list:
            # print(_)
            _.stop()
            _.destroy()
        
        # 停止车辆运动
        control = carla.VehicleControl()
        control.throttle = 0.0  # 不加速
        control.brake = 1.0     # 使用刹车
        for v in vehicle_dict.values():
            v.set_autopilot(False)
            v.apply_control(control)
        # world.tick()
        now_frame = world.get_snapshot().frame
        #  初始化实例拍摄摄像头
        inseg_camera = creat_inseg_camera(location=SEG_LOCATION, kind = 'instance_segmentation')
        inseg_camera.listen(lambda image: save_img_callback_2(image, pic_seg_info_dir, seg_queue))
        world.tick()
        
    except KeyboardInterrupt: 
        print('\nKeyboardInterrupt ...')

    finally:
        # 拍摄实例分割信息并销毁车辆
        for k,v in vehicle_dict.items():
            now_frame = world.get_snapshot().frame
            vehicle_name = k
            v.set_transform(carla.Transform(location=carla.Location(SEG_LOCATION.x, SEG_LOCATION.y, 0.6), rotation=carla.Rotation(0, 0, 0)))
            for i in range(3):
                try:
                    world.tick()
                except:
                    world.tick()
                time.sleep(0.1)
                seg_queue.get(True, 1)

            v.destroy() # 拍摄完成后销毁车辆
            # world.tick()

        # 停止监听并销毁实例信息摄像头
        inseg_camera.stop()
        inseg_camera.destroy()
        # world.tick()

        set_sync_mode(False)
        time.sleep(0.1)
