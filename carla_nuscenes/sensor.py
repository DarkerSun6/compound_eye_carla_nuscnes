import numpy as np
import carla
from .actor import Actor
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib


def parse_image(image):
    array = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8, buffer=image.raw_data,order="C")
    return array

def parse_lidar_data(lidar_data):
    # for i in range(len(lidar_data)):
    #     lidar_data[i].point.y *= -1
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    points[:, 1] = -points[:, 1]
    return points

    # points = []
    # current_channel = 0
    # end_idx = lidar_data.get_point_count(current_channel)
    # for idx,data in enumerate(lidar_data):
    #     point = [data.point.x,data.point.y,data.point.z,data.intensity,current_channel]
    #     if idx==end_idx:
    #         current_channel+=1
    #         end_idx+=lidar_data.get_point_count(current_channel)
    #     points.append(point)
    # return np.array(points)

def parse_radar_data(radar_data):
    points_1 = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points_1 = np.reshape(points_1, (len(radar_data), 4))
    # print(points_1)
    l=np.cos(points_1[:,2])*points_1[:,3]
    z=np.sin(points_1[:,2])*points_1[:,3]
    x=np.cos(points_1[:,1])*l
    y=np.sin(points_1[:,1])*l
    
    # plt.figure("3D Scatter", facecolor="lightgray",figsize=(20,20),dpi=80)
    # ax3d = plt.gca(projection="3d") 
    
    # ax3d.scatter(x, y, z, s=10, cmap="jet", marker="o")
    # ax3d.view_init(elev=0, azim=-70)
    # #ax3d.set_yticks(np.arange(0, 100, 10))
    # plt.show()

    pcd = o3d.geometry.PointCloud()
    # 将雷达数据转换为点云坐标
    points = []
    for i in range(len(x)):
        points.append([x[i], y[i], z[i]])
    # for detection in radar_data:
    #     depth = detection.depth
    #     azimuth = np.radians(detection.azimuth)
    #     altitude = np.radians(detection.altitude)
        
    #     x = depth * np.cos(altitude) * np.sin(azimuth)
    #     y = depth * np.cos(altitude) * np.cos(azimuth)
    #     z = depth * np.sin(altitude)
        
    #     points.append([x, y, z])
    
    # 将点添加到点云中
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 保存点云到PCD文件
    # o3d.io.write_point_cloud(filename, pcd)

    # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).copy()
    # points = np.reshape(points, (-1, 4))
    # return points
    return pcd

def parse_data(data):
    if isinstance(data,carla.Image):
        return parse_image(data)
    elif isinstance(data,carla.RadarMeasurement):
        return parse_radar_data(data)
    elif isinstance(data,carla.LidarMeasurement):
        return parse_lidar_data(data)

def get_data_shape(data):
    if isinstance(data,carla.Image):
        return data.height,data.width
    else:
        return 0,0
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
            
    def add_data(self,data):
        self.data_list.append((self.actor.parent.get_transform(),data))

    def get_transform(self):
        return self.actor.get_transform()