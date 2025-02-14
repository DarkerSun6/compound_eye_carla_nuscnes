import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import carla

out_dir = '/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan'
# 极坐标转换为笛卡尔坐标
def polar2cart(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

def get_points_self(r,theta,pitch):
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

    column_names = ['X', 'Y', 'Z']
    df = pd.DataFrame(points, columns=column_names)
    df.to_csv(os.path.join(out_dir, 'points/points_17_or.csv'), index=False)

    return points

def get_points(out_dir_name):
    '''
    功能：读取文件中的坐标点并储存为列表
    out_dir_name:摄像机点位储存位置
    返回：摄像机点位列表
    '''
    df = pd.read_csv(out_dir_name) 
    points2 = []
    points = df[['X', 'Y', 'Z']].values.tolist()
    for i in range(0,len(points)):
        points2.append((float(format(points[i][0],'.5f')),float(format(points[i][1],'.5f')),float(format(points[i][2],'.5f'))))
    return points2

def cartesian_to_spherical(r,x, y, z):
    """
    功能：直角坐标系转换为球坐标系
    直角坐标系坐标点：(x, y, z)
    球坐标系坐标点：(r, theta, phi)
    """
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    """
    功能：球坐标系转换为直角坐标系
    球坐标系坐标点;(r, theta, phi)
    直角坐标系坐标点：(x, y, z)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def set_lim(ax,r,mul):
    '''
    功能：设置坐标轴范围、名称及比例
    ax:3D坐标系
    r:球体半径
    mul:成像距离与球体半径的倍数
    '''
    axis_range = r*mul
    ax.set_xlim([-axis_range, axis_range])
    ax.set_ylim([-axis_range, axis_range])
    ax.set_zlim([-axis_range, axis_range])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.gca().set_box_aspect((1, 1, 1))
 
def plot_globe(r,ax,alpha):
    '''
    功能：三维坐标系下绘制球体
    r:半径
    ax:3d坐标系
    alpha:透明度

    '''
    # 获得绘制半球球面的坐标
    t1=np.linspace(0,np.pi,1000)
    t2=np.linspace(0,np.pi*2,2000)
    t1,t2=np.meshgrid(t1,t2)

    x1=abs((r*0.99)*np.sin(t1)*np.cos(t2))
    y1=(r*0.99)*np.sin(t1)*np.sin(t2)
    z1=(r*0.99)*np.cos(t1)

    x2=abs((r*0.95)*np.sin(t1)*np.cos(t2))  # x始终为正
    y2=(r*0.95)*np.sin(t1)*np.sin(t2)
    z2=(r*0.95)*np.cos(t1)
    # 绘制3D球体

    # 绘制球面
    ax.plot_surface(x1,y1,z1,color="#606060",alpha=alpha)
    # ax.plot_surface(x2,y2,z2,color="#606060",alpha=alpha)

def plot_points(points,ax,is_one=False):
    '''
    功能：绘制摄像机点位
    points:摄像机点位列表
    ax:3d坐标系
    is_one:是否只绘制一个点
    '''
    if is_one:
        len_plot = 1
    else:
        len_plot = len(points)

    for i in range(0,len_plot):
        ax.scatter(points[i][0], points[i][1], points[i][2], color='r', s=30)
        ax.text(points[i][0], points[i][1], points[i][2], '%s' % (str(i)), size=16, zorder=1, color='k',verticalalignment='bottom', horizontalalignment='center')

def plot_camera_points(camera_positions, fov_angle, radius, ax,is_axpand=False,mul=1.0):
    """
    功能：绘制指向曲线
    camera_positions:摄像机点位列表
    fov_angle:视场角
    radius:球体半径
    ax:3d坐标系
    """
    fov_radius = radius * np.tan(np.radians(fov_angle) / 2)
    for x, y, z in camera_positions:
        # ax.plot([0, x], [0, y], [0, z],'-',color="#00FF00")     # 绿色
        ax.plot([0, x], [0, y], [0, z],'-',color="#FF0000")     # 红色
    if is_axpand:
        for x, y, z in camera_positions:
            _, theta, phi = cartesian_to_spherical(radius,x, y, z)
            # Calculate the end point of the FOV line
            fov_end_x = x + fov_radius * np.sin(theta) * np.cos(phi)*mul
            fov_end_y = y + fov_radius * np.sin(theta) * np.sin(phi)*mul
            fov_end_z = z + fov_radius * np.cos(theta)*mul
            # ax.plot([x, fov_end_x], [y, fov_end_y], [z, fov_end_z], '-',color="#00FF00")    # 绿色
            ax.plot([x, fov_end_x], [y, fov_end_y], [z, fov_end_z], '-',color="#FF0000")    # 红色

def plot_camera_views(r,fov,points,ax,mul,alpha,is_one=False,toward=True,plot='surface'):
    '''
    功能：绘制摄像头成像区域
    r:球体半径
    fov:视场角
    points:摄像机点位
    ax:3d坐标系
    mul:成像距离是球体半径的倍数
    alpha:透明度
    is_one:是否只绘制一个成像
    toward:是否绘制摄像头到成像四点的虚线
    plot:成像区域显示形式:surface-面,line-线
    '''
    distance = r*mul
    h = distance * np.sin(np.radians(fov / 2))      # 成像区域正方形的边长
    H = np.sqrt(np.power(distance,2) - 2*np.power(h,2))  # 初始旋转矩阵的纵坐标
    if is_one:
        len_plot = 1
    else:
        len_plot = len(points)

    for j in range(0,len_plot):
        # 当yz坐标为零时，摄像机点位在x轴上，因此初始旋转矩阵的纵坐标放在z轴上
        if (points[j][1]==0.0) & (points[j][2]==0.0):
            square = np.array([
            [-h, -h, H],
            [-h, h, H],
            [h, h, H],
            [h, -h, H]
            ])
            z_unit = np.array([0, 0, 1])
        # 当yz坐标不为零时，由于保证成像在x轴上没有旋转，因此初始旋转矩阵的纵坐标放在x轴上
        else:
            square = np.array([
            [H, -h, -h],
            [H, -h,  h],
            [H,  h,  h],
            [H,  h, -h]
            ])
            z_unit = np.array([1, 0, 0])
        
        # 摄像头指向原点的单位方向向量
        direction = points[j] / np.linalg.norm(points[j])
        # 构建旋转到原点方向的旋转矩阵
        rot_axis = np.cross(z_unit, direction)
        rot_angle = np.arccos(np.dot(z_unit, direction))
        rotation = R.from_rotvec(rot_angle * rot_axis / np.linalg.norm(rot_axis)) # type: ignore
        # 应用旋转矩阵
        rotated_square = rotation.apply(square)
        # 平移到摄像头位置
        translated_square = rotated_square + points[j]
        # print("translated_square:")
        # print(translated_square)

        # 绘制摄像头位置
        # ax.scatter(*points[j], color='red', s=100, label='Camera Position%02d' % j)
        # 绘制成像平面
        if plot == 'line':
            for i, point in enumerate(translated_square):
                next_point = translated_square[(i + 1) % len(translated_square)]
                ax.plot(*zip(point, next_point), color='blue')
        elif plot == 'surface':
            square_points = []
            for i in range(0,4):
                square_points.append([float(format(translated_square[i][0],'.5f')),float(format(translated_square[i][1],'.5f')),float(format(translated_square[i][2],'.5f'))])
                # print(square_points)
                # square_points = np.array(square_points)
                # print(square_points.shape)
            square_points =[square_points]
            pc = Poly3DCollection(square_points, alpha=alpha, facecolors='grey')
            ax.add_collection3d(pc)

        # 从相机到成像画线
        if toward:
            for point in translated_square:
                ax.plot(*zip(point, points[j]), color='green', linestyle='--')
        
        
def carla_point_transform(points_transform, center_transform,relative_location,type = 'forward'):
    # 获取中心参照的位置和朝向
    center_location = center_transform.location
    center_rotation = center_transform.rotation

    # 计算视野的位置和朝向
    view_location = carla.Location()
    view_location = center_location + \
        relative_location.x * center_transform.get_forward_vector() + \
        relative_location.y * center_transform.get_right_vector() + \
        relative_location.z * center_transform.get_up_vector()
    
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
            view_rotation.yaw -= 180

    view_transform = carla.Transform(view_location, view_rotation)

    points_transform.append((
        float(format(view_transform.location.x,'.5f')),float(format(view_transform.location.y,'.5f')),float(format(view_transform.location.z,'.5f')),float(format(view_transform.rotation.pitch,'.5f')),float(format(view_transform.rotation.yaw,'.5f')),float(format(view_transform.rotation.roll,'.5f'))
                            ))
    return points_transform
def save_point_transform(points,center_transform):
    points_transform = []
    for j in range(len(points)):
        points_transform = carla_point_transform(points_transform, carla.Transform(center_transform),carla.Location(*points[j]), 'backward')
    column_names = ['X', 'Y', 'Z', 'pitch', 'yaw', 'roll']
    df = pd.DataFrame(points_transform, columns=column_names)
    df.to_csv(os.path.join(out_dir, 'points/points_transform.csv'), index=False)

    return points_transform

def main():
    r = 0.2  # 半球半径
    fov = 60  # 视场角

    # 从文件中得到点的坐标
    # points = []
    # out_dir = 'F:/carla0.9.14/collect_fuyan/'
    # out_name = 'points/points_17.csv'
    # out_dir_name = os.path.join(out_dir,out_name)
    # points = get_points(out_dir_name)

    # 自己生成摄像头点的坐标
    theta = 30
    pitch = 15
    points = get_points_self(r,theta,pitch) #保存为points.csv，仅xyz坐标
    print(points)

    center_transform = carla.Location(0.3, 0, 1.8)
    points_transform = save_point_transform(points, center_transform)
    print(points_transform)


    points_sph = []
    for i in range(0,len(points)):
        points_sph.append(cartesian_to_spherical(r, points[i][0], points[i][1], points[i][2] ))
    # print(points_sph)

    fig=plt.figure()
    # ax=plt.axes(projection="3d")
    ax = fig.add_subplot(111, projection='3d')

    set_lim(ax,r,mul=1)                           # 设置坐标轴
    plot_globe(r,ax,alpha=0.3)                       # 绘制球体
    plot_points(points,ax,is_one=False)                        # 绘制摄像头点位
    # plot_points(points,ax,is_one=True)                        # 绘制一个摄像头点位
    # plot_camera_points(points,fov,r,ax,is_axpand=True,mul=1.75)           # 绘制摄像头朝向
    # plot_camera_views(r,fov,points,ax,alpha=0.5,mul=2,is_one=False,toward=False,plot='surface')   # 绘制摄像头成像
    # plot_camera_views(r,fov,points,ax,alpha=0.5,mul=1.5,is_one=True,toward=True,plot='surface')   # 绘制摄像头成像

    
    ax.axis('off')              # 设置坐标轴不可见
    # ax.view_init(elev=180, azim=0)               # 设置观察角度       
    # ax.view_init(elev=180, azim=60)            # 设置观察角度，单个成像
    # ax.view_init(elev=-150, azim=20)
    ax.view_init(elev=0, azim=0)                # 匹配照片裁剪

    plt.show()

if __name__ == "__main__":
    main()