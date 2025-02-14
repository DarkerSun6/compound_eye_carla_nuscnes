import hashlib
import numpy as np
from pyquaternion import Quaternion
import json
import carla

def transform_timestamp(timestamp):
    return int(timestamp*10e5)

def generate_token(key,data):
    obj = hashlib.md5(str(key).encode('utf-8'))     # 创建一个 MD5 对象，并使用密钥进行初始化 
    obj.update(str(data).encode('utf-8'))           # 更新 MD5 对象，添加数据
    result = obj.hexdigest()                        # 获取最终的 MD5 哈希值
    return result

def dump(data,path):
    with open(path, "w") as filedata:
        json.dump(data, filedata, indent=0, separators=(',',':'))
        # indent:参数根据数据格式缩进显示，读起来更加清晰。
        # separators:是分隔符的意思，参数意思分别为不同dict项之间的分隔符和dict项内key和value之间的分隔符，把：和，后面的空格都除去了。

def load(path):
    with open(path, "r") as filedata:
        return json.load(filedata)

def get_intrinsic(fov, image_size_x,image_size_y):
    # 相机内参矩阵
    focal = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_size_x / 2.0
    K[1, 2] = image_size_y / 2.0
    return K

def get_nuscenes_rt(transform,mode=None):
    translation = [transform.location.x,
                -transform.location.y,
                transform.location.z]
    if mode == "zxy":
        rotation_matrix1 = np.array([
            [0,0,1],
            [1,0,0],
            [0,-1,0]
        ])
    else:
        rotation_matrix1 = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])

    rotation_matrix2 = np.array(transform.get_matrix())[:3,:3]  #相机外参数矩阵
    rotation_matrix3 = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
    rotation_matrix = rotation_matrix3@rotation_matrix2@rotation_matrix1
    quat = Quaternion(matrix=rotation_matrix,rtol=1, atol=1).elements.tolist()  #四元数转换
    return quat,translation

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))