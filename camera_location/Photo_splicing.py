import cv2
import numpy as np
import os
from PIL import Image

# 原始采集数据集照片位置
# path = 'F:/carla0.9.14/collect_fuyan/fuyan_17/pic_rgb/multi_camera_rgb_'
path = r"F:\carla0.9.14\collect_fuyan\fuyan_17\4_120x120_60_ZYX\pic_rgb\multi_camera_rgb_"
path2 = ['00','01','02','03','04','05','06','07','08','09',
         '10','11','12','13','14','15','16']
directories = [path + _ for _ in path2]
# print(directories)

# 生成照片储存位置
# out_path = 'F:/carla0.9.14/collect_fuyan/fuyan/pic_rgb/multi_camera_rgb_re'
out_path = r"F:\carla0.9.14\collect_fuyan\fuyan_17\4_120x120_60_ZYX\pic_rgb\multi_camera_rgb_re"

os.makedirs(out_path, exist_ok=True)

# 设置照片的预期尺寸
new_width = 1440  # 这里设置每张图片的预期宽度
new_height = 720 # 这里设置每张图片的预期高度
dim = (new_width, new_height)

# 确保列表中有17个文件夹路径
if len(directories) != 17:
    raise ValueError('需要提供17个文件夹路径。')



# 获取第一个文件夹内的所有图片名称，其他文件夹中的图片应与此处的图片名称一致
file_names = os.listdir(directories[0])
# print(file_names)

# 为每个名称，从每个文件夹中读取相应的图片并进行拼接
for file_name in file_names:
    pics = []
    for dir in directories:
        pic_path = os.path.join(dir, file_name)
        img = cv2.imread(pic_path)
        assert img is not None, f'无法读取图片: {pic_path}'
        pics.append(img)

    for i in range(0,len(pics)):
        pics[i] = cv2.copyMakeBorder(pics[i],2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))

    # 分割成三行，每行分别有6, 5, 6张图
    pos1 = [1,3,5,7,8,10]
    row1 = [pics[pos] for pos in pos1 if pos < len(pics) ]
    pos2 = [0,2,4,6,8,]
    row2 = [pics[pos] for pos in pos2 if pos < len(pics) ]
    pos3 = [9,11,12,13,14,15]
    row3 = [pics[pos] for pos in pos3 if pos < len(pics) ]

    # 添加空白图片在第二行的两端
    blank_image = np.ones((row2[0].shape[0], row2[0].shape[1] // 2, row2[0].shape[2]), dtype=np.uint8)*255
    row2 = [blank_image] + row2 +[blank_image]
    
    # 拼接构建大图
    row1 = np.hstack(row1)
    row2 = np.hstack(row2)
    row3 = np.hstack(row3)
    final_img = np.vstack((row1, row2, row3))

    # 写入合并后的图片
    outfile_name = os.path.join(out_path, file_name)
    resized = cv2.resize(final_img, dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite(outfile_name, resized)
    print(f'已成功生成拼接图片: {outfile_name}')