import cv2
import numpy as np
import os
from PIL import Image

# 原始采集数据集照片位置
images_path_1 = 'F:/carla0.9.14/collect_fuyan/fuyan_17/7_120x120_60_hyd/pic_rgb/'

images_path_2 = ['00','01','02','03','04','05','06','07','08','09',
         '10','11','12','13','14','15','16']
images_path = [images_path_1 + 'multi_camera_rgb_' + _ for _ in images_path_2]
# print(directories)
 
# images_path_1 = 'F:/carla0.9.14/data/aaa/'
# images_path = [images_path_1 + 'multi_camera_rgb_' + _ for _ in images_path_2]


# 生成照片储存位置
# pohto_splicing_path_1 = 'multi_camera_rgb_re'
pohto_splicing_path_1 = 'main_camera_rgb'
pohto_splicing_path = images_path_1 + pohto_splicing_path_1
os.makedirs(pohto_splicing_path, exist_ok=True)

# 设置视频储存位置
video_name= "collect_main.mp4"
video_path_name = images_path_1 + video_name

def set_image_size(horizontal,vertical,mul):
    '''
    功能:设置大图的预期尺寸
    horizontal:横排图像数量
    vertical:竖排图像数量
    mul:大图的放大倍数
    '''
    image_name = os.listdir(images_path[0])
    image_path = os.path.join(images_path[0], image_name[0])
    print(image_path)
    img = Image.open(image_path)
    w = img.width
    h = img.height
    print("the size of image:(%d,%d)" %(w,h))

    new_width = w*horizontal*mul  # 这里设置每张图片的预期宽度
    new_height = h*vertical*mul # 这里设置每张图片的预期高度
    print("the size of new image:(%d,%d)" %(new_width,new_height))
    
    dim = (int(new_width), int(new_height))
    return dim

def photo_splicing(directories,dim):
    '''
    功能：将复眼摄像头采集到的图像拼接为大图
    directories:摄像头采集图像储存位置
    dim:大图的宽和高
    注:对不同复眼摄像头点位,需修改hstack及vstack,并修改pos中对应的图像位置,位置可参考plot_camera程序绘制的摄像头点位图
    '''
    # 获取第一个文件夹内的所有图片名称，其他文件夹中的图片与此处的图片名称一致
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

        # 给原始照片生成白边
        for i in range(0,len(pics)):
            pics[i] = cv2.copyMakeBorder(pics[i],2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))

        # pow中设置图片位置
        pos1 = [5,6,7,8,9,10]
        row1 = [pics[pos] for pos in pos1 if pos < len(pics) ]
        pos2 = [0,1,2,3,4]
        row2 = [pics[pos] for pos in pos2 if pos < len(pics) ]
        pos3 = [11,12,13,14,15,16]
        row3 = [pics[pos] for pos in pos3 if pos < len(pics) ]
        # pos4 = [9,3,5,12]
        # row4 = [pics[pos] for pos in pos4 if pos < len(pics) ]
        # pos5 = [13,8,15]
        # row5 = [pics[pos] for pos in pos5 if pos < len(pics) ]

        # 添加空白图片在第二行的两端
        blank_image = np.ones((row2[0].shape[0], row2[0].shape[1] // 2, row2[0].shape[2]), dtype=np.uint8)*255
        # row1 = [blank_image] + row1 +[blank_image]
        row2 = [blank_image] + row2 +[blank_image]
        # row3 = [blank_image] + row3 +[blank_image]
        # row4 = [blank_image] + row4 +[blank_image]
        # row5 = [blank_image] + row5 +[blank_image]
        
        # 拼接构建大图
        row1 = np.hstack(row1)
        row2 = np.hstack(row2)
        row3 = np.hstack(row3)
        # row4 = np.hstack(row4)
        # row5 = np.hstack(row5)
        final_img = np.vstack((row1,row2,row3))

        # 写入合并后的图片
        outfile_name = os.path.join(pohto_splicing_path, file_name)
        resized = cv2.resize(final_img, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite(outfile_name, resized)
        print(f'已成功生成拼接图片: {outfile_name}')

def photo_splicing_2(directories,dim):
    '''
    collect_17
    功能：将复眼摄像头采集到的图像拼接为大图
    directories:摄像头采集图像储存位置
    dim:大图的宽和高
    注:对不同复眼摄像头点位,需修改hstack及vstack,并修改pos中对应的图像位置,位置可参考plot_camera程序绘制的摄像头点位图
    '''
    # 获取第一个文件夹内的所有图片名称，其他文件夹中的图片与此处的图片名称一致
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

        # 给原始照片生成白边
        for i in range(0,len(pics)):
            pics[i] = cv2.copyMakeBorder(pics[i],2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))

        # pow中设置图片位置
        pos1 = [14,7,16]
        row1 = [pics[pos] for pos in pos1 if pos < len(pics) ]
        pos2 = [10,4,6,11]
        row2 = [pics[pos] for pos in pos2 if pos < len(pics) ]
        pos3 = [2,0,1]
        row3 = [pics[pos] for pos in pos3 if pos < len(pics) ]
        pos4 = [9,3,5,12]
        row4 = [pics[pos] for pos in pos4 if pos < len(pics) ]
        pos5 = [13,8,15]
        row5 = [pics[pos] for pos in pos5 if pos < len(pics) ]

        # 添加空白图片在第二行的两端
        blank_image = np.ones((row2[0].shape[0], row2[0].shape[1] // 2, row2[0].shape[2]), dtype=np.uint8)*255
        row1 = [blank_image] + row1 +[blank_image]
        # row2 = [blank_image] + row2 +[blank_image]
        row3 = [blank_image] + row3 +[blank_image]
        # row4 = [blank_image] + row4 +[blank_image]
        row5 = [blank_image] + row5 +[blank_image]
        
        # 拼接构建大图
        row1 = np.hstack(row1)
        row2 = np.hstack(row2)
        row3 = np.hstack(row3)
        row4 = np.hstack(row4)
        row5 = np.hstack(row5)
        final_img = np.vstack((row1,row2,row3,row4,row5))

        # 写入合并后的图片
        outfile_name = os.path.join(pohto_splicing_path, file_name)
        resized = cv2.resize(final_img, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite(outfile_name, resized)
        print(f'已成功生成拼接图片: {outfile_name}')

def photo_splicing_3(directories,dim):
    '''
    功能：将复眼摄像头采集到的图像拼接为大图
    directories:摄像头采集图像储存位置
    dim:大图的宽和高
    注:对不同复眼摄像头点位,需修改hstack及vstack,并修改pos中对应的图像位置,位置可参考plot_camera程序绘制的摄像头点位图
    '''
    # 获取第一个文件夹内的所有图片名称，其他文件夹中的图片与此处的图片名称一致
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

        # 给原始照片生成白边
        for i in range(0,len(pics)):
            pics[i] = cv2.copyMakeBorder(pics[i],2,2,2,2,cv2.BORDER_CONSTANT,value=(255,255,255))

        # pow中设置图片位置
        pos1 = [14,7,16]
        row1 = [pics[pos] for pos in pos1 if pos < len(pics) ]
        pos2 = [10,4,6,11]
        row2 = [pics[pos] for pos in pos2 if pos < len(pics) ]
        pos3 = [2,0,1]
        row3 = [pics[pos] for pos in pos3 if pos < len(pics) ]
        pos4 = [9,3,5,12]
        row4 = [pics[pos] for pos in pos4 if pos < len(pics) ]
        pos5 = [13,8,15]
        row5 = [pics[pos] for pos in pos5 if pos < len(pics) ]

        # 添加空白图片在第二行的两端
        blank_image = np.ones((row2[0].shape[0], row2[0].shape[1] // 2, row2[0].shape[2]), dtype=np.uint8)*255
        row1 = [blank_image] + row1 +[blank_image]
        # row2 = [blank_image] + row2 +[blank_image]
        row3 = [blank_image] + row3 +[blank_image]
        # row4 = [blank_image] + row4 +[blank_image]
        row5 = [blank_image] + row5 +[blank_image]
        
        # 拼接构建大图
        row1 = np.hstack(row1)
        row2 = np.hstack(row2)
        row3 = np.hstack(row3)
        row4 = np.hstack(row4)
        row5 = np.hstack(row5)
        final_img = np.vstack((row1,row2,row3,row4,row5))

        # 写入合并后的图片
        outfile_name = os.path.join(pohto_splicing_path, file_name)
        resized = cv2.resize(final_img, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite(outfile_name, resized)
        print(f'已成功生成拼接图片: {outfile_name}')


def video_create(image_folder, video_path_name, fps):
    '''
    功能: 将大图拼接为视频
    image_folder: 大图储存位置
    fps: 视频帧率
    '''
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 根据文件名对图片进行排序

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 设置视频编码器和输出路径
    video = cv2.VideoWriter(video_path_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    video.release()

if __name__ == '__main__':
    dim = set_image_size(horizontal=4,vertical=5,mul=2)

    photo_splicing_2(images_path,dim)
    print("Photo splicing completed.")

    # fps = 10
    # video_create(pohto_splicing_path, video_path_name, fps)
    # print("Video creation completed.")