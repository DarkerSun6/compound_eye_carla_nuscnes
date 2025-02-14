import pandas as pd
import os
import yaml

int_dir = '/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan/points/points_transform.csv'
out_dir_2 = '/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan/points/points_carla_transform.yaml'
out_dir_3 = '/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan/points/points_carla_transform.txt'
df = pd.read_csv(int_dir)
# 将每列数据转化为字典存储，键为列名，值为该列数据的列表
columns_to_lists = {col: df[col].tolist() for col in df.columns}
list_len = len(columns_to_lists['X'])

def sensors_data():
    for i in range(list_len):
        print('-')
        print('    name: "multi_cam_rgb_'+str(i)+'"')
        print("    modality: 'camera'")
def calibrated_sensors_data():
    for i in range(list_len):
        print('-') 
        print('  name: "multi_cam_rgb_'+str(i)+'"')

        print("  bp_name: 'sensor.camera.rgb'")
        print('  location: ')
        print("    x: " + str(columns_to_lists['X'][i]))
        print("    y: " + str(columns_to_lists['Y'][i]))
        print("    z: " + str(columns_to_lists['Z'][i]))
        print('  rotation: ')
        print("    yaw: " + str(columns_to_lists['yaw'][i]))
        print("    pitch: " + str(columns_to_lists['pitch'][i]))
        print("    roll: " + str(columns_to_lists['roll'][i]))
        print('  options: ')
        print('    <<: *CameraRGBBaseOptions')
        print('    "fov": "60"')


def calibrated_sensors_data_txt():
    with open(out_dir_3, 'r') as file:
        content = file.read()

    if content.strip():
        with open(out_dir_3, 'w') as file:
            file.write('')
        print(f"The file '{out_dir_3}' has been cleared.")
    else:
        print(f"The file '{out_dir_3}' is already empty.")

    for i in range(list_len):
        data = []
        data.append('  -') 
        data.append('    name: "MULTI_CAM_RGB_'+str(i)+'"')

        data.append("    bp_name: 'sensor.camera.rgb'")
        data.append('    location: ')
        data.append("      x: " + str(columns_to_lists['X'][i]))
        data.append("      y: " + str(columns_to_lists['Y'][i]))
        data.append("      z: " + str(columns_to_lists['Z'][i]))
        data.append('    rotation: ')
        data.append("      yaw: " + str(columns_to_lists['yaw'][i]))
        data.append("      pitch: " + str(columns_to_lists['pitch'][i]))
        data.append("      roll: " + str(columns_to_lists['roll'][i]))
        data.append('    options: ')
        data.append('      <<: *CameraRGBBaseOptions')
        data.append('      "fov": "60"')
        with open(out_dir_3, 'a') as f:
            for line in data:
                f.write(line+ '\n')

def calibrated_sensors_data_yaml():
    data_all = []
    for i in range(list_len):
        data = {
            'name': 'multi_cam_rgb_'+str(i),
            'bp_name': 'sensor.camera.rgb',
            'location': {
                'x': columns_to_lists['X'][i],
                'y': columns_to_lists['Y'][i],
                'z': columns_to_lists['Z'][i]
            },
            'rotation': {
                'yaw': columns_to_lists['yaw'][i],
                'pitch': columns_to_lists['pitch'][i],
                'roll': columns_to_lists['roll'][i]
            }
            ,
            'options': {
                '<<': '*CameraRGBBaseOptions',
                'fov': '60'
            }
        }
        data_all.append(data)
    xxx = {"camera":data_all}
    column_names = {'sensors': xxx}
    with open(out_dir_2, 'w', encoding='utf-8') as f:
        yaml.dump(data=column_names, stream=f, allow_unicode=True,default_flow_style= False)

def main():
    # sensors_data()
    calibrated_sensors_data_txt()


if __name__ == '__main__':
    main()

