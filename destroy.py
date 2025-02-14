import carla
import os
import pandas as pd
spawn_points_path = "/home/sunbs/carla_0.9.14/carla_nuscenes/fuyan/points/"

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15.0)
world = client.get_world()
world = client.load_world('Town03_Opt')

# spawn_points_list = []
# spawn_points = world.get_map().get_spawn_points()
# for i in range(len(spawn_points)): 
#     spawn_points_list.append([spawn_points[i].location.x, spawn_points[i].location.y, spawn_points[i].location.z])

# column_names = ['X', 'Y', 'Z']
# df = pd.DataFrame(spawn_points_list, columns=column_names)
# df.to_csv(os.path.join(spawn_points_path, 'spawn_points_all.csv'), index=False)

for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.destroy()
for walker in world.get_actors().filter('*walker*'):
    walker.destroy()
for sensor in world.get_actors().filter('*sensor*'):
    sensor.destroy()
for controller in world.get_actors().filter('*controller*'):
    controller.stop()
print("All actors destroyed")