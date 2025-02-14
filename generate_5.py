from carla_nuscenes.generator import Generator
import os
import yaml 
import time
import sys
import traceback
from yamlinclude import YamlIncludeConstructor
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)
config_path = "configs/config_fuyan_pcd_one_scene_2.yaml"
# config_path = "configs/config_fuyan_world.yaml"
# config_path = "carla_nuscenes/configs/config.yaml"
# config_path = "carla_nuscenes/configs/config_fuyan_copy.yaml"

# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.error_terminal = sys.stderr
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        """处理标准输出"""
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后立即刷新

    def write_error(self, message):
        """专门用于写入错误信息的方法"""
        self.error_terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后立即刷新

    def flush(self):
        """刷新终端和日志文件的缓冲区"""
        self.terminal.flush()
        self.error_terminal.flush()
        self.log.flush()

    def close(self):
        """关闭日志文件"""
        self.flush()
        self.log.close()

    def __enter__(self):
        """进入上下文管理器时重定向 stdout 和 stderr"""
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """退出上下文管理器时恢复 stdout 和 stderr 并处理异常"""
        if exc_type is not None:
            error_message = f"Exception occurred: {exc_type.__name__}: {exc_value}\n"
            error_message += ''.join(traceback.format_exception(exc_type, exc_value, traceback_obj))
            self.write_error(error_message)

        self.close()
        sys.stdout = self.terminal
        sys.stderr = self.error_terminal


with open(config_path,'r') as f:
    config = yaml.load(f.read( ),Loader=yaml.FullLoader)#读取yaml文件
    runner = Generator(config)

t = time.strftime("-%Y%m%d-%H%M%S", time.localtime())  # 时间戳
filename = 'carla_bev_world5_count20_time20_6_' + t + '.txt'
log = Logger(filename)  
sys.stdout = log

if os.path.exists(config["dataset"]["root"]):       #存在数据集路径
    runner.generate_dataset(True)       #
else:       #不存在数据集路径
    runner.generate_dataset(False)