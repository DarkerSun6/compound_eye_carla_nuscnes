import time
from datetime import datetime



capture_current_timestamp = time.time()+  86400*4.0
capture_local_time_str = datetime.fromtimestamp(capture_current_timestamp) 
date_part = capture_local_time_str.date()  # 仅日期部分
time_part = capture_local_time_str.time()  # 仅时间部分

print(f"Date: {date_part}")
print(f"Time: {time_part}")