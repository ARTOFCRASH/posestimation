import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs


# information
persimmon_Number = 89
roll_angle = 40
pitch_angle = 56

output_folder = r"D:\files\persimmon data\RealSenseD405_raw"
os.makedirs(output_folder, exist_ok=True)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
depth = frames.get_depth_frame()
img = frames.get_color_frame()

depth16 = np.asanyarray(depth.get_data())  
img_np  = np.asanyarray(img.get_data())      

# print(depth16)
# print("shape:", depth16.shape, "dtype:", depth16.dtype)

# 获取 depth scale（深度单位→米）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("depth_scale =", depth_scale)

depth_mm = depth16 * depth_scale * 1000          # 单位 = mm
depth_mm = depth_mm.astype(np.uint16)
print(depth_mm)

# 查看深度范围
valid_depth = depth_mm[depth_mm > 0]

if valid_depth.size == 0:
    print("No valid depth pixels!")
else:
    min_depth = valid_depth.min()
    max_depth = valid_depth.max()

    print(f"Depth min (mm): {min_depth}")
    print(f"Depth max (mm): {max_depth}")


depth_mm_path = os.path.join(output_folder,
    f"persimmon{persimmon_Number}_{roll_angle}_{pitch_angle}_depth.png")
color_path = os.path.join(output_folder,
    f"persimmon{persimmon_Number}_{roll_angle}_{pitch_angle}_color.png")


cv2.imwrite(depth_mm_path, depth_mm)
cv2.imwrite(color_path, img_np)


print("Files saved:")
print("Depth image (unit: mm):", depth_mm_path)
print("RGB image:             ", color_path)


pipeline.stop()

