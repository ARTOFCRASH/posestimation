import cv2
import numpy as np


# 1️⃣ 读入深度图（保持16位）
depth = cv2.imread(r"D:\files\persimmon data\RealSenseD405_raw\persimmon89_40_56_depth.png",cv2.IMREAD_UNCHANGED)


# 2️⃣ 转换为可显示范围（归一化到0~255）
#    注意：这里会自动把最小/最大深度映射到[0,255]
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = np.uint8(depth_norm)


# 3️⃣ 应用伪彩色映射（可选: COLORMAP_JET, COLORMAP_TURBO, COLORMAP_INFERNO）
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


# 4️⃣ 显示或保存
cv2.imshow("depth color", depth_color)
cv2.waitKey(0)



