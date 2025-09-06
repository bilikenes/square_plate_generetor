import cv2
import numpy as np
import os
import random
import math

input_folder = r"D:\Medias\normal_plates\TR\images"
output_folder = r"D:\Medias\normal_plates\TR\result"
os.makedirs(output_folder, exist_ok=True)

def get_strong_perspective(src):
    h, w = src.shape[:2]
    f = 0.8 * w

    yaw = random.uniform(-50, 50) * math.pi / 180   # Yatay açı
    pitch = random.uniform(-30, 30) * math.pi / 180 # Dikey açı
    roll = random.uniform(-45, 45) * math.pi / 180  # Hafif döndürme

    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)]
    ])
    R_y = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    R_z = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll),  math.cos(roll), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x

    pts_3d = np.array([
        [-w/2, -h/2, 0],
        [ w/2, -h/2, 0],
        [ w/2,  h/2, 0],
        [-w/2,  h/2, 0]
    ])

    pts_3d = pts_3d @ R.T

    pts_2d = pts_3d[:, :2] / (pts_3d[:, 2].reshape(-1,1)/f + 1)
    pts_2d[:,0] += w/2
    pts_2d[:,1] += h/2

    min_x, min_y = np.min(pts_2d, axis=0)
    max_x, max_y = np.max(pts_2d, axis=0)

    dst_width = int(np.ceil(max_x - min_x))
    dst_height = int(np.ceil(max_y - min_y))

    translation = np.array([[1, 0, -min_x],
                            [0, 1, -min_y],
                            [0, 0, 1]])

    src_pts = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst_pts = np.float32(pts_2d)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M = translation @ M 

    warped = cv2.warpPerspective(src, M, (dst_width, dst_height), borderMode=cv2.BORDER_REPLICATE)
    return warped

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)
        warped_img = get_strong_perspective(img)
        cv2.imwrite(output_path, warped_img)
        print(f"{filename} ok.")
