# import cv2
# import numpy as np
# import os

# input_folder = "plates/generated_plates"
# output_folder = "plates/perspective_plates"
# os.makedirs(output_folder, exist_ok=True)

# def get_warped_image(src):
#     h, w = src.shape[:2]

#     src_pts = np.float32([
#         [0, 0],     # sol üst
#         [w, 0],     # sağ üst
#         [w, h],     # sağ alt
#         [0, h]      # sol alt
#     ])

#     dst_pts = np.float32([
#         [1, 6],     # sol üst
#         [123, 19],  # sağ üst
#         [123, 88],  # sağ alt
#         [1, 72]     # sol alt
#     ])

#     matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     out_w = int(max(dst_pts[:,0])) + 10
#     out_h = int(max(dst_pts[:,1])) + 10

#     warped = cv2.warpPerspective(src, matrix, (out_w, out_h))
#     return warped

# for filename in os.listdir(input_folder):
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)

#         img = cv2.imread(input_path)

#         warped_img = get_warped_image(img)
#         cv2.imwrite(output_path, warped_img)
#         print(f"{filename} ok.")

import cv2
import numpy as np
import os
import random

input_folder = r"C:\Users\PC\Desktop\square_plates\plates"
output_folder = r"C:\Users\PC\Desktop\square_plates\perspectived_plates"
os.makedirs(output_folder, exist_ok=True)

def get_warped_image(src):
    h, w = src.shape[:2]

    src_pts = np.float32([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    max_warp_x = int(0.15 * w)
    max_warp_y = int(0.15 * h)

    dst_pts = src_pts.copy()
    for i in range(4):
        dx = random.randint(-max_warp_x, max_warp_x)
        dy = random.randint(-max_warp_y, max_warp_y)
        dst_pts[i][0] += dx
        dst_pts[i][1] += dy

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(src, matrix, (w, h))
    return warped

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        img = cv2.imread(input_path)

        warped_img = get_warped_image(img)
        cv2.imwrite(output_path, warped_img)
        print(f"{filename} ok.")
