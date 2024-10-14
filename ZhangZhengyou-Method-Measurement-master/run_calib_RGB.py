# -*- coding: utf-8 -*-
"""
Calibrate the Camera with Zhang Zhengyou Method.
Picture File Folder: "./pic/RGB_camera_calib_img/", Without Distortion. 

By You Zhiyuan, 2022.07.04, zhiyuanyou@foxmail.com
"""

import os
import cv2

from calibrate_helper import Calibrator


def cut_gear():
    # Define the directory path where the files are located
    directory_path = './pic/RGB_distortion'  # Replace with your directory path

    # Common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # Loop through all the files in the directory
    for file in os.listdir(directory_path):
        # Check if the file starts with 'g' and has an image file extension
        if file.startswith('g') and any(file.lower().endswith(ext) for ext in image_extensions):
            # Construct the full file path
            file_path = os.path.join(directory_path, file)

            # Read the image using cv2
            image = cv2.imread(file_path)

            # Check if the image was successfully loaded
            if image is not None:
                # Display the image (optional)
                cv2.imshow('Image', image)


def calibrate_camera(img_dir, distortion_dir,dedistortion_dir,para_path, shape_inner_corner, size_grid):
    # img_dir = "./pic/mydata"
    # distortion_dir = "./pic/RGB_distortion"
    # dedistortion_dir = "pic/RGB_dedistortion"
    # para_path = "./para/RGB_camera_calib_para/"
    #
    # shape_inner_corner = (11, 8)
    # size_grid = 6.
    # create calibrator
    calibrator = Calibrator(img_dir, distortion_dir,para_path,shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    # dedistort and save the dedistortion result
    save_dir = dedistortion_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    calibrator.dedistortion(save_dir)
    cv2.destroyAllWindows()



