import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
import yaml
from GearMeasure import measure_gear
from PanMeasure import measure_pan


def on_click(event,points,canvas,calibration_data):
    # 添加点到列表
    points.append((event.x, event.y))
    # 如果已经有两个点，绘制直线并计算距离
    if len(points) == 2:
        # 绘制直线
        canvas.create_line(points[0], points[1], fill="red", tags="line")
        # 计算距离并显示
        distance = calculate_distance(calibration_data,points[0], points[1])
        # 显示距离标签
        show_distance_label(canvas,points[0], points[1], distance)
        points.clear()


def calculate_distance(calibration_data,point1=None, point2=None):
    # 获得内参以及外参矩阵
    K = np.array(calibration_data["intrinsic matrix"])
    rvec = np.mean(np.array(calibration_data["rotation vectors"]).reshape(-1, 3), axis=0).reshape(3, 1)
    tvec = np.mean(np.array(calibration_data["translation vectors"]).reshape(-1, 3), axis=0).reshape(3, 1)
    # K = np.array(([3.8865e3, 0, 1.2438e3], [0, 3.8907e3, 978.8508], [0, 0, 1]))
    # rvec = np.array(([0.0517, -0.0402, 1.6121])).reshape(3, 1)
    # tvec = np.array([11.3881, -29.7727, 239.5093]).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)


    if point1 is None or point2 is None:
        point1 = [0,0]
        point2 = [0,1.025]
    else:
        pass
    RT = np.hstack((R, tvec))
    # # 水平拼接 000 到 K 的右侧
    K_temp = np.hstack((K, np.zeros((3, 1))))
    # 垂直拼接 0001 到 K_temp 的下方
    K_qi = np.vstack((K_temp, np.array([0, 0, 0, 1])))
    #
    # 合并两个点的坐标
    points_temp = np.array([np.append(point1, 1), np.append(point2, 1)])
    #  转换
    temp = np.linalg.inv(K).dot((points_temp.T))

    camera_points = np.linalg.pinv(RT).dot(temp)


    point11 = camera_points[:, 0]
    point22 = camera_points[:, 1]

    point11 = point11[:]/point11[3]
    point22 = point22[:]/point22[3]


    distance = np.linalg.norm(point11 - point22)
    # print(float(distance))
    return float(distance)


def show_distance_label(canvas,point1, point2, distance):
    # 计算标签的位置（直线的中间位置）
    x = (point1[0] + point2[0]) / 2
    y = (point1[1] + point2[1]) / 2
    # 创建距离标签
    label = f"{distance:.3f} mm"
    canvas.create_text(x, y, text=label, fill="blue", font=("Arial", 12))


def load_calibration_data(calib_file):
    with open(calib_file, "r") as f:
        return yaml.safe_load(f)


def load_and_resize_image(image_path, scale):
    img = Image.open(image_path)
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    return img.resize((new_width, new_height), Image.ANTIALIAS)




