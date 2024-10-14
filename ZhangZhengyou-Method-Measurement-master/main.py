# Function: 主程序入口
from measurePic import *
from run_calib_RGB import *

if __name__ == '__main__':

    chessboard_dir = "./pic/RGB_camera_calib_img"# 棋盘格文件夹
    distortion_dir = "./pic/RGB_distortion"# 需要矫正畸变图片文件夹
    dedistortion_dir = "pic/RGB_dedistortion"# 矫正畸变后图片文件夹
    para_path = "./para/RGB_camera_calib_para/"# 矫正参数的保存文件夹

    shape_inner_corner = (11, 8)# 棋盘格内角点个数
    size_grid = 6.# 棋盘格格子大小

    file = 'pic/RGB_dedistortion/p5.bmp'# 需要测量的图片
    calibration_data_file = para_path + "para.yaml"# 矫正参数文件
    run_mode = 'measure'# 运行模式选择calibrate或者measure

    if run_mode == 'calibrate':
        # 标定图片
        calibrate_camera(chessboard_dir, distortion_dir,dedistortion_dir,para_path, shape_inner_corner, size_grid)
    else:
        # 读取标定数据
        calibration_data = load_calibration_data(calibration_data_file)
        # 选择测量模式
        mode = input('请选择任务：'
                     '1.自动测量齿轮'
                     '2.自动测量薄板'
                     '3.手动测量距离 ')
        # mode = '2'
        if mode == '1':
            gear = measure_gear(file)
            fact = calculate_distance(calibration_data)
            gear.print_circ(fact=fact)

        elif mode == '2':
            pan = measure_pan(file)
            fact = calculate_distance(calibration_data)
            pan.print_pan(fact=fact)

        else:

            img = Image.open(file)
            root = tk.Tk()
            root.title("Draw Line and Show Distance")

            canvas = tk.Canvas(root, width=img.width, height=img.height)
            canvas.pack()

            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)

            points = []

            canvas.bind("<Button-1>", lambda event: on_click(event, points, canvas,calibration_data))

            root.mainloop()

            # calculate_distance([0,0],[1,1])