# Zhang Zhengyou Calibration and Measurement Program
## Usage
1. Ensure Python=3.7, install the dependency packages by running `requirements.txt`.
2. `main.py` is the main program, select either `calibrate` or `measure` mode for calibration and measurement respectively. Modify the folder name as needed before running.
3. `run_calib_RGB.py` is the calibration program, and `measurePic.py` is the measurement program.
4. `GearMeasure.py` is used for automatically measuring gear addendum circle and bore diameter, while `PanMeasure.py` is used for measuring the length and width of square plates.
5. For secondary development: The `calculate_distance` function in `measurePic.py` is the core function for measurement. It takes the saved `para.yaml` as input. If two point coordinates are provided, it returns the distance between the points; if not, it returns the scale factor per unit pixel.
## Configuration
1. Python=3.7
2. opencv-python=3.4.2.16
## References
1. [Zhang Zhengyou Calibration Reference Program](https://github.com/zhiyuanyou/Calibration-ZhangZhengyou-Method.git)
2. [Gear Measurement Reference Program](https://github.com/AltafBagwan/calc-gear-parameters.git)

# 张正友标定和测量程序
## 使用方法
1. python=3.7,运行requirements.txt安装依赖包。
2. `main.py`是主程序，选择`calibrate`或`measure`模式，分别进行标定和测量，运行前根据需要更改文件夹名称。
3. `run_calib_RGB.py`是标定程序，`measurePic.py`是测量程序。
4. `GearMeasure.py`是用于自动测量齿轮齿顶圆和内孔直径的程序，`PanMeasure.py`是用于测量方形薄板长款的程序。
5. 二次开发：`measurePic.py`中的`calculate_distance`函数是测量的核心函数，输入保存的`para.yaml`，如果输入两点坐标，就返回两点距离，如果不输入，就返回单位像素代表的尺度因子。

## 相关配置
1. python=3.7
2. opencv-python=3.4.2.16

## 引用
1. [张正友标定参考程序](https://github.com/zhiyuanyou/Calibration-ZhangZhengyou-Method.git)
2. [齿轮测量参考程序](https://github.com/AltafBagwan/calc-gear-parameters.git)
