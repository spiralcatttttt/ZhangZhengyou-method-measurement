import cv2
import numpy as np
import matplotlib.pyplot as plt

class measure_pan():
    def __init__(self,file):
        self.trans_top = {'x': 200, 'y': 200}
        self.shrink_img, self.shrink_gray, self.shrink_edges = self.preprocess(file)
        self.result_img = cv2.imread(file)
        self.largest_contour = None
        self.rect =None
        self.find_rect()

    def preprocess(self,file):
        image = cv2.imread(file)
        # 裁剪图片（目的是不让照明边框影响轮廓提取）
        img = image[self.trans_top['x']:image.shape[0]-self.trans_top['x'] , self.trans_top['y']:image.shape[1] - self.trans_top['y']]
        # 变为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯滤波
        blurred = cv2.GaussianBlur(img, (13, 13), 0)
        # 边缘提取
        edges = cv2.Canny(blurred, 100, 200)

        return img,gray,edges
    def find_rect(self):
        # 寻找所有轮廓
        contours, _ = cv2.findContours(self.shrink_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 找到最大的封闭轮廓
        self.largest_contour = max(contours, key=cv2.contourArea)
        # 寻找最大封闭轮廓（齿廓）最小外接圆（齿顶圆）
        self.rect = cv2.minAreaRect(self.largest_contour)
        # 这里rect的格式是((x,y),(w,h),angle)，用rect[0][1]获得x
        return

    def print_pan(self,fact=None):
        image = self.result_img
        # 轮廓坐标平移并画出
        trans_largest_contour = np.zeros_like(self.largest_contour)
        trans_largest_contour[:, 0, 0] = self.largest_contour[:, 0, 0] + self.trans_top['x']
        trans_largest_contour[:, 0, 1] = self.largest_contour[:, 0, 1] + self.trans_top['y']
        cv2.drawContours(image, [trans_largest_contour], 0, (0, 255, 0), 20)

        # 矩阵坐标平移并画出
        box = cv2.boxPoints(self.rect)
        box = np.int0(box)
        box_like = np.zeros_like(box)
        box_like[:, 0] = box[:, 0] +self.trans_top['x']
        box_like[:, 1] = box[:, 1] +self.trans_top['y']
        cv2.drawContours(image, [box_like], 0, (255, 0, 0), 5)
        # 画出文字
        if fact is not None:
            text = "width={:.2f}mm".format(fact*self.rect[1][0])
            cv2.putText(image, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 10, cv2.LINE_AA)
            text = "height={:.2f}mm".format(fact*self.rect[1][1])
            cv2.putText(image, text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 10, cv2.LINE_AA)
        plt.imshow(image)
        plt.show()
        return


if __name__ == '__main__':
    pan = measure_pan('pic/RGB_dedistortion/p4.bmp')
    pan.print_pan(fact=1)