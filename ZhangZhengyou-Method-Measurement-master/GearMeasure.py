import cv2
import numpy as np
import matplotlib.pyplot as plt

class measure_gear():
    def __init__(self,file):
        self.trans_top = {'x': 200, 'y': 200}
        self.shrink_img,self.shrink_gray,self.shrink_edges=self.preprocess(file)
        self.result_img = cv2.imread(file)
        self.largest_contour =None
        self.top_circle = None
        self.trans_inside = None
        self.inside_circle = None
        self.find_outside_contour()
        self.find_inside_circle()

    def preprocess(self,file):
        image = cv2.imread(file)
        # 裁剪图片（目的是不让照明边框影响轮廓提取）
        img = image[self.trans_top['x']:image.shape[0]-self.trans_top['x'] , self.trans_top['y']:image.shape[1] - self.trans_top['y']]
        # 变为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯滤波
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        # 边缘提取
        edges = cv2.Canny(blurred, 100, 200)

        return img,gray,edges

    def find_outside_contour(self):
        # 寻找所有轮廓
        contours, _ = cv2.findContours(self.shrink_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 找到最大的封闭轮廓
        self.largest_contour = max(contours, key=cv2.contourArea)
        # 寻找最大封闭轮廓（齿廓）最小外接圆（齿顶圆）
        (x, y), radius = cv2.minEnclosingCircle(self.largest_contour)
        # 保存齿顶圆
        self.top_circle = (x,y,radius)
        # cv2.circle(self.shrink_img, (int(x), int(y)), int(radius), (255, 0, 0), 3)
        return radius

    def fit_img_to_gear(self):
        # 将齿顶圆的圆心作为新图像的中心
        x,y,radius = self.top_circle
        # 裁剪灰度图
        img_cut_gray = self.shrink_gray[int(y - radius - 10):int(y + radius + 10), int(x - radius - 10):int(x + radius + 10)]
        # 裁剪彩色图
        img_cut = self.shrink_img[int(y - radius - 10):int(y + radius + 10), int(x - radius - 10):int(x + radius + 10)]
        # 定义变换参数
        self.trans_inside = {'y': int(y - radius - 10), 'x': int(x - radius - 10)}
        return img_cut,img_cut_gray

    def find_inside_circle(self):
        # 裁剪图片，防止外轮廓干扰霍夫变换
        img_cut,img_cut_gray = self.fit_img_to_gear()
        # 霍夫变换找中间的圆
        # 设置dp=1，以尽可能减少计算量以及找到圆的数量
        # 设置minDist=100000，以尽可能保证只找到一个圆
        # param1=100是canny变换的参数，验证后可行
        # param2=20是霍夫变换的参数，验证后可行
        # 设置minRadius=1，以尽可能找到小圆
        # 设置maxRadius=int(radius/2)，以尽可能找到小圆
        circles = cv2.HoughCircles(img_cut_gray, cv2.HOUGH_GRADIENT, 1, 100000,
                                   param1=100, param2=20, minRadius=1, maxRadius=int(self.top_circle[2] / 2))
        if circles is None :
            print('No circle found')
        if len(circles[0])>1:
            print('More than one circle found')
        else:
            centerx, centery, radius_inside = circles[0,0,:]
            self.inside_circle = (centerx,centery,radius_inside)
            # cv2.circle(img_cut, (int(centerx), int(centery)), int(radius_inside), (255, 0, 0), 3)
            # print(radius_inside)
    def print_circ(self,fact=None):
        # 坐标平移
        trans_largest_contour = np.zeros_like(self.largest_contour)
        trans_largest_contour[:, 0, 0] = self.largest_contour[:, 0, 0] + self.trans_top['x']
        trans_largest_contour[:, 0, 1] = self.largest_contour[:, 0, 1] + self.trans_top['y']
        # 画出轮廓和圆
        cv2.drawContours(self.result_img,trans_largest_contour, -1, (0, 255, 0), 3)
        cv2.circle(self.result_img, (int(self.top_circle[0]+self.trans_top['x']), int(self.top_circle[1]+self.trans_top['y'])), int(self.top_circle[2]), (255, 0, 0), 3)
        cv2.circle(self.result_img, (int(self.inside_circle[0]+self.trans_inside['x']+self.trans_top['x']),
                                     int(self.inside_circle[1]+self.trans_inside['y']+self.trans_top['y'])), int(self.inside_circle[2]), (255, 0, 0), 3)
        if fact is not None:
            text = "top_d={:.2f}mm".format(2*fact * self.top_circle[2])
            cv2.putText(self.result_img, text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 10, cv2.LINE_AA)
            text = "inside_d={:.2f}mm".format(2*fact * self.inside_circle[2])
            cv2.putText(self.result_img, text, (10,200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 10, cv2.LINE_AA)
        # 显示画出的圆和半径
        plt.imshow(self.result_img)
        plt.show()
        # cv2.imshow('detected circles', self.result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    file = 'pic/RGB_dedistortion/g1.bmp'
    gear=measure_gear(file)
    gear.print_circ(fact=1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()