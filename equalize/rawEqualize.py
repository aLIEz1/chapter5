import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 计算图像灰度直方图
def calcGrayHist(image):
    # 灰度图像矩阵的宽高
    rows, cols,_ = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint32)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist


def equalHist(image):
    # 灰度图像矩阵的宽高
    rows, cols, _ = image.shape
    # 计算灰度直方图
    grayHist = calcGrayHist(image)
    # 计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 根据直方图均衡化得到的输入灰度级和输出灰度级的映射
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (rows * cols)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
    if q >= 0:
        outPut_q[p] = math.floor(q)  # 小于q的最大整数
    else:
        outPut_q[p] = 0
    # 得到直方图均衡化后的图像
    equalHistImage = np.zeros(image.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistImage[r][c] = outPut_q[image[r][c]]
    return equalHistImage


# 主函数
image = cv.imread("../img/1.png", cv.IMREAD_ANYCOLOR)

dst = equalHist(image)  # 直方图均衡化
# 显示图像
cv.imshow("image", image)  # 显示原图像
cv.imshow("dst", dst)  # 显示均衡化图像
# 显示原始图像直方图
plt.figure("原始直方图")
plt.hist(image.ravel(), 256)  # ravel()多维数组转换为一维数组的功能
# 显示均衡化后的图像直方图
plt.figure("均衡化直方图")
plt.hist(dst.ravel(), 256)
plt.show()
cv.waitKey()
cv.destroyAllWindows()
