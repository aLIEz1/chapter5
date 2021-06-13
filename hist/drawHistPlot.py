import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("../img/1.png")
# 转换为一维数组
plt.hist(image.ravel(), 256)
plt.show()
