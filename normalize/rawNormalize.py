import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread("../img/1.png")
imageMax = np.max(image)
imageMin = np.min(image)
min_l = 0
max_l = 255
m = float(max_l - min_l) / imageMax - imageMin
n = min_l - min_l * m
image1 = m * image + n
image1 = image1.astype(np.uint8)
plt.figure("原始直方图")
plt.hist(image.ravel(), 256)
plt.figure("正规化后直方图")
plt.hist(image1.ravel(), 256)
plt.show()
# 显示原始图像
cv.imshow("image", image)
cv.waitKey()
cv.destroyAllWindows()
