import cv2 as cv
import matplotlib.pyplot as plt

# 读取一幅图像
image = cv.imread("../img/1.png", cv.IMREAD_GRAYSCALE)
cv.imshow("cartree", image)  # 显示原始图像
equ = cv.equalizeHist(image)  # 直方图均衡化处理
cv.imshow("equcartree", equ)  # 显示均衡化后的图像
plt.figure("原始直方图")  # 显示原始图像直方图
plt.hist(image.ravel(), 256)
plt.figure("均衡化直方图")  # 显示均衡化后的图像直方图
plt.hist(equ.ravel(), 256)
plt.show()
cv.waitKey()
cv.destroyAllWindows()
