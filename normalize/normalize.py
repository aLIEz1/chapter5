import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("../img/1.png")
cv.imshow("image", image)
# 显示原始图像的直方图
plt.figure("原始直方图")
# 画出图像直方图
plt.hist(image.ravel(), 256)
# 直方图正规化
image1 = cv.normalize(image, image, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
# 显示正规化后的图像
plt.figure("正规化后直方图")
# 画出图像直方图
plt.hist(image.ravel(), 256)
plt.show()
cv.waitKey()
cv.destroyAllWindows()
