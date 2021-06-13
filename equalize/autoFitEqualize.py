import cv2 as cv
import matplotlib.pyplot as plt

# 读取图像
image = cv.imread("../img/1.png", cv.IMREAD_GRAYSCALE)
# 创建CLAHE对象
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值均衡化
dst = clahe.apply(image)
# 显示图像
cv.imshow("image", image)
cv.imshow("clahe", dst)
plt.figure("原始直方图")  # 显示原始图像直方图
plt.hist(image.ravel(), 256)
plt.figure("均衡化直方图")  # 显示均衡化后的图像直方图
plt.hist(dst.ravel(), 256)
plt.show()
cv.waitKey()
cv.destroyWindow()
