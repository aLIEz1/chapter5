import matplotlib.pyplot as plt

# 构建两个列表
arr1 = [1, 1.2, 1.5, 1.6, 2, 2.5, 2.8, 3.5, 4.3]
arr2 = [5, 4.5, 4.3, 4.2, 3.6, 3.4, 3.1, 2.5, 2.1, 1.5]
plt.plot(arr1)  # 绘制arr1的图像
plt.plot(arr2, 'r')  # 绘制arr2的图像，'r'表示用红色绘制
plt.show()
