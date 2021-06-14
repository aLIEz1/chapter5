# 第五章常用函数

## 计算直方图

```python
hist = cv2.calcHist(image,channel,mask,histSize,range, accumulate)

·hist表示返回的统计直方图，数组内的元素是各个灰度级的像素个数。
·image表示原始图像，该图像需要用“[]”括起来。
·channel表示指定通道编号。通道编号需要用“[]”括起来。
·mask表示掩模图像。当统计整幅图像的直方图时，将这个值设为None。当统计图像某一部分的直方图时，需要用到掩模图像。
·histSize表示BINS的值，该值需要用“[]”括起来。
·range表示像素值范围。
·accumulate表示累计标识，默认值为False。
如果被设置为True，则直方图在开始计算时不会被清零，计算的是多个直方图的累积结果，用于对一组图像计算直方图。该参数是可选的，一般情况下不需要设置。

# 或者

matplotlib.pyplot中的hist()
```

### 使用`OpenCV`计算直方图

示例

```python
import cv2 as cv
image = cv.imread("F:/picture/panda.jpg")                   
# 导入一幅图像
hist = cv.calcHist([image],[0],None, [256], [0,255])       
 # 计算其统计直方图信息
print(hist)     # 输出统计直方图信息，为一维数组
```



#### 使用`plot()`函数绘制直方图

- 使用`plot()`绘制曲线

```python
import matplotlib.pyplot as plt        #导入绘图模块
# 构建两个列表
arr1 = [1,1.2,1.5,1.6,2,2.5,2.8,3.5,4.3]
arr2 = [5,4.5,4.3,4.2,3.6,3.4,3.1,2.5,2.1,1.5]
plt.plot(arr1)                        # 绘制arr1的图像
plt.plot(arr2,'r')                    # 绘制arr2的图像，'r'表示用红色绘制
plt.show() 
```

![image.png](https://ae02.alicdn.com/kf/H1b539d7164e747408c9e4cc0b41badc4F.png)



- 使用`plot()`函数将`calcHist()`的返回值绘制出来

```python
import cv2 as cv
import matplotlib.pyplot as plt
image = cv.imread("F:/picture/panda.png")                   
# 导入一幅图像
hist = cv.calcHist([image],[0],None, [256], [0,255])        
# 得到统计直方图的信息
plt.plot(hist)  # 显示直方图
plt.show() 

```

![image.png](https://ae05.alicdn.com/kf/Hb93f79d683d14de1918f47b3a058c53fV.png)



### 使用`hist()`函数直接绘制图像直方图

```python
matplotlib.pyplot.hist(image,BINS)
·BINS表示灰度级的分组情况。
·image表示原始图像数据，必须将其转换为 一维数据 。*******************
import cv2 as cv
import matplotlib.pyplot as plt                        # 导入绘图模块
image = cv.imread("F:/picture/panda.png")              # 读取一幅图像
image = image.ravel()                                  # 将图像转换为一维数组
plt.hist(image,256)                                    # 绘制直方图
```



![image.png](https://ae04.alicdn.com/kf/H542ec62897df4ea69ea2270e1108741aB.png)



## 直方图正规化



### 正规化代码实现

```python
image = cv.imread("F:/picture/img4.jpg",0)        # 读取一幅灰度图像
imageMax = np.max(image)                          # 计算image的最大值
imageMin = np.min(image)                          # 计算image的最小值
min_l = 0   max_l = 255
   m = float(max_l-min_l)/(imageMax-imageMin)# 计算m、n的值
n = min_l -min_l*m
image1 = m*image + n                              # 矩阵的线性变换
image1 = image1.astype(np.uint8)                  # 数据类型转换
cv.imshow("image",image)
plt.figure("原始直方图")
plt.hist(image.ravel(),256)
plt.figure("正规化后直方图")
plt.hist(image1.ravel(),256)
plt.show()
# 显示原始图像
cv.imshow("image",image)
plt.figure("原始直方图")
plt.hist(image.ravel(),256)
# 显示正规化后的图像
plt.figure("正规化后直方图")
plt.hist(image1.ravel(),256)
plt.show()
cv.waitKey()
cv.destroyAllWindows() 

```

![image.png](https://ae02.alicdn.com/kf/H4ced6960e4024549a65315d2194afd0fq.png)



### 使用`normalize`实现

- `cv2.normalize()`函数来实现图像直方图正规化
- 一般令`norm_type=NORM_MINMAX`，其计算原理与前面提到的计算方法基本相同。
- `cv2.normalize(src,dst,alpha,beta,norm_type,dtype)`

```python
cv.imshow("image",image)
# 显示原始图像的直方图
plt.figure("原始直方图")
# 画出图像直方图
plt.hist(image.ravel(),256)
# 直方图正规化
image1 = cv.normalize(image,image,255, 0, cv.NORM_MINMAX, cv.CV_8U)
# 显示正规化后的图像
plt.figure("正规化后直方图")
# 画出图像直方图
plt.hist(image.ravel(),256)
plt.hist(image.ravel(),256)
plt.show()

```

![image.png](https://ae01.alicdn.com/kf/He3a4fced86fc43edb10e341e919227e0C.png)



## 直方图均衡化

- `cv2.equalHist()`函数，用于实现图像的直方图均衡化：
- `dst = cv2. equalHist (src)`

```python
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# 计算图像灰度直方图
def calcGrayHist(image):
    # 灰度图像矩阵的宽高
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint32)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist


def equalHist(image):
    # 灰度图像矩阵的宽高
    rows, cols = image.shape
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
            outPut_q[p] = math.floor(q)  #小于q的最大整数
        else:
            outPut_q[p] = 0
    # 得到直方图均衡化后的图像
    equalHistImage = np.zeros(image.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            equalHistImage[r][c] = outPut_q[image[r][c]]
    return equalHistImage
# 主函数
image = cv.imread("F:/picture/cartree.jpg",cv.IMREAD_ANYCOLOR)

dst = equalHist(image)                # 直方图均衡化
# 显示图像
cv.imshow("image", image)             # 显示原图像
cv.imshow("dst",dst)                  # 显示均衡化图像
# 显示原始图像直方图
plt.figure("原始直方图")
plt.hist(image.ravel(),256)  # ravel()多维数组转换为一维数组的功能
# 显示均衡化后的图像直方图
plt.figure("均衡化直方图")  
plt.hist(dst.ravel(),256)
plt.show()
cv.waitKey()
cv.destroyAllWindows()


```

![image.png](https://ae06.alicdn.com/kf/He147b8dc0b434a3d939e6b854f449c48s.png)



### 使用`cv2.equalHist()`实现直方图均衡化

```python
import cv2 as cv
import matplotlib.pyplot as plt
# 读取一幅图像
image = cv.imread("F:/picture/cartree.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("cartree", image)         # 显示原始图像
equ = cv.equalizeHist(image)        # 直方图均衡化处理
cv.imshow("equcartree", equ)        # 显示均衡化后的图像
plt.figure("原始直方图")            # 显示原始图像直方图
plt.hist(image.ravel(),256)
plt.figure("均衡化直方图")          # 显示均衡化后的图像直方图
plt.hist(equ.ravel(),256)
plt.show()
cv.waitKey()
cv.destroyAllWindows()

```



### 自适应直方图均衡化

#### 使用`cv.createCLAHE()`函数实现限制对比度的直方图均衡化

```python
import cv2 as cv
import matplotlib.pyplot as plt
# 读取图像
image = cv.imread("F:/picture/img4.jpg",cv.IMREAD_GRAYSCALE)
# 创建CLAHE对象
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# 限制对比度的自适应阈值均衡化
dst = clahe.apply(image)
# 显示图像
cv.imshow("image", image)
cv.imshow("clahe",dst)
plt.figure("原始直方图")                  # 显示原始图像直方图
plt.hist(image.ravel(),256)
plt.figure("均衡化直方图")                # 显示均衡化后的图像直方图
plt.hist(dst.ravel(),256)
plt.show()

```

![image.png](https://ae02.alicdn.com/kf/H7994e814fade44d988216e25171f3560u.png)





