import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("../img/1.png")
hist = cv.calcHist([image], [0], None, [256], [0, 255])
print(hist)

