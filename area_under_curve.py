import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d

img = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\cafar_direction539.png", 0)
ske = skeletonize_3d(img)
plt.imshow(ske)
plt.show()

row, col = img.shape
point_x, point_y = [], []

y_max, y_min = 0, row
for x in range(0, col):
    for y in range(0, row):
        # print(y_max, y_min)
        if (ske[y][x] > 0):
            # print(x, y)
            point_x.append(x)
            point_y.append(y)
            if (y > y_max):
                y_max = y
                x_ymax = x
            if (y < y_min):
                y_min = y
                x_ymin = x
print("max x, y: ", x_ymax, y_max)
print("min x, y: ", x_ymin, y_min)
delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
if (delta_x == 0):
    delta_x = 1
ratio_delta = delta_y / delta_x
angle = math.degrees(math.atan(ratio_delta))


canvas = np.zeros(img.shape, np.float32)
print("angle: ", angle)
if (5 <= angle <= 70):
    if (y_max < row):
        for i in range(y_max, row):
            point_x.append(x_ymax)
            point_y.append(i)

    if (x_ymin > x_ymax):
        mode = "right"
        print("right /")
        for k in range(0, len(point_x)):
            for i in range(0, col):
                for j in range(0, row):
                    if (i >= point_x[k] and j >= point_y[k]):
                        canvas[j][i] = 1

    elif (x_ymin < x_ymax):
        mode = "left"
        print("left \\")
        for k in range(0, len(point_x)):
            for i in range(0, col):
                for j in range(0, row):
                    if (i <= point_x[k] and j >= point_y[k]):
                        canvas[j][i] = 1


elif (75 <= angle <= 90):
    mode = "straight"
    print("straight |")
    x_pos = int((x_ymin+x_ymax)/2)
    canvas[x_pos: col, 0: row]
else:
    mode = "straight"
    print("cannot approximate")

plt.imshow(canvas)
plt.show()