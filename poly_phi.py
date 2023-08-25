import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage.measurements import label


phi = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\phi_20.png", 0)
plt.imshow(phi)
plt.show()

gy, gx = np.gradient(phi)
plt.imshow(gx+gy)
plt.show()

edge = gx+gy
edge = (1*(edge>0))+(0*(edge==0))
plt.imshow(edge)
plt.show()


row, col = phi.shape
point_x = []
point_y = []
y_max, y_min = 0, row
for x in range(0, col):
    for y in range(0, row):
        # print(y_max, y_min)
        if (edge[y][x] > 0):
            # print("ooooooo")
            # img_ske[i][j]=55
            point_x.append(x)
            point_y.append(y)
            if(y>y_max):
                y_max = y
                x_ymax = x
            elif(y<y_min):
                y_min = y
                x_ymin = x

print(x_ymax, x_ymin)
delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
ratio_delta = delta_y / delta_x
angle = math.degrees(math.atan(ratio_delta))

# """curve"""
# if(45 <= angle <=75):
#     p

point_x = np.asarray(point_x)
point_y = np.asarray(point_y)

z = np.polyfit(point_x, point_y, 7)  # degree 15
poly_function = np.poly1d(z)
x_new = np.linspace(point_x[0], point_x[-1], num=len(point_x) * 10)

canvas = np.zeros(phi.shape)
x_new = sorted(x_new)
x_new = np.linspace(x_new[0], x_new[-1], num=len(x_new) * 10)
# print("x_new3: ", x_new)
x_1, y_1 = x_new[0], poly_function(x_new[0])
x_2, y_2 = x_new[-1], poly_function(x_new[-1])
cnt = []
for k in range(0, len(x_new)):
    cv2.circle(canvas, (int(x_new[k]), int(poly_function(x_new[k]))), 1, (255, 255, 255), -1)
    cnt.append([int(x_new[k]), int(poly_function(x_new[k]))])

if(y_max != row):
    x = x_ymax
    for i in range(y_max, row):
        if(i==y_max):
            y = y_max
        else:
            y = i
        cv2.circle(canvas, (int(x), y), 1, (255, 255, 255), -1)

plt.title("canvas")
plt.imshow(canvas)
plt.show()

# canvas2 = np.zeros(phi.shape)
# cnt = np.asarray(cnt)
# cv2.fillPoly(canvas2, cnt, [255, 255, 255])
# plt.imshow(canvas2)
# plt.show()


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
add = phi+canvas
closing = cv2.morphologyEx(add, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
plt.show()
res = (1*(closing>0))+(0*(closing==0))
plt.imshow(res)
plt.show()

closing = closing.astype(np.uint8)
result = cv2.addWeighted(phi, 0.3, closing, 0.7, 0)
plt.imshow(result)
plt.show()

# canvas = canvas.astype(np.uint8)
# res = cv2.bitwise_and(phi, canvas)
# plt.imshow(res)
# plt.show()
