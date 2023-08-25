import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

def phi_est(phi, mode):
    re_phi = False
    phi = (255*(phi>=0))+(0*phi<0)
    gy, gx = np.gradient(phi)

    edge = gx + gy
    edge = (1 * (edge > 0)) + (0 * (edge == 0))
    # plt.title("gradient phi")
    # plt.imshow(edge)
    # plt.show()

    row, col = phi.shape

    # plt.title("g")
    # plt.imshow(edge)
    # plt.show()g")
    # plt.imshow(edge)
    # plt.show()
    """all pixel must very closer"""
    canvas2 = np.zeros(phi.shape, np.float32)
    test = canvas2.copy()
    pipe_arr = np.nonzero(edge)
    # print("pipe_arr: ", pipe_arr)
    pos_y, pos_x = pipe_arr[0], pipe_arr[1]
    # print("pos_y, pos_x: ", pos_y, pos_x)
    # print("len: ", len(pos_y))
    # points = []
    for i in range(0, len(pos_y)):
        if (0 < pos_y[i] < row  and 0 < pos_x[i] < col ):  # for win 3*3
            # window = ske_pad[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
            window = edge[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
            # print("window", window)
            conv = np.mean(window)
            # print("conv", conv)
            test[pos_y[i], pos_x[i]] = conv

            if(conv > (0.23)):
                canvas2[pos_y[i], pos_x[i]] = 1

        # elif (pos_y[i] == 0) or (pos_y[i] == row):
        #     # points.append([pos_y[i], pos_x[i], 1])
        #     canvas2[pos_y[i], pos_x[i]] = 1

    # plt.title("""filter""")
    # plt.imshow(test)
    # plt.show()
    #
    # plt.title("""after filter""")
    # plt.imshow(canvas2)
    # plt.show()

    point_x = []
    point_y = []
    y_max, y_min = 0, row
    for x in range(0, col):
        for y in range(0, row):
            # print(y_max, y_min)
            if (canvas2[y][x] > 0):
                # print("ooooooo")
                # img_ske[i][j]=55
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
    # # r_group = math.sqrt(math.pow((point_x[-1]-point_x[-2]), 2) + math.pow((point_y[-1] - point_y[-2]), 2))
    # # print("r group: ", r_group)
    # # if(r_group>10):
    # #     print("pop!")
    # #     point_x.pop(-1)
    # #     point_y.pop(-1)

    delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
    if (delta_x == 0):
        print("deta_x is zero")
        delta_x = 1
    ratio_delta = delta_y / delta_x
    angle = math.degrees(math.atan(ratio_delta))

    # """curve"""
    # if(45 <= angle <=75):
    #     p
    canvas = np.zeros(phi.shape)
    point_x = np.asarray(point_x)
    point_y = np.asarray(point_y)
    # print("point x: ", point_x)
    # print("point y: ", point_y)
    #
    # print("+++++++++++++++++")
    # for k in range (0, len(point_x)):
    #     print(point_x[k])
    # print("------------------")
    # for k in range (0, len(point_y)):
    #     print(point_y[k])
    if(mode == "left" or mode == "right"):
        # z = np.polyfit(point_x, point_y, 5)  # degree 15
        # poly_function = np.poly1d(z)
        # x_new = np.linspace(point_x[0], point_x[-1], num=len(point_x) * 10)
        #
        # x_new = sorted(x_new)
        # x_new = np.linspace(x_new[0], x_new[-1], num=len(x_new) * 10)
        # # print("x_new3: ", x_new)
        # cnt = []
        # for k in range(0, len(x_new)):
        #     cv2.circle(canvas, (int(x_new[k]), int(poly_function(x_new[k]))), 1, (255, 255, 255), -1)
        #     cnt.append([int(x_new[k]), int(poly_function(x_new[k]))])
        # # plt.title("poly estimation")
        # # plt.imshow(canvas)
        # # plt.show()
        #
        # # canvas2 = np.zeros(phi.shape)
        # # cnt = np.asarray(cnt)
        # # cv2.fillPoly(canvas2, cnt, [255, 255, 255])
        # # plt.imshow(canvas2)
        # # plt.show()
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #
        # res = phi + canvas
        # closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
        # result = (1 * (closing > 0)) + (0 * (closing == 0))
        result = phi
        """reinit"""
        r = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
        print("distance result: ", r)
        if (r < 50) or (delta_x <= 50) or (angle >= 75):
            print("re initial because the distance from polynomial estimation less than 50")
            re_phi = True

    elif(mode == "straight"):
        # print("point x: ", point_x)
        # print("point y: ", point_y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(point_x, point_y)
        # plt.plot(point_x, intercept + slope * point_x, 'r', label='fitted line')
        # plt.imshow(canvas2)
        # plt.show()
        # z = np.polyfit(point_x, point_y, 1)  # degree 15
        # slope = z[0]
        # intercept = z[1]
        print("linear regression: ", slope, intercept)
        x_top = (y_min - intercept) / slope
        x_bot = (y_max - intercept) / slope
        # if(abs(slope) < 5):
        #     """y = mx+c"""
        #     x_top = (y_min - intercept) / slope
        #     x_bot = (y_max - intercept) / slope
        # elif(abs(slope)>= 5):
        #     x_top = x_ymin
        #     x_bot = x_ymax

        # x_top = x_ymin
        # x_bot = x_ymax
        print("x_top and x_bot: ", x_top, x_bot)
        pts = np.array([[x_top, 0], [x_bot, row], [col, row], [col, 0]], np.int32)  # rectangle
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(canvas, pts, 2)
        # plt.imshow(canvas)
        # plt.show()
        result = canvas
        """reinit"""

        if (abs(slope) <= 10):
            print("re initial because the distance from polynomial estimation less than 50")
            re_phi = True

    else:
        print("cannot estimate phi")
        result = phi


    # plt.title("phi update")
    # plt.imshow(closing)
    # plt.show()
    return result, re_phi