import cv2
import numpy as np
import sys
import math
from scipy import stats
import matplotlib.pyplot as plt
from ParamsSetting import levelsetparams
from ParamsSetting import postprocessingparams

def eliminateSmallSegment(img):
    minArea = postprocessingparams.minimum_area
    canvas = 255 * np.ones(img.shape, 'float32')
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    """Allow just 5 parts of pipe"""
    for j in range(0, len(cnts)):
        cnts = contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        area = cv2.contourArea(cnts[j])
        if (area >= minArea):
            cv2.drawContours(canvas, [cnts[j]], -1, (0, 0, 255), -1)
    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY_INV)
    return canvas

def postProcess(phi, shape, num):
    lengthCurved = postprocessingparams.length_threshold
    deltaXCurved = postprocessingparams.delta_x_threshold_curved
    angleCurved = postprocessingparams.angle_threshold_curved
    angleStraight = postprocessingparams.angle_threshold_straight
    ReInitPhi = False
    print("\nPostprocessing of frame %d..." %num)
    phi_norm = (255 * (phi >= 0)) + (0 * phi < 0)
    phi_norm = eliminateSmallSegment(phi_norm.astype(np.uint8))
    gy, gx = np.gradient(phi_norm)
    edge = gx + gy
    edge = (1 * (edge > 0)) + (0 * (edge == 0))
    row, col = phi.shape
    if(shape == "left" or shape == "right"):
        if (shape == "right"):
            x_ymax, x_ymin = 0, int(col/2)
        elif (shape == "left"):
                x_ymax, x_ymin = int(col/2), col
    elif(shape == "straight"):
        x_ymax, x_ymin = int(col / 2), int(col / 2)
    else:
        sys.exit()
    canvas2 = edge.copy()
    point_x = []
    point_y = []
    y_max, y_min = 0, row
    for x in range(0, col):
        for y in range(0, row):
            # print(y_max, y_min)
            if (canvas2[y][x] > 0):
                point_x.append(x)
                point_y.append(y)
                if (y > y_max):
                    y_max = y
                    x_ymax = x
                if (y < y_min):
                    y_min = y
                    x_ymin = x
    delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
    if (delta_x == 0):
        print("deta_x is zero")
        delta_x = 1
    ratio_delta = delta_y / delta_x
    angle = math.degrees(math.atan(ratio_delta))
    # print("max(x, y): (%.2f, %.2f), min(x, y): (%.2f, %.2f), angle: %.2f" % (x_ymax, y_max, x_ymin, y_min, angle))
    canvas = np.zeros(phi.shape)
    point_x = np.asarray(point_x)
    point_y = np.asarray(point_y)
    if (shape == "left" or shape == "right"):
        print("mode: left or right")
        result = phi
        """reinit"""
        r = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
        print("distance result: ", r)
        if (r < lengthCurved) or (delta_x <= deltaXCurved) or (angle > angleCurved):
            print("re initial because the distance from polynomial estimation less than 50")
            ReInitPhi = True
    elif (shape == "straight"):
        print("mode: straight")
        slope, intercept, r_value, p_value, std_err = stats.linregress(point_x, point_y)
        print("linear regression: y = %.2fx + %.2f " % (slope, intercept))
        x_top = (y_min - intercept) / slope
        x_bot = (y_max - intercept) / slope
        print("x top: %.2f, x bottom: %.2f " % (x_top, x_bot))
        pts = np.array([[x_top, 0], [x_bot, row], [col, row], [col, 0]], np.int32)  # rectangle
        pts = pts.reshape((-1, 1, 2))
        cv2.fillConvexPoly(canvas, pts, 1)
        # phi_coef = levelsetparams.phi_coef
        # canvas = (-1 * phi_coef * (canvas == 0)) + (1* phi_coef * (canvas == 1))
        result = phi
        """reinit"""
        theta = math.degrees(math.atan(abs(slope)))
        # x_m = 490 * math.tan(math.radians(90 - theta))  # the distance from the middle frame (pixels)
        # print("theta: %.2f, x middle: %.2f" % (theta, x_m))
        print("theta: %.2f" %(theta))
        if (theta <= angleStraight):
            print("re initial because theta less than 85")
            ReInitPhi = True

    else:
        print("cannot estimate phi")
        result = phi
    return result, ReInitPhi