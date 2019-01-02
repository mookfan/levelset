import cv2
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.ndimage.measurements import label
from scipy.spatial.distance import cdist, pdist
import math
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def reduce_noise(img):
    img_red = 255 * np.ones(img.shape, 'float32')
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    """Allow just 5 parts of pipe"""
    for j in range(0, len(cnts)):
        cnts = contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        area = cv2.contourArea(cnts[j])
        if (area>=40):
            cv2.drawContours(img_red, [cnts[j]], -1, (0, 0, 255), -1)
    _, img_red = cv2.threshold(img_red, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\res_bin.png", img_red)
    # cv2.imshow("contour", img_red)
    # cv2.waitKey(-1)
    return img_red


def endpoint_cal(img_gray):
    ske_input = (1*(img_gray==255)) + (0*(img_gray==0))
    ske = skeletonize_3d(ske_input)
    # plt.imshow(ske)
    # plt.title("skeleton")
    # plt.show()

    rows, cols = ske.shape
    canvas = np.zeros(ske.shape, np.float32)

    ske_pad = np.pad(ske, [1, 1], mode='constant')
    # pipe_arr = np.nonzero(ske_pad)
    pipe_arr = np.nonzero(ske)
    pos_y, pos_x = pipe_arr[0], pipe_arr[1]
    points = []
    # branch = []
    # plt.imshow(ske_pad)
    # plt.title("ske_pad")
    # plt.show()
    for i in range(0, len(pos_y)):
        if (pos_y[i] < rows  and pos_x[i] < cols ):  # for win 3*3
            # window = ske_pad[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
            window = ske[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
            conv = np.mean(window)
            # canvas[pos_y[i], pos_x[i]] = conv
            """end point"""
            if (0< conv <= (0.25)):
                points.append([pos_y[i], pos_x[i], 1])
                cv2.circle(canvas, (pos_x[i], pos_y[i]), 5, (255, 255, 255), -1)
            # if (conv>=0.4): #Not child (4/9)
            #     print("branch point")
            #     branch.append([rows[i], cols[i], 1])
            #     cv2.circle(canvas, (pos_x[i], pos_y[i]), 5, (255, 255, 255), -1)
    # print("points of skeleton: ", points)
    # plt.imshow(canvas)
    # plt.title("skeleton conv")
    # plt.show()

    # for i in range(0, len(branch)):
    #     dis_i = cdist(branch[i], points)
    #     min_dis_i = np.min(dis_i)
    #     if (min_dis_i <= 30):
    #         index = np.where(dis_i[:, :] == min_dis_i)
    #         index_points = index[1][0]
    #         points.pop(index_points)
    #     else:
    #         points.append(branch[i])

    # canvas2 = np.zeros(ske.shape, np.float32)
    # for i in range(0, len(points)):
    #     cv2.circle(canvas2, (points[i][1], points[i][0]), 5, (255, 255, 255), -1)
    # plt.imshow(canvas2)
    # plt.title("eliminate end point which is branch")
    # plt.show()

    """group each points"""
    labeled, num_labeled = label(img_gray)
    # plt.imshow(labeled)
    # plt.title("labeled")
    # plt.show()
    for p in points:
        class_value = labeled[p[0], p[1]]
        p[2] = class_value
    values = set(map(lambda x: x[2], points))
    new_point = [[(y[0], y[1], y[2]) for y in points if y[2] == x] for x in values]
    print("class point: ", new_point)


    two_endpoint = []
    for i in range (0, len(new_point)):
        group = new_point[i]
        print("group: ", group)
        if(len(group)>2):
            point_elimination = np.asarray(group)
            dist = pdist(point_elimination, 'euclidean')
            dist = squareform(dist)
            max_value = np.max(dist)
            max_pos_arr = np.where(dist[:, :] == max_value)
            point_1_pos = max_pos_arr[0][0]
            point_2_pos = max_pos_arr[1][0]
            two_endpoint.append(group[point_1_pos])
            two_endpoint.append(group[point_2_pos])
        elif(len(group)==2): #normal case
            two_endpoint.append(group[0])
            two_endpoint.append(group[1])
        elif(len(group)==1): # one point case
            two_endpoint.append(group[0])
    print("two point: ", two_endpoint)

    two_endpoint_arr = np.asarray(two_endpoint)
    p = np.where(two_endpoint_arr[:2]==0)
    for i in range (len(p[0])-1, -1, -1):
        index_zero_class = p[0][i]
        two_endpoint.pop(index_zero_class)
    print("eliminate zero class: ", two_endpoint)


    """assign priority to each points"""
    priority_label = []
    """all pixels of labeled"""
    total_pix_label = len((np.nonzero(labeled))[0])
    # print("Total pixels of labeled: ", total_pix_label)
    for i in range(1, num_labeled+1):
        id = np.where(labeled[:, :] == i)
        num = len(id[0])
        priority_label.append(((num/total_pix_label), i))
    # # """sort max to min area (labeled)"""
    priority_label = sorted(priority_label, reverse=True)
    print("probability of labeled pixels : ", priority_label)

    """map probability to each point"""
    points = []
    for i in range (0, len(priority_label)):
        group_id = priority_label[i][1]
        array = np.asarray(two_endpoint)
        search = np.where(array[:,2]==group_id)
        # print("search: ", search)
        if(len(search[0])==2):
            pos_1 = search[0][0]
            pos_2 = search[0][1]
            points.append((two_endpoint[pos_1][0], two_endpoint[pos_1][1], priority_label[i][0]))
            points.append((two_endpoint[pos_2][0], two_endpoint[pos_2][1], priority_label[i][0]))
        elif(len(search[0])==1):
            pos_1 = search[0][0]
            points.append((two_endpoint[pos_1][0], two_endpoint[pos_1][1], priority_label[i][0]))
    print("point: ", points)

    return points, priority_label, canvas

def connect_estimation(points, paper, map_label):
    img_gray = paper.copy()
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    labeled, num_labeled = label(img_gray)
    max_num=0
    max_area = 1
    for i in range (1, num_labeled):
        id = np.where(labeled[:, :] == i)
        num = len(id[0])
        # print("i: ", i)
        # print("num: ", num)
        if(num>max_num):
            max_num=num
            max_area=i

    """point:  [(82, 701, 0.9), (348, 403, 0.9), (5, 724, 0.04), (20, 725, 0.04), (55, 708, 0.03), (56, 707, 0.03)]"""
    """check which directions of pipeline (most probability)"""
    straigth_dis = 10

    dif_class = False
    if(len(points)==2):
        class_1 = points[0][2]
        class_2 = points[1][2]
        if(class_1!=class_2):
            dif_class = True
            points = sorted(points)

    row_1, col_1 = points[0][0], points[0][1]
    row_2, col_2 = points[1][0], points[1][1]

    curve_bool = False
    left_curve_bool = False
    rigth_curve_bool = False
    straigth_bool = False
    # print(row_1, col_1, row_2, col_2)
    """pipe \ , /"""
    if(abs(row_1-row_2)>straigth_dis or (abs(col_1-col_2)>straigth_dis)):
        degree = 7 #estimate to curve line
        curve_bool = True
        straigth_bool = False
        if((col_1-col_2)>0):  #"""pipe /"""
            print("left")
            left_curve_bool = True
        else:
            print("right")    #"""pipe \"""
            rigth_curve_bool = True
    else:
        degree = 2 #estimate to straigth line
        straigth_bool = True
        curve_bool = False
        left_curve_bool = False
        rigth_curve_bool = False
        print("straigth")

    """cal minimum distance between max area and other point"""


    new_point = [(y[0], y[1], y[2]) for y in points if y[2] >= 0.2]
    print("new point: ", new_point)

    filter_area = []
    # count=0
    if(len(new_point)==2):
        if(new_point[0][2]==new_point[1][2]): # same label
            connect_bool = False
        else:
            connect_bool = True
    """confirm that there are more than 1 area that must be connected"""
    # if(len(points)>2 or connect_bool):
    if(len(points)>=2):
        for i in range (0, len(new_point)):
            """ref_1 = points[0] #top
                ref_2 = points[1] #bottom"""
            if(curve_bool): #curve case
                if(left_curve_bool): #/
                    """quardant 1"""
                    top_angle_min = -80
                    top_angle_max = -5
                    """quardant 3"""
                    bot_angle_min = -80
                    bot_angle_max = -5
                else: #\
                    """quardant 1"""
                    top_angle_min = 5
                    top_angle_max = 80
                    """quardant 3"""
                    bot_angle_min = 5
                    bot_angle_max = 80
            else: #|
                top_angle_min = -10
                top_angle_max = 10
                bot_angle_min = -10
                bot_angle_max = 10

            ref_point = []
            ref = new_point[i]
            ref_point.append(ref[:len(ref) - 1])
            # print(ref_point)
            group = ref[2]
            # print("group: ", group)
            if(len(new_point)>0):
                points_i = points.copy()
                # print("pointttt: ", points_i)
                points_i_arr = np.asarray(points_i)
                b = np.where(points_i_arr[:]==group)
                # print("b: ", b)
                if(len(b[0])==2): #normal case
                    b1, b2 = b[0][0], b[0][1]
                    """pop 2 maximum"""
                    points_i.pop(b1)
                    points_i.pop(b2-1)
                elif(len(b[0])==1):
                    b1 = b[0][0]
                    points_i.pop(b1)
                # print("points i: ", points_i)

                points_xy = []
                for l in range(0, len(points_i)):
                    a = points_i[l]
                    points_xy.append(a[:len(a) - 1])
                print("points xy: ", points_xy)

            if(points_xy==[]): #two points are same label
                for l in range(0, len(points)):
                    a = points[l]
                    points_xy.append(a[:len(a) - 1])
                print("points xy(2 points same area): ", points_xy)

            dis = cdist(ref_point, points_xy)
            # print(dis)
            dist_pos = np.where(dis[:]<200)
            # print(dist_pos)
            dist_pos = dist_pos[1]
            # print(dist_pos)
            for j in range (0, len(dist_pos)):
                ref_x, ref_y = ref_point[0][1], ref_point[0][0]
                p_x, p_y = points_xy[dist_pos[j]][1], points_xy[dist_pos[j]][0]
                delta_x, delta_y = ref_x-p_x, ref_y-p_y
                ratio_delta = delta_x / delta_y
                angle = math.degrees(math.atan(ratio_delta))
                print("angle: ", angle)
                print("delta y: ", delta_y)
                # cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (0, 255, 255))
                if(dif_class): #only teo element and different class
                    if (top_angle_min <= angle <= top_angle_max):
                        print("two diff class")
                        filter_area.append(group)
                        filter_area.append(points_i[dist_pos[j]][2])
                        cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (255, 255, 0), thickness=4)

                elif(i%2==0 or i==0): #top
                    if(delta_y > 0):
                        if(top_angle_min<=angle<=top_angle_max):
                            print("top")
                            # print("angle: ", angle)
                            # print("delta y: ", delta_y)
                            # print(p_x, p_y,points_i[dist_pos[j]][2] )
                            filter_area.append(group)
                            filter_area.append(points_i[dist_pos[j]][2])
                            cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (255, 0, 0), thickness=4)
                else:
                    if (delta_y < 0):
                        if (bot_angle_min <= angle <= bot_angle_max):
                            print("bottom")
                            # print("angle: ", angle)
                            # print("delta y: ", delta_y)
                            filter_area.append(group)
                            filter_area.append(points_i[dist_pos[j]][2])
                            cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (0, 255, 0), thickness=4)
                print()
                # cv2.imshow("paper", paper)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()
                # plt.imshow(paper)
                # plt.title("drawn")
                # plt.show()
    lab = np.zeros(img_gray.shape, np.float32)
    if (filter_area != []):
        filter_area = list(set(filter_area))
        filter_area = sorted(filter_area, reverse=True)
        print("filter area: ", filter_area)
        map_label = sorted(map_label, key=lambda x: x[0], reverse=True)
        print("map label: ", map_label)
        values = set(map(lambda x: (x[0], x[1]), map_label))
        mapped_area = [[(x[1]) for y in filter_area if y == x[0]] for x in values]
        mapped_area = sorted(mapped_area)
        print("mapped area: ", mapped_area)
        for i in range (0, len(mapped_area)):
            lab = lab + (1 * (labeled == mapped_area[i])) + (0 * (labeled != mapped_area[i]))
    else:

        lab = (1 * (labeled == max_area)) + (0 * (labeled != max_area))

    # plt.imshow(paper)
    # plt.title("after")
    # plt.show()
    # print()
    return lab, paper



def estimate_polynomial(labeled):
    rows, cols = labeled.shape
    degree = 7
    p, x_new = None, None
    lab_ske = skeletonize_3d(labeled)
    # lab_ske = (1 * (lab_ske == True)) + (0 * (lab_ske == False))
    lab_ske = lab_ske.astype(np.float32)
    # plt.imshow(lab_ske)
    # plt.title("ske labeled")
    # plt.show()
    point_x = []
    point_y = []
    for x in range(0, cols):
        for y in range(0, rows):
            if (lab_ske[y][x] != 0):
                # img_ske[i][j]=55
                point_x.append(x)
                point_y.append(y)
    # point_x=sorted(point_x)
    # print("x: ", point_x)
    # print("y: ", point_y)
    point_x = np.asarray(point_x)
    point_y = np.asarray(point_y)
    if ((len(point_x)!=0) and (len(point_y!=0))):
        z = np.polyfit(point_x, point_y, degree) #degree 15
        p = np.poly1d(z)
        x_new = np.linspace(point_x[0], point_x[-1], num=len(point_x) * 10)
    # x_new = np.linspace(0, cols, num=cols * 10)
    # plt.imshow(lab_ske)
    # plt.plot(x_new, p(x_new))
    # plt.title("estimate")
    # plt.show()
    return p, x_new

def join(img, scene, poly_function_pre):
        rows, cols = img.shape
        img = img.astype(np.uint8)
        img = reduce_noise(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        points, map_label, point_ske = endpoint_cal(img)
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\021219(ver.4.3)\frame_" + str(scene) + "_points.png", point_ske)
        paper = img.copy()
        # print(paper.shape)
        paper = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
        labeled, paper = connect_estimation(points, paper, map_label)

        plt.imshow(paper)
        plt.title("connected "+ str(scene))
        plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\021219(ver.4.3)\frame_" + str(scene) + "_coonected_points.png")

        lab_save = (0*(labeled==0))+(255*(labeled!=0))
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\021219(ver.4.3)\frame_" + str(scene) + "_labeled.png", lab_save)
        # plt.imshow(labeled)
        # plt.title("labeled frame"+str(scene))
        # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\301218(ver4.3)\frame_" + str(scene) + "_labeled.png")
        # # plt.show()

        poly_function, x_new = estimate_polynomial(labeled)
        if(poly_function is None or x_new is None):
            poly_function=poly_function_pre

        canvas = np.zeros(img.shape)

        for k in range (0, len(x_new)):
                cv2.circle(canvas, (int(x_new[k]), int(poly_function(x_new[k]))), 6, (255, 255, 255), -1)

        # plt.imshow(canvas)
        # plt.title("canvas frame"+str(scene))
        # plt.show()

        img=(0*(img==0))+(1*(img==255))
        img=img.astype(np.uint8)
        canvas=(0*(canvas==0))+(1*(canvas>0))
        canvas=canvas.astype(np.uint8)
        return img, canvas, labeled, poly_function