import cv2
import numpy as np
import pickle as pl
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
        if (area >= 20):
            cv2.drawContours(img_red, [cnts[j]], -1, (0, 0, 255), -1)
    _, img_red = cv2.threshold(img_red, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\res_bin.png", img_red)
    # cv2.imshow("contour", img_red)
    # cv2.waitKey(-1)
    return img_red


def endpoint_cal(img_gray):
    ske_input = (1 * (img_gray == 255)) + (0 * (img_gray == 0))
    ske = skeletonize_3d(ske_input)
    ske = (1 * (ske > 0))+(0 * (ske < 0))
    # plt.imshow(ske)
    # plt.title("skeleton")
    # plt.show()

    rows, cols = ske.shape
    canvas = np.zeros(ske.shape, np.float32)
    canvas2 = canvas.copy()

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
            canvas2[pos_y[i], pos_x[i]] = conv
            """end point"""
            if (0< conv <= (0.25)):
                points.append([pos_y[i], pos_x[i], 1])
                cv2.circle(canvas, (pos_x[i], pos_y[i]), 5, (255, 255, 255), -1)
            # if (conv>=0.4): #Not child (4/9)
            #     print("branch point")
            #     branch.append([rows[i], cols[i], 1])
            #     cv2.circle(canvas, (pos_x[i], pos_y[i]), 5, (255, 255, 255), -1)
    # print("points of skeleton: ", points)

    # plt.figure()
    # plt.imshow(canvas2)
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
    for i in range(0, len(new_point)):
        group = new_point[i]
        # print("group: ", group)
        if(len(group) > 2):
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
    p = np.where(two_endpoint_arr[:2] == 0)
    for i in range(len(p[0])-1, -1, -1):
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
    for i in range(0, len(priority_label)):
        group_id = priority_label[i][1]
        array = np.asarray(two_endpoint)
        search = np.where(array[:,2]==group_id)
        # print("search: ", search)
        if(len(search[0]) == 2):
            pos_1 = search[0][0]
            pos_2 = search[0][1]
            points.append((two_endpoint[pos_1][0], two_endpoint[pos_1][1], priority_label[i][0]))
            points.append((two_endpoint[pos_2][0], two_endpoint[pos_2][1], priority_label[i][0]))
        elif(len(search[0]) == 1):
            pos_1 = search[0][0]
            points.append((two_endpoint[pos_1][0], two_endpoint[pos_1][1], priority_label[i][0]))
    print("point: ", points)

    return points, priority_label, canvas, canvas2

def connect_estimation(points, paper, map_label, re_phi):
    img_gray = paper.copy()
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    lab = np.zeros(img_gray.shape, np.float32)
    labeled, num_labeled = label(img_gray)
    rows, cols = img_gray.shape
    max_num = 0
    max_area = 1
    for i in range (1, num_labeled+1):
        id = np.where(labeled[:, :] == i)
        num = len(id[0])
        # print("i: ", i)
        # print("num: ", num)
        if(num > max_num):
            max_num = num
            max_area = i

    """point:  [(82, 701, 0.9), (348, 403, 0.9), (5, 724, 0.04), (20, 725, 0.04), (55, 708, 0.03), (56, 707, 0.03)]"""
    """check which directions of pipeline (most probability)"""
    straigth_dis = 20

    dif_class = False

    if(len(points)==1):
        no_pair_point = True
    else:
        no_pair_point = False

    """points have only one element"""
    if(no_pair_point):
        print("len points == 1")
        map_label = sorted(map_label, key=lambda x: x[0], reverse=True)
        print("map label: ", map_label)
        for i in range (0, len(map_label)):
            if(points[0][0][2]==map_label[i][0]):
                pr_class_1 = map_label[i][0]
                class_1 = map_label[i][1]
                break
        if (pr_class_1>0.3):
            lab = lab + (1 * (labeled == class_1)) + (0 * (labeled != class_1))
            ske_lab = skeletonize_3d(lab)
            ske_lab = (1 * (ske_lab > 0)) + (0 * (ske_lab < 0))
            area_po = np.nonzero(lab)
            pos_y, pos_x = area_po[0], area_po[1]
            points = []
            canvas = np.zeros(img_gray.shape, np.float32)
            for i in range(0, len(pos_y)):
                if (pos_y[i] < rows and pos_x[i] < cols):  # for win 3*3
                    # window = ske_pad[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
                    window = ske_lab[pos_y[i] - 1:pos_y[i] + 2, pos_x[i] - 1:pos_x[i] + 2]
                    conv = np.mean(window)
                    # canvas[pos_y[i], pos_x[i]] = conv
                    """end point"""
                    if (0 < conv <= (0.25)):
                        points.append([pos_y[i], pos_x[i], 1])
                        cv2.circle(canvas, (pos_x[i], pos_y[i]), 5, (255, 255, 255), -1)
                        plt.imshow(canvas)
                        plt.show()
            if(points!=[]):
                row_1, col_1 = points[0][0], points[0][1]
                row_2, col_2 = points[1][0], points[1][1]
            else:
                re_phi = True
        else:
            re_phi = True


    """points have more than 1 elements"""
    if(not no_pair_point):
        if(len(points)==2):
            class_1 = points[0][2]
            class_2 = points[1][2]
            if(class_1!=class_2):
                dif_class = True
                points = sorted(points)
                row_1, col_1 = points[0][0], points[0][1]
                row_2, col_2 = points[1][0], points[1][1]
                print("two diff class is assigned")
            else:
                row_1, col_1 = points[0][0], points[0][1]
                row_2, col_2 = points[1][0], points[1][1]
                print("two same class is assigned")

        else:  #find pair point
            n_diff = True
            for i in range (0, len(points)-1):
                class_1 = points[i][2]
                class_2 = points[i+1][2]
                if(class_1==class_2):
                    #if (class_1<1.0):
                       # “find end point of max_area“
                    row_1, col_1 = points[i][0], points[i][1]
                    row_2, col_2 = points[i+1][0], points[i+1][1]
                    print("pair point: ", class_1)
                    n_diff = False
                    break

            # """more than 2 but diff group"""
            #


    """find the connection"""
    if(not re_phi):
        curve_bool = False
        left_curve_bool = False
        rigth_curve_bool = False
        straigth_bool = False
        # print(row_1, col_1, row_2, col_2)
        """pipe \ , /"""
        # if(abs(row_1-row_2)>straigth_dis or (abs(col_1-col_2)>straigth_dis)):
        if(abs(row_1-row_2)>straigth_dis and (abs(col_1-col_2)>straigth_dis)):
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
            degree = 7 #estimate to straigth line
            straigth_bool = True
            curve_bool = False
            left_curve_bool = False
            rigth_curve_bool = False
            print("straigth")

        """cal minimum distance between max area and other point"""


        new_point = [(y[0], y[1], y[2]) for y in points if y[2] >= 0.15]
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
                    top_angle_min = -5
                    top_angle_max = 5
                    bot_angle_min = -5
                    bot_angle_max = 5

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
                    print("b: ", b)
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

                if(straigth_bool or dif_class):
                    max_dis = 500
                else:
                    max_dis = 200
                dis = cdist(ref_point, points_xy)
                # print(dis)
                dist_pos = np.where(dis[:]<max_dis)
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
                    """the relationship between range and angle"""
                    if (straigth_bool and abs(delta_y) > 170):
                        print("reduce limited angle")
                        top_angle_min, bot_angle_min = -2, -2
                        top_angle_max, bot_angle_max = 2, 2
                    # cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (0, 255, 255))
                    if(dif_class): #only two element and different class
                        if (top_angle_min <= angle <= top_angle_max):
                            print("two diff class")
                            print(p_x, p_y)
                            filter_area.append(group)
                            filter_area.append(points_i[dist_pos[j]][2])
                            cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (255, 255, 0), thickness=4)

                    elif(i%2==0 or i==0): #top
                        if(delta_y > 0):
                            if(top_angle_min<=angle<=top_angle_max):
                                print("top")
                                print(p_x, p_y)
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
                                print(p_x, p_y)
                                # print("angle: ", angle)
                                # print("delta y: ", delta_y)
                                filter_area.append(group)
                                filter_area.append(points_i[dist_pos[j]][2])
                                cv2.line(paper, (ref_x, ref_y), (p_x, p_y), (0, 255, 0), thickness=4)
                    # print(filter_area)
                    print()
                    # cv2.imshow("paper", paper)
                    # cv2.waitKey(-1)
                    # cv2.destroyAllWindows()
                    # plt.imshow(paper)
                    # plt.title("drawn")
                    # plt.show()
        if (filter_area != []):
            print("filter area: ", filter_area)
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
            print("filter area is []")
            print("max area: ", max_area)
            lab = (1 * (labeled == max_area)) + (0 * (labeled != max_area))

        # plt.imshow(paper)
        # plt.title("after")
        # plt.show()
        # print()
    return lab, paper, curve_bool, re_phi



def estimate_polynomial(labeled, degree):
    rows, cols = labeled.shape
    # degree = 7
    p, x_new = None, None
    lab_ske = skeletonize_3d(labeled)
    # lab_ske = (1 * (lab_ske == True)) + (0 * (lab_ske == False))
    lab_ske = lab_ske.astype(np.float32)
    # plt.imshow(lab_ske)
    # plt.title("ske labeled")
    # plt.show()
    point_x = []
    point_y = []
    y_max, y_min = 0, rows
    y_real = 0
    mid_x = int(cols/2)
    for x in range(0, cols):
        for y in range(0, rows):
            if (lab_ske[y][x] != 0):
                # img_ske[i][j]=55
                point_x.append(x)
                point_y.append(y)
                # print("y value: ", y)
                if (y > y_max):
                    y_max = y
                    print("chang y max: ", y_max)
                    x_made_y_max = x
                elif (y < y_min):
                    y_min = y
                    x_made_y_min = x
        if (x == mid_x):
            print("add middle x")
            point_x.append(x)
            point_y.append(rows)
            y_mid = rows
            x_made_y_max = x

    # point_x=sorted(point_x)
    # print("x: ", point_x)
    # print("y: ", point_y)
    point_x = np.asarray(point_x)
    point_y = np.asarray(point_y)

    """decide to keep y middle or delete it"""
    y_de = abs(y_mid-y_max)
    print("y mid, y max: ", y_mid, y_max)
    if(y_de < 50):
        print("delete y mid")
        pos = np.where(point_y == y_mid)
        print("position del: ", pos)
        pos = pos[0][0]
        point_y = np.delete(point_y, pos)
        point_x = np.delete(point_x, pos)
    else:
        y_max = y_mid

    min_max = [(x_made_y_min, y_min), (x_made_y_max, y_max)]

    z = np.polyfit(point_x, point_y, degree)  # degree 15
    p = np.poly1d(z)
    x_new = np.linspace(point_x[0], point_x[-1], num=len(point_x) * 10)
    # if ((len(point_x)!=0) and (len(point_y)!=0)):
    #     if(curve_bool):
    #         z = np.polyfit(point_x, point_y, degree) #degree 15
    #         p = np.poly1d(z)
    #         x_new = np.linspace(point_x[0], point_x[-1], num=len(point_x) * 10)
    #     else: #|
    #         p_min = np.where(point_y==point_y.min())
    #         p_max = np.where(point_y==point_y.max())
    #         x_new = np.linspace(point_x[p_min[0][0]], point_x[p_max[0][0]], num=2)
    #         p = np.linspace(point_y.min(), point_y.max(), num=2)
    # # x_new = np.linspace(0, cols, num=cols * 10)
    #
    # # plt.imshow(lab_ske)
    # # plt.plot(x_new, p(x_new))
    # # plt.title("estimate")
    # # plt.show()
    return p, x_new, lab_ske, min_max

def join(img, scene, poly_params):
        poly_function_pre = poly_params[0]
        # x_pre = poly_params[1]
        # print("x_pre: ", x_pre)
        re_phi = False
        canvas = np.zeros(img.shape)
        rows, cols = img.shape
        img = img.astype(np.uint8)
        img = reduce_noise(img)
        img = (1*(img>0))+(0*(img<=0)) #to be an input of skeletonize3d
        labeled = skeletonize_3d(img)
        degree = 5
        pipe_width = 4
        plt.title("skeleton_" + str(scene))
        plt.imshow(labeled)
        plt.show()
        # # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        # # img = cv2.dilate(img, kernel, iterations=2)
        # # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\090119\frame_" + str(scene) + "_morp.png", img)
        # # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # points, map_label, point_ske, canvas2 = endpoint_cal(img)
        # fig = plt.figure()
        # plt.title("skeleton_" + str(scene))
        # plt.imshow(canvas2)
        # pl.dump(fig, open(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\skeleton_" + str(scene) + ".pickle", 'wb'))
        #
        # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(scene) + "_points.png", point_ske)
        # paper = img.copy()
        # # print(paper.shape)
        # paper = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
        # labeled, paper, curve_bool, re_phi = connect_estimation(points, paper, map_label, re_phi)
        #
        # plt.imshow(paper)
        # plt.title("connected "+ str(scene))
        # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(scene) + "_conected_points.png")
        #
        # lab_save = (0*(labeled==0))+(255*(labeled!=0))
        # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(scene) + "_labeled.png", lab_save)
        # # plt.imshow(labeled)
        # # plt.title("labeled frame"+str(scene))
        # # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\301218(ver4.3)\frame_" + str(scene) + "_labeled.png")
        # # # plt.show()
        # if(not re_phi):
        #     if (curve_bool): #/, \
        #         pipe_width = 4 #radius
        #         degree = 7
        #     else: # |
        #         pipe_width = 7
        #         degree = 5

        if(not re_phi):
            poly_function, x_new, lab_ske, min_max = estimate_polynomial(labeled, degree)
            print("poly function: ", poly_function)
            # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_"+ str(scene) + "_estimate.png", lab_ske)

            """except first time"""
            if(poly_function_pre != 0):
                error_pow = []
                """check error function"""
                for id in range(0, len(x_new)):
                    diff = (poly_function_pre(x_new[id])-poly_function(x_new[id]))
                    error_pow.append(diff**2)
                sum_err = sum(error_pow)
                rmse = math.sqrt((sum_err)/len(x_new))
                print("rmse: ", rmse)

            """in case of polynomial function cannot be found"""
            if(poly_function is None or x_new is None):
                poly_function=poly_function_pre

            """assign shape of a pipeline"""
            x_top, y_top = min_max[0][0], min_max[0][1]
            x_bot, y_bot = min_max[1][0], min_max[1][1]
            print("min_max: ", x_top, y_top, x_bot, y_bot)
            # #top and bottom point
            # if(yi > yj):
            #     x_top, y_top = xi, yi
            #     x_bot, y_bot = xj, yj
            # else:
            #     x_top, y_top = xj, yj
            #     x_bot, y_bot = xi, yi
            delta_x, delta_y = abs(x_top - x_bot), (y_bot - y_top)
            ratio_delta = delta_y / delta_x
            angle = math.degrees(math.atan(ratio_delta))
            print("angle: ", angle)
            if(10 <= angle <= 60):
                print("curve bool")
                curve_bool = True
            elif(60 < angle <= 90):
                print("not curve bool")
                curve_bool = False

            """draw polynomial function"""
            if(curve_bool):
                # # print("x_new1: ", x_new)
                # if(poly_function(x_made_y_max)<poly_function(x_pre)):
                #     print("append case")
                #     x_new = np.append(x_new, x_pre)
                # # print("x_new2: ", x_new)
                x_new = sorted(x_new)
                x_new = np.linspace(x_new[0], x_new[-1], num=len(x_new) * 10)
                print("x_new3: ", x_new)
                x_1, y_1 = x_new[0], poly_function(x_new[0])
                x_2, y_2 = x_new[-1], poly_function(x_new[-1])
                for k in range (0, len(x_new)):
                        cv2.circle(canvas, (int(x_new[k]), int(poly_function(x_new[k]))), pipe_width, (255, 255, 255), -1)
            else: # | poly_function is y values min and max
                # if (poly_function(x_new[-1]) < poly_params(x_pre)):
                #     np.append(x_new, x_pre)
                x_1, y_1 = x_new[0], poly_function[0]
                x_2, y_2 = x_new[-1], poly_function[-1]
                cv2.line(canvas, (int(x_new[0]), int(poly_function[0])), (int(x_new[-1]), int(poly_function[-1])), (255, 0, 0), thickness=pipe_width)

            """if pipeline is very low -> reinit"""
            del_x, del_y = x_1 - x_2, y_1 - y_2
            r = math.sqrt(math.pow(del_x, 2)+math.pow(del_y,2))
            print("distance result: ", r)
            if(r<100):
                print("re initial because the distance from polynomial estimation less than 100")
                re_phi = True
            # plt.imshow(canvas)
            # plt.title("canvas frame"+str(scene))
            # plt.show()

            img=(0*(img==0))+(1*(img==255))
            img=img.astype(np.uint8)
            canvas=(0*(canvas==0))+(1*(canvas>0))
            canvas=canvas.astype(np.uint8)

            poly_params = [poly_function]
        return img, canvas, labeled, poly_params, re_phi