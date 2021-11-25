import numpy as np
import cv2
import math


def image_mask(list_point, color_value, size):
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros(size, dtype=np.uint8)

    # 初始化撞线polygon
    ndarray_pts = np.array(list_point, np.int32)
    polygon_color_value = cv2.fillPoly(mask_image_temp, [ndarray_pts], color=color_value)
    polygon_color_value = polygon_color_value[:, :, np.newaxis]

    return polygon_color_value

def polygon_mask(point_list_first, point_list_second, size):
    polygon_value_first = image_mask(point_list_first, 1, size)
    polygon_value_second = image_mask(point_list_second, 2, size)
    
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_first_and_second = polygon_value_first + polygon_value_second

    # set the first  polygon to blue
    blue_color_plate = [255, 0, 0]
    blue_image = np.array(polygon_value_first * blue_color_plate, np.uint8)
    # set the first  polygon to yelllow
    yellow_color_plate = [0, 255, 255]
    yellow_image = np.array(polygon_value_second * yellow_color_plate, np.uint8)
    # 彩色图片（值范围 0-255）用于图片显示
    polygon_color_image = blue_image + yellow_image

    return polygon_mask_first_and_second,  polygon_color_image

def  line2polygon(line, width, height, size):
    polygon_0  = [line[0], line[1], [line[1][0] + width, line[1][1] + height], [line[0][0] + width, line[0][1] + height]]
    polygon_1  = [line[0], line[1], [line[1][0] - width, line[1][1] - height], [line[0][0] - width, line[0][1] - height]]
    polygon_value_first = image_mask(polygon_0, 1, size)
    polygon_value_second = image_mask(polygon_1, 2, size)
    # set the first  polygon to blue
    blue_color_plate = [255, 0, 0]
    blue_image = np.array(polygon_value_first * blue_color_plate, np.uint8)
    # set the first  polygon to yelllow
    yellow_color_plate = [0, 255, 255]
    yellow_image = np.array(polygon_value_second * yellow_color_plate, np.uint8)

    polygon_mask_first_and_second = polygon_value_first + polygon_value_second
    polygon_color_image = blue_image + yellow_image
    
    return polygon_mask_first_and_second,  polygon_color_image

def roi_count(roi_point, list_bboxes, list_classes,  color, size):
    class_num = [0]*len(list_classes)

    roi_value = image_mask(roi_point, 1, size)
    # set the roi to red
    color_plate = color
    roi_color_image = np.array(roi_value * color_plate, np.uint8)
    
    for i in range(0, len(list_bboxes)):
        x1 = list_bboxes[i].xmin
        y1 = list_bboxes[i].ymin
        x2 = list_bboxes[i].xmax
        y2 = list_bboxes[i].ymax
        cls = list_bboxes[i].Class

        # 撞线的点(中心点)
        x = int(x1 + ((x2 - x1) * 0.5))
        y = int(y1 + ((y2 - y1) * 0.5))

        # 判断车辆（中心点）是否在roi区域
        if roi_value[y, x] == 1:
            cls_index = list_classes.index(cls)
            class_num[cls_index] += 1

    return class_num, roi_color_image

def roi_count_queue(roi_point, list_bboxes, list_classes, stop_point, color, size, is_center = False):
    class_num = [0]*len(list_classes)
    classes_length = [3, 8, 8, 1, 1]

    roi_value = image_mask(roi_point, 1, size)
    # set the roi to red
    color_plate = color
    roi_color_image = np.array(roi_value * color_plate, np.uint8)

    queue_info = {}
    distances = []
    roi_id = []
    roi_v = []

    # # 设置roi区域的1，2点为停止线，并选择其中点为停止点
    # stop_x = int(roi_point[0][0]+roi_point[1][0])
    # stop_y = int(roi_point[0][1]+roi_point[1][1])
    
    for i in range(0, len(list_bboxes)):
        track_id = list_bboxes[i].id
        x1 = list_bboxes[i].xmin
        y1 = list_bboxes[i].ymin
        x2 = list_bboxes[i].xmax
        y2 = list_bboxes[i].ymax
        cls = list_bboxes[i].Class

        ground_x = list_bboxes[i].x
        ground_y = list_bboxes[i].y
        vx = list_bboxes[i].vx
        vy = list_bboxes[i].vy

        # 撞线的点(中心点)

        x = int(x1 + ((x2 - x1) * 0.5))
        if is_center:
            y = int(y1 + ((y2 - y1) * 0.5))
        else:
            y = int(y2)

        # 判断车辆（中心点）是否在roi区域
        if roi_value[y, x] == 1:
            cls_index = list_classes.index(cls)
            class_num[cls_index] += 1

            dis = math.sqrt(math.pow(ground_x - stop_point[0], 2) + math.pow(ground_y - stop_point[1], 2))
            v =round(math.sqrt(vx*vx + vy*vy), 2)
            distances.append(dis)
            roi_id.append(track_id)
            roi_v.append(v)

    classified_statistic =[]
    sum_car = 0
    for i in range(0, len(list_classes)):
        sum_car += class_num[i]
        classified_count = {
            "class":list_classes[i],
            "num":class_num[i]
        }
        classified_statistic.append(classified_count)   
    

    if len(distances) > 0:
        max_dis = max(distances)
        min_dis = min(distances)

        tail_index = distances.index(max_dis)
        head_index = distances.index(min_dis)
        tail_v = roi_v[tail_index]
        head_v = roi_v[head_index]

        mean_v = np.mean(roi_v)
        mean_dis = (max_dis - min_dis) / len(distances)



        queue_info = {
            "car_num": sum_car,
            "count_list": classified_statistic,
            "ave_car_speed": round(mean_v, 2),
            "car_distribute": round(mean_dis, 2),
            "head_car_pos": round(min_dis, 2),
            "head_car_speed": round(head_v, 2),
            "tail_car_pos": round(max_dis, 2),
            "tail_car_speed": round(tail_v, 2),
            "car_count": len(distances)
        }

        if mean_v <= 5:
            queue_info.update({"is_queue": "True", "queue_len": round(max_dis, 2)})
        else:
            queue_info.update({"is_queue": "False", "queue_len": 0})

    return class_num, roi_color_image,  queue_info

def image_count(list_bboxes, list_classes):
    class_num = [0]*len(list_classes)
  
    for i in range(0, len(list_bboxes)):
        x1 = list_bboxes[i].xmin
        y1 = list_bboxes[i].ymin
        x2 = list_bboxes[i].xmax
        y2 = list_bboxes[i].ymax
        cls = list_bboxes[i].Class
        # 撞线的点(中心点)
        x = int(x1 + ((x2 - x1) * 0.5))
        y = int(y1 + ((y2 - y1) * 0.5))

        cls_index = list_classes.index(cls)
        class_num[cls_index] += 1

    return class_num

def traffic_count(image, list_bboxes, list_classes,  polygon_mask_first_and_second, first_list, second_list,  up_count, down_count):
    class_num = len(list_classes)
    point_radius = 3

    first_num = [0]*class_num
    second_num = [0]*class_num

    if len(list_bboxes) > 0:
        for i in range(0, len(list_bboxes)):
            # conf = list_bboxes[i].probability
            x1 = list_bboxes[i].xmin
            y1 = list_bboxes[i].ymin
            x2 = list_bboxes[i].xmax
            y2 = list_bboxes[i].ymax
            cls = list_bboxes[i].Class

            if (cls in list_classes):
                cls_index = list_classes.index(cls)
            
            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))
            # 判断目标在是否在多边形0和1内
            if polygon_mask_first_and_second[y, x] == 1 or polygon_mask_first_and_second[y, x]  == 3:
                first_num[cls_index] += 1
            elif polygon_mask_first_and_second[y, x] == 2:
                second_num[cls_index] += 1

        for index in range(0, class_num):
            first_list[0][index] = first_list[1][index] 
            first_list[1][index] = first_num[index] 
            second_list[0][index] = second_list[1][index] 
            second_list[1][index] = second_num[index]

        # print("first_list",first_list)
        # print("second_list",second_list)

        for i in range(0, class_num):
            if first_list[0][i] > first_list[1][i]:
                first_diff = first_list[0][i] - first_list[1][i]
                second_diff =  second_list[1][i] - second_list[0][i]
                if first_diff == second_diff:
                    up_count[i] += first_diff
            elif second_list[0][i] > second_list[1][i]:
                first_diff = first_list[1][i] - first_list[0][i]
                second_diff =  second_list[0][i] - second_list[1][i]
                if first_diff == second_diff:
                    down_count[i] += second_diff

    return up_count, down_count              
    
def bboxes_mask(tracks_msg,  size,  color_value = 1):
    img_bboxes_mask = np.zeros(size, dtype=np.uint8)
    img_bboxes_mask = img_bboxes_mask[:, :, np.newaxis]

    if len(tracks_msg.bbox_coordinate) > 0:
        for i in range(0, len(tracks_msg.bbox_coordinate)):
            x1 = tracks_msg.bbox_coordinate[i].xmin
            y1 = tracks_msg.bbox_coordinate[i].ymin
            x2 = tracks_msg.bbox_coordinate[i].xmax
            y2 = tracks_msg.bbox_coordinate[i].ymax
            point_list = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
            bboxes_mask = image_mask(point_list, 1, size)
            img_bboxes_mask += bboxes_mask

    # set the first  polygon to yelllow
    yellow_color_plate = [255, 255, 255]
    yellow_image = np.array(img_bboxes_mask * yellow_color_plate, np.uint8)

    return img_bboxes_mask, yellow_image

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)

def occupancy(tracks_msg, img_bboxes_mask, line, line_occupy_flag, line_occupy_time):
    width = 0
    height =  10
    sum_pixel = 0 
    l_x1, l_y1, l_x2, l_y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    rec1 = (l_x1, l_y1, l_x2 + width , l_y2 + height)
    # # 判断boundingboxes是否与检测线生成的polygon相交
    # for x in range(x1, x2 + 1 + width ):
    #     for y in range(y1, y2 + 1 + height):
    #         if img_bboxes_mask[y,x] >= 1:
    #             sum_pixel += 1
    # if sum_pixel > 100:
    #     occupy_flag = 1
    # else:
    #     occupy_flag = 0
    
    occupy_flag = 0
    if len(tracks_msg.bbox_coordinate) > 0:
        for i in range(0, len(tracks_msg.bbox_coordinate)):
            track_id = tracks_msg.bbox_coordinate[i].id
            x1 = tracks_msg.bbox_coordinate[i].xmin
            y1 = tracks_msg.bbox_coordinate[i].ymin
            x2 = tracks_msg.bbox_coordinate[i].xmax
            y2 = tracks_msg.bbox_coordinate[i].ymax
            rec2 = (x1, y1, x2, y2)
            iou = compute_IOU(rec1,rec2)
            if iou > 0:
                occupy_flag |= 1

    #  判断boundingboxes是否与检测线相交，如果相交则为有车存在，并记录有车->无车，无车->有车的时间点
    if (line_occupy_flag == 0 and occupy_flag ==1) or (line_occupy_flag == 1 and occupy_flag == 0):
        line_occupy_time.append(tracks_msg.header.stamp.to_sec())
    #
    if occupy_flag:
        line_occupy_flag = 1
    else:
        line_occupy_flag = 0

    return line_occupy_flag


def traffic_count_track(image, tracks_msg, list_classes,  polygon_mask_first_and_second, first_list, second_list,  
                                                up_count, down_count, car_head_passtime, car_speed, is_center=False):
    class_num = len(list_classes)

    if len(tracks_msg.bbox_coordinate) > 0:
        for i in range(0, len(tracks_msg.bbox_coordinate)):
            track_id = tracks_msg.bbox_coordinate[i].id
            x1 = tracks_msg.bbox_coordinate[i].xmin
            y1 = tracks_msg.bbox_coordinate[i].ymin
            x2 = tracks_msg.bbox_coordinate[i].xmax
            y2 = tracks_msg.bbox_coordinate[i].ymax
            cls = tracks_msg.bbox_coordinate[i].Class
            vx = tracks_msg.bbox_coordinate[i].vx
            vy = tracks_msg.bbox_coordinate[i].vy  

            if (cls in list_classes):
                cls_index = list_classes.index(cls)
            else:
                break

            # #  判断boundingboxes是否与检测线相交，如果相交则为有车存在，并记录有车->无车，无车->有车的时间点
            # line_center_x = int(line[0][0] + ((line[1][0] - line[0][0]) * 0.5))
            # line_center_y = int(line[0][1] + ((line[1][1] - line[0][1]) * 0.5))
            # print(line_center_x, line_center_y)
            # if img_bboxes_mask[line_center_y, line_center_x] >= 0:
            #     line_occupy = 1
            # print("line_occupy {0}: {1}  track_id:{2}".format(i, line_occupy, track_id))
            # occupy_flag |= line_occupy
            
            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            if is_center:
                y = int(y1 + ((y2 - y1) * 0.5))
            else:
                y = int(y2)
                y2 = int(y1)

            # 判断目标在是否在多边形0和1内
            if polygon_mask_first_and_second[y,x]==1 or polygon_mask_first_and_second[y, x] == 3:
                # 记录通过第一个polygon的时间戳以及数度
                car_head_passtime.append(tracks_msg.header.stamp.to_sec())
                car_speed.append(round(math.sqrt(vx*vx + vy*vy), 2))

                # 如果撞 第一个 polygon
                if track_id not in first_list:
                    first_list.append(track_id)
                # 判断 第二个 polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是 2--->1 方向
                if track_id in second_list:
                    # 2--->1方向 +1
                    down_count[cls_index] += 1
                    print('2--->1 count:', down_count, ', 2--->1 id:', second_list)
                    # 删除 黄polygon list 中的此id
                    second_list.remove(track_id)


            elif polygon_mask_first_and_second[y, x] == 2:
                # 如果撞第二个 polygon
                if track_id not in second_list:
                    second_list.append(track_id)
                # 判断第一个polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是 1--->2 方向
                if track_id in first_list:
                    #  1--->2 方向 +1
                    up_count[cls_index] += 1
                    print('1--->2 count:', up_count, ', 1--->2  id:', first_list)
                    # 删除 蓝polygon list 中的此id
                    first_list.remove(track_id)
        pass

        # print("occupy_flag: ", occupy_flag)
        # if (line_occupy_flag == 0 and occupy_flag ==1) or (line_occupy_flag == 1 and occupy_flag == 0):
        #     line_occupy_time.append(tracks_msg.header.stamp.to_sec())
        
        # if occupy_flag:
        #     line_occupy_flag = 1
        #     print("-----------------------------------------------")
        # else:
        #     line_occupy_flag = 0
        # print("line_occupy_flag2: ", line_occupy_flag)

        # ----------------------清除无用id----------------------
        list_overlapping_all = second_list + first_list
        for id in list_overlapping_all:
            is_found = False
            for i in range(0, len(tracks_msg.bbox_coordinate)):
                bbox_id = tracks_msg.bbox_coordinate[i].id
                if bbox_id == id:
                    is_found = True
                    break

            if not is_found:
                # 如果没找到，删除id
                if id in second_list:
                    second_list.remove(id)
                if id in first_list:
                    first_list.remove(id)
        list_overlapping_all.clear()

        # # 清空list
        # tracks_msg.bbox_coordinate.clear()

    else:
        # 如果图像中没有任何的bbox，则清空list
        first_list.clear()
        second_list.clear()

    return up_count, down_count

