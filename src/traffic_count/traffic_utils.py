import numpy as np
import cv2


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

def roi_count(roi_point, list_bboxs, list_classes,  color, size):
    class_num = [0]*len(list_classes)

    roi_value = image_mask(roi_point, 1, size)
    # set the roi to red
    color_plate = color
    roi_color_image = np.array(roi_value * color_plate, np.uint8)
    
    for i in range(0, len(list_bboxs)):
        x1 = list_bboxs[i].xmin
        y1 = list_bboxs[i].ymin
        x2 = list_bboxs[i].xmax
        y2 = list_bboxs[i].ymax
        cls = list_bboxs[i].Class

        # 撞线的点(中心点)
        x = int(x1 + ((x2 - x1) * 0.5))
        y = int(y1 + ((y2 - y1) * 0.5))

        # 判断车辆（中心点）是否在roi区域
        if roi_value[y, x] == 1:
            cls_index = list_classes.index(cls)
            class_num[cls_index] += 1

    return class_num, roi_color_image

def image_count(list_bboxs, list_classes):
    class_num = [0]*len(list_classes)
  
    for i in range(0, len(list_bboxs)):
        x1 = list_bboxs[i].xmin
        y1 = list_bboxs[i].ymin
        x2 = list_bboxs[i].xmax
        y2 = list_bboxs[i].ymax
        cls = list_bboxs[i].Class
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

def traffic_count_track(image, list_bboxes, list_classes,  polygon_mask_first_and_second, first_list, second_list,  up_count, down_count):
    class_num = len(list_classes)
    point_radius = 3

    if len(list_bboxes) > 0:
        for i in range(0, len(list_bboxes)):
            track_id = list_bboxes[i].id
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
            if polygon_mask_first_and_second[y,x]==1 or polygon_mask_first_and_second[y, x] == 3:
                # 如果撞 蓝polygon
                if track_id not in first_list:
                    first_list.append(track_id)
                # 判断 黄polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是 外出方向
                if track_id in second_list:
                    # 外出+1
                    down_count[cls_index] += 1
                    print('up count:', up_count, ', up id:', second_list)
                    # 删除 黄polygon list 中的此id
                    second_list.remove(track_id)


            elif polygon_mask_first_and_second[y, x] == 2:
                # 如果撞 黄polygon
                if track_id not in second_list:
                    second_list.append(track_id)
                # 判断 蓝polygon list 里是否有此 track_id
                # 有此 track_id，则 认为是 进入方向
                if track_id in first_list:
                    # 进入+1
                    up_count[cls_index] += 1
                    print('down count:', down_count, ', down id:', first_list)
                    # 删除 蓝polygon list 中的此id
                    first_list.remove(track_id)
        pass

        # ----------------------清除无用id----------------------
        list_overlapping_all = second_list + first_list
        for id in list_overlapping_all:
            is_found = False
            for i in range(0, len(list_bboxes)):
                bbox_id = list_bboxes[i].id
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
        # list_bboxes.clear()

    else:
        # 如果图像中没有任何的bbox，则清空list
        first_list.clear()
        second_list.clear()

    return up_count, down_count    

