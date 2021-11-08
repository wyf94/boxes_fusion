import numpy as np
import cv2
"""
#生成一个尺寸为size的图片mask，包含1个polygon，（值范围 0、1、2），供撞线计算使用
#list_point：点数组
#color_value: polygon填充的值
#size：图片尺寸
"""
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

def roi_count(image, roi_point, list_bboxs, list_classes,  color, size):
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

def image_count(image, list_bboxs, list_classes):
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
    
def traffic_count(image, frame_count, list_bboxs, polygon_mask_first_and_second, first_list, second_list,  up_count, down_count):
    first_num = 0
    second_num = 0
    point_radius = 3

    if len(list_bboxs) > 0:
        for i in range(0, len(list_bboxs)):
            conf = list_bboxs[i].probability
            x1 = list_bboxs[i].xmin
            y1 = list_bboxs[i].ymin
            x2 = list_bboxs[i].xmax
            y2 = list_bboxs[i].ymax
            cls = list_bboxs[i].Class
            
            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))

            #画出中心list_bboxs的中心点
            list_pts = []
            list_pts.append([x-point_radius, y-point_radius])
            list_pts.append([x-point_radius, y+point_radius])
            list_pts.append([x+point_radius, y+point_radius])
            list_pts.append([x+point_radius, y-point_radius])
            ndarray_pts = np.array(list_pts, np.int32)
            image = cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))           

            # 判断目标在是否在多边形0和1内
            if polygon_mask_first_and_second[y, x] == 1 or polygon_mask_first_and_second[y, x]  == 3:
                first_num += 1
            elif polygon_mask_first_and_second[y, x] == 2:
                second_num += 1

        if frame_count > 2:
            second_list.pop(0)
            first_list.pop(0)

        first_list.append(first_num)
        second_list.append(second_num)

        if frame_count > 2 and first_list[0] > first_list[1]:
            first_diff = first_list[0] - first_list[1]
            second_diff =  second_list[1] - second_list[0]
            if first_diff == second_diff:
                up_count += first_diff
                print('up count:', up_count)
        elif frame_count >2 and second_list[0] > second_list[1]:
            second_diff =  second_list[0] - second_list[1]
            first_diff = first_list[1] - first_list[0]
            if first_diff == second_diff:
                down_count += first_diff  
                print('down count:', down_count)

    return up_count, down_count             