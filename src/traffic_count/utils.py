import numpy as np
import cv2

class TrafficCount(object):
    def __init__(self, image, list_bboxes, list_classes, line):
        self.image = image
        self.list_bboxes = list_bboxes
        self.list_classes = list_classes
        self.line  = line
        self.size = (self.image.shape[0], self.image.shape[1])

    def image_mask(self, list_point, color_value):
        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        mask_image_temp = np.zeros(self.size, dtype=np.uint8)

        # 初始化撞线polygon
        ndarray_pts = np.array(list_point, np.int32)
        polygon_color_value = cv2.fillPoly(mask_image_temp, [ndarray_pts], color=color_value)
        polygon_color_value = polygon_color_value[:, :, np.newaxis]

        return polygon_color_value

    def  line2polygon(self, width, height):
        polygon_0  = [self.line[0], self.line[1], [self.line[1][0] + width, self.line[1][1] + height], [self.line[0][0] + width, self.line[0][1] + height]]
        polygon_1  = [self.line[0], self.line[1], [self.line[1][0] - width, self.line[1][1] - height], [self.line[0][0] - width, self.line[0][1] - height]]
        #生成一个尺寸为size的图片mask，包含1个polygon，（值范围 0、1、2, 3(交界线)），供撞线计算使用
        polygon_value_first = self.image_mask(polygon_0, 1)
        polygon_value_second = self.image_mask(polygon_1, 2)
        polygon_mask_first_and_second = polygon_value_first + polygon_value_second

        # set the first  polygon to blue
        blue_color_plate = [255, 0, 0]
        blue_image = np.array(polygon_value_first * blue_color_plate, np.uint8)
        # set the first  polygon to yelllow
        yellow_color_plate = [0, 255, 255]
        yellow_image = np.array(polygon_value_second * yellow_color_plate, np.uint8)
        polygon_color_image = blue_image + yellow_image
        
        return polygon_mask_first_and_second,  polygon_color_image

    def image_count(self):
        class_num = [0]*len(self.list_classes)
    
        for i in range(0, len(self.list_bboxes)):
            x1 = self.list_bboxes[i].xmin
            y1 = self.list_bboxes[i].ymin
            x2 = self.list_bboxes[i].xmax
            y2 = self.list_bboxes[i].ymax
            cls = self.list_bboxes[i].Class
            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))

            cls_index = self.list_classes.index(cls)
            class_num[cls_index] += 1

        return class_num

    def roi_count(self, roi_point, color):
        class_num = [0]*len(self.list_classes)

        roi_value = self.image_mask(roi_point, 1)
        # set the roi to red
        color_plate = color
        roi_color_image = np.array(roi_value * color_plate, np.uint8)
        
        for i in range(0, len(self.list_bboxes)):
            x1 = self.list_bboxes[i].xmin
            y1 = self.list_bboxes[i].ymin
            x2 = self.list_bboxes[i].xmax
            y2 = self.list_bboxes[i].ymax
            cls = self.list_bboxes[i].Class

            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y1 + ((y2 - y1) * 0.5))

            # 判断车辆（中心点）是否在roi区域
            if roi_value[y, x] == 1:
                cls_index = self.list_classes.index(cls)
                class_num[cls_index] += 1

        return class_num, roi_color_image

    def traffic_count(self,  polygon_mask_first_and_second, first_list, second_list,  up_count, down_count):
        class_num = len(self.list_classes)
        point_radius = 3

        first_num = [0]*class_num
        second_num = [0]*class_num

        if len(self.list_bboxes) > 0:
            for i in range(0, len(self.list_bboxes)):
                conf = self.list_bboxes[i].probability
                x1 = self.list_bboxes[i].xmin
                y1 = self.list_bboxes[i].ymin
                x2 = self.list_bboxes[i].xmax
                y2 = self.list_bboxes[i].ymax
                cls = self.list_bboxes[i].Class

                cls_index = self.list_classes.index(cls)
                
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
                self.image = cv2.fillPoly(self.image, [ndarray_pts], color=(0, 0, 255))           

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


