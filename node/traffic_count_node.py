#!/usr/bin/python3
#!coding=utf-8

import rospy
import numpy as np
from rospy.topics import _PublisherImpl
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import requests
import json
import os
from threading import Timer

from traffic_count.msg import BoundingBox
from traffic_count.msg import BoundingBoxes

import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


from traffic_count.cameratool import CameraTools
from traffic_count.utils import TrafficCount
# import traffic_count.traffic_count as traffic_count
import traffic_count.traffic_utils as utils
# from traffic_count.yolo_classes import CLASSES_LIST
from traffic_count.classes import CLASSES_LIST


def point_in_polygon(x, y, verts):
    """
    - PNPoly算法
    - 参考网站:https://www.jianshu.com/p/3187832cb6cc
    功能: 判断一个点是否在多边形内部
    参数:
        x,y: 需要检测的点的坐标
        verts: 多边形各点的坐标, [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    try:
        x, y = float(x), float(y)
    except:
        return False
    #print('verts:',verts)
    vertx = [xyvert[0] for xyvert in verts]
    #print('vertx:',vertx)
    verty = [xyvert[1] for xyvert in verts]

    # N个点中，横坐标和纵坐标的最大值和最小值，判断目标坐标点是否在这个四边形之内
    if not verts or not min(vertx) <= x <= max(vertx) or not min(verty) <= y <= max(verty):
        return False

    # 上一步通过后，核心算法部分
    nvert = len(verts)
    is_in = False
    for i in range(nvert):
        j = nvert - 1 if i == 0 else i - 1
        if ((verty[i] > y) != (verty[j] > y)) and (
                x < (vertx[j] - vertx[i]) * (y - verty[i]) / (verty[j] - verty[i]) + vertx[i]):
            is_in = not is_in

    return is_in

def roi_point_detect(roi, bboxes, class_list):
    count_list = [0]*len(class_list)
    count = 0
    for bbox in bboxes:
        x = int(bbox.xmin + (bbox.xmax - bbox.xmin) * 0.5)
        y = int(bbox.ymin + (bbox.ymax - bbox.ymin) * 0.5)
        if roi[x][y].any != 0:
            print(roi[x][y])
            count_list[class_list.index(bbox.Class)] += 1
            count += 1
    return count, count_list

def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('clicking: ', x, y)

def callback(image, boxes):
    global frame_count, up_count, down_count, blue_list, yellow_list, classes_list, point_roi, line, lines, polygons, multi_roi, multi_line, roi_num,json_path, Publisher_json
    # ct = CameraTools(905.8602,516.4283,1626.513816,1624.574619,1200 ,11)
    # x,y = ct.pixel2world(500,600)
    # print("相机投影坐标：",x,y)
    # print('像素坐标:',ct.world2pixel([x,y,0]))
    bridge = CvBridge()
    # cv_image = bridge.compressed_imgmsg_to_cv2(image, "bgr8")
    cv_image = bridge.imgmsg_to_cv2(image,"bgr8")
    size = (cv_image.shape[0], cv_image.shape[1])


    # 在图像上画出每个bounding_boxes的中兴点
    point_radius = 3
    for i in range(0, len(boxes.bounding_boxes)):
        conf = boxes.bounding_boxes[i].probability
        x1 = boxes.bounding_boxes[i].xmin
        y1 = boxes.bounding_boxes[i].ymin
        x2 = boxes.bounding_boxes[i].xmax
        y2 = boxes.bounding_boxes[i].ymax
        cls = boxes.bounding_boxes[i].Class

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
        cv_image = cv2.fillPoly(cv_image, [ndarray_pts], color=(0, 0, 255))     


    # 整张图像的各个类别数量
    classes_num = utils.image_count(boxes.bounding_boxes, classes_list)


    # 每个roi区域的各个类别数量
    for index in range(0, len(multi_roi)):
        roi_num[index], roi_color_image = utils.roi_count(multi_roi[index], boxes.bounding_boxes, classes_list,  [0, 0, 255], size)
        cv_image = cv2.add(cv_image, roi_color_image)
        classified_statistic =[]
        sum = 0
        for i in range(0, len(classes_list)):
            sum += roi_num[index][i]
            classified_count = {
                "class":classes_list[i],
                "num":roi_num[index][i]
            }
            classified_statistic.append(classified_count)
        area_json = {
            'area_id': polygons[index]['road_number'], 
            'car_num': sum,
            'count_list': classified_statistic
        }
        ROI_statistics.append(area_json)
    # print('ROI_statistics',ROI_statistics)

    # 各个类别穿过每条线的统计情况
    for index in range(0, len(multi_line)):
        polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(multi_line[index], 0, 10, size)
        up_count[index], down_count[index] = utils.traffic_count(cv_image, boxes.bounding_boxes, classes_list,  polygon_mask_blue_and_yellow, 
                                                                blue_list[index], yellow_list[index],  up_count[index], down_count[index])
        cv_image = cv2.add(cv_image, polygon_color_image)
        classified_statistic =[]
        sum = 0
        for i in range(0, len(classes_list)):
            sum += up_count[index][i]
            classified_count = {
                "class":classes_list[i],
                "up_count":up_count[index][i],
                "down_count":down_count[index][i]
            }
            classified_statistic.append(classified_count)
        line_json = {
            'channel_id': lines[index]['name'], 
            'total_car': sum,
            'classified_statistic':classified_statistic
        }
        Line_statistics.append(line_json)
    # print('Line_statistics',Line_statistics)

    # Publisher_json = {
    #     "period_statistical_info":Line_statistics,
    #     "area_statistical_info":ROI_statistics
    # }
    # print('Publisher_json',Publisher_json)

    # 实时更新ROI区域内的信息，并写入json文件
    Publisher_json.update({"area_statistical_info":ROI_statistics})
    json_str = json.dumps(Publisher_json, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)


    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(1224 * 0.01), int(1024 * 0.05))
    text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count) + \
        '  , Class Num: ' + str(classes_num) + '  , Roi Num: ' + str(roi_num)

    # print(cv_image.shape)
    # print (polygon_color_image.shape)
    # cv_image = cv2.add(cv_image, polygon_color_image)
    # cv_image = cv2.add(cv_image, roi_color_image)
    # cv_image = cv2.add(cv_image, polygon_color_01)
    cv_image = cv2.putText(img=cv_image, text=text_draw,
                           org=draw_text_postion,
                           fontFace=font_draw_number,
                           fontScale=0.3, color=(0, 0, 255), thickness=2)

    # cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5,
    #                      interpolation=cv2.INTER_AREA)
    cv2.setMouseCallback('cv_image', get_point, cv_image    )
    # cv2.imshow("cv_image", cv_image)
    cv2.waitKey(1)

    print("sub frame: ", frame_count)
    frame_count += 1


def read_json():
    current_dir = os.path.dirname(__file__)
    f = open(current_dir + "/../json/polygon.json", encoding="UTF-8")
    file = json.load(f)
    lines = file['reference_point']['collision_lines']
    polygons = file['reference_point']['roads']
    return lines, polygons



def dump_json():
    global Publisher_json, json_path, Line_statistics, ROI_statistics,up_count,down_count
    Publisher_json = {
        "period_statistical_info":Line_statistics,
        "area_statistical_info":ROI_statistics
    }
    # up_count, down_count 数据清零
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    # 把数据dump到json文件里
    json_str = json.dumps(Publisher_json, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    print("Dump data into json successed.")

class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

if __name__ == '__main__':
    # rospy.init_node('showImage',anonymous = True)/
    rospy.init_node('traffic_count', anonymous=True)
    img_pub = rospy.Publisher('/traffic_count_publisher', Image, queue_size=10)
    rate = rospy.Rate(25)

    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir + "/../json/yolo_statistics.json")

    classes_list = CLASSES_LIST
    lines, polygons = read_json()

    multi_line = [[0]]*len(lines)
    multi_roi = [[0]]*len(polygons)

    roi_num = [[0]]*len(polygons)
    
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    blue_list = np.zeros((len(lines), 2, len(classes_list)))
    yellow_list = np.zeros((len(lines), 2, len(classes_list)))

    Publisher_json = {}
    Line_statistics = []
    ROI_statistics = []
    
    frame_count = 0

    count = 0
    for line in lines:
        multi_line[count] = line['line_points']
        print(multi_line[count])
        count = count + 1

    count = 0
    for polygon in polygons:
        multi_roi[count] = polygon['points']
        print(multi_roi[count])
        count = count + 1

    # 每60秒更新一次周期统计信息，并把统计信息置零
    t = RepeatingTimer(60.0,dump_json)
    t.start()


    # image_sub0 = message_filters.Subscriber('/bitcq_camera/image_source0/compressed', CompressedImage)
    # boxes_sub1 = message_filters.Subscriber('/bounding_boxes', BoundingBoxes)
    image_sub0= message_filters.Subscriber('/output_image_nms', Image)
    boxes_sub1 = message_filters.Subscriber('/bounding_boxes_nms', BoundingBoxes)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub0, boxes_sub1], queue_size=5, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()
