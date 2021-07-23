#!/usr/bin/python3
#!coding=utf-8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import requests
import json
import os

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
    global frame_count, up_count, down_count, blue_list, yellow_list, classes_list, point_roi, line, lines, polygons, multi_roi, multi_line, roi_num

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
    ROI_statistics = []
    for index in range(0, len(multi_roi)):
        roi_num[index], roi_color_image = utils.roi_count(multi_roi[index], boxes.bounding_boxes, classes_list,  [0, 0, 255], size)
        cv_image = cv2.add(cv_image, roi_color_image)
        # print("roi_num ", index, " :", roi_num[index] )
        car_num =  roi_num[index][1] + roi_num[index][2] + roi_num[index][6]
        area_json = {
            'name': polygons[index]['road_number'], 
            'car_num': car_num,
            'count_list': roi_num[index]
        }
        ROI_statistics.append(area_json)
    # print(ROI_statistics)
        # json_str = json.dumps(Line_statistics, indent=4)
    # with open('test_data.json', 'w') as json_file:
    #     json_file.write(json_str)


    # payload = []

    # for i in range(0, len(multi_roi)):
    #     roi_count, roi_count_list = roi_point_detect(multi_roi[i], boxes.bounding_boxes, classes_list)
    #     area_json = {
    #         'name': polygons[i]['road_number'],
    #         'count_num': roi_count,
    #         'count_list': roi_count_list
    #     }
    #     payload.append(area_json)
    
    # print(payload)

    # 各个类别穿过每条线的统计情况
    Line_statistics = []
    for index in range(0, len(multi_line)):
        polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(multi_line[index], 0, 10, size)
        up_count[index], down_count[index] = utils.traffic_count(cv_image, boxes.bounding_boxes, classes_list,  polygon_mask_blue_and_yellow, 
                                                                                                    blue_list[index], yellow_list[index],  up_count[index], down_count[index])
        cv_image = cv2.add(cv_image, polygon_color_image)
        classified_statistic = {}
        for i in range(0, len(classes_list)):

            classified_statistic[classes_list[i]] = up_count[index][i]
        
        # print('classified_statistic:', classified_statistic)
        line_json = {
            'channel_id': lines[index]['name'], 
            'up_count': up_count[index].tolist(),
            'down_count':down_count[index].tolist()
        }
        Line_statistics.append(line_json)
    # print(Line_statistics)

    # current_dir = os.path.dirname(__file__)
    # json_str = json.dumps(Line_statistics, indent=4)
    # with open('test_data.json', 'w') as json_file:
    #     json_file.write(json_str)


    # tc = TrafficCount(cv_image, boxes.bounding_boxes, classes_list,  line)

    # polygon_mask_blue_and_yellow, polygon_color_image = tc.line2polygon(0, 20)
    # # classes_num = tc.image_count()
    # roi_num, roi_color_image = tc.roi_count(multi_roi[0], [0, 0, 255])
    # up_count, down_count = tc.traffic_count(
    #     polygon_mask_blue_and_yellow, blue_list, yellow_list,  up_count, down_count)

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(1224 * 0.01), int(1024 * 0.05))
    text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count) + \
        '  , Class Num: ' + str(classes_num) + '  , Roi Num: ' + str(roi_num)

    # print(cv_image.shape)
    # print (polygon_color_image.shape)
    cv_image = cv2.add(cv_image, polygon_color_image)
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


if __name__ == '__main__':
    # rospy.init_node('showImage',anonymous = True)/
    rospy.init_node('traffic_count', anonymous=True)
    img_pub = rospy.Publisher('/traffic_count_publisher', Image, queue_size=10)
    rate = rospy.Rate(25)

    classes_list = CLASSES_LIST
    lines, polygons = read_json()

    multi_line = [[0]]*len(lines)
    multi_roi = [[0]]*len(polygons)

    roi_num = [[0]]*len(polygons)
    
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    blue_list = np.zeros((len(lines), 2, len(classes_list)))
    yellow_list = np.zeros((len(lines), 2, len(classes_list)))
    
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

    current_dir = os.path.dirname(__file__)

    Line_statistics = []
    ROI_statistics = []

    line_json_path = os.path.join(current_dir + "../json/line_statistics.json")
    roi_json_path = os.path.join(current_dir + "../json/roi_statistics.json")

    # for polygon in polygons:
    #     # image_mask = np.zeros(image_size, dtype=np.uint8)
    #     # ndarray_pts = np.array(polygon['points'], np.int32)
    #     # polygon_color_value = cv2.fillPoly(image_mask, [ndarray_pts], 1)
    #     # polygon_color_value = polygon_color_value[:, :, np.newaxis]
    #     # color_plate = (0, 0, 255)
    #     # multi_roi[count] = np.array(polygon_color_value * color_plate, np.uint8)
    #     # # multi_roi[count] = image_mask
    #     multi_roi[count] = polygon['points']
    #     print(multi_roi[count])

    #     # image_mask = cv2.resize(multi_roi[count], None, fx=0.3, fy=0.3)
    #     # cv2.imshow(str(count), image_mask)

    #     count = count + 1

    # cv2.waitKey(0)

    # point_roi = [[321*2, 356*2], [369*2, 357*2],
    #             [364*2, 287*2], [337*2, 292*2]]
    # point_roi = multi_roi[0]

    
    # print(CLASSES_LIST)
    # # line = [[205*2, 385*2], [425*2, 385*2]]
    # line = [[425*2, 325*2], [570*2, 325*2]]

    # category_number = len(classes_list)

    # up_count = [0]*category_number
    # down_count = [0]*category_number
    # blue_list = np.zeros((2, category_number))
    # yellow_list = np.zeros((2, category_number))

    # image_sub0 = message_filters.Subscriber('/bitcq_camera/image_source0/compressed', CompressedImage)
    # boxes_sub1 = message_filters.Subscriber('/bounding_boxes', BoundingBoxes)
    image_sub0= message_filters.Subscriber('/output_image_nms', Image)
    boxes_sub1 = message_filters.Subscriber('/bounding_boxes_nms', BoundingBoxes)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub0, boxes_sub1], queue_size=5, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()
