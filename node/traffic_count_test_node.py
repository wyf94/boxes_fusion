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
import threading
import time
import signal
import math
import random

from traffic_count.msg import BoundingBox
from traffic_count.msg import BoundingBoxes
from sort_track.msg import Tracks as TracksMsg
from sort_track.msg import Targets as TargetsMsg
from traffic_count.msg import BboxCoordinate as BboxCoordinateMsg
from traffic_count.msg import BboxesCoordinates as BboxesCoordinatesMsg

import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


from traffic_count.cameratool import CameraTools
# import traffic_count.utils as utils
import traffic_count.traffic_utils as utils
import sort.sort as sort
from traffic_count.sort_track_wyf import Sort_Track
# from traffic_count.yolo_classes import CLASSES_LIST
from traffic_count.classes import CLASSES_LIST, TRACK_CLASSES_LIST, TRACK_CLASSES_LEN

def callback(image, boxes):
    # start  = time.time()
    global frame_count, up_count, down_count, blue_list, yellow_list, classes_list, lines, polygons, multi_roi, multi_line, roi_num,json_path, Publisher_json, Line_statistics
    global is_show_image, publish_image, tracker, car_speed,car_head_passtime, line_occupy_flag, line_occupy_time, padding, track_time

    t1 = time.time()
    # print("sub frame: ", frame_count)
    # print("diff time: ", image.header.stamp.to_sec() - boxes.header.stamp.to_sec())
    frame_count += 1

    track1 = time.time()
    # 跟踪
    BboxesCoordinates_msg = BboxesCoordinatesMsg()
    BboxesCoordinates_msg = tracker.sort_track(boxes)
    # print("BboxesCoordinates_msg:", BboxesCoordinates_msg)
    track_time = BboxesCoordinates_msg.header.stamp.to_sec()
    track2 = time.time()

    # 读取image msg并转换为opencv格式
    bridge = CvBridge()
    if is_CompressedImage:
        cv_image = bridge.compressed_imgmsg_to_cv2(image, "bgr8")
    else:
        cv_image = bridge.imgmsg_to_cv2(image,"bgr8")
    size = (cv_image.shape[0], cv_image.shape[1])
    # print("size: ", size)

    # 整张图像的各个类别数量
    classes_num = utils.image_count(BboxesCoordinates_msg.bbox_coordinate, classes_list)

    r1 = time.time()
    # 每个roi区域的各个类别数量
    roi_color =  [0, 0, 255]
    ROI_statistics = []
    ROI_queue = []
    for index in range(0, len(multi_roi)):
        # 设置roi区域的1，2点为停止线，并选择其中点为停止点
        stop_x = int(multi_roi[index][0][0]+multi_roi[index][1][0])
        stop_y = int(multi_roi[index][0][1] + multi_roi[index][1][1])
        ground_stop_x, ground_stop_y = tracker.pixel2world(stop_x, stop_y)
        stop_point = (ground_stop_x, ground_stop_y)
        # roi_num[index], roi_color_image, queue_info = utils.roi_count_queue(multi_roi[index], BboxesCoordinates_msg.bbox_coordinate, 
        #                                                                                                                         track_classes_list,  stop_point, roi_color, size, is_show_image)                                                    
        # if is_show_image:
        #     cv_image = cv2.add(cv_image, roi_color_image)

        # area_json = {
        #     'area_id': polygons[index]['road_number'], 
        #     'car_num': 0,
        #     'count_list': 0,
        #     "ave_car_speed": 0,
        #     "car_distribute": 0,
        #     "head_car_pos": 0,
        #     "head_car_speed": 0,
        #     "tail_car_pos": 0,
        #     "tail_car_speed": 0,
        #     "car_count": 0
        # }
        # area_json.update(queue_info)
        # ROI_statistics.append(area_json)
        roi_num[index], roi_color_image, area_info, queue_info = utils.roi_count_queue(multi_roi[index], BboxesCoordinates_msg.bbox_coordinate, 
                                                                                                                                track_classes_list,  stop_point, roi_color, size, is_show_image)                                                    
        if is_show_image:
            cv_image = cv2.add(cv_image, roi_color_image)
        area_json = {
            'area_id': polygons[index]['road_number'], 
            'car_num': 0,
            # 'count_list': 0,
            "ave_car_speed": 0,
            "car_distribute": 0,
            "head_car_pos": 0,
            "head_car_speed": 0,
            "tail_car_pos": 0,
            "tail_car_speed": 0,
        }
        queue_up_info = {
            'lane_id': polygons[index]['road_number'], 
            "queue_len": 0,
            "head_car_pos": 0,
            "tail_car_pos": 0,
            "car_count": random.randint(2,10)
            }
        area_json.update(area_info)
        # queue_up_info.update(queue_info)
        ROI_queue.append(queue_up_info)
        ROI_statistics.append(area_json)
    print('ROI_statistics:',ROI_statistics)
    r2 = time.time()

    # # lock
    # lock.acquire()
    # # 实时更新ROI区域内的信息，并写入json文件
    # Publisher_json.update({"area_statistical_info":ROI_statistics})
    # json_str = json.dumps(Publisher_json, indent=4)
    # with open(json_path, 'w') as json_file:
    #     json_file.write(json_str)
    # # unlock
    # lock.release()

    # l1 = time.time()
    # # 各个类别穿过每条线的统计情况
    # Line_statistics = []
    # for index in range(0, len(multi_line)):
    #     # 判断line中点加上padding之后是否超出图片范围
    #     for i in range(0, 2):
    #         if multi_line[index][i][0]+padding[0] >= size[1] or multi_line[index][i][0]+padding[0] <= 0:
    #             print(" The point of lines out off range or padding out off range")
    #         if multi_line[index][i][1]+padding[1] >= size[0] or multi_line[index][i][1]+padding[1] <= 0:
    #             print(" The point of lines out off range or padding out off range")

    #     polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(multi_line[index], padding, size, is_show_image)
    #     up_count[index], down_count[index] = utils.traffic_count_track(BboxesCoordinates_msg, track_classes_list,  polygon_mask_blue_and_yellow, 
    #                                                             blue_list[index], yellow_list[index],  up_count[index], down_count[index], car_head_passtime[index], car_speed[index])
    #     # 
    #     line_occupy_flag[index] = utils.occupancy(BboxesCoordinates_msg,  multi_line[index], padding, line_occupy_flag[index], line_occupy_time[index])

    #     if is_show_image:
    #         cv_image = cv2.add(cv_image, polygon_color_image)

    # l2 = time.time()

    if is_show_image:
        # 在图像上画出每个bounding_boxes
        point_radius = 3
        # for item_bbox in list_track:
        for i in range(0, len(BboxesCoordinates_msg.bbox_coordinate)):
            track_id = BboxesCoordinates_msg.bbox_coordinate[i].id
            x1 = BboxesCoordinates_msg.bbox_coordinate[i].xmin
            y1 = BboxesCoordinates_msg.bbox_coordinate[i].ymin
            x2 = BboxesCoordinates_msg.bbox_coordinate[i].xmax
            y2 = BboxesCoordinates_msg.bbox_coordinate[i].ymax
            cls = BboxesCoordinates_msg.bbox_coordinate[i].Class
            vx = BboxesCoordinates_msg.bbox_coordinate[i].vx
            vy = BboxesCoordinates_msg.bbox_coordinate[i].vy

            v =round(math.sqrt(vx*vx + vy*vy), 2) 

            # 撞线的点(中心点)
            x = int(x1 + ((x2 - x1) * 0.5))
            y = int(y2)

            #画出中心list_bboxs的中心点
            list_pts = []
            list_pts.append([x-point_radius, y-point_radius])
            list_pts.append([x-point_radius, y+point_radius])
            list_pts.append([x+point_radius, y+point_radius])
            list_pts.append([x+point_radius, y-point_radius])
            ndarray_pts = np.array(list_pts, np.int32)
            cv_image = cv2.fillPoly(cv_image, [ndarray_pts], color=(0, 0, 255))
            # # 绘制 检测框
            cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
            # 绘制 跟踪ID
            cv2.putText(cv_image , str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)
            # 绘制 目标类别
            cv2.putText(cv_image , str(cls) + ", v: " + str(v), (int(x1), int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0  , 0, 255), lineType=cv2.LINE_AA)     

        # show data in image
        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_postion = (10, 50)
        text_draw = 'Class Num: ' + str(classes_num)
        cv_image = cv2.putText(img=cv_image, text=text_draw,
                            org=draw_text_postion,
                            fontFace=font_draw_number,
                            fontScale=1, color=(0, 0, 255), thickness=2)

        cv2.namedWindow("YOLO+SORT", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO+SORT", cv_image)
        cv2.waitKey(1)

    # if publish_image:
    #     msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    #     img_pub.publish(msg)
    #     rate.sleep()
    t2 = time.time()
    print("track time: ", (track2 - track1)*1000)
    # print("roi time: ", (r2 - r1)*1000)
    # print("line time: ", (l2 - l1)*1000)
    print("run time: ", (t2 - t1)*1000)
    


def read_json():
    current_dir = os.path.dirname(__file__)
    f = open(current_dir + "/../json/polygon_chengdu_test.json", encoding="UTF-8")
    file = json.load(f)
    lines = file['reference_point']['collision_lines']
    polygons = file['reference_point']['roads']
    return lines, polygons

def dump_json():
    global Publisher_json, json_path, Line_statistics, up_count, down_count, dump_num, lock, track_time, \
                  car_speed, car_head_passtime, line_occupy_flag, line_occupy_time, track_classes_list
    # 周期性统计各个类别穿过每条线的情况
    Line_statistics = []
    for index in range(0, len(multi_line)):
        # 分类统计
        classified_statistic =[]
        sum_count = 0
        sum_car_len = 0
        for i in range(0, len(track_classes_list)):
            count = up_count[index][i] + down_count[index][i]
            sum_count += count
            sum_car_len += (count * track_classes_len[i])
            classified_count = {
                "class":track_classes_list[i],
                "num":up_count[index][i] + down_count[index][i]
            }
            classified_statistic.append(classified_count)

        # 平均车长
        if sum_count != 0:
            ave_car_len = round(sum_car_len / sum_count, 2)
        else:
            ave_car_len = 0

        # 时间占有率
        sum_occupy_time = 0
        if len(line_occupy_time[index]) > 1:
            if len(line_occupy_time[index]) % 2 == 0:
                for i in range(0, len(line_occupy_time[index]), 2):
                    sum_occupy_time += (line_occupy_time[index][i+1] - line_occupy_time[index][i])
            else:
                time_now = track_time
                for i in range(0, len(line_occupy_time[index]) - 1, 2):
                    sum_occupy_time += (line_occupy_time[index][i+1] - line_occupy_time[index][i])
                sum_occupy_time += (time_now - line_occupy_time[index][-1] )
        line_occupy = sum_occupy_time / period

        # 平均车头时距
        sum_car_head_passtime = 0
        for i in range(0, len(car_head_passtime[index]) - 1):
            sum_car_head_passtime += (car_head_passtime[index][i+1] - car_head_passtime[index][i])
        if sum_count != 0:
            ave_car_head_dis = round(sum_car_head_passtime / sum_count, 2)
        else:
            ave_car_head_dis =0
        
        # 平均车速
        sum_car_speed  = 0
        for i in range(0, len(car_speed[index])):
            sum_car_speed += car_speed[index][i]
        if sum_count != 0:
            ave_car_speed = round(sum_car_speed / sum_count, 2)
        else:
            ave_car_speed = 0

        line_json = {
            'channel_id': lines[index]['name'], 
            'total_car': sum_count,
            'classified_statistic':classified_statistic,
            'ave_car_head_dis': ave_car_head_dis,
            'ave_car_len': ave_car_len,
            'ave_car_speed': ave_car_speed
        }
        Line_statistics.append(line_json)
    print("-------------------------------------------")
    print("Line_statistics: ", Line_statistics)
    print("-------------------------------------------")

    # lock
    lock.acquire()
    Publisher_json.update({"period_statistical_info": Line_statistics})
    # 把数据dump到json文件里
    json_str = json.dumps(Publisher_json, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    print("Dump data into json successed: ", dump_num)
    dump_num += 1
    # unlock
    lock.release()

    # lines 周期统计参数变量清零
    Line_statistics = []
    up_count = np.zeros((len(lines),  len(track_classes_list)))
    down_count = np.zeros((len(lines),  len(track_classes_list)))
    car_head_passtime = [[] for i in range(len(lines))]
    car_speed = [[] for i in range(len(lines))]
    line_occupy_time = [[] for i in range(len(lines))]

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

    is_show_image = rospy.get_param('/traffic_count/show_image')
    publish_image = rospy.get_param('/traffic_count/publish_image')
    bounding_boxes_topic = rospy.get_param('/traffic_count/bounding_boxes_topic')
    detect_image_topic = rospy.get_param('/traffic_count/detect_image_topic')

    max_age = rospy.get_param('/traffic_count/max_age')
    min_hits = rospy.get_param('/traffic_count/min_hits')
    camera_config_path = rospy.get_param('/traffic_count/camera_config_path')

    period = rospy.get_param('/traffic_count/period')
    is_CompressedImage = rospy.get_param('/traffic_count/is_CompressedImage')

    classes_list = CLASSES_LIST
    track_classes_list = TRACK_CLASSES_LIST
    track_classes_len = TRACK_CLASSES_LEN

    # 读取josn文件里的lines, polygons
    lines, polygons = read_json()
    multi_line = [[0] for i in range(len(lines))]
    multi_roi = [[0] for i in range(len(polygons))]
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

    frame_count = 0
    dump_num = 0
    track_time = 0

    # json 变量
    Publisher_json = {}

    # area 统计参数变量
    roi_num = [[0] for i in range(len(polygons))]

    # lines 周期统计参数变量    
    padding = (0, 20) # line填充为矩形的(weight, heigeht)
    Line_statistics = []
    up_count = np.zeros((len(lines),  len(track_classes_list)))
    down_count = np.zeros((len(lines),  len(track_classes_list)))
    blue_list = [[] for i in range(len(lines))]
    yellow_list = [[] for i in range(len(lines))]
    car_head_passtime = [[] for i in range(len(lines))]
    car_speed = [[] for i in range(len(lines))]
    line_occupy_flag = [0]*len(lines)
    line_occupy_time = [[] for i in range(len(lines))]

    
    # 每60秒更新一次周期统计信息，并把统计信息置零
    lock = threading.Lock()
    # t = RepeatingTimer(float(period), dump_json)
    # t.start()


    tracker = Sort_Track(max_age, min_hits, camera_config_path)

    if is_CompressedImage:
        image_sub0= message_filters.Subscriber(detect_image_topic, CompressedImage)
    else:
        image_sub0= message_filters.Subscriber(detect_image_topic, Image)
    boxes_sub1 = message_filters.Subscriber(bounding_boxes_topic, BoundingBoxes)
    # image_sub0= message_filters.Subscriber(detect_image_topic, CompressedImage)
    # boxes_sub1 = message_filters.Subscriber(bounding_boxes_topic, TracksMsg)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub0, boxes_sub1], queue_size=15, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()





