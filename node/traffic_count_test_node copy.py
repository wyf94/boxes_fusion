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
from traffic_count.classes import CLASSES_LIST

def callback(image, boxes):
    # start  = time.time()
    global frame_count, up_count, down_count, blue_list, yellow_list, classes_list, point_roi, lines, polygons, multi_roi, multi_line, roi_num,json_path, Publisher_json, Line_statistics
    global show_image, publish_image,list_boxes, tracker,car_speed,car_head_passtime, line_occupy_flag, line_occupy_time
    # ct = CameraTools(905.8602,516.4283,1626.513816,1624.574619,1200 ,11)
    # x,y = ct.pixel2world(500,600)
    # print("相机投影坐标：",x,y)
    # print('像素坐标:',ct.world2pixel([x,y,0]))
    print("sub frame: ", frame_count)
    print("diff time: ", image.header.stamp.to_sec()*1000 - boxes.header.stamp.to_sec()*1000)
    frame_count += 1

    tracks_msg = TracksMsg()
    targets_msg = TargetsMsg()
    BboxesCoordinates_msg = BboxesCoordinatesMsg()
    # tracks_msg, targets_msg, BboxesCoordinates_msg = tracker.sort_track(boxes)
    BboxesCoordinates_msg = tracker.sort_track(boxes)

    bridge = CvBridge()
    # cv_image = bridge.compressed_imgmsg_to_cv2(image, "bgr8")
    cv_image = bridge.imgmsg_to_cv2(image,"bgr8")
    size = (cv_image.shape[0], cv_image.shape[1])

    # # 整张图像的各个类别数量
    # classes_num = utils.image_count(BboxesCoordinates_msg.bbox_coordinate, classes_list)
    # print(classes_num)
    classes_list = ["car", "bus", "truck", "motor", "bike"]
    classes_length = [3, 8, 8, 1, 1]

    # # 每个roi区域的各个类别数量
    # ROI_statistics = []
    # for index in range(0, len(multi_roi)):
    #     # roi_num[index], roi_color_image = utils.roi_count(multi_roi[index], tracks_msg.bounding_boxes, classes_list,  [0, 0, 255], size)
    #     # 设置roi区域的1，2点为停止线，并选择其中点为停止点
    #     stop_x = int(multi_roi[index][0][0]+multi_roi[index][1][0])
    #     stop_y = int(multi_roi[index][0][1] + multi_roi[index][1][1])
    #     ground_stop_x, ground_stop_y = tracker.pixel2world(stop_x, stop_y)

    #     roi_num[index], roi_color_image, queue_info = utils.roi_count_queue(multi_roi[index], BboxesCoordinates_msg.bbox_coordinate, 
    #                                                                                                                             classes_list,  (ground_stop_x, ground_stop_y), [0, 0, 255], size)
    #     cv_image = cv2.add(cv_image, roi_color_image)
    #     # classified_statistic =[]
    #     # sum = 0
    #     # for i in range(0, len(classes_list)):
    #     #     sum += roi_num[index][i]
    #     #     classified_count = {
    #     #         "class":classes_list[i],
    #     #         "num":roi_num[index][i]
    #     #     }
    #     #     classified_statistic.append(classified_count)
    #     area_json = {
    #         'area_id': polygons[index]['road_number'], 
    #         'car_num': 0,
    #         'count_list': 0,
    #         "ave_car_speed": 0,
    #         "car_distribute": 0,
    #         "head_car_pos": 0,
    #         "head_car_speed": 0,
    #         "tail_car_pos": 0,
    #         "tail_car_speed": 0,
    #         "car_count": 0
    #     }
    #     area_json.update(queue_info)
    #     # print("index: ", index)
    #     # if area_occupy_flag[index] == 0 and area_json['car_num'] > 0:
    #     #     area_occupy_time[index].append(BboxesCoordinates_msg.header.stamp.to_sec())
    #     # # print("car_num: ", area_json['car_num'])
    #     #     print("area_occupy_time[index]: ", area_occupy_time[index])
    #     # if area_json['car_num'] == 0:
    #     #     area_occupy_flag[index] = 0
    #     # else:
    #     #     area_occupy_flag[index] = 1

    #     ROI_statistics.append(area_json)
    # # print('ROI_statistics:',ROI_statistics)
    
    img_bboxes_mask, yellow_image= utils.bboxes_mask(BboxesCoordinates_msg, size)
    cv_image = cv2.add(cv_image, yellow_image)
    # print("img_bboxes_mask: ", img_bboxes_mask)
    # 各个类别穿过每条线的统计情况
    Line_statistics = []
    for index in range(0, len(multi_line)):
        polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(multi_line[index], 0, 10, size)
        # up_count[index], down_count[index] = utils.traffic_count(cv_image, boxes.bounding_boxes, classes_list,  polygon_mask_blue_and_yellow, 
        #                                                         blue_list[index], yellow_list[index],  up_count[index], down_count[index])
        up_count[index], down_count[index] = utils.traffic_count_track(cv_image, BboxesCoordinates_msg, classes_list,  polygon_mask_blue_and_yellow, 
                                                                blue_list[index], yellow_list[index],  up_count[index], down_count[index], 
                                                                car_head_passtime, car_speed)

        line_occupy_flag[index] = utils.occupancy(BboxesCoordinates_msg, img_bboxes_mask, multi_line[index], line_occupy_flag[index], line_occupy_time[index])
        
        cv_image = cv2.add(cv_image, polygon_color_image)
        classified_statistic =[]
        sum_count = 0
        sum_car_len = 0
        for i in range(0, len(classes_list)):
            count = up_count[index][i] + down_count[index][i]
            sum_count += count
            sum_car_len += (count * classes_length[i])
            classified_count = {
                "class":classes_list[i],
                "num":up_count[index][i] + down_count[index][i]
            }
            classified_statistic.append(classified_count)

        # 时间占有率
        T =  10
        sum_occupy_time = 0
        if len(line_occupy_time[index]) > 1:
            if len(line_occupy_time[index]) % 2 == 0:
                for i in range(0, len(line_occupy_time[index]), 2):
                    sum_occupy_time += (line_occupy_time[index][i+1] - line_occupy_time[index][i])
            else:
                time_now = BboxesCoordinates_msg.header.stamp.to_sec()
                for i in range(0, len(line_occupy_time[index]) - 1, 2):
                    sum_occupy_time += (line_occupy_time[index][i+1] - line_occupy_time[index][i])
                sum_occupy_time += (time_now - line_occupy_time[index][-1] )
        line_occupy = sum_occupy_time / T
        print("-------------------------------------------")
        print("line_occupy_flag: ", line_occupy_flag)
        print("line_occupy_time: ", line_occupy_time)
        print("line_occupy: ", line_occupy)
        print("-------------------------------------------")

        # 平均车长
        if sum_count != 0:
            ave_car_len = round(sum_car_len / sum_count, 2)
        else:
            ave_car_len = 0

        # 平均车头时距
        sum_car_head_passtime = 0
        for i in range(0, len(car_head_passtime) - 1):
            sum_car_head_passtime += (car_head_passtime[i+1] - car_head_passtime[i])
        if sum_count != 0:
            ave_car_head_dis = round(sum_car_head_passtime / sum_count, 2)
        else:
            ave_car_head_dis =0
        
        # 平均车速
        sum_car_speed  = 0
        for i in range(0, len(car_speed)):
            sum_car_speed += car_speed[i]
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
    print('Line_statistics:',Line_statistics)

    # Publisher_json = {
    #     "period_statistical_info":Line_statistics,
    #     "area_statistical_info":ROI_statistics
    # }
    # print('Publisher_json',Publisher_json)

    # # lock
    # lock.acquire()
    # # 实时更新ROI区域内的信息，并写入json文件
    # Publisher_json.update({"area_statistical_info":ROI_statistics})
    # Publisher_json.update({"period_statistical_info":Line_statistics})
    # json_str = json.dumps(Publisher_json, indent=4)
    # with open(json_path, 'w') as json_file:
    #     json_file.write(json_str)
    # # unlock
    # lock.release()

    if show_image:
        # 在图像上画出每个bounding_boxes的中兴点
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
            # x = int(int(x1) + ((int(x2) - int(x1)) * 0.5))
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

        # # show data in image
        # font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        # draw_text_postion = (10, 50)
        # text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count) +  '  , Class Num: ' + str(classes_num)
        # cv_image = cv2.putText(img=cv_image, text=text_draw,
        #                     org=draw_text_postion,
        #                     fontFace=font_draw_number,
        #                     fontScale=1, color=(0, 0, 255), thickness=2)

        # cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5,
        #                     interpolation=cv2.INTER_AREA)
        cv2.namedWindow("YOLO+SORT", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO+SORT", cv_image)
        cv2.waitKey(1)

    # if publish_image:
    #     msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    #     img_pub.publish(msg)
    #     rate.sleep()

    # print("sub frame: ", frame_count)
    # frame_count += 1


def read_json():
    current_dir = os.path.dirname(__file__)
    f = open(current_dir + "/../json/polygon_chengdu_test2.json", encoding="UTF-8")
    file = json.load(f)
    lines = file['reference_point']['collision_lines']
    polygons = file['reference_point']['roads']
    return lines, polygons



def dump_json():
    global Publisher_json, json_path, Line_statistics, up_count, down_count, dump_num, lock
    # lock
    lock.acquire()
    Publisher_json.update({"period_statistical_info":Line_statistics})
    # up_count, down_count 数据清零
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    car_head_passtime = []
    # 把数据dump到json文件里
    json_str = json.dumps(Publisher_json, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    print("Dump data into json successed: ", dump_num)
    dump_num += 1
    # unlock
    lock.release()

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

    lock = threading.Lock()

    classes_list = CLASSES_LIST
    lines, polygons = read_json()

    multi_line = [[0]]*len(lines)
    multi_roi = [[0]]*len(polygons)

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


    # area_occupy_flag = [0]*len(multi_roi)
    # print("area_occupy_flag", area_occupy_flag)
    # area_occupy_time = [[]]*len(multi_roi)
    # print("area_occupy_flag", area_occupy_time)

    roi_num = [[0]]*len(polygons)
    
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    # blue_list = np.zeros((len(lines), 2, len(classes_list)))
    # yellow_list = np.zeros((len(lines), 2, len(classes_list)))
    blue_list = [[]]*len(lines)
    yellow_list = [[]]*len(lines)

    Publisher_json = {}
    Line_statistics = []
    
    frame_count = 0
    dump_num = 0

    list_boxes = []
    car_head_passtime = []
    car_speed = []

    line_occupy_flag = [0]*len(lines)
    line_occupy_time = [[]]*len(lines)

    
    # 每60秒更新一次周期统计信息，并把统计信息置零
    # t = RepeatingTimer(10.0, dump_json)
    # t.start()

    show_image = rospy.get_param('/traffic_count/show_image')
    publish_image = rospy.get_param('/traffic_count/publish_image')
    bounding_boxes_topic = rospy.get_param('/traffic_count/bounding_boxes_topic')
    detect_image_topic = rospy.get_param('/traffic_count/detect_image_topic')

    max_age = rospy.get_param('/traffic_count/max_age')
    min_hits = rospy.get_param('/traffic_count/min_hits')
    camera_config_path = rospy.get_param('/traffic_count/camera_config_path')

    tracker = Sort_Track(max_age, min_hits, camera_config_path)

    image_sub0= message_filters.Subscriber(detect_image_topic, Image)
    boxes_sub1 = message_filters.Subscriber(bounding_boxes_topic, BoundingBoxes)
    # image_sub0= message_filters.Subscriber(detect_image_topic, CompressedImage)
    # boxes_sub1 = message_filters.Subscriber(bounding_boxes_topic, TracksMsg)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub0, boxes_sub1], queue_size=15, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()





