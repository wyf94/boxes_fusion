#!/home/wyf/anaconda3/envs/test/bin/python
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

from traffic_count.msg import BoundingBox
from traffic_count.msg import BoundingBoxes

import message_filters
from sensor_msgs.msg import Image, CameraInfo, CompressedImage


from traffic_count.cameratool import CameraTools
import traffic_count.traffic_utils as utils
# from traffic_count.yolo_classes import CLASSES_LIST
from traffic_count.classes import CLASSES_LIST

def get_point(event, x, y, flags, param):
    # 鼠标单击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 输出坐标
        print('clicking: ', x, y)

def callback(image, boxes):
    # start  = time.time()
    global frame_count, up_count, down_count, blue_list, yellow_list, classes_list, point_roi, lines, polygons, multi_roi, multi_line, roi_num,json_path, Publisher_json, Line_statistics
    global show_image, publish_image
    # ct = CameraTools(905.8602,516.4283,1626.513816,1624.574619,1200 ,11)
    # x,y = ct.pixel2world(500,600)
    # print("相机投影坐标：",x,y)
    # print('像素坐标:',ct.world2pixel([x,y,0]))
    print("sub frame: ", frame_count)

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
    # print('ROI_statistics:',ROI_statistics)
    

    # 各个类别穿过每条线的统计情况
    Line_statistics = []
    for index in range(0, len(multi_line)):
        polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(multi_line[index], 0, 10, size)
        up_count[index], down_count[index] = utils.traffic_count(cv_image, boxes.bounding_boxes, classes_list,  polygon_mask_blue_and_yellow, 
                                                                blue_list[index], yellow_list[index],  up_count[index], down_count[index])
        cv_image = cv2.add(cv_image, polygon_color_image)
        classified_statistic =[]
        sum = 0
        for i in range(0, len(classes_list)):
            sum = sum + up_count[index][i] + down_count[index][i]
            classified_count = {
                "class":classes_list[i],
                "num":up_count[index][i] + down_count[index][i]
            }
            classified_statistic.append(classified_count)
        line_json = {
            'channel_id': lines[index]['name'], 
            'total_car': sum,
            'classified_statistic':classified_statistic
        }
        Line_statistics.append(line_json)
    # print('Line_statistics:',Line_statistics)

    # Publisher_json = {
    #     "period_statistical_info":Line_statistics,
    #     "area_statistical_info":ROI_statistics
    # }
    # print('Publisher_json',Publisher_json)

    # lock
    lock.acquire()
    # 实时更新ROI区域内的信息，并写入json文件
    Publisher_json.update({"area_statistical_info":ROI_statistics})
    json_str = json.dumps(Publisher_json, indent=4)
    with open(json_path, 'w') as json_file:
        json_file.write(json_str)
    # unlock
    lock.release()

    # show data in image
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(1224 * 0.01), int(1024 * 0.05))
    text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count) +  '  , Class Num: ' + str(classes_num)
    cv_image = cv2.putText(img=cv_image, text=text_draw,
                        org=draw_text_postion,
                        fontFace=font_draw_number,
                        fontScale=0.5, color=(0, 0, 255), thickness=2)

    if show_image:
        cv_image = cv2.resize(cv_image, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA)
        # cv2.setMouseCallback('cv_image', get_point, cv_image    )
        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(1)

    if publish_image:
        msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        img_pub.publish(msg)
        rate.sleep()

    print("sub frame: ", frame_count)
    frame_count += 1


def read_json():
    current_dir = os.path.dirname(__file__)
    f = open(current_dir + "/../json/polygon_chengdu.json", encoding="UTF-8")
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

    roi_num = [[0]]*len(polygons)
    
    up_count = np.zeros((len(lines),  len(classes_list)))
    down_count = np.zeros((len(lines),  len(classes_list)))
    blue_list = np.zeros((len(lines), 2, len(classes_list)))
    yellow_list = np.zeros((len(lines), 2, len(classes_list)))

    Publisher_json = {}
    Line_statistics = []
    
    frame_count = 0
    dump_num = 0

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
    t = RepeatingTimer(10.0, dump_json)
    t.start()

    show_image = rospy.get_param('/traffic_count/show_image')
    publish_image = rospy.get_param('/traffic_count/publish_image')
    bounding_boxes_topic = rospy.get_param('/traffic_count/bounding_boxes_topic')
    detect_image_topic = rospy.get_param('/traffic_count/detect_image_topic')

    image_sub0= message_filters.Subscriber(detect_image_topic, Image)
    boxes_sub1 = message_filters.Subscriber(bounding_boxes_topic, BoundingBoxes)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub0, boxes_sub1], queue_size=15, slop=0.3)
    ts.registerCallback(callback)
    rospy.spin()





