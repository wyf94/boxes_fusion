#!/usr/bin/python3
#!coding=utf-8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys

from traffic_count.msg import BoundingBox
from traffic_count.msg import BoundingBoxes

import message_filters
from sensor_msgs.msg import Image, CameraInfo   

from traffic_count.cameratool import CameraTools
from traffic_count.utils import TrafficCount
# import traffic_count.traffic_count as traffic_count
from traffic_count.yolo_classes import CLASSES_LIST
 

def callback(image, boxes):
    global frame_count, up_count, down_count, blue_list, yellow_list, Classes_List, point_roi, line

    # ct = CameraTools(905.8602,516.4283,1626.513816,1624.574619,1200 ,11)
    # x,y = ct.pixel2world(500,600)
    # print("相机投影坐标：",x,y)
    # print('像素坐标:',ct.world2pixel([x,y,0]))
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image,"bgr8")
    
    

    tc = TrafficCount(cv_image, boxes.bounding_boxes, Classes_List,  line)

    polygon_mask_blue_and_yellow, polygon_color_image= tc.line2polygon(0, 20)
    classes_num = tc.image_count()
    roi_num, roi_color_image = tc.roi_count(point_roi, [0, 0, 255])
    up_count, down_count = tc.traffic_count(polygon_mask_blue_and_yellow, blue_list, yellow_list,  up_count, down_count)

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(1224 * 0.01), int(1024 * 0.05))
    text_draw = 'DOWN: ' + str(down_count) + ' , UP: ' + str(up_count) + '  , Class Num: ' + str(classes_num)+ '  , Roi Num: ' + str(roi_num)

    # print(cv_image.shape)
    # print (polygon_color_image.shape)
    cv_image = cv2.add(cv_image, polygon_color_image)
    cv_image = cv2.add(cv_image, roi_color_image)
    # cv_image = cv2.add(cv_image, polygon_color_01)
    cv_image = cv2.putText(img=cv_image, text=text_draw,
                                        org=draw_text_postion,
                                        fontFace=font_draw_number,
                                        fontScale=0.5   , color=(0, 0, 255), thickness=2)    

    new_img = cv2.resize(cv_image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
    cv2.imshow("new_img", new_img)
    cv2.waitKey(1) 

    print("sub frame: ", frame_count)
    frame_count +=1

if __name__ == '__main__':
    # rospy.init_node('showImage',anonymous = True)/
    rospy.init_node('traffic_count', anonymous=True)
    img_pub = rospy.Publisher('/traffic_count_publisher', Image, queue_size=10)
    rate = rospy.Rate(25)

    frame_count = 0
 
    point_roi = [[321*2, 356*2],[369*2, 357*2], [364*2, 287*2],[337*2, 292*2]]

    Classes_List = CLASSES_LIST
    print(CLASSES_LIST)
    line = [[205*2, 385*2], [425*2, 385*2]]

    category_number = len(Classes_List)

    up_count = [0]*category_number
    down_count = [0]*category_number
    blue_list = np.zeros((2, category_number))
    yellow_list = np.zeros((2, category_number))

    image_sub0= message_filters.Subscriber('/output_image_nms', Image)
    boxes_sub1 = message_filters.Subscriber('/bounding_boxes_nms', BoundingBoxes)
    ts = message_filters.TimeSynchronizer([image_sub0, boxes_sub1], 15)
    ts.registerCallback(callback)
    rospy.spin()
