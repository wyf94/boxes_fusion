#!/home/wyf/anaconda3/envs/test/bin/python
#!coding=utf-8

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys

sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
 
def imagePublisher():
    rospy.init_node('video_publisher', anonymous=True)

    img_pub_0 = rospy.Publisher('/image_source_0', Image, queue_size=10)
    rate = rospy.Rate(25)
    img_pub_1 = rospy.Publisher('/image_source_1', Image, queue_size=10)
    rate = rospy.Rate(25)
 
    cap_0 = cv2.VideoCapture("/home/wyf/ros_ws/src/video_publisher/video/video_8mm.avi")
    frames_num_0=cap_0.get(7)
    bridge_0 = CvBridge()
    if not cap_0.isOpened():
        print('Video_0 open failed!')
        return -1

    cap_1 = cv2.VideoCapture("/home/wyf/ros_ws/src/video_publisher/video/video_35mm.avi")
    frames_num_1=cap_1.get(7)
    bridge_1 = CvBridge()
    if not cap_1.isOpened():
        print('Video_1 open failed!')
        return -1

    count = 0

    print('Video_0 and Video_1 open succeed!')
    while not rospy.is_shutdown():
        if count >= frames_num_1:
            cap_0 = cv2.VideoCapture("/home/wyf/ros_ws/src/video_publisher/video/video_8mm.avi")
            cap_1 = cv2.VideoCapture("/home/wyf/ros_ws/src/video_publisher/video/video_35mm.avi")
            count = 0

        ret_0, frame_0 = cap_0.read()
        ret_1, frame_1 = cap_1.read()
        msg_0 = bridge_0.cv2_to_imgmsg(frame_0, encoding="bgr8")
        msg_1 = bridge_1.cv2_to_imgmsg(frame_1, encoding="bgr8")

        now = rospy.Time.now()
        # print('time:',now)
        # print('time1:',msg_0.header.stamp()) 
        img_pub_0.publish(msg_0)
        img_pub_0.sent_time = now
        img_pub_1.publish(msg_1)
        img_pub_1.sent_time = now

        count = count + 1
        print('publish', count, 'frame')
        rate.sleep()
 
if __name__ == '__main__':
    try:
        imagePublisher()
    except rospy.ROSInterruptException:
        pass
