
"""
ROS节点使用SORT TRACKER和YOLOv4检测器跟踪对象(yolov4_trt_ros)
从 yolov4_trt_ros 获取检测到的bounding boxes，并使用它们来计算跟踪的 bounding boxes
被跟踪的对象及其ID被发布到sort_track节点
这里没有延迟
"""

from sys import setdlopenflags
from numpy.lib import index_tricks
import rospy
import numpy as np
# from yolov4_trt_ros.msg import BoundingBoxes
# from darknet_ros_msgs.msg import BoundingBoxes
from sort import sort 
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image,CompressedImage
from sort_track.msg import Tracks as TracksMsg
from sort_track.msg import BoundingBox
from sort_track.msg import Target as TargetMsg
from sort_track.msg import Targets as TargetsMsg
from traffic_count.msg import BboxCoordinate as BboxCoordinateMsg
from traffic_count.msg import BboxesCoordinates as BboxesCoordinatesMsg
from traffic_count.cameratool import CameraTools
import yaml
import time
import message_filters

class Target(object):

    def __init__(self,target_id,x,y,Class,time_stamp):
        self.id = target_id
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.Class = Class 
        self.last_time = time_stamp
        self.skip_frames = 0

    def _update(self,x,y,Class,time_stamp):
        dt = time_stamp - self.last_time
        self._set_velocity(x,y,dt)
        # print(self.x)
        self.x = round(x,2)
        # print(self.x)

        self.y = round(y,2)
        self.last_time = time_stamp
        self.Class = Class
        self.skip_frames = 0

    def _set_velocity(self,x,y,dt):
        if dt != 0:
            vx = (x - self.x) / dt
            vy = (y - self.y) / dt
            self.vx = (vx + self.vx) / 2
            self.vy = (vy + self.vy) / 2

    def toString(self):
        res = '{id:'+str(self.id)+',x:'+str(self.x)+',y:'+str(self.y)+\
        ',vx:'+str(self.vx)+',vy:'+str(self.vy)+',class:'+str(self.Class)+'}'
        return res

class Targets(object):

    def __init__(self,max_frames_to_skip):
        self.targets = []
        self.max_frames_to_skip = max_frames_to_skip

    def get_id_index(self,target_id):
        for i in range(len(self.targets)):
            if target_id == self.targets[i].id:
                return i
        return -1

    def get_residue_id_list(self,id_list):
        '''
        获取除查询id外剩下的其他id列表的index
        参数: 
            id_list: id列表,[id1,id2,id3],如: [1,3,6]
        返回值:
            id_index_list: 不在id列表里的目标索引列表,[index1,index2,index3],如: [0,2,5]
        '''
        id_index_list = []
        for i in range(len(self.targets)):
            if self.targets[i].id not in id_list:
                id_index_list.append(i)
        return id_index_list

    def update(self,targets):
        '''
        传入最新数据进行更新
        参数:
            targets: 数据列表,内容形式为[[id,x,y,class,time_stamp]],
            如: [[1,10.0,8.9,'car','1627838388888'],[2,10.0,8.9,'person','1627838388888']]
        '''
        if len(self.targets) == 0:
            for target in targets:
                # 创建新的Target对象
                t = Target(target[0],target[1],target[2],target[3],target[4])
                self.targets.append(t)

            return self.targets

        for target in targets:
            id_index = self.get_id_index(target[0])
            # 如果目标不在列表里
            if id_index == -1:
                # 创建新的Target对象
                t = Target(target[0],target[1],target[2],target[3],target[4])
                self.targets.append(t)
            else:
                # 使用新的数据进行更新
                self.targets[id_index]._update(target[1],target[2],target[3],target[4])

        # 获取传入目标数据的id        
        id_list = []
        for item in targets:
            id_list.append(item[0])
        
        # 待删除的列表
        det_tracks_index = []
        # 对跳过对帧进行处理
        id_index_list = self.get_residue_id_list(id_list)
        for item in id_index_list:
            self.targets[item].skip_frames += 1
            if self.targets[item].skip_frames > self.max_frames_to_skip:
                det_tracks_index.append(item)

        # 删除要删除的列表
        if len(det_tracks_index) > 0:
            temp = [i for num,i in enumerate(self.targets) if num not in det_tracks_index]
            self.targets = temp

        # 获取返回数据
        ret = []
        for item in self.targets:
            if item.id in id_list:
                ret.append(item)
        return ret
    def toString(self):
        if len(self.targets)<1:
            return '[]'
        res = '['
        for item in self.targets:
            # print(item)
            res += str(item.toString() )+','
        res = res[0:-1]
        res += ']'

        return res

class Sort_Track(object):
    def __init__(self, max_age, min_hits, camera_config_path ):
        # 获取参数
        self.max_age = max_age
        self.min_hits = min_hits
        self.camera_config_path = camera_config_path
        # 获取相机参数
        (cx,cy,fx,fy,h,pitch_angle) = self.get_camera_config(self.camera_config_path)
        # 实例化CameraTools
        self.cameratool = CameraTools(cx,cy,fx,fy,h,pitch_angle)
        
        #创建SORT跟踪器的实例
        self.tracker = sort.Sort(self.max_age, self.min_hits)
        # 检测数据列表
        self.detections = []
        # 跟踪数据列表
        self.trackers = []
        self.tracks = []

        self.tracks_msg = TracksMsg()
        self.targets_msg = TargetsMsg()
        self.BboxesCoordinates_msg = BboxesCoordinatesMsg()

        # 创建Targets类的实例
        self.targets = Targets(self.max_age)

    def get_camera_config(self,config_path):
        '''
        获取相机参数
        '''
        with open (config_path,'r') as f:
            config_data = yaml.safe_load(f)
            cx = config_data['camera']['cx']
            cy = config_data['camera']['cy']
            fx = config_data['camera']['fx']
            fy = config_data['camera']['fy']
            h  = config_data['camera']['h']
            pitch_angle = config_data['camera']['pitch_angle']

        return (cx,cy,fx,fy,h,pitch_angle)

    def pixel2world(self, x, y):
        world_x, world_y = self.cameratool.pixel2world(x, y)
        return world_x / 1000, world_y / 1000

    def sort_track(self, boxes):
        # 接收到的检测数据
        detections = []
        for box in boxes.bounding_boxes:
            # 过滤掉不需要的类别
            if box.Class == 'traffic light':
                continue
            detections.append(np.array([box.xmin, box.ymin, box.xmax, box.ymax, round(box.probability,2),box.Class]))
            
        self.detections = np.array(detections)
        # 调用跟踪器
        self.trackers = self.tracker.update(self.detections)

        # self.tracks_msg = self.publish_data(self.trackers)
        # self.targets_msg = self.publish_targets(self.trackers)
        self.BboxesCoordinates_msg = self.publish_BboxesCoordinates(self.trackers)

        return self.BboxesCoordinates_msg
        # return self.tracks_msg, self.targets_msg, self.BboxesCoordinates_msg

    
    def publish_data(self,tracks):
        '''
        发布跟踪后的数据(bounding boxes)
        '''
        track_msg = TracksMsg()
        track_msg.header.stamp = rospy.Time.now()
        track_msg.header.frame_id = 'sort_track'            

        for track in tracks:
            bounding_box_msg = BoundingBox()
            item = np.array(track[:5],dtype='float')
            bounding_box_msg.xmin  = int(item[0])
            bounding_box_msg.ymin  = int(item[1])
            bounding_box_msg.xmax  = int(item[2])
            bounding_box_msg.ymax  = int(item[3])
            bounding_box_msg.id    = int(item[4])
            bounding_box_msg.Class = track[5]

            track_msg.bounding_boxes.append(bounding_box_msg)

        return track_msg

    def publish_targets(self,tracks,is_center=False):
        '''
        发布跟踪后的数据（世界坐标）
        '''
        targets_msg = TargetsMsg()
        targets_msg.header.stamp = rospy.Time.now()
        targets_msg.header.frame_id = 'sort_track'

        time_stamp = time.time()
        targets = []
        for track in tracks:
            item = np.array(track[:5],dtype='float')
            center_x = (item[0] + item[2]) / 2
            # 如果取中心
            if is_center:
                center_y = (item[1] + item[3]) / 2
            # 取最下边
            else:
                center_y = item[3]
            x,y = self.cameratool.pixel2world(center_x,center_y)
            x = x / 1000
            y = y / 1000

            targets.append([int(item[4]),x,y,track[5],time_stamp])

        ret_targets = self.targets.update(targets)
        for item in ret_targets:
            target_msg = TargetMsg()
            target_msg.id    = int(item.id)
            target_msg.x     = round(item.x,2)
            target_msg.y     = round(item.y,2)
            target_msg.vx    = round(item.vx,2)
            target_msg.vy    = round(item.vy,2)
            target_msg.Class = item.Class

            targets_msg.data.append(target_msg)
            
        return targets_msg


    def publish_BboxesCoordinates(self, tracks, is_center=True):
        '''
        发布跟踪后的数据(bounding boxes)
        发布跟踪后的数据（世界坐标）
        '''
        BboxesCoordinates_msg = BboxesCoordinatesMsg()
        
        BboxesCoordinates_msg.header.stamp = rospy.Time.now()
        BboxesCoordinates_msg.header.frame_id = 'sort_track'

        time_stamp = time.time()
        targets = []
        for track in tracks:
            item = np.array(track[:5],dtype='float')
            center_x = (item[0] + item[2]) / 2
            # 如果取中心
            if is_center:
                center_y = (item[1] + item[3]) / 2
            # 取最下边
            else:
                center_y = item[3]
            x,y = self.cameratool.pixel2world(center_x,center_y)
            x = x / 1000
            y = y / 1000

            targets.append([int(item[4]),x,y,track[5],time_stamp])

        ret_targets = self.targets.update(targets)

        for item in ret_targets:
            BboxCoordinate_msg = BboxCoordinateMsg()
            BboxCoordinate_msg.id    = int(item.id)
            BboxCoordinate_msg.x     = round(item.x,2)
            BboxCoordinate_msg.y     = round(item.y,2)
            BboxCoordinate_msg.vx    = round(item.vx,2)
            BboxCoordinate_msg.vy    = round(item.vy,2)
            BboxCoordinate_msg.Class = item.Class

            for track in tracks:
                track_array = np.array(track[:5],dtype='float')
                if int(track_array[4]) == int(item.id):
                    BboxCoordinate_msg.xmin  = int(track_array[0])
                    BboxCoordinate_msg.ymin  = int(track_array[1])
                    BboxCoordinate_msg.xmax  = int(track_array[2])
                    BboxCoordinate_msg.ymax  = int(track_array[3])
                    break
            BboxesCoordinates_msg.bbox_coordinate.append(BboxCoordinate_msg)
            
        return BboxesCoordinates_msg