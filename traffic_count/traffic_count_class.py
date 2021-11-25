import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import String
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType
# from sort_track_msg.msg import Targets as TargetsMsg
from sort_track.msg import Targets as TargetsMsg
from sensor_msgs.msg import Image, CompressedImage

import os
import sys
import yaml
import json
import numpy as np
from threading import Timer
import threading
import requests
from cv_bridge import CvBridge, CvBridgeError

import message_filters 

from message_filters import ApproximateTimeSynchronizer, Subscriber

sys.path.insert(0, './src/traffic_count')
import traffic_count_utils.traffic_utils_10_28 as utils
from traffic_count_utils.classes import CLASSES_LIST, TRACK_CLASSES_LIST, TRACK_CLASSES_LEN

class RepeatingTimer(Timer): 
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

class TrafficCountPublisher(Node):

    def __init__(self):
        super().__init__('traffic_count_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0
        self.dump_num = 0
        self.roi_color =  [0, 0, 255]
        self.frame_count = 0
        self.cv_bridge = CvBridge()

        self.declare_parameter('bounding_boxes_topic', 'world')
        self.declare_parameter('detect_image_topic', 'world')
        self.declare_parameter('is_static_matrix', 'world')
        self.declare_parameter('static_matrix_config_path', 'world')
        self.declare_parameter('camera_config_path', 'world')
        self.declare_parameter('period', 'world')
        self.declare_parameter('is_CompressedImage', 'world')
        self.declare_parameter('polygon_path', 'world')
        self.declare_parameter('is_Web', 'world')
        self.declare_parameter('url', 'world')
        self.declare_parameter('size', 'world')
        self.declare_parameter('padding', 'world')
        self.declare_parameter('queue_speed', 'world')
        self.declare_parameter('show_image', True)
        self.declare_parameter('publish_image', True)        

        self.bounding_boxes_topic = self.get_parameter('bounding_boxes_topic').get_parameter_value().string_value
        self.detect_image_topic = self.get_parameter('detect_image_topic').get_parameter_value().string_value
        self.period = self.get_parameter('period').get_parameter_value().integer_value
        self.is_CompressedImage = self.get_parameter('is_CompressedImage').get_parameter_value().bool_value
        self.polygon_path = self.get_parameter('polygon_path').get_parameter_value().string_value
        self.is_Web = self.get_parameter('is_Web').get_parameter_value().bool_value
        self.url = self.get_parameter('url').get_parameter_value().string_value
        self.size = self.get_parameter('size').get_parameter_value().integer_array_value
        self.size = (int(self.size[1]), int(self.size[0]))
        self.padding = self.get_parameter('padding').get_parameter_value().integer_array_value
        self.queue_speed = self.get_parameter('queue_speed').get_parameter_value().integer_value
        self.is_show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.publish_image = self.get_parameter('publish_image').get_parameter_value().bool_value


        self.json_path = os.path.join("/root/ros2_ws/src/traffic_count/json/yolo_statistics.json")

        # 坐标转化
        self.is_static_matrix = self.get_parameter('is_static_matrix').get_parameter_value().bool_value
        if self.is_static_matrix:
            from traffic_count_utils.tf_static_matrix import CameraTools
            static_matrix_config_path = self.get_parameter('static_matrix_config_path').get_parameter_value().string_value
            T, H = self.get_transfromation_matrix(static_matrix_config_path)
            # 初始化了坐标转换函数
            self.cameratool = CameraTools(T, H)
        else:
            from traffic_count_utils.tf_camera_intrinsic import CameraTools
            camera_config_path = self.get_parameter('camera_config_path').get_parameter_value().string_value
            # 获取相机参数
            (cx,cy,fx,fy,h,pitch_angle) = self.get_camera_config(camera_config_path)
            # 实例化CameraTools
            self.cameratool = CameraTools(cx,cy,fx,fy,h,pitch_angle)

        self.classes_list = CLASSES_LIST 
        self.track_classes_list = TRACK_CLASSES_LIST
        self.track_classes_len = TRACK_CLASSES_LEN

        # 读取josn文件里的lines, polygons
        self.lines, self.polygons = self.read_json(self.polygon_path)
        self.multi_line = [[0] for i in range(len(self.lines))]
        self.multi_roi = [[0] for i in range(len(self.polygons))]
        self.multi_stopline_pixel = [[0] for i in range(len(self.polygons))]
        self.multi_stopline_world = [[0] for i in range(len(self.polygons))]

        count = 0
        for line in self.lines:
            self.multi_line[count] = line['points']
            count = count + 1
            # print(self.multi_line[count-1])
        count = 0
        for polygon in self.polygons:
            self.multi_roi[count] = polygon['points']
            self.multi_stopline_pixel[count] = polygon["stop_line"][0]
            count = count + 1
            # print(self.multi_roi[count-1])
            # print(self.multi_stopline_pixel[count-1])

        # json 变量
        self.pub_header = {
            "time_stamp": 0, 
            "device_id": 1, 
            "msg_source": "camera"
        }
        self.Publisher_json = {}

        # area 统计参数变量
        self.roi_num = [[0] for i in range(len(self.polygons))]

        # lines 周期统计参数变量    
        self.Line_statistics = []
        self.up_count = np.zeros((len(self.lines),  len(self.track_classes_list)))
        self.down_count = np.zeros((len(self.lines),  len(self.track_classes_list)))
        self.blue_list = [[] for i in range(len(self.lines))]
        self.yellow_list = [[] for i in range(len(self.lines))]
        self.car_head_passtime = [[] for i in range(len(self.lines))]
        self.car_speed = [[] for i in range(len(self.lines))]
        self.line_occupy_flag = [0]*len(self.lines)
        self.line_occupy_time = [[] for i in range(len(self.lines))]

        targets = TargetsMsg()
        # targets.data.id = 0
        print("targets: ", targets)  
        # tracks = Tracks()
        # # targets.data.id = 0
        # print("tracks: ", tracks)  

        # 每60秒更新一次周期统计信息，并把统计信息置零      
        self.lock = threading.Lock()
        self.t = RepeatingTimer(float(self.period), self.dump_json)
        self.t.start()

        if self.is_show_image:
            if self.is_CompressedImage:
                self.image_sub0= message_filters.Subscriber(self, CompressedImage, self.detect_image_topic, qos_profile=qos_profile_sensor_data)
            else:
                self.image_sub0= message_filters.Subscriber(self, Image, self.detect_image_topic, qos_profile=qos_profile_sensor_data)
            self.boxes_sub1 = message_filters.Subscriber(self, TargetsMsg, self.bounding_boxes_topic)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub0, self.boxes_sub1], 15, 1) 
            self.ts.registerCallback(self.image_boxes_callback)
        else:
            # self.subscription = self.create_subscription(TargetsMsg, self.bounding_boxes_topic, self.boxes_callback, 10)
            # self.subscription  # prevent unused variable warning

            self.subscription = self.create_subscription(TargetsMsg, self.bounding_boxes_topic, self.boxes_callback, 10)
            self.subscription  # prevent unused variable warning 

    def get_camera_config(self, config_path):
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

    def get_transfromation_matrix(self, config_path):
        '''
        获取转换矩阵
        '''
        with open(config_path, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            distance_to_pixel_matrix = json_data['staticMatrixOption']['world2pixel']
            pixel_to_distance_matrix = json_data['staticMatrixOption']['pixel2world']

            return distance_to_pixel_matrix, pixel_to_distance_matrix

    def read_json(self, polygon_path):
        '''
        读取json文件 获取碰撞线collision_lines，统计区域polygons
        '''
        # current_dir = os.path.dirname(__file__)
        f = open(polygon_path, encoding="UTF-8")
        file = json.load(f)
        lines = file['reference_point']['collision_lines']
        polygons = file['reference_point']['measure_filter_range']
        return lines, polygons



    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i

        print("pwd: ", os.getcwd())
        print("size: ", self.size)
        print("size: ", type(self.size))
        print("show_image: ", self.is_show_image)
        print("show_image: ", type(self.is_show_image))
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def dump_json(self):
        '''
        周期性更新碰撞线的数据
        '''
        # 周期性统计各个类别穿过每条线的情况
        Line_statistics = []

        for index in range(0, len(self.multi_line)):
            # 分类统计
            classified_statistic =[]
            sum_count = 0
            sum_car_len = 0
            for i in range(0, len(self.track_classes_list)):
                count = self.up_count[index][i] + self.down_count[index][i]
                sum_count += count
                sum_car_len += (count * self.track_classes_len[i])
                classified_count = {
                    "class":self.track_classes_list[i],
                    "num":self.up_count[index][i] + self.down_count[index][i]
                }
                classified_statistic.append(classified_count)

            # 平均车长
            if sum_count != 0:
                ave_car_len = round(sum_car_len / sum_count, 2)
            else:
                ave_car_len = 0

            # 时间占有率
            sum_occupy_time = 0
            if len(self.line_occupy_time[index]) > 1:
                if len(self.line_occupy_time[index]) % 2 == 0:
                    for i in range(0, len(self.line_occupy_time[index]), 2):
                        sum_occupy_time += (self.line_occupy_time[index][i+1] - self.line_occupy_time[index][i])
                else:
                    time_now = track_time
                    for i in range(0, len(self.line_occupy_time[index]) - 1, 2):
                        sum_occupy_time += (self.line_occupy_time[index][i+1] - self.line_occupy_time[index][i])
                    sum_occupy_time += (time_now - self.line_occupy_time[index][-1] )
            line_occupy = sum_occupy_time / self.period

            # 平均车头时距
            sum_car_head_passtime = 0
            # print("self.car_head_passtime: ", self.car_head_passtime)
            for i in range(0, len(self.car_head_passtime[index]) - 1):
                sum_car_head_passtime += (self.car_head_passtime[index][i+1] - self.car_head_passtime[index][i])
            if sum_count != 0:
                ave_car_head_dis = round(sum_car_head_passtime / sum_count, 2)
            else:
                ave_car_head_dis =0
            
            # 平均车速
            sum_car_speed  = 0
            # print("self.car_speed: ", self.car_speed)
            for i in range(0, len(self.car_speed[index])):
                sum_car_speed += self.car_speed[index][i]
            if sum_count != 0:
                ave_car_speed = round(sum_car_speed / sum_count, 2)
            else:
                ave_car_speed = 0

            line_json = {
                'channel_id': self.lines[index]['road_number'], 
                'total_car': sum_count,
                'occupancy': line_occupy,
                'classified_statistic':classified_statistic,
                'ave_car_head_dis': ave_car_head_dis,
                'ave_car_len': ave_car_len,
                'ave_car_speed': ave_car_speed
            }
            Line_statistics.append(line_json)

        # lock
        self.lock.acquire()
        self.Publisher_json.update({"period_statistical_info": Line_statistics})
        # print("self.Publisher_json: ", self.Publisher_json)
        # 把数据dump到json文件里
        json_str = json.dumps(self.Publisher_json, indent=4)
        with open(self.json_path, 'w') as json_file:
            json_file.write(json_str)
        print("Dump data into json successed: ", self.dump_num)
        self.dump_num += 1
        # unlock
        self.lock.release()

        print("-------------------------------------------")
        print("Publisher_json: ", self.Publisher_json)
        print("-------------------------------------------")
        # 删除周期更新信息

        if self.is_Web:
            # url = "http://10.31.200.139:8001/api/dataView/create" 
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url = self.url, headers = headers, data = json.dumps(self.Publisher_json))

        # lines 周期统计参数变量清零
        Line_statistics = []
        self.up_count = np.zeros((len(self.lines),  len(self.track_classes_list)))
        self.down_count = np.zeros((len(self.lines),  len(self.track_classes_list)))
        self.car_head_passtime = [[] for i in range(len(self.lines))]
        self.car_speed = [[] for i in range(len(self.lines))]
        self.line_occupy_time = [[] for i in range(len(self.lines))]

    def image_boxes_callback(self, image, BoxesMsg):
        '''
        订阅图像与追踪数据话题，并实时更新区域内的统计信息
        '''
        # start  = time.time()
        t1 = time.time()
        print("sub frame: ", self.frame_count)

        self.frame_count += 1

        # 读取image msg并转换为opencv格式
        if self.is_CompressedImage:
            cv_image = self.cv_bridge.compressed_imgmsg_to_cv2(image, "bgr8")
        else:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image,"bgr8")
        # self.size = (cv_image.shape[0], cv_image.shape[1])
        # print("size: ", (cv_image.shape[0], cv_image.shape[1]))


        track_time = BoxesMsg.header.stamp.to_sec()
        self.pub_header.update({"time_stamp": int(track_time)})

        # 整张图像的各个类别数量
        classes_num = utils.image_count(BoxesMsg.data, self.classes_list)

        # 每个roi区域的各个类别数量
        ROI_statistics = []
        ROI_queue = []
        for index in range(0, len(self.multi_roi)):

            # 设置roi区域的1，2点为停止线，并选择其中点为停止点
            stop_x = int(self.multi_roi[index][0][0] + self.multi_roi[index][1][0])
            stop_y = int(self.multi_roi[index][0][1] + self.multi_roi[index][1][1])
            ground_stop_x, ground_stop_y = self.cameratool.pixel2camera_projection(stop_x * 0.5, stop_y * 0.5)
            stop_point = (ground_stop_x/1000, ground_stop_y/1000)
    
            self.roi_num[index], roi_color_image, area_info, queue_info = utils.roi_count_queue(self.multi_roi[index], BoxesMsg.data, 
                                                                        self.track_classes_list,  stop_point, self.roi_color, self.size, self.queue_speed, self.is_show_image)                                                    

            if self.is_show_image:
                cv_image = cv2.add(cv_image, roi_color_image)
                
            area_json = {
                'area_id': self.polygons[index]['road_number'], 
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
                'lane_id': self.polygons[index]['road_number'], 
                "queue_len": 0,
                "head_car_pos": 0,
                "tail_car_pos": 0,
                "car_count": 0
                }
            area_json.update(area_info)
            queue_up_info.update(queue_info)
            ROI_queue.append(queue_up_info)
            ROI_statistics.append(area_json)
        # print("+++++++++++++++++++++++++++++++++")
        # print('ROI_statistics:',ROI_statistics)

        # lock
        self.lock.acquire()
        # 实时更新ROI区域内的信息，并写入json文件
        self.Publisher_json.update(self.pub_header)
        self.Publisher_json.update({"area_statistical_info":ROI_statistics})
        self.Publisher_json.update({"queue_up_info":ROI_queue})
        json_str = json.dumps(self.Publisher_json, indent=4)
        with open(self.json_path, 'w') as json_file:
            json_file.write(json_str)
        # unlock
        self.lock.release()

        # 删除周期更新信息
        if "period_statistical_info" in self.Publisher_json:
            del self.Publisher_json["period_statistical_info"]
        if is_Web:
            # 推送区域信息到前端
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url,headers = headers, data = json.dumps(self.Publisher_json))

        # 各个类别穿过每条线的统计情况
        Line_statistics = []
        for index in range(0, len(self.multi_line)):
            # 判断line中点加上padding之后是否超出图片范围
            for i in range(0, 2):
                if self.multi_line[index][i][0]+self.padding[0] >= self.size[1] or self.multi_line[index][i][0]+self.padding[0] <= 0:
                    print(" The point of lines out off range or padding out off range")
                if self.multi_line[index][i][1]+self.padding[1] >= self.size[0] or self.multi_line[index][i][1]+self.padding[1] <= 0:
                    print(" The point of lines out off range or padding out off range")

            # 周期性统计各个类别穿过每条线的情况
            polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(self.multi_line[index], self.padding, self.size, self.is_show_image)
            self.up_count[index], self.down_count[index] = utils.traffic_count(BoxesMsg, self.size,  self.track_classes_list,  polygon_mask_blue_and_yellow, 
                                                                    self.blue_list[index], self.yellow_list[index],  self.up_count[index], self.down_count[index], self.car_head_passtime[index], self.car_speed[index])

            # 计算occupancy
            self.line_occupy_flag[index] = utils.occupancy(BoxesMsg,  self.multi_line[index], self.padding, self.line_occupy_flag[index], self.line_occupy_time[index])

            if self.is_show_image:
                cv_image = cv2.add(cv_image, polygon_color_image)

        if self.is_show_image:
            # 在图像上画出每个bounding_boxes
            point_radius=3
            # for item_bbox in list_track:
            for i in range(0, len(BoxesMsg.data)):
                track_id = BoxesMsg.data[i].id
                x1 = BoxesMsg.data[i].xmin
                y1 = BoxesMsg.data[i].ymin
                x2 = BoxesMsg.data[i].xmax
                y2 = BoxesMsg.data[i].ymax
                cls = BoxesMsg.data[i].target_class
                vx = BoxesMsg.data[i].vx
                vy = BoxesMsg.data[i].vy

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

            cv_image = cv2.resize(cv_image, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_NEAREST)
            cv2.namedWindow("YOLO+SORT", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO+SORT", cv_image)
            cv2.waitKey(1)

    def boxes_callback(self, BoxesMsg):
        '''
        订阅追踪数据话题，并实时更新区域内的统计信息和排队信息
        '''
        # start  = time.time()
        print("sub frame: ", self.frame_count)
        self.frame_count += 1
        print("BoxesMsg", BoxesMsg)


        track_time = BoxesMsg.header.stamp.sec + 10E-9 * BoxesMsg.header.stamp.nanosec
        self.pub_header.update({"time_stamp": int(track_time)})

        # 整张图像的各个类别数量
        classes_num = utils.image_count(BoxesMsg.data, self.classes_list)

        # 每个roi区域的各个类别数量
        ROI_statistics = []
        ROI_queue = []
        for index in range(0, len(self.multi_roi)):
            # 设置roi区域的1，2点为停止线，并选择其中点为停止点
            stop_x = int(self.multi_roi[index][0][0] + self.multi_roi[index][1][0])
            stop_y = int(self.multi_roi[index][0][1] + self.multi_roi[index][1][1])
            ground_stop_x, ground_stop_y = self.cameratool.pixel2camera_projection(stop_x * 0.5, stop_y * 0.5)
            stop_point = (ground_stop_x/1000, ground_stop_y/1000)
            
            self.roi_num[index], roi_color_image, area_info, queue_info = utils.roi_count_queue(self.multi_roi[index], BoxesMsg.data, 
                                                                        self.track_classes_list,  stop_point, self.roi_color, self.size, self.queue_speed, self.is_show_image)                                                    

            area_json = {
                'area_id': self.polygons[index]['road_number'], 
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
                'lane_id': self.polygons[index]['road_number'], 
                "queue_len": 0,
                "head_car_pos": 0,
                "tail_car_pos": 0,
                "car_count": 0
                }
            area_json.update(area_info)
            queue_up_info.update(queue_info)
            ROI_queue.append(queue_up_info)
            ROI_statistics.append(area_json)
        # print("+++++++++++++++++++++++++++++++++")
        # print('ROI_statistics:',ROI_statistics)

        # lock
        self.lock.acquire()
        # 实时更新ROI区域内的信息，并写入json文件
        self.Publisher_json.update(self.pub_header)
        self.Publisher_json.update({"area_statistical_info":ROI_statistics})
        self.Publisher_json.update({"queue_up_info":ROI_queue})
        json_str = json.dumps(self.Publisher_json, indent=4)
        with open(self.json_path, 'w') as json_file:
            json_file.write(json_str)
        # unlock
        self.lock.release()

        # 删除周期更新信息
        if "period_statistical_info" in self.Publisher_json:
            del self.Publisher_json["period_statistical_info"]
        # 推送区域信息到前端
        if self.is_Web:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url,headers = headers, data = json.dumps(self.Publisher_json))

        # 各个类别穿过每条线的统计情况
        Line_statistics = []
        for index in range(0, len(self.multi_line)):
            # 判断line中点加上padding之后是否超出图片范围
            for i in range(0, 2):
                if self.multi_line[index][i][0]+self.padding[0] >= self.size[1] or self.multi_line[index][i][0]+self.padding[0] <= 0:
                    print(" The point of lines out off range or padding out off range")
                if self.multi_line[index][i][1]+self.padding[1] >= self.size[0] or self.multi_line[index][i][1]+self.padding[1] <= 0:
                    print(" The point of lines out off range or padding out off range")

            # 周期性统计各个类别穿过每条线的情况
            polygon_mask_blue_and_yellow, polygon_color_image = utils.line2polygon(self.multi_line[index], self.padding, self.size, self.is_show_image)
            self.up_count[index], self.down_count[index] = utils.traffic_count(BoxesMsg, self.size,  self.track_classes_list,  polygon_mask_blue_and_yellow, 
                                                                    self.blue_list[index], self.yellow_list[index],  self.up_count[index], self.down_count[index], self.car_head_passtime[index], self.car_speed[index])
                                                                
            # 计算occupancy
            self.line_occupy_flag[index] = utils.occupancy(BoxesMsg,  self.multi_line[index], self.padding, self.line_occupy_flag[index], self.line_occupy_time[index])
