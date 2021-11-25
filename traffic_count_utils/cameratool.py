# -*- coding: utf-8 -*-
'''
描述: 相机\雷达各坐标系之间的转换
最后修改时间: 2020/08/06 14:43
'''
import numpy as np
import math

class CameraTools(object):
    '''
    🦋
    本例中，选取相机投影坐标系作为世界坐标系
    参数:
        cx: 相机x方向光轴偏移
        cy: 相机y方向光轴偏移
        fx: 相机x方向焦距
        fy: 相机y方向焦距
        h: 相机安装高度，单位: mm
        pitch_angle: 相机俯仰角,单位: 度
    🦋
    '''
    def __init__(self,cx,cy,fx,fy,h,pitch_angle):
        
        # 世界到相机旋转的角度
        rotation_angle_w2c = - 90 - pitch_angle

        # 转为弧度
        rotation_angle_w2c = rotation_angle_w2c * math.pi / 180
        pitch_angle = pitch_angle * math.pi/180
        
        # 相机俯仰角
        self.pitch_angle = pitch_angle
        
        # 相机安装高度
        self.h = h

        # 相机参数值
        self.camera_intrinsic = {
            "R": [[1,0,0],
              [0,math.cos(rotation_angle_w2c),math.sin(rotation_angle_w2c)],
              [0,-math.sin(rotation_angle_w2c),math.cos(rotation_angle_w2c)]],
            "T": [0,0,h],
            "f": [fx,fy],
            "c": [cx,cy]
        }

        # 相机内参矩阵
        camera_matrix = [[fx,   0.,   cx,  0.],
                         [0.,   fy,   cy,  0.],
                         [0.,   0.,   1.,  0.]]
        self.camera_matrix = np.asarray(camera_matrix)
        
        # 相机转世界旋转、平移矩阵
        RT_matrix = [[1.,  0.,                      0.,                     0.],
                     [0.,  -math.sin(pitch_angle),  math.cos(pitch_angle),  0.],
                     [0.,  -math.cos(pitch_angle),  -math.sin(pitch_angle), h],
                     [0.,  0.,                      0.,                     1.]]
        self.RT_matrix = np.asarray(RT_matrix)

        

    def pixel2world(self,u,v):
        '''
        🦋

        功能: 像素坐标 -> 相机投影坐标

        参数: 
            u: 像素坐标系u方向的值
            v: 像素坐标系v方向的值

        返回值: 
            xcw: 相机投影坐标系下的x方向的数值，右为正，单位: mm
            ycw: 相机投影坐标系下的y方向的数值，相机光轴方向为正,单位: mm

        注:
            1）该公式使用的前提为目标在水平面上
        🦋
        '''
        right_matrix = self._getRightMatrix(u,v)

        m = right_matrix[0,1]
        l = right_matrix[0,0]
        # 俯仰角的cos
        cos_pitch = math.cos(self.pitch_angle)
        # 俯仰角的sin
        sin_pitch = math.sin(self.pitch_angle)

        ycw = m * sin_pitch * self.h/(1 - m * cos_pitch)

        xcw = (ycw * cos_pitch + sin_pitch * self.h) * l

        return xcw,ycw

    def _getRightMatrix(self,u,v):
        '''
        🦋

        功能: 获取右边矩阵

        参数:
            u: 像素坐标系下 u 方向的值
            v: 像素坐标系下 v 方向的值

        返回值:
            r_matrix:  右边矩阵
        🦋
        '''
        # 像素矩阵
        pixel_matrix = [u, v, 1]
        pixel_matrix = np.asarray(pixel_matrix)
        '''
        numpy中 np.linalg.inv()只能求方阵的逆
        '''
        # camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        c_mat = np.matrix(self.camera_matrix)

        camera_matrix_inv = c_mat.I

        # print(camera_matrix_inv)

        # print(camera_matrix_inv)

        r_matrix = np.dot(np.dot(self.RT_matrix,camera_matrix_inv), pixel_matrix.T)

        # print('right_matrix:',r_matrix)
        # print('right_matrix of type:',type(r_matrix))
        return r_matrix
   
    def world2camera(self,joint_world):
        '''
        功能: 世界坐标系 -> 相机坐标系 : R * (pt - T)
        参数: 
            world_coord: 世界坐标 [x,y,z]
        '''
        joint_world = np.asarray(joint_world)
        R = np.asarray(self.camera_intrinsic["R"])
        T = np.asarray(self.camera_intrinsic["T"])
        # joint_num = len(joint_world)
        joint_cam = np.dot(R, (joint_world - T).T).T  # R * (pt - T)
        return joint_cam
 
    def camera2world(self,joint_cam):
        """
        功能: 相机坐标系 -> 世界坐标系: inv(R) * pt +T 
            joint_cam = np.dot(inv(R), joint_world.T)+T
            :return:
        """
        
        joint_cam = np.asarray(joint_cam)
        R = np.asarray(self.camera_intrinsic["R"])
        T = np.asarray(self.camera_intrinsic["T"])
        # 相机坐标系 -> 世界坐标系
        joint_world = np.dot(np.linalg.inv(R), joint_cam.T).T + T
        return joint_world

    def cam2pixel(self,cam_coord):
        '''
        🦋
        功能: 
            实现相机坐标到转像素坐标的转换，
            将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
            u = X * fx / Z + cx
            v = Y * fy / Z + cy
            D(v,u) = Z / Alpha
        参数: 
            cam_coord: 相机坐标[x,y,z]
        返回值:
            u: 像素u方向的值
            v: 像素v方向的值
            d: 深度
        🦋
        '''
        f = self.camera_intrinsic["f"]
        c = self.camera_intrinsic["c"]

        u = cam_coord[0] / cam_coord[2] * f[0] + c[0] 
        v = cam_coord[1] / cam_coord[2] * f[1] + c[1]
        d = cam_coord[2]

        return int(u),int(v),d

    def __cam2pixel(self,cam_coord, f, c):
        '''将删除'''
        """
        相机坐标系 -> 像素坐标系: (f / dx) * (X / Z) = f * (X / Z) / dx
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
        将从3D(X,Y,Z)映射到2D像素坐标P(u,v)计算公式为：
        u = X * fx / Z + cx
        v = Y * fy / Z + cy
        D(v,u) = Z / Alpha
        =====================================================
        camera_matrix = [[428.30114, 0.,   316.41648],
                        [   0.,    427.00564, 218.34591],
                        [   0.,      0.,    1.]])
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        =====================================================
        :param cam_coord:
        :param f: [fx,fy]
        :param c: [cx,cy]
        :return:
        """
        # 等价于：(f / dx) * (X / Z) = f * (X / Z) / dx
        # 三角变换， / dx, + center_x
        # print(cam_coord[..., 0])
        u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
        v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
        d = cam_coord[..., 2]
        return u, v, d

    def camera2pixel(self,joint_cam):
        '''将删除'''
        """
        相机坐标系 -> 像素坐标系
        :param joint_cam:
        :return:
        """
        # 相机坐标系 -> 像素坐标系，并 get relative depth
        # Subtract center depth
        # 选择 Pelvis骨盆 所在位置作为相机中心，后面用之求relative depth
        root_idx = 0
        center_cam = joint_cam[root_idx]  # (x,y,z) mm
        joint_num = len(joint_cam)
        f = self.camera_intrinsic["f"]
        c = self.camera_intrinsic["c"]
        # print('c:',c)
        # joint image_dict，像素坐标系，Depth 为相对深度 mm
        joint_img = np.zeros((joint_num, 3))
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = self.__cam2pixel(joint_cam, f, c)  # x,y
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
        return joint_img

    def world2pixel(self,joint_world):
        """
        🦋

        功能: 世界坐标系转像素坐标系

        参数: 
            joint_world: 世界坐标系列表 [[x,y,z]]

        返回: 
            pixel_coord: 像素坐标系列表 [[u,v,d]]
        🦋
        """
        joint_cam = self.world2camera(joint_world)
        # pixel_coord = self.camera2pixel(joint_cam)
        pixel_coord = self.cam2pixel(joint_cam)

        return pixel_coord
    
    def radar2camera_projection(self, radar_angle, lx, ly, xr, yr):
        '''
        🦋

        功能: 雷达坐标转相机投影坐标

        参数: 
            radar_angle: 雷达转相机投影坐标的旋转角度
            lx: 雷达坐标转相机投影坐标系x方向的位移
            ly: 雷达坐标转相机投影坐标系y方向的位移
            xr: 雷达x方向的坐标值
            yr: 雷达y方向的坐标值

        返回值:
            x: 相机投影坐标系下x方向的值
            y: 相机投影坐标系下y方向的值

        注: 
            1）绕原点逆时针方向 radar_angle 为正。
            2）相机在雷达坐标x轴正方向lx取负
            3）相机在雷达坐标y轴正方向ly取负
            4) CW = RT * R
        🦋
        '''
        # 雷达转相机投影坐标的旋转角度
        # 转为弧度值
        radar_angle = radar_angle * math.pi / 180

        cos_radar = math.cos(radar_angle)
        sin_radar = math.sin(radar_angle)

        # 雷达转相机投影坐标的旋转平移矩阵
        rt_matrix = [[cos_radar,  sin_radar,  lx],
                     [-sin_radar, cos_radar,  ly],
                     [0.,         0.,         1.]]
        rt_matrix = np.mat(rt_matrix)
        
        radar_matrix = np.mat([xr, yr, 1])

        camera_matrix = np.dot(rt_matrix, radar_matrix.T)
        # print(camera_matrix)
        camera_matrix = camera_matrix.tolist()

        x = camera_matrix[0][0]
        y = camera_matrix[1][0]

        return x,y

    def camera_projection2radar(self, radar_angle, lx, ly, xcw, ycw):
        '''
        🦋

        功能: 相机投影坐标转雷达坐标

        参数: 
            radar_angle: 雷达转相机投影坐标的旋转角度
            lx: 雷达坐标转相机投影坐标系x方向的位移
            ly: 雷达坐标转相机投影坐标系y方向的位移
            xcw: 相机投影坐标系下x方向的坐标值
            ycw: 相机投影坐标系下y方向的坐标值

        返回值:
            x: 相机投影坐标系下x方向的值
            y: 相机投影坐标系下y方向的值

        注: 
            1）绕原点逆时针方向 radar_angle 为正。
            2）相机在雷达坐标x轴正方向lx取负
            3）相机在雷达坐标y轴正方向ly取负
            4) R = RT.I * CW
        🦋
        '''
        # 雷达转相机投影坐标的旋转角度
        # 转为弧度值
        radar_angle = radar_angle * math.pi / 180

        cos_radar = math.cos(radar_angle)
        sin_radar = math.sin(radar_angle)

        # 雷达转相机投影坐标的旋转平移矩阵
        rt_matrix = [[cos_radar,  sin_radar,  lx],
                     [-sin_radar, cos_radar,  ly],
                     [0.,         0.,         1.]]
        rt_matrix = np.mat(rt_matrix)
        
        camera_projection_matrix = np.mat([xcw, ycw, 1])

        radar_matrix = np.dot(rt_matrix.I, camera_projection_matrix.T)

        radar_matrix = radar_matrix.tolist()

        x = radar_matrix[0][0]
        y = radar_matrix[1][0]

        return x,y
    
    def radar2pixel(self,xr,yr,radar_angle,lx,ly):
        '''
        🦋

        功能: 雷达坐标系转像素坐标系

        参数: 
            xr: 雷达x方向的坐标值
            yr: 雷达y方向的坐标值
            radar_angle: 雷达转相机投影坐标的旋转角度
            lx: 雷达坐标转相机投影坐标系x方向的位移
            ly: 雷达坐标转相机投影坐标系y方向的位移
            
        返回值:
            x: 相机投影坐标系下x方向的值
            y: 相机投影坐标系下y方向的值
        🦋
        '''

        xcw,ycw= self.radar2camera_projection(radar_angle,lx,ly,xr,yr)
        # 建立相机投影坐标，z=0
        joint_world = [xcw,ycw,0]
        pixel_coord = self.world2pixel(joint_world)
        u = int(pixel_coord[0])
        v = int(pixel_coord[1])
        return u,v
  
    def pixel2radar(self,u,v,radar_angle,lx,ly):
        '''
        🦋

        功能: 像素坐标系转雷达坐标系
        
        参数: 
            u: 像素坐标系下u方向的值
            v: 像素坐标系下v方向的值
            radar_angle: 雷达转相机投影坐标的旋转角度
            lx: 雷达坐标转相机投影坐标系x方向的位移
            ly: 雷达坐标转相机投影坐标系y方向的位移

        返回值:
            x: 相机投影坐标系下x方向的值
            y: 相机投影坐标系下y方向的值

        注: 
            1）绕原点逆时针方向 radar_angle 为正。
            2）相机在雷达坐标x轴正方向lx取负
            3）相机在雷达坐标y轴正方向ly取负
            4) R = RT.I * CW
        🦋
        '''
        xcw,ycw = self.pixel2world(u,v)
        x,y = self.camera_projection2radar(radar_angle,lx,ly,xcw,ycw)
        return x,y


# ct = CameraTools(905.8602,516.4283,1626.513816,1624.574619,1200 ,11)
# x,y = ct.pixel2world(500,600)
# print("相机投影坐标：",x,y)

# print('像素坐标:',ct.world2pixel([x,y,0]))

# # print(ct.radar2camera_projection(10,1,2,9,8))
# x,y = ct.radar2camera_projection(10,1,2,9,8)

# print(ct.camera_projection2radar(10,1,2,x,y))
# # print(ct.camera_projection2radar(10,1,2,9,8))


# xr,yr = ct.pixel2radar(500,600,11,1,2)

# print('雷达的值:',xr,yr)

# print('像素的值:')
# print(ct.radar2pixel(xr,yr,11,1,2))
