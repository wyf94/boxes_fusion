# -*-coding:utf-8-*-
import numpy as np
import math

'''
功能: 实现各个坐标系之间的转换(直接得到转换关系版)
最后修改时间: 2021/09/16
'''

class CameraTools(object):
    def __init__(self, T, H):
        # T: 像素转世界转换矩阵
        self.T = np.asarray(T)
        # H: 像素转世界转换矩阵
        self.H = np.asarray(H)

    def camera_projection2pixel(self, x, y):
        '''
        功能: 实现相机投影(默认为世界坐标)转像素
        参数: 
            x: 目标的横行距离,单位:米
            y: 目标的纵向距离,单位:米

        返回值: 
            u: 像素坐标系u方向的值
            v: 像素坐标系v方向的值
        '''
        rst = np.dot(self.T, np.array([x, y, 1]).T)
        # print('rst:',rst)
        u = rst[0] / rst[2]
        v = rst[1] / rst[2]

        return int(u), int(v)

    def pixel2camera_projection(self, u, v):
        '''
        功能: 实现像素转世界坐标功能

        参数:
            u: 像素坐标系u方向的值
            v: 像素坐标系v方向的值
        返回值: 
            x: 目标的横行距离,单位:米
            y: 目标的纵向距离,单位:米
        '''
        rst = np.dot(self.H, np.array([u, v, 1]).T)
        # print('H:',H)
        x = rst[0] / rst[2]
        y = rst[1] / rst[2]

        return x, y

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

        return x, y

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

        return x, y

    def radar2pixel(self, xr, yr, radar_angle, lx, ly):
        '''
        🦋

        功能: 雷达坐标系转像素坐标系

        参数: 
            xr: 雷达x方向的坐标值,单位:米
            yr: 雷达y方向的坐标值,单位:米
            radar_angle: 雷达转相机投影坐标的旋转角度
            lx: 雷达坐标转相机投影坐标系x方向的位移,单位:米
            ly: 雷达坐标转相机投影坐标系y方向的位移,单位:米

        返回值:
            u,v: 像素坐标
        🦋
        '''

        xcw, ycw = self.radar2camera_projection(radar_angle, lx, ly, xr, yr)
        # 建立相机投影坐标，z=0
        # joint_world = [xcw,ycw,0]
        u, v = self.camera_projection2pixel(xcw, ycw)

        return int(u), int(v)

    def pixel2radar(self, u, v, radar_angle, lx, ly):
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
        xcw, ycw = self.pixel2camera_projection(u, v)
        x, y = self.camera_projection2radar(radar_angle, lx, ly, xcw, ycw)
        return x, y


'''
测试代码


# 距离转像素矩阵
T = [[4747.90330564942,	581.787506791903,	104.333339893983],
    [81.9906969275870,	223.639112481143,	33088.3393285115],
    [-0.0139641472013332,	0.463836426989393,	1]
]

# 像素转距离矩阵
H = [[-0.0146272247905266,	-0.000469236200641869,	18.5587158710807],
    [8.36487005907202e-05,	-0.000110763741803657,	-147.518457762086],
    [4.29972972849467e-05,	-0.00215935697523353,	1]]

u = 1729 
v = 903

ct = CameraTools(T,H)

x,y = ct.pixel2camera_projection(u,v)
print("像素转相机投影:")
print('x:',x)
print('y:',y)
u,v = ct.camera_projection2pixel(x,y)
print("相机投影转像素:")
print('u:',u)
print('v:',v)

u,v = ct.radar2pixel(x,y,0,0,0)
print("雷达转像素:")
print('u:',u)
print('v:',v)

x,y = ct.pixel2radar(u,v,0,0,0)
print("像素转雷达:")
print('x:',x)
print('y:',y)


'''
