from sys import setdlopenflags
from numpy.lib import index_tricks
import rospy
import numpy as np

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