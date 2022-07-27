from dis import dis
from re import S, X
from tracemalloc import start
import cv2
import os
import numpy as np
import pickle
import ipdb
import torch
import matplotlib.pyplot as plt
import icp
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

intrinsics = np.array([
            [386.2352294921875,0,327.5399169921875],
            [0,385.2908630371094,240.36749267578125],
            [0,0,1]]).astype(float)

init_post = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).astype(float)

class NICP():
    def __init__(self,intrinsics,root_path,devidend_legth=0.1,device = "cuda:0") -> None:
        self.intrinsics = intrinsics
        self.root_path = root_path
        self.devidend_legth =devidend_legth
        self.plot_init = False
    def read_human_points(self):
        '''
        读取分割之后的人体点云
        '''
        with open(os.path.join(self.root_path,"00000_D2986_C2932.pkl"),'rb') as file:
            self.frame_1 = pickle.load(file)
        with open(os.path.join(self.root_path,"00010_D2996_C2942.pkl"),'rb') as file:
            self.frame_2 = pickle.load(file)
        print(f"load pkl file and pkl file keys {self.frame_1.keys()}")
        self.points_1 = self.frame_1["points"].numpy() # (x,d,y)
        self.points_2 = self.frame_2["points"].numpy()

    def map_uv_to_wh(self,tex,height,width):
        '''
        换算uv 到图像坐标系
        (0,0) is mapped to (0,0)
        (0,1) is mapped to (0,height-1)
        (1,0) is mapped to (width-1,0)
        (1,1) is mapped to (width-1,height-1)
        '''
        y = (width-1)*tex[:,0] # 列
        x = (height-1)*tex[:,1] # 行
        # 返回n*2 维度 (n,0) 行索引 (n,1) 列索引
        return np.vstack((x,y)).astype(int).T

    def filter_tex(self,verts:np.ndarray,tex:np.ndarray):
        # 过滤掉不规范的纹理点
        u_valid = np.logical_and(tex[:,0] > 0,tex[:,0] <= 1)
        v_valid = np.logical_and(tex[:,1] > 0,tex[:,1] <= 1)
        uv_valid = np.logical_and(u_valid == True,v_valid == True)
        uv_index = np.where(uv_valid == True)
        verts = verts[uv_index[0],:]
        tex = tex[uv_index[0],:]
        return verts,tex

    def random_choice(self,points_arr,num_pts):
        '''
        随机采样 返回索引点
        '''
        is_torch = isinstance(points_arr, torch.Tensor)
        N = points_arr.size(0) if is_torch else points_arr.shape[0]
        samp_inds = torch.arange(N) if is_torch else np.arange(N)
        if N == 0:
            return samp_inds
        
        if N > num_pts:
            samp_inds = np.random.choice(np.arange(N), size=num_pts, replace=False)
        elif N < num_pts:
            while samp_inds.shape[0] < num_pts:
                pad_size = num_pts - (samp_inds.size(0) if is_torch else samp_inds.shape[0])
                if is_torch:
                    samp_inds = torch.cat([samp_inds, samp_inds[:pad_size]], dim=0)
                else:
                    samp_inds = np.concatenate([samp_inds, samp_inds[:pad_size]], axis=0)
        return samp_inds

    def read_points(self,down_sample = 4096*5):
        '''
        读取未分割的人体点云
        '''
        self.color_1 = cv2.imread(os.path.join(self.root_path,"00000_D2986_C2932.jpg"))
        # BGR - RGB
        self.color_1 = self.color_1[:,:,[2,1,0]]
        self.color_2 = cv2.imread(os.path.join(self.root_path,"00010_D2996_C2942.jpg"))
        self.color_2 = self.color_2[:,:,[2,1,0]]
        with open(os.path.join(self.root_path,"00000_D2986_C2932.pkl"),'rb') as file:
            self.frame_1 = pickle.load(file)
        with open(os.path.join(self.root_path,"00010_D2996_C2942.pkl"),'rb') as file:
            self.frame_2 = pickle.load(file)
        print(f"load pkl file and pkl file keys {self.frame_1.keys()}")
        # keys : 'verts', 'tex'
        self.points_1 = self.frame_1["verts"][:,[0,2,1]] # (x,d,y)
        self.points_1[:,2]*=-1
        self.points_2 = self.frame_2["verts"][:,[0,2,1]]
        self.points_2[:,2]*=-1
        self.tex_1 = self.frame_1["tex"]
        self.tex_2 = self.frame_2["tex"]
        # 过滤掉不规范的点
        self.points_1,self.tex_1 = self.filter_tex(self.points_1,self.tex_1)
        color_shape = self.color_1.shape
        tex_1_filter_wh = self.map_uv_to_wh(self.tex_1,color_shape[0],color_shape[1])
        self.points_2,self.tex_2 = self.filter_tex(self.points_2,self.tex_2)
        tex_2_filter_wh = self.map_uv_to_wh(self.tex_2,color_shape[0],color_shape[1])

        self.tex_color_1 = self.color_1[tex_1_filter_wh[:,0],tex_1_filter_wh[:,1],:]/255
        self.tex_color_2 = self.color_2[tex_2_filter_wh[:,0],tex_2_filter_wh[:,1],:]/255

        # 下采样
        random_tex_index_1 = self.random_choice(self.points_1,down_sample)
        self.points_1 = self.points_1[random_tex_index_1,:]
        self.tex_1 = self.tex_1[random_tex_index_1,:]
        self.tex_color_1 = self.tex_color_1[random_tex_index_1,:]

        random_tex_index_2 = self.random_choice(self.points_2,down_sample)
        self.points_2 = self.points_2[random_tex_index_2,:]
        self.tex_2 = self.tex_2[random_tex_index_2,:]
        self.tex_color_2 = self.tex_color_2[random_tex_index_2,:]

        self.frame_1["color"] = self.tex_color_1
        self.frame_2["color"] = self.tex_color_2
        self.frame_1["verts"] = self.points_1
        self.frame_2["verts"] = self.points_2
        self.frame_1["tex"] = self.tex_1
        self.frame_2["tex"] = self.tex_2

    def draw(self):
        '''
        dest_point 分割后的点云
        tex_color 目标点云颜色
        '''
        if not self.plot_init:
            fig = plt.figure()
            self.single_ax = plt.axes(projection='3d')
            plt.xlabel("x")
            plt.ylabel("y")
            self.plot_init = True

        x_2 = self.points_1[:,0]
        y_2 = self.points_1[:,1]
        z_2 = self.points_1[:,2]
        self.single_ax.scatter3D(x_2, y_2, z_2,s=0.1,c = self.frame_1["color"])
        self.single_ax.set_xlim3d((-2,2))
        self.single_ax.set_ylim3d((1,4.5))
        self.single_ax.set_zlim3d((-3,1))
        plt.show()

    def sample_point(self,points:np.ndarray,start_coordinate:list,end_coordinate:list):
        '''
        points (N * 3(x,d,y))
        coordinate x,y,d
        '''
        start_x,start_d,start_y = start_coordinate
        end_x,end_d,end_y = end_coordinate
        valid_x = np.logical_and(points[:,0]>=start_x,points[:,0] < end_x)
        valid_d = np.logical_and(points[:,1]>=start_d,points[:,1] < end_d)
        valid_y = np.logical_and(points[:,2]>=start_y,points[:,2] < end_y)
        valid = np.logical_and.reduce([valid_x == True,valid_y == True,valid_d == True])
        valid_index = np.where(valid == True)
        # ipdb.set_trace()
        return points[valid_index,:].squeeze()

    def calculate_nearly_index(self,index:int,xnum:int,ynum:int,dnum:int):
        max_index = xnum*ynum*dnum
        min_index = 1
        table_one = np.array([[index-xnum-1,index-xnum,index-xnum+1],
                                [index-1,index,index+1],
                                [index+xnum-1,index+xnum,index+xnum+1]])
        table_two = np.array([[index-xnum-1-xnum*ynum,index-xnum-xnum*ynum,index-xnum-xnum*ynum+1],
                                [index-xnum*ynum-1,index-xnum*ynum,index-xnum*ynum+1],
                                [index+xnum-1-xnum*ynum,index+xnum-xnum*ynum,index+xnum-xnum*ynum+1]])
        table_thi = np.array([[index-xnum-1+xnum*ynum,index-xnum+xnum*ynum,index-xnum+xnum*ynum+1],
                                [index-1+xnum*ynum,index+xnum*ynum,index+1+xnum*ynum],
                                [index+xnum-1+xnum*ynum,index+xnum+xnum*ynum,index+xnum+xnum*ynum+1]])
        tables = np.vstack([table_two,table_one,table_thi]).flatten()
        valid_ = np.logical_and(tables>=min_index,tables <= max_index)
        tables = tables[valid_]
        return tables

    def devide_point_2_patch(self,points:np.ndarray):
        '''
        
        将点云划分成不同大小的patch
        横轴，深度轴，竖轴
        '''

        x_max,d_max,y_max = np.max(points,axis=0) + self.devidend_legth/2 # 点云边界点
        x_min,d_min,y_min = np.min(points,axis=0) - self.devidend_legth/2
        # 每个轴上的采样个数
        points_nums_x = int((x_max-x_min) / self.devidend_legth)
        points_nums_y = int((y_max-y_min) / self.devidend_legth)
        points_nums_d = int((d_max-d_min) / self.devidend_legth)
        print(f"sample nums is [{[points_nums_x,points_nums_y,points_nums_d]}]")
        x_lines = np.linspace(x_min,x_max,points_nums_x)
        y_lines = np.linspace(y_min,y_max,points_nums_y)
        d_lines = np.linspace(d_min,d_max,points_nums_d)
        # ipdb.set_trace()       
        block_list = {}
        block_index = 0
        f_point_list = None
        points_nums_sum = 0
        for d in range(1,len(d_lines)):
            for y in range(1,len(y_lines)):
                for x in range(1,len(x_lines)):
                    start_d = d_lines[d-1]
                    start_y = y_lines[y-1]
                    start_x = x_lines[x-1]
                    end_d = d_lines[d]
                    end_y = y_lines[y]
                    end_x = x_lines[x]
                    block = {}
                    block["index"] = block_index
                    tables = self.calculate_nearly_index(block_index,points_nums_x,points_nums_y,points_nums_d)
                    block["nearly_index"] = tables
                    block["start"] = [start_x,start_d,start_y]
                    block["end"] = [end_x,end_d,end_y]
                    block["center"] = [(start_x + end_x) / 2,(start_d + end_d) / 2,(start_y + end_y) / 2]
                    f_points = self.sample_point(points,block["start"],block["end"]).reshape(-1,3)
                    if len(f_points) > 0:
                        block["points_set"] = f_points
                        points_nums_sum += len(f_points)
                        block["points_num_sum"] = points_nums_sum
                        # print(f"from point ([{x,d,y}]) to point ({x+self.devidend_legth,d+self.devidend_legth,y+self.devidend_legth}) have [{len(f_points)}] points!")
                        block_list[block_index] = block
                        if f_point_list is None:
                            f_point_list = f_points
                        else:
                            f_point_list = np.vstack([f_point_list,f_points])

                    block_index += 1
        # for d in d_lines:
        #     for y in y_lines:
        #         for x in x_lines:
        #             block = {}
        #             block["index"] = block_index
        #             tables = self.calculate_nearly_index(block_index,points_nums_x,points_nums_y,points_nums_d)
        #             block["nearly_index"] = tables
        #             block["start"] = [x,d,y]
        #             block["end"] = [x+self.devidend_legth,d+self.devidend_legth,y+self.devidend_legth]
        #             block["center"] = [x+self.devidend_legth/2,d+self.devidend_legth/2,y+self.devidend_legth/2]
        #             f_points = self.sample_point(points,block["start"],block["end"])
        #             if len(f_points) > 0:
        #                 block["points_set"] = f_points
        #                 print(f"from point ([{x,d,y}]) to point ({x+self.devidend_legth,d+self.devidend_legth,y+self.devidend_legth}) have [{len(f_points)}] points!")
        #                 block_list[block_index] = block
        #             block_index += 1
        return block_list,f_point_list

    def get_points_1(self):
        return self.points_1,self.frame_1["tex"]
    def get_points_2(self):
        return self.points_2,self.frame_2["tex"]

def resize_points(points_arr, num_pts):
    '''
    下采样
    Either randomly subsamples or pads the given points_arr to be of the desired size.
    - points_arr : N x 3
    - num_pts : desired num point
    '''
    is_torch = isinstance(points_arr, torch.Tensor)
    N = points_arr.size(0) if is_torch else points_arr.shape[0]
    if N > num_pts:
        samp_inds = np.random.choice(np.arange(N), size=num_pts, replace=False)
        points_arr = points_arr[samp_inds]
    elif N < num_pts:
        while N < num_pts:
            pad_size = num_pts - N
            if is_torch:
                points_arr = torch.cat([points_arr, points_arr[:pad_size]], dim=0)
                N = points_arr.size(0)
            else:
                points_arr = np.concatenate([points_arr, points_arr[:pad_size]], axis=0)
                N = points_arr.shape[0]
    return points_arr

def down_sample(first_point_cloud:np.ndarray,second_point_cloud:np.ndarray):
    N = 0
    first_point_cloud = first_point_cloud.reshape(-1,3)
    second_point_cloud = second_point_cloud.reshape(-1,3)
    if first_point_cloud.shape[0] != second_point_cloud.shape[0]:
        if first_point_cloud.shape[0] < second_point_cloud.shape[0]:
            N = first_point_cloud.shape[0]
            second_point_cloud = resize_points(second_point_cloud, first_point_cloud.shape[0])
        else:
            N = second_point_cloud.shape[0]
            first_point_cloud = resize_points(first_point_cloud, second_point_cloud.shape[0])
    return first_point_cloud,second_point_cloud

def icp_two_point_clouds(first_point_cloud:np.ndarray,second_point_cloud:np.ndarray,init_post=init_post):
    first_point_cloud,second_point_cloud = down_sample(first_point_cloud,second_point_cloud)
    # print(f"N is {N} first point shape {first_point_cloud.shape} second point shape {second_point_cloud.shape}")
    T, distance, i = icp.icp(first_point_cloud, second_point_cloud,init_post, max_iterations=500, tolerance=1e-5)
    # print("iterator nums is ",i,"distance is ", distance,"sum is ",np.sum(distance))
    return T,distance,i

def transform_points(points:np.ndarray,T:np.ndarray):
    # ipdb.set_trace()
    # Transform C
    N = points.shape[0]
    C = np.ones((N, 4))
    C[:, 0:3] = np.copy(points)
    # 旋转矩阵 与下面两行等价
    C = np.dot(T, C.T).T[:, :3]
    # 旋转
    # C = np.dot(R1, C.T).T
    # 平移
    # C += t1
    return C

def icp_two_v2(points_1_patch:dict,points_2_patch:dict):
    for k,v in points_1_patch.items():
        points = v["points_set"]
        tables = v["nearly_index"]
        T_dict = {}
        T_list = []
        distance_list = []
        for index in tables:
            if index in points_2_patch.keys():
                s_points = points_2_patch[index]["points_set"]
                T,distance,i = icp_two_point_clouds(points,s_points)
                T_dict[index] = T
                T_list.append(T)
                temp_distance = np.sum(distance)
                distance_list.append(temp_distance)
        # ipdb.set_trace()
        if len(T_list) > 0:
            
            # ipdb.set_trace()
            # weight_T = weight * np.array(T_list)
            d_i = np.argmin(distance_list)
            weight_T = T_list[d_i]
            # print(f"get min distance [{distance_list}]:[{distance_list[d_i]}] ")
            points_1_patch[k]["distance"] = distance_list[d_i]
            points_1_patch[k]["weight_T"] = weight_T
            points_1_patch[k]["T"] = T_dict
            after_trans = transform_points(points,weight_T)
            points_1_patch[k]["after_trans"] = after_trans
        else:
            points_1_patch[k]["distance"] = 0
            points_1_patch[k]["weight_T"] = init_post
            points_1_patch[k]["T"] = []
            points_1_patch[k]["after_trans"] = points
    return points_1_patch

def icp_two(points_1_patch:dict,points_2_patch:dict):
    for k,v in points_1_patch.items():
        points = v["points_set"]
        tables = v["nearly_index"]
        T_dict = {}
        T_list = []
        weight = []
        sum_distance = 0
        for index in tables:
            if index in points_2_patch.keys():
                s_points = points_2_patch[index]["points_set"]
                T,distance,i = icp_two_point_clouds(points,s_points)
                T_dict[index] = T
                T_list.append(T)
                temp_distance = np.sum(distance)
                temp_weight = 1/(1 + np.exp(temp_distance))
                weight.append(temp_weight)
        # ipdb.set_trace()
        if len(T_list) > 0:
            weight = np.array(weight)/np.linalg.norm(np.array(weight),2)
            weight /= np.sum(weight)
            print(f"get mean transformation matrix and weight is [{weight}] ")
            # ipdb.set_trace()
            weight_T = np.zeros((4,4))
            for index,weight_temp in enumerate(weight):
                weight_T += weight_temp*T_list[index]
            # weight_T = weight * np.array(T_list)
            points_1_patch[k]["weight_T"] = weight_T
            points_1_patch[k]["T"] = T_dict
            after_trans = transform_points(points,weight_T)
            points_1_patch[k]["after_trans"] = after_trans
        else:
            points_1_patch[k]["weight_T"] = init_post
            points_1_patch[k]["T"] = []
            points_1_patch[k]["after_trans"] = points


def draw(points):
    '''
    dest_point 分割后的点云
    tex_color 目标点云颜色
    '''
    fig = plt.figure()
    single_ax = plt.axes(projection='3d')
    plt.xlabel("x")
    plt.ylabel("y")

    x_2 = points[:,0]
    y_2 = points[:,1]
    z_2 = points[:,2]
    single_ax.scatter3D(x_2, y_2, z_2,s=0.1)
    single_ax.set_xlim3d((-2,2))
    single_ax.set_ylim3d((1,4.5))
    single_ax.set_zlim3d((-3,1))
    plt.show()

def draw_2(points_one,points_two):
    '''
    dest_point 分割后的点云
    tex_color 目标点云颜色
    '''
    fig = plt.figure(figsize=(12,6))
    single_ax = fig.add_subplot(121, projection='3d')
    second_ax = fig.add_subplot(122, projection='3d')
    plt.xlabel("x")
    plt.ylabel("y")

    x_2 = points_one[:,0]
    y_2 = points_one[:,1]
    z_2 = points_one[:,2]

    x_3 = points_two[:,0]
    y_3 = points_two[:,1]
    z_3 = points_two[:,2]

    single_ax.scatter3D(x_2, y_2, z_2,s=0.1,c='r')
    second_ax.scatter3D(x_3, y_3, z_3,s=0.1,c='y')
    single_ax.set_xlim3d((-2,2))
    single_ax.set_ylim3d((1,4.5))
    single_ax.set_zlim3d((-3,1))
    second_ax.set_xlim3d((-2,2))
    second_ax.set_ylim3d((1,4.5))
    second_ax.set_zlim3d((-3,1))
    plt.show()
    
def calculate_diff(points_1_patch:dict,points_2_patch : dict):
    points_list_temp = None
    distance_sum = 0
    for k,v in points_1_patch.items():
        after_trans = points_1_patch[k]["after_trans"] if "after_trans" in points_1_patch[k].keys() else points_1_patch[k]["points_set"]
        distance_sum += points_1_patch[k]["distance"] if "distance" in points_1_patch[k].keys() else 0
        if points_list_temp is None:
            points_list_temp = after_trans
        else:
            points_list_temp = np.vstack([points_list_temp,after_trans])

    points_origin_2 = None
    for k,v in points_2_patch.items():
        origin_points = points_2_patch[k]["points_set"]
        if points_origin_2 is None:
            points_origin_2 = origin_points
        else:
            points_origin_2 = np.vstack([points_origin_2,origin_points])

    diff = points_list_temp - points_origin_2
    print(f"################### distance sum is {distance_sum} and diff is {np.sqrt(np.sum(diff**2))}")
    return points_list_temp,points_origin_2

if __name__ == "__main__":
    nicp = NICP(intrinsics,"./data",0.02)
    # nicp.read_points()
    nicp.read_human_points()
    # nicp.draw()
    points_1,tex_1 = nicp.get_points_1()
    points_2,tex_2 = nicp.get_points_2()
    points_1,points_2 = down_sample(points_1,points_2)
    points_1_patch,points_1 = nicp.devide_point_2_patch(points_1)
    points_2_patch,points_2 = nicp.devide_point_2_patch(points_2)

    calculate_diff(points_1_patch,points_2_patch)
    # ipdb.set_trace()
    points_1_patch = icp_two_v2(points_1_patch,points_2_patch)

    points_list_temp,points_origin_2 = calculate_diff(points_1_patch,points_2_patch)

    draw_2(points_2,points_list_temp)
