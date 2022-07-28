from dis import dis
from math import ceil
import os
import torch
import numpy as np
import pickle
import ipdb
from sklearn.neighbors import KDTree
import open3d as o3d
import timeit


class NICP():
    def __init__(self,frame_one,frame_two,max_iterator = 5,lr = 0.001,device = "cuda:0",tolerance = 5) -> None:
       
        self.device = device
        self.points_one = frame_one["points"].numpy()
        self.points_two = frame_two["points"].numpy()
        self.color_one = frame_one["color"]
        self.color_two = frame_two["color"]
        self.max_iterator = max_iterator
        self.lr = lr
        self.tolerance = tolerance

    def downsample(self,points_arr, num_pts):
        '''
        下采样
        Either randomly subsamples or pads the given points_arr to be of the desired size.
        - points_arr : N x 3
        - num_pts : desired num point
        '''
        is_torch = isinstance(points_arr, torch.Tensor)
        N = points_arr.size(0) if is_torch else points_arr.shape[0]
        print(f"### down sample point cloud [{N}] to [{num_pts}]")
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

    def downsample_index(self,points_arr,num_pts):
        '''
        随机采样 返回索引点
        '''
        is_torch = isinstance(points_arr, torch.Tensor)
        N = points_arr.size(0) if is_torch else points_arr.shape[0]
        print(f"### down sample point cloud [{N}] to [{num_pts}]")
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

    def get_nearly_points(self,points_one:np.ndarray,points_two:np.ndarray,K=4,dis_threshold = 0.1):
        '''
        v1:
        获取临近点,使用KDtree
        K 表示获取临近点的数量 默认是4个
        v2:
        此处没有使用查找半径
        可以使用半径阈值，对于距离超过这个阈值可以视为没有连接点
        '''
        tree = KDTree(points_two,leaf_size = ceil(K/2)) # 使用点云2构建KDtree
        dist,ind = tree.query(points_one,k=K) # 用点云1查询临近点
        # ipdb.set_trace()
        print(f"### query points min distance {np.min(dist)} max distance [{np.max(dist)}]")
        points_num = points_one.shape[0] # sample_num  = 4096
        filter_dist = dist < dis_threshold
        lines_num = np.sum(filter_dist)
        M = np.zeros((lines_num,points_num)) # (m * n) = (3n * n) m条边 n个顶点
        # 构建 M矩阵
        line_index = 0
        for row in range(points_num):
            for i in range(0,ind.shape[1]):
                if filter_dist[row,i]:
                    M[line_index,row] = 1
                    M[line_index,i] = -1
                    line_index += 1

        assert K == 4
        '''
        如果K!=4 下面验证是否有距离小于阈值的临近点的判断条件就需要改
        '''
        valid = np.logical_or.reduce([dist[:,0] < dis_threshold,dist[:,1] < dis_threshold,dist[:,2] < dis_threshold,dist[:,3] < dis_threshold])
        W = np.diag(valid.astype(int)) # n *n 的权重矩阵，如果点的距离大于阈值 那么权重值为0否则为1 表示有临近点
        print(f"### build node-arc matrix shape is [{M.shape}]")
        print(f"### build W matrix shape is [{W.shape}]")
        return M,ind,W

    def create_graph(self,points_one:np.ndarray,points_two:np.ndarray,G_lam = 0.5,K = 4,dis_threshold = 0.1):
        '''
        构建图，并创建图矩阵 M
        更具距离阈值创建 W矩阵 
        '''
        G = np.diag([1,1,1,G_lam])
        M,ind,W = self.get_nearly_points(points_one,points_two,K,dis_threshold)
        matrix_up = np.kron(M,G) # (m * n ) \otimes (4*4) = 4m * 4n
        # self.draw_graph(points,ind)
        return matrix_up,W

    def draw_graph(self,points,lines):
        '''
        绘制图
        '''
        points_num = points.shape[0]
        lines_matrix = np.zeros((3*points_num,2)) 
        
        for i in range(1,lines.shape[1]):
            lines_matrix[lines[:,0]+points_num*(i-1),:] = lines[:,[0,i]]
        graph_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines_matrix),
        )
        colors = [[0, 0, 0] for i in range(len(lines_matrix))]
        graph_set.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([graph_set])
    
    def build_distance_matrix(self,points:np.ndarray):
        '''
        构建 距离项 矩阵D n* 4n
        '''
        points_num = points.shape[0]
        D = np.zeros((points_num,4*points_num))
        one = np.ones((points_num,1))
        points = np.hstack((points,one))
        for i in range(points_num):
            # 0 0,1,2,3
            # 1 4,5,6,7
            D[i,[4*i,4*i+1,4*i+2,4*i+3]] = points[i]
        return D

    def precess_with_color(self,down_sample = 4096,G_lam = 0.5,K = 4,dis_threshold = 0.1):
        # 更具索引下采样，保留颜色信息
        one_index = self.downsample_index(self.points_one,down_sample)
        points_one = self.points_one[one_index,:]
        color_one = self.color_one[one_index,:]

        # 构建矩阵 U # 更具索引下采样，保留颜色信息
        two_index = self.downsample_index(self.points_two,down_sample)
        U = self.points_two[two_index,:]  # n * 3
        color_two = self.color_two[two_index,:]

        # 构建图 返回M x G 矩阵
        matrix_up,W = self.create_graph(points_one,U,G_lam,K,dis_threshold) # 4m * 4n
        # 构建矩阵 D
        matrix_down = self.build_distance_matrix(points_one) # n* 4n
        

        # W = np.random.randn(down_sample,down_sample*4) # n* 4n
        A = np.vstack((matrix_up,matrix_down)) # (4m+n) * 4n
        B = np.vstack((np.zeros((matrix_up.shape[0],3)),U)) # (4m+n) * 3
        # inv(4n*4n) * (4n * (4m+n)) *(4m+n * 3) = 4n*3
        # 每4行3列是一个旋转矩阵
        device = "cuda:0"
        A = torch.from_numpy(A).to(device)
        B = torch.from_numpy(B).to(device)
        W = torch.mm(torch.inverse(torch.mm(A.T,A)),torch.mm(A.T,B)).cpu().numpy()
        # W = np.linalg.inv(A.T @ A) @ (A.T @ B)
        return points_one,color_one,U,color_two,W

    def process_cpu(self,down_sample = 4096,lam = 0.5,K = 4):
        # 下采样
        points_one = self.downsample(self.points_one,down_sample)
        # 构建矩阵 U
        U = self.downsample(self.points_two,down_sample) # n * 3
        # 构建图 返回M x G 矩阵
        matrix_up,W = self.create_graph(points_one,U,lam,K,dis_threshold = 0.001) # 4m * 4n
        # 构建矩阵 D
        matrix_down = self.build_distance_matrix(points_one) # n* 4n
        # X每4行3列是一个旋转矩阵
        X = np.random.rand(4*down_sample,3) # 4*n * 3
        alpha = np.linspace(0.1,1,10)[::-1] # 刚性系数
        # min(||AX-B||)
        B = np.vstack((np.zeros((matrix_up.shape[0],3)),W @ U,U)) # (4m+n) * 3
        
        for a in alpha:
            X_temp = np.copy(X) # 4*n * 3
            A = np.vstack((a * matrix_up,W @ matrix_down,matrix_down)) # (4m+n) * 4n
            for i in range(self.max_iterator):

                # inv(4n*4n) * (4n * (4m+n)) *(4m+n * 3) = 4n*3
                loss = np.sqrt(np.sum(np.power(A @ X_temp - B,2))) # ||AX-B||^2
                # E'(x) = A.T(AX-B)
                X_temp -= self.lr * (A.T @ (A @ X_temp - B)) # 4n * 3

                diff = np.sqrt(np.sum(np.power(X_temp - X,2)))

                print(f"#### alpha is [{a}] inner iterator nums [{i}] loss is [{loss}] and X diff is [{diff}]")
                '''
                原文这里使用的是 前后旋转矩阵的差值diff小于阈值既可以退出,这里改成了点集差 loss
                '''
                if loss < self.tolerance:
                    break
            X = X_temp

        return points_one,U,X

    def precess_torch(self,down_sample = 4096,lam = 0.5,K = 4):
        # 下采样
        points_one = self.downsample(self.points_one,down_sample)
        # 对点云2 下采样 同时 构建矩阵 U
        U = self.downsample(self.points_two,down_sample) # n * 3
        # 构建图 返回M x G 矩阵
        matrix_up,W = self.create_graph(points_one,U,lam,K,dis_threshold = 0.001) # matrix_up 4m * 4n  W: n* n
        # 构建矩阵 D
        matrix_down = self.build_distance_matrix(points_one) # n* 4n

        X = np.random.rand(4*down_sample,3) # 4*n * 3
        alpha = np.linspace(0.1,1,10)[::-1] # 刚性系数
        # min(||AX-B||)
        # ipdb.set_trace()
        matrix_up = torch.from_numpy(matrix_up).float().to(self.device)
        matrix_down = torch.from_numpy(matrix_down).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)
        U = torch.from_numpy(U).float().to(self.device)
        X = torch.from_numpy(X).float().to(self.device)

        B = torch.cat((torch.zeros((matrix_up.shape[0],3)).to(self.device) , torch.mm(W,U),U),dim = 0) # (4m+n) * 3

        for a in alpha:
            X_temp = torch.clone(X).to(self.device) # 4*n * 3
            A = torch.cat((a * matrix_up , torch.mm(W,matrix_down),matrix_down),dim = 0) # (4m+n) * 4n
            
            for i in range(self.max_iterator):
                # inv(4n*4n) * (4n * (4m+n)) *(4m+n * 3) = 4n*3
                # 每4行3列是一个旋转矩阵
                loss = torch.sqrt(torch.sum(torch.pow(torch.mm(A,X_temp) - B,2)))
                # ipdb.set_trace()
                # X_temp -= self.lr * torch.mm(torch.linalg.inv(torch.mm(A.T,A)),torch.mm(A.T,B)) # 4n * 3
                X_temp -= self.lr * torch.mm(A.T,torch.mm(A,X_temp) - B) # 4n * 3

                diff = torch.sqrt(torch.sum(torch.pow(X_temp - X,2)))
                print(f"#### alpha is [{a}] inner iterator nums [{i}] loss is [{loss}] and X diff is [{diff}]")
                if loss < self.tolerance:
                    break
            X = X_temp

        U = U.cpu().numpy()
        X = X.cpu().numpy()
        return points_one,U,X

def draw_points(points_one,colors_one,points_two,colors_two):
    pcd_one = o3d.geometry.PointCloud()
    pcd_two = o3d.geometry.PointCloud()
    if colors_one is None:
        colors_one = [[0, 0, 0] for i in range(len(points_one))]
    pcd_one.points = o3d.utility.Vector3dVector(points_one)
    pcd_one.colors = o3d.utility.Vector3dVector(colors_one)

    if colors_two is None:
        colors_two = [[0, 1, 0] for i in range(len(points_two))]
    pcd_two.points = o3d.utility.Vector3dVector(points_two)
    pcd_two.colors = o3d.utility.Vector3dVector(colors_two)

    # pcd_one.estimate_normals()
    # distances = pcd_one.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_one,o3d.utility.DoubleVector([radius, radius * 2]))
    # o3d.visualization.draw_geometries([mesh])

def nicp_v2():
    root_path = os.path.join("./data")
    with open(os.path.join(root_path,"00000_D2986_C2932.pkl"),'rb') as file:
        frame_one = pickle.load(file)

    with open(os.path.join(root_path,"00005_D2991_C2937.pkl"),'rb') as file:
        frame_two = pickle.load(file)
    print(f"load pkl file and pkl file keys {frame_one.keys()}")
    points_one = frame_one["points"].numpy()
    points_two = frame_two["points"].numpy()
    nicp = NICP(frame_one,frame_two)
    points_one,color_one,points_two,color_two,W = nicp.precess_with_color(down_sample=4096)
    # n * 3 W = 4n * 3
    one = np.ones((points_one.shape[0],1))
    temp_points = np.hstack((points_one,one))

    new_points_list = []

    for index,row in enumerate(temp_points):
        row = row.reshape(1,-1)
        new_points = row @ W[index*4:index*4+4,:] # 1*4 4*3
        new_points_list.append(new_points)

    new_points_list = np.array(new_points_list).squeeze()
    # ipdb.set_trace()
    diff = new_points_list - points_two
    diff = np.sqrt(np.sum(diff ** 2))
    print(f"### transformation diff is {diff}")
    # draw_points(new_points_list,color_one,points_two,color_two)
    
    with open("./output/points_one.pkl",'wb') as file:
        points_write = {}
        points_write["points"] = points_one
        points_write["color"] = color_one
        points_write["transform_points"] = new_points_list
        pickle.dump(points_write,file)

    with open("./output/points_two.pkl",'wb') as file:
        points_write = {}
        points_write["points"] = points_two
        points_write["color"] = color_two
        pickle.dump(points_write,file)

def nicp():
    root_path = os.path.join("./data")
    with open(os.path.join(root_path,"00000_D2986_C2932.pkl"),'rb') as file:
        frame_one = pickle.load(file)

    with open(os.path.join(root_path,"00050_D3036_C2982.pkl"),'rb') as file:
        frame_two = pickle.load(file)
    print(f"load pkl file and pkl file keys {frame_one.keys()}")
    points_one = frame_one["points"].numpy()
    points_two = frame_two["points"].numpy()
    start = timeit.default_timer()
    nicp = NICP(frame_one,frame_two,max_iterator=200)
    points_one,points_two,W = nicp.precess_torch(down_sample=4096*3)
    print(f"##### nicp time [{(timeit.default_timer() - start)*1000}] ms")
    

    matrix_down = nicp.build_distance_matrix(points_one) # D矩阵 n* 4n 
    # matrix_down (n * 4n) * W(4n * 3) = n * 3
    new_points_list = matrix_down @ W

    diff = new_points_list - points_two
    diff = np.sqrt(np.sum(diff ** 2))
    print(f"### transformation diff is {diff}")
    # draw_points(new_points_list,color_one,points_two,color_two)
    
    with open("./output/points_one.pkl",'wb') as file:
        points_write = {}
        points_write["points"] = points_one
        points_write["transform_points"] = new_points_list
        pickle.dump(points_write,file)

    with open("./output/points_two.pkl",'wb') as file:
        points_write = {}
        points_write["points"] = points_two
        pickle.dump(points_write,file)

if __name__ == "__main__":
   nicp()