import os
import pickle
import open3d as o3d
import numpy as np

root = os.path.join("./output_0_50")
with open(os.path.join(root,"points_one.pkl"),'rb') as file:
    frame_one = pickle.load(file)

with open(os.path.join(root,"points_two.pkl"),'rb') as file:
    frame_two = pickle.load(file)



def draw_points():
    points_one = frame_one["transform_points"]
    origin_points = frame_one["points"]
    color_one = frame_one["color"]
    points_two = frame_two["points"]
    color_two = frame_two["color"]
    pcd_one = o3d.geometry.PointCloud()
    origin = o3d.geometry.PointCloud()
    pcd_two = o3d.geometry.PointCloud()
    if colors_one is None:
        colors_one = [[0, 0, 0] for i in range(len(points_one))]
    pcd_one.points = o3d.utility.Vector3dVector(points_one)
    pcd_one.colors = o3d.utility.Vector3dVector(colors_one)

    if colors_two is None:
        colors_two = [[0, 1, 0] for i in range(len(points_two))]
    pcd_two.points = o3d.utility.Vector3dVector(points_two)
    pcd_two.colors = o3d.utility.Vector3dVector(colors_two)

    origin.points = o3d.utility.Vector3dVector(origin_points)
    origin.colors = o3d.utility.Vector3dVector(colors_one)

    o3d.visualization.draw_geometries([origin])
    o3d.visualization.draw_geometries([pcd_one])
    o3d.visualization.draw_geometries([pcd_two])

def draw_points_2():
    points_one = frame_one["transform_points"]
    origin_points = frame_one["points"]
    points_two = frame_two["points"]
    colors_one = None
    pcd_one = o3d.geometry.PointCloud()
    origin = o3d.geometry.PointCloud()
    pcd_two = o3d.geometry.PointCloud()
    if colors_one is None:
        colors_one = [[0, 0, 0] for i in range(len(points_one))]
    pcd_one.points = o3d.utility.Vector3dVector(points_one)
    pcd_one.colors = o3d.utility.Vector3dVector(colors_one)
    colors_two = None
    if colors_two is None:
        colors_two = [[0, 1, 0] for i in range(len(points_two))]
    pcd_two.points = o3d.utility.Vector3dVector(points_two)
    pcd_two.colors = o3d.utility.Vector3dVector(colors_two)

    origin.points = o3d.utility.Vector3dVector(origin_points)
    origin.colors = o3d.utility.Vector3dVector(colors_one)

    # o3d.visualization.draw_geometries([origin,pcd_two])
    # o3d.visualization.draw_geometries([pcd_one,pcd_two])
    o3d.visualization.draw_geometries([pcd_one])
    o3d.visualization.draw_geometries([pcd_two])

    # pcd_one.estimate_normals()
    # distances_one = pcd_one.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances_one)
    # radius = 1.5 * avg_dist   
    # print(radius)
    # tetra_mesh_1, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_one)
    # mesh_one = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_one,0.02,tetra_mesh_1,pt_map)
    # mesh_one.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_one], mesh_show_back_face=True)

    pcd_two.estimate_normals()
    distances_two = pcd_two.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances_two)
    radius = 1.5 * avg_dist   
    print(radius)
    tetra_mesh_2, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_two)
    mesh_two = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_two,20,tetra_mesh_2,pt_map)
    mesh_two.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_two], mesh_show_back_face=True)

draw_points_2()