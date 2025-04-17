"""
Project 589: 3D Point Cloud Processing
Description:
3D point cloud processing involves working with sets of data points in three-dimensional space. These point clouds are used in applications like 3D reconstruction, robotics, and autonomous driving. In this project, we will implement 3D point cloud processing using techniques like point cloud segmentation and point cloud classification, utilizing libraries such as Open3D or PyTorch3D.
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Load a 3D point cloud (assuming you have a .ply or .pcd file)
point_cloud = o3d.io.read_point_cloud("path_to_point_cloud.ply")  # Replace with actual file path
 
# 2. Visualize the original point cloud
o3d.visualization.draw_geometries([point_cloud], window_name="Original Point Cloud")
 
# 3. Perform voxel downsampling to reduce the size of the point cloud (simplifying for processing)
voxel_size = 0.05  # Voxel size for downsampling
downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
 
# 4. Visualize the downsampled point cloud
o3d.visualization.draw_geometries([downsampled_point_cloud], window_name="Downsampled Point Cloud")
 
# 5. Segment the point cloud using plane segmentation (e.g., finding a flat surface like a table)
plane_model, inliers = downsampled_point_cloud.segment_plane(distance_threshold=0.01, 
                                                             ransac_n=3, 
                                                             num_iterations=1000)
 
# 6. Extract the inliers (points belonging to the detected plane) and outliers (remaining points)
inlier_cloud = downsampled_point_cloud.select_by_index(inliers)
outlier_cloud = downsampled_point_cloud.select_by_index(inliers, invert=True)
 
# 7. Visualize the segmentation result
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for inliers (plane)
outlier_cloud.paint_uniform_color([0, 1.0, 0])  # Green for outliers (objects)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Point Cloud Segmentation")
 
# 8. Optionally, save the segmented point clouds to files
o3d.io.write_point_cloud("segmented_plane.ply", inlier_cloud)
o3d.io.write_point_cloud("segmented_objects.ply", outlier_cloud)