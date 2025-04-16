"""
Project 234. 3D reconstruction from images
Description:
3D reconstruction involves creating a three-dimensional representation of an object or scene using one or more 2D images. It is widely used in robotics, AR/VR, gaming, architecture, and digital heritage preservation. In this project, we’ll demonstrate multi-view 3D reconstruction using COLMAP, a popular SfM (Structure-from-Motion) and MVS (Multi-View Stereo) pipeline.

About:
What It Does:
This project takes a folder of images from different angles of an object or scene and reconstructs a 3D mesh model (PLY format). It’s used in photogrammetry, 3D modeling, surveying, and VR world building. For Python-native deep learning alternatives, explore tools like NeRF (Neural Radiance Fields) or OpenMVG/OpenMVS.
"""
import os
import subprocess
 
# Paths (adjust accordingly)
colmap_path = "colmap"  # or full path to colmap binary
images_path = "./images"  # Folder containing multi-view input images
workspace_path = "./3d_model_output"
 
# Step 1: Feature extraction
subprocess.run([
    colmap_path, "feature_extractor",
    "--database_path", f"{workspace_path}/database.db",
    "--image_path", images_path
])
 
# Step 2: Feature matching
subprocess.run([
    colmap_path, "exhaustive_matcher",
    "--database_path", f"{workspace_path}/database.db"
])
 
# Step 3: Sparse reconstruction (Structure from Motion)
os.makedirs(f"{workspace_path}/sparse", exist_ok=True)
subprocess.run([
    colmap_path, "mapper",
    "--database_path", f"{workspace_path}/database.db",
    "--image_path", images_path,
    "--output_path", f"{workspace_path}/sparse"
])
 
# Step 4: Dense reconstruction
os.makedirs(f"{workspace_path}/dense", exist_ok=True)
 
# Undistort images
subprocess.run([
    colmap_path, "image_undistorter",
    "--image_path", images_path,
    "--input_path", f"{workspace_path}/sparse/0",
    "--output_path", f"{workspace_path}/dense",
    "--output_type", "COLMAP"
])
 
# Stereo depth maps
subprocess.run([
    colmap_path, "patch_match_stereo",
    "--workspace_path", f"{workspace_path}/dense",
    "--workspace_format", "COLMAP",
    "--PatchMatchStereo.geom_consistency", "true"
])
 
# Step 5: Fuse depth maps into 3D mesh
subprocess.run([
    colmap_path, "stereo_fusion",
    "--workspace_path", f"{workspace_path}/dense",
    "--workspace_format", "COLMAP",
    "--input_type", "geometric",
    "--output_path", f"{workspace_path}/dense/fused.ply"
])
 
print("\n🧱 3D Reconstruction Completed!")
print(f"🔍 You can now open the PLY model in MeshLab or Blender from:\n{workspace_path}/dense/fused.ply")