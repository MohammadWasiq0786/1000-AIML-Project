"""
Project 237. Face swapping implementation
Description:
Face swapping involves replacing one person's face with another's in an image or video, while preserving expressions and orientation. It’s widely used in entertainment, social media filters, and visual effects. In this project, we’ll implement a basic face swap using OpenCV and Dlib, involving face detection, landmark alignment, and seamless cloning.

About:
What It Does:
This project swaps one face onto another with realistic blending using landmark detection, affine transforms, and seamless cloning. It’s great for deepfake alternatives, fun apps, and content creation tools. For higher accuracy, try advanced libraries like FaceSwap, DeepFaceLab, or use GAN-based approaches for identity-preserving swaps.
"""
# Install required packages:
# pip install opencv-python dlib imutils numpy
 
import cv2
import dlib
import numpy as np
import imutils
 
# Load Dlib’s face detector and shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from dlib model zoo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
 
# Load source and target images
src_img = cv2.imread("face1.jpg")  # Source face
dst_img = cv2.imread("face2.jpg")  # Destination face
 
# Resize for easier processing
src_img = imutils.resize(src_img, width=500)
dst_img = imutils.resize(dst_img, width=500)
 
# Function to get face landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(gray, faces[0]).parts()])
 
# Get landmarks for both faces
src_points = get_landmarks(src_img)
dst_points = get_landmarks(dst_img)
 
if src_points is None or dst_points is None:
    print("❌ Could not detect faces in one or both images.")
    exit()
 
# Compute convex hull to blend only facial region
hull_index = cv2.convexHull(np.array(dst_points), returnPoints=False)
src_hull = [src_points[int(i)] for i in hull_index]
dst_hull = [dst_points[int(i)] for i in hull_index]
 
# Compute Delaunay triangulation on the destination face
size = dst_img.shape
rect = (0, 0, size[1], size[0])
subdiv = cv2.Subdiv2D(rect)
for p in dst_hull:
    subdiv.insert((p[0, 0], p[0, 1]))
triangles = subdiv.getTriangleList()
 
# Convert triangles to indices
def find_index(p, points):
    for i, pt in enumerate(points):
        if abs(p[0] - pt[0, 0]) < 1 and abs(p[1] - pt[0, 1]) < 1:
            return i
    return -1
 
tri_indices = []
for t in triangles:
    pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
    idx = [find_index(p, dst_hull) for p in pts]
    if -1 not in idx:
        tri_indices.append(idx)
 
# Warp each triangle from source to destination
warped_img = np.copy(dst_img)
for tri in tri_indices:
    src_tri = np.float32([src_hull[i] for i in tri])
    dst_tri = np.float32([dst_hull[i] for i in tri])
 
    # Compute affine transform
    matrix = cv2.getAffineTransform(src_tri, dst_tri)
 
    # Apply warp to triangle region
    r_dst = cv2.boundingRect(dst_tri)
    warped_triangle = cv2.warpAffine(
        src_img, matrix, (r_dst[2], r_dst[3]),
        borderMode=cv2.BORDER_REFLECT_101
    )
 
    # Create mask and insert warped triangle into destination image
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri - dst_tri.min(axis=0)), (1, 1, 1), 16)
    warped_img[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] *= (1 - mask)
    warped_img[r_dst[1]:r_dst[1]+r_dst[3], r_dst[0]:r_dst[0]+r_dst[2]] += warped_triangle * mask
 
# Create mask for seamless cloning
hull8U = [(int(p[0, 0]), int(p[0, 1])) for p in dst_hull]
mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
cv2.fillConvexPoly(mask, np.array(hull8U), (255, 255, 255))
 
# Find center for cloning
r = cv2.boundingRect(np.float32([hull8U]))
center = (r[0] + r[2] // 2, r[1] + r[3] // 2)
 
# Use seamless cloning to blend the faces
output = cv2.seamlessClone(warped_img, dst_img, mask, center, cv2.NORMAL_CLONE)
 
# Show the result
cv2.imshow("Face Swapped Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()