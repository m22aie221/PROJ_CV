# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:36:52 2024

@author: ranji
"""

img_folder = 'C:/Users/ranji/Documents/IITJ/Course/Sem3/ComputerVision/Project/CV_PROJ/homo_sim/H_test_im/'



import cv2
import numpy as np


# Load the source and target images
source_image = cv2.imread(img_folder + '71_i110.PNG')
target_image = cv2.imread(img_folder + '71_l6c1.PNG')

# Convert images to Lab color space
source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2Lab)
target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2Lab)

import cv2
import numpy as np

def extract_chromaticity(image):
    # Convert image to Lab color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Compute chromaticity coordinates
    chroma = lab[:, :, 1:] / 255.0
    
    return chroma

def color_transfer(source, target):
    # Extract chromaticity coordinates
    source_chroma = extract_chromaticity(source)
    target_chroma = extract_chromaticity(target)
    
    # Compute homography
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(source, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    
    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Get corresponding points
    source_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography
    H, _ = cv2.findHomography(source_pts, target_pts, cv2.RANSAC, 5.0)
    
    # Apply color transfer
    transferred_chroma = cv2.perspectiveTransform(source_chroma.reshape(1, -1, 2), H).reshape(target_chroma.shape)
    
    # Combine with luminance of target image
    lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    result_lab = np.dstack((lab_target[:, :, 0], (transferred_chroma * 255).astype(np.uint8)))
    
    # Convert back to BGR color space
    #result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_Lab2RGB)
    
    return result_bgr



# Perform color transfer
result_image = color_transfer(source_image, target_image)

# Display result
cv2.imshow('source_image', source_image)
cv2.imshow('target_image', target_image)
cv2.imshow('Color Transferred Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
