# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:51:32 2024

@author: ranji
"""

import cv2
import numpy as np

img_folder = 'C:/Users/ranji/Documents/IITJ/Course/Sem3/ComputerVision/Project/CV_PROJ/homo_sim/H_test_im/'


# Load the source and target images
source_image = cv2.imread(img_folder +'71_i110.PNG')
target_image = cv2.imread(img_folder +'71_l6c1.PNG')

# Convert images to Lab color space
source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2Lab)
target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2Lab)

# Extract chromaticity coordinates (a* and b*) from Lab images
source_ab = source_lab[:,:,1:].astype(np.float32)
target_ab = target_lab[:,:,1:].astype(np.float32)

# Reshape the chromaticity coordinates for further processing
source_ab_reshaped = source_ab.reshape(-1, 2)
target_ab_reshaped = target_ab.reshape(-1, 2)

# Estimate the homography using RANSAC
homography_matrix, _ = cv2.findHomography(source_ab_reshaped, target_ab_reshaped, cv2.RANSAC)

# Apply the homography transformation to the chromaticity coordinates
warped_ab = cv2.perspectiveTransform(target_ab_reshaped.reshape(-1, 1, 2), homography_matrix)

# Reshape the transformed chromaticity coordinates
warped_ab_reshaped = warped_ab.reshape(-1, 2)

# Reconstruct the corrected Lab image
corrected_lab = np.zeros_like(source_lab)
corrected_lab[:,:,0] = source_lab[:,:,0]  # Keep the original L channel
corrected_lab[:,:,1:] = warped_ab_reshaped.reshape(source_lab[:,:,1:].shape)

# Convert the corrected Lab image back to BGR color space
corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_Lab2BGR)
#corrected_image = corrected_lab

# Display the original and corrected images
cv2.imshow('Original Image', source_image)
cv2.imshow('Target Image', target_image)
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
