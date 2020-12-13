import cv2
import numpy as np

def find_green_box(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert image to HSV colour space
    mask = cv2.inRange(hsv, np.array([50, 0, 0], dtype='uint8'), np.array([70, 255, 255], dtype='uint8')) # Apply thresholding to obtain binary mask
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Extract contours (edges) from binary mask

    if len(contours) == 0:  # Exit if no contours (edges) found in binary image
        return

    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]
    if cv2.contourArea(largest_contour) < 10:
        return

    moments = cv2.moments(contours[np.argmax(map(cv2.contourArea, contours))])  # Calculate moments from largest found contours

    X_Pos = moments["m10"] / moments["m00"]  # Calculate X position of centroid from moments
    Y_Pos = moments["m01"] / moments["m00"]  # Calculate Y position of centroid from moments

    return [X_Pos, Y_Pos]


