import cv2

import numpy as np


def find_green_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([50, 0, 0], dtype='uint8'), np.array([70, 255, 255], dtype='uint8'))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) is 0:
        return
    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]
    if cv2.contourArea(largest_contour) < 10:
        return
    moments = cv2.moments(largest_contour)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
