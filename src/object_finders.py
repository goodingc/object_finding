import cv2
import numpy as np


def find_green_box(image):
    """
    @author Oliver
    @param image: Colour image from camera
    @return: The screen-space coordinates of the center of the green box to home towards
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV colour space
    mask = cv2.inRange(hsv, np.array([50, 0, 0], dtype='uint8'),
                       np.array([70, 255, 255], dtype='uint8'))  # Apply thresholding to obtain binary mask
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # Extract contours (edges) from binary mask

    if len(contours) == 0:  # Exit if no contours (edges) found in binary image
        return

    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]

    if cv2.contourArea(largest_contour) < 10:  # Ignore areas below a certain size, which are erroneous responses
        return

    moments = cv2.moments(
        contours[np.argmax(map(cv2.contourArea, contours))])  # Calculate moments from largest found contours

    X_Pos = moments["m10"] / moments["m00"]  # Calculate X position of centroid from moments
    Y_Pos = moments["m01"] / moments["m00"]  # Calculate Y position of centroid from moments

    return [X_Pos, Y_Pos]


def find_fire_hydrant(image):
    """
    @author Oliver
    @param image: Colour image from camera
    @return: The screen-space coordinates of the center of the fire hydrant to home towards
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV colour space
    mask = cv2.inRange(hsv, np.array([0, 1, 0], dtype='uint8'),
                       np.array([1, 255, 255], dtype='uint8'))  # Apply thresholding to obtain binary mask

    # Apply morphological transformations to join segments of fire hydrant together 
    # And to remove erroneous responses 
    kernel = np.ones((3,3),np.uint8)    
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 3)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Extract contours (edges) from binary mask

    if len(contours) == 0:  # Exit if no contours (edges) found in binary image
        return

    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]

    if cv2.contourArea(largest_contour) < 10000:  # Ignore areas below a certain size, which are erroneous responses
        return

    moments = cv2.moments(
        contours[np.argmax(map(cv2.contourArea, contours))])  # Calculate moments from largest found contours

    X_Pos = moments["m10"] / moments["m00"]  # Calculate X position of centroid from moments
    Y_Pos = moments["m01"] / moments["m00"]  # Calculate Y position of centroid from moments

    return [X_Pos, Y_Pos]


def find_mail_box(image):
    """
    @author Oliver
    @param image: Colour image from camera
    @return: The screen-space coordinates of the center of the mailbox to home towards
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV colour space
    mask = cv2.inRange(hsv, np.array([110, 1, 0], dtype='uint8'),
                       np.array([113, 255, 255], dtype='uint8'))  # Apply thresholding to obtain binary mask

    # Apply morphological transformations to join segments of mailbox together
    # And to remove erroneous responses 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=1)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)  # Extract contours (edges) from binary mask

    if len(contours) == 0:  # Exit if no contours (edges) found in biAdnary image
        return

    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]

    if cv2.contourArea(largest_contour) < 10000:  # Ignore areas below a certain size, which are erroneous responses
        return

    moments = cv2.moments(
        contours[np.argmax(map(cv2.contourArea, contours))])  # Calculate moments from largest found contours

    X_Pos = moments["m10"] / moments["m00"]  # Calculate X position of centroid from moments
    Y_Pos = moments["m01"] / moments["m00"]  # Calculate Y position of centroid from moments

    return [X_Pos, Y_Pos]

def find_number_5(image):
    """
        @author Mohamed
        @param image: Colour image from camera
        @return: The screen-space coordinates of the number 5 box
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0], dtype='uint8'),np.array([0, 0, 0], dtype='uint8'))
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) is 0:
        return
    largest_contour = contours[np.argmax(map(cv2.contourArea, contours))]
    if cv2.contourArea(largest_contour) < 10:
        return
    moments = cv2.moments(largest_contour)
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
