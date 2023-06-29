import numpy as np
import cv2

# Function to extract information only from road in edge detected image
def region_of_interest(image):
    height, width = image.shape
    # Create mask in shape of trapezoid with road lanes as region of interest
    trapezoid = np.array([[(0, height), (465, 320), (515, 320), (width, height)]])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, trapezoid, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

# Function to check which side of car is shoulder
def check_shoulder_side(img_file):
    height, width, _ = frame.shape
    # Create mask with right half of frame as white pixels
    mask_right = img_file[:,round(width/2):]
    # Create mask with left half of frame as white pixels
    mask_left = img_file[:,:round(width/2)]
    # Count number of white pixels in right image frame
    number_of_white_pixels_right_mask = np.sum(mask_right == 255)
    # Count number of white pixels in left image frame
    number_of_white_pixels_left_mask = np.sum(mask_left == 255)
    # If right half of image frame has more white pixels than left, then road shoulder is on right, or otherwise
    if(number_of_white_pixels_right_mask>number_of_white_pixels_left_mask):
        shoulder_right = True
    else:
        shoulder_right = False
    return shoulder_right

# Function gives the starting and ending points for each line using with the average slope and y_intercept of the line
def create_points_for_single_line(image, average): 
    slope, y_intercept = average
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_intercept)//slope)
    x2 = int((y2 - y_intercept)//slope)
    return np.array([x1, y1, x2, y2])

# Function to average out the lines produced by hough transform
def average_of_hough_lines(image, lines):
    # List to hold all left side lines
    left = []
    # List to hold all right side lines
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
    # Average out all right side lines
    right_average = np.average(right, axis=0)
    # Average out all left side lines
    left_average = np.average(left, axis=0)
    # Create points for single left side lane
    left_side_line = create_points_for_single_line(image, left_average)
    # Create points for single right side lane
    right_side_line = create_points_for_single_line(image, right_average)
    return np.array([left_side_line, right_side_line])

# Function to draw single green line on solid lane and single red line on broken lane
def plot_broken_and_solid_lines(image, lines, shoulder_side):
    h, w, _ = image.shape
    lines_frame = np.zeros_like(image)
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line
        # If shoulder is on right side of car
        if(shoulder_side):
            if(x1 < round(w/2) and x2 < round(w/2)):
                cv2.line(lines_frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            else:
                cv2.line(lines_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
        # If shoulder is on left side of car
        else:
            if(x1 < round(w/2) and x2 < round(w/2)):
                cv2.line(lines_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
            else:
                cv2.line(lines_frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return lines_frame

cap = cv2.VideoCapture("prob2_dataset/whiteline.mp4")
total_frames = cap.get(7)
# Fetch single frame from video sequence
cap.set(1,8)
ret, frame = cap.read()
# Flag to know whether shoulder is on right or left side of the car
shoulder_right = False
# frame = cv2.flip(frame,1)
# Convert image from RGB colorspace to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Blurr the with GaussainBlur of kernel size 5x5
blurred = cv2.GaussianBlur(gray_image,(5,5),0)
# Canny edge detection to find lane outlines/edges
edge_detected = cv2.Canny(blurred, 50, 100)
# Focus on only road out of edge detected region
ROI = region_of_interest(edge_detected)
# Use Hough transform to find lines along lanes strips
lines = cv2.HoughLinesP(ROI, rho=2, theta=np.pi/180, threshold=100, lines=None, minLineLength=40, maxLineGap=5)
# Make of copy of frame/image
input_frame = frame.copy()
# Function to check on which side is the shoulder in image frame
shoulder_right = check_shoulder_side(ROI)
# Function to average out the lines produced by hough transform
hough_lines_averaged = average_of_hough_lines(input_frame,lines)
# Function to draw single green line on solid lane and single red line on broken lane
broken_and_solid_lines = plot_broken_and_solid_lines(input_frame,hough_lines_averaged,shoulder_right)
# To make alpha blending of lane line over the input image frame
output_with_lane_detection = cv2.addWeighted(input_frame, 0.8, broken_and_solid_lines, 1, 1)
cv2.imshow('input',frame)
cv2.imshow('output',output_with_lane_detection)
cv2.imwrite('prob2_output/broken_and_solid_lane_lines_detected.png',output_with_lane_detection)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()