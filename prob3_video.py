import numpy as np
import cv2
import utility_functions as uf
from utility_functions import LaneLine


# Read input video file
cap = cv2.VideoCapture("prob3_dataset/challenge.mp4")
# Create object to write output video file
output = cv2.VideoWriter("prob3_output/challenge_output1.mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (1280,720))

left_line = LaneLine()
right_line = LaneLine()

while(cap.isOpened()):
    ret, frame = cap.read()
    if np.any(frame) == None:
        break
    w, h =  frame.shape[:2]
    # frame = cv2.flip(frame,1)
    # Colour filtered to get yellow line highlighted
    colour_filtered = uf.filter_colors(frame)
    # Grayscale conversion
    grayscale = cv2.cvtColor(colour_filtered, cv2.COLOR_BGR2GRAY)
    # Gaussain blur
    blurred = cv2.GaussianBlur(grayscale, (3,3), 0)
    # Binary threshold
    (T, thres) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    # Warp perspective
    warped, _, m, minv = uf.warp_perspective(thres)
    # Function to detect lanes
    lane_detected, cur = uf.lane_lines_detected(warped, left_line, right_line)
    # Function to compute road metadata
    road_info, curvature, deviation = uf.compute_metadata(left_line, right_line)
    # Function to visualize lanes
    weighted_visual, visual = uf.visualize_lanes(lane_detected, left_line, right_line)
    # Warp back the image
    warp_back = cv2.warpPerspective(weighted_visual, minv, (h, w))
    text = str(round(curvature,2)) + ' m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255,255,255)
    thickness = 2
    cv2.putText(warp_back, 'Radius of curvature: '+ text, (20, 30), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(warp_back, 'Direction: '+ road_info, (20,60), font, fontScale, color, thickness, cv2.LINE_AA)
    # Overlay input frame with lane detection and metadata
    result = cv2.addWeighted(frame, 1, warp_back, 1, 0)
    output.write(result)
    cv2.imshow('output',result)
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        break

output.release()
cv2.destroyAllWindows()