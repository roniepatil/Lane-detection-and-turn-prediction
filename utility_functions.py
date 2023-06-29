import numpy as np
import cv2

class LaneLine:
    def __init__(self):
        # Lane line found in the previous iteration
        self.found = False
        # Window width with limits
        self.window_size_limits = 56
        # X values of last iterations
        self.previous_x = []
        # Coefficients of recent fit polynomial
        self.present_fit = [np.array([False])]
        # Radius of curvature of lane line
        self.r_o_c = None
        # Starting x value
        self.x_start = None
        # Ending x value
        self.x_end = None
        # Values of x for found lane line
        self.all_of_x = None
        # Values of y for found lane line
        self.all_of_y = None
        # Metadata
        self.information_of_road = None
        self.curvature = None
        self.divergence = None

def compute_curvature(left_lane_line, right_lane_line):
    y = left_lane_line.all_of_y
    left_x, right_x = left_lane_line.all_of_x, right_lane_line.all_of_x
    # Flip to match openCV Y direction
    left_x = left_x[::-1]
    right_x = right_x[::-1]
    # Max value of y -> bottom
    y_value = np.max(y)
    # Calculation and conversion roc meter per pixel
    lane_width = abs(right_lane_line.x_start - left_lane_line.x_start)
    y_m_per_pixel = 30/720
    x_m_per_pixel = 3.7*(720/1280)/lane_width
    # Fit polynomial in world space
    left_curve_fit = np.polyfit(y * y_m_per_pixel, left_x * x_m_per_pixel, 2)
    right_curve_fit = np.polyfit(y * y_m_per_pixel, right_x * x_m_per_pixel, 2)
    # Compute new roc
    left_curve_fit_radius = ((1 + (2 * left_curve_fit[0] * y_value * y_m_per_pixel + left_curve_fit[1]) ** 2) ** 1.5)/np.absolute(2 * left_curve_fit[0])
    right_curve_fit_radius = ((1 + (2 * right_curve_fit[0] * y_value * y_m_per_pixel + right_curve_fit[1] ** 2) ** 1.5)/np.absolute(2 * right_curve_fit[0]))
    # ROC
    left_lane_line.r_o_c = left_curve_fit_radius
    right_lane_line.r_o_c = right_curve_fit_radius
    return left_curve_fit_radius

def even_off(lines, number_of_previous_lines=3):
    # Average out lines
    lines = np.squeeze(lines)
    averaged_line = np.zeros((720))
    for i, line in enumerate(reversed(lines)):
        if i == number_of_previous_lines:
            break
        averaged_line += line
    averaged_line = averaged_line/number_of_previous_lines
    return averaged_line

def readjust_line_search(img, left_lane, right_lane):
    # Histogram in lower half of image along columns
    histogram = np.sum(img[int(img.shape[0]/2):, :], axis=0)
    # Blank canvas
    res_img = np.dstack((img, img, img)) * 255
    # Find peak values of left and right halves of histogram
    middle = np.int(histogram.shape[0]/2)
    left_p = np.argmax(histogram[:middle])
    right_p = np.argmax(histogram[middle:]) + middle
    # Number of sliding windows
    window_number = 9
    # Define window height
    height_of_window = np.int(img.shape[0]/window_number)
    # Find all non zero pixel
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    # Present positions to be updated for each window
    present_left_x = left_p
    present_right_x = right_p
    # Min number of pixels found to recenter window
    min_number_pixel = 50
    # Empty lists to save left and right lane pixel indices
    window_left_lane = []
    window_right_lane = []
    window_margin = left_lane.window_size_limits
    # Go through the windows one after other
    for window in range(window_number):
        # Find window boundaries
        win_y_low = img.shape[0] - (window + 1) * height_of_window
        win_y_high = img.shape[0] - window * height_of_window
        win_leftx_min = present_left_x - window_margin
        win_leftx_max = present_left_x + window_margin
        win_rightx_min = present_right_x - window_margin
        win_rightx_max = present_right_x + window_margin
        # Draw windows on canvas
        cv2.rectangle(res_img, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(res_img, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)
        # Find nonzero pixels inside window
        left_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_leftx_min) & (nonzero_x <= win_leftx_max)).nonzero()[0]
        right_window_inds = ((nonzero_y >= win_y_low) & (nonzero_y <= win_y_high) & (nonzero_x >= win_rightx_min) & (nonzero_x <= win_rightx_max)).nonzero()[0]
        # Append indices to list
        window_left_lane.append(left_window_inds)
        window_right_lane.append(right_window_inds)
        # If found > minpixels, recenter next window to their mean position
        if len(left_window_inds) > min_number_pixel:
            present_left_x = np.int(np.mean(nonzero_x[left_window_inds]))
        if len(right_window_inds) > min_number_pixel:
            present_right_x = np.int(np.mean(nonzero_x[right_window_inds]))
    
    # Concatenate the arrays of indoices
    window_left_lane = np.concatenate(window_left_lane)
    window_right_lane = np.concatenate(window_right_lane)
    # Extract left and right line pixel positions
    leftx= nonzero_x[window_left_lane]
    lefty =  nonzero_y[window_left_lane]
    rightx = nonzero_x[window_right_lane]
    righty = nonzero_y[window_right_lane]
    res_img[lefty, leftx] = [255, 0, 0]
    res_img[righty, rightx] = [0, 0, 255]
    # Fit second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    # print(righty)
    right_fit = np.polyfit(righty, rightx, 2)
    left_lane.present_fit = left_fit
    right_lane.present_fit = right_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    left_lane.previous_x.append(left_plotx)
    right_lane.previous_x.append(right_plotx)
    if len(left_lane.previous_x) > 10:
        left_avg_line = even_off(left_lane.previous_x, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.present_fit = left_avg_fit
        left_lane.all_of_x, left_lane.all_of_y = left_fit_plotx, ploty
    else:
        left_lane.present_fit = left_fit
        left_lane.all_of_x, left_lane.all_of_y = left_plotx, ploty
    if len(right_lane.previous_x) > 10:
        right_avg_line = even_off(right_lane.previous_x, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_lane.present_fit = right_avg_fit
        right_lane.all_of_x, right_lane.all_of_y = right_fit_plotx, ploty
    else:
        right_lane.present_fit = right_fit
        right_lane.all_of_x, right_lane.all_of_y = right_plotx, ploty
    
    left_lane.x_start, right_lane.x_start = left_lane.all_of_x[len(left_lane.all_of_x)-1], right_lane.all_of_x[len(right_lane.all_of_x)-1]
    left_lane.x_end, right_lane.x_end = left_lane.all_of_x[0], right_lane.all_of_x[0]
    left_lane.found, right_lane.found = True, True
    cur = compute_curvature(left_lane, right_lane)
    return res_img, cur

# Search based on previous found line
def track_line_search(img, left_lane, right_lane):
    # Canvas image
    res_img = np.dstack((img, img, img)) * 255
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Get margin of windows from Line class. Adjust this number.
    window_margin = left_lane.window_size_limits
    left_line_fit = left_lane.present_fit
    right_line_fit = right_lane.present_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin
    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]
    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]
    res_img[lefty, leftx] = [255, 0, 0]
    res_img[righty, rightx] = [0, 0, 255]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)
    left_lane.previous_x.append(left_plotx)
    right_lane.previous_x.append(right_plotx)
    if len(left_lane.previous_x) > 10:
        left_avg_line = even_off(left_lane.previous_x, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_lane.present_fit = left_avg_fit
        left_lane.all_of_x, left_lane.all_of_y = left_fit_plotx, ploty
    else:
        left_lane.present_fit = left_fit
        left_lane.all_of_x, left_lane.all_of_y = left_plotx, ploty
    if len(right_lane.previous_x) > 10:
        right_avg_line = even_off(right_lane.previous_x, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_lane.present_fit = right_avg_fit
        right_lane.all_of_x, right_lane.all_of_y = right_fit_plotx, ploty
    else:
        right_lane.present_fit = right_fit
        right_lane.all_of_x, right_lane.all_of_y = right_plotx, ploty
    # Compute Standard Deviation of the distance between X positions of pixels of left and right lines
    # If this STDDEV is too high, then we need to reset our line search, using line_search_reset
    stddev = np.std(right_lane.all_of_x - left_lane.all_of_x)
    if (stddev > 80):
        left_lane.found = False
    left_lane.x_start, right_lane.x_start = left_lane.all_of_x[len(left_lane.all_of_x) - 1], right_lane.all_of_x[len(right_lane.all_of_x) - 1]
    left_lane.x_end, right_lane.x_end = left_lane.all_of_x[0], right_lane.all_of_x[0]
    cur = compute_curvature(left_lane, right_lane)
    return res_img, cur

def lane_lines_detected(img, left_lane, right_lane):
    #check if the line detected before
    if left_lane.found == False:
        return readjust_line_search(img, left_lane, right_lane)
    else:
        return track_line_search(img, left_lane, right_lane)

def visualize_lanes(img, left_line, right_line, lane_color=(0, 255, 255), road_color=(0, 255, 0)):
    # Create an empty image to draw on
    window_img = np.zeros_like(img)
    window_margin = left_line.window_size_limits
    left_plotx, right_plotx = left_line.all_of_x, right_line.all_of_x
    ploty = left_line.all_of_y
    mid_plotx = (left_line.all_of_x + right_line.all_of_x)//2
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    mid_line = np.array([np.transpose(np.vstack([mid_plotx, ploty]))])
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), lane_color)
    cv2.fillPoly(window_img, np.int_([right_line_pts]), lane_color)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    
    # Draw arrows onto the warped blank image
    for i in range(0, len(mid_line[0])-1, 36):
        if (i/36)%2 == 0:
            cv2.arrowedLine(window_img, np.int_(mid_line[0][i+35]), np.int_(mid_line[0][i]), (255,0,0), 9, tipLength=0.5)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)
    return result, window_img

def compute_metadata(left_line, right_line):
    # take average of radius of left curvature and right curvature
    # curvature = (left_line.r_o_c + right_line.r_o_c) / 2
    curvature = left_line.r_o_c
    # calculate direction using X coordinates of left and right lanes
    direction = ((left_line.x_end - left_line.x_start) + (right_line.x_end - right_line.x_start)) / 2
    if curvature > 2000 and abs(direction) < 100:
        road_info = 'Straight'
        curvature = -1
    elif curvature <= 2000 and direction < - 50:
        road_info = 'Turning to Left'
    elif curvature <= 2000 and direction > 50:
        road_info = 'Turning to Right'
    else:
        if left_line.information_of_road != None:
            road_info = left_line.information_of_road
            curvature = left_line.curvature
        else:
            road_info = 'None'
            curvature = curvature
    
    center_lane = (right_line.x_start + left_line.x_start) / 2
    lane_width = right_line.x_start - left_line.x_start

    center_car = 720 / 2
    if center_lane > center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Left'
    elif center_lane < center_car:
        deviation = str(round(abs(center_lane - center_car), 3)) + 'm Right'
    else:
        deviation = 'by 0 (Centered)'
    left_line.information_of_road = road_info
    left_line.curvature = curvature
    left_line.divergence = deviation
    return road_info, curvature, deviation

def filter_colors(image):
    # Filter white pixels
    white_threshold = 200
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)
    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([0,80,80])
    upper_yellow = np.array([110,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2

def warp_perspective(img):
    #warp the image
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[200, 720],[1100, 720],[595, 450],[685, 450]])
    dst = np.float32([[300, 720],[980, 720],[300, 0],[980, 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    return warped, unwarped, m, m_inv