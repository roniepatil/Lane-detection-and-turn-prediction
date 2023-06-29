from matplotlib import pyplot as plt
import numpy as np
import cv2
import glob

img_array = []
for filename in glob.glob('prob1_dataset/*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('prob1_output//adaptive_histogram_equalization.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, size)

def compute_histogram(image, bins=256):
    # Array with size of bins, set to zeros
    histogram = np.zeros(bins)
    # Loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    # Return our final result
    return histogram

def cumsum(values):
    result = [values[0]]
    for i in values[1:]:
        result.append(result[-1] + i)
    return result

def equalize(entries):
    numerator = (entries - np.min(entries))*255
    denominator = np.max(entries) - np.min(entries)
    # Re-normalize the cdf
    result = numerator/denominator
    # Convert float to int
    result = result.astype('uint8')
    return result

def histogram_equalization(image):
    # Convert image into a numpy array
    image_array = np.asarray(image)
    # Convert array to into 1D array
    flatten_image = image_array.flatten()
    # Compute histogram
    computed_histogram_for_input = compute_histogram(flatten_image)
    # Compute cumulative sum 
    cumulative_sum = cumsum(computed_histogram_for_input)
    # Perform equalization over cumulative sum
    cumulative_sum_normalised = equalize(cumulative_sum)
    # Get the value from cumulative sum normalised for every index in flatten_image, and set that as computed_histogram_for_output
    computed_histogram_for_output = cumulative_sum_normalised[flatten_image]
    # Convert array back to original image shape
    final_image = np.reshape(computed_histogram_for_output,image.shape)
    return flatten_image, cumulative_sum, final_image, computed_histogram_for_output, cumulative_sum_normalised

def perform_equalization_and_merge_channels(img):
    # Split blue, green and red channels of the image
    b, g, r = cv2.split(img)
    flatten_image_b, cumulative_sum_b, result_b, histogram_equalized_b, cum_sum_norm_b = histogram_equalization(b)
    flatten_image_g, cumulative_sum_g, result_g, histogram_equalized_g, cum_sum_norm_g = histogram_equalization(g)
    flatten_image_r, cumulative_sum_r, result_r, histogram_equalized_r, cum_sum_norm_r = histogram_equalization(r)

    # Merge blue, green and red channels of the image
    merged_result = cv2.merge([result_b,result_g,result_r])
    return merged_result

def slice_image_and_compute_AHE(img_file):
    block_img = np.zeros(img_file.shape,dtype=np.uint8)
    # tile_size = 800
    tile_size_x = 47
    tile_size_y = 153
    i=0
    j=0
    for j in range(j, img_file.shape[1], tile_size_y):
        i=0
        for i in range(i, img_file.shape[0], tile_size_x):
            tile = img_file[i:i+tile_size_x,j:j+tile_size_y,:]
            hist_tile = perform_equalization_and_merge_channels(tile)
            block_img[i:i+tile_size_x,j:j+tile_size_y,:] = hist_tile
    return block_img

for i in range(len(img_array)):
    output = slice_image_and_compute_AHE(img_array[i])
    cv2.imwrite('prob1_output/adaptive_histogram_equalization'+ str(i) +'.png',output)
    out.write(output)
    
out.release()