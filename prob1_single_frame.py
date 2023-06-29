from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread("prob1_dataset/0000000000.png")
# Split blue, green and red channels of the image
b, g, r = cv2.split(img)

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

flatten_image_b, cumulative_sum_b, result_b, histogram_equalized_b, cum_sum_norm_b = histogram_equalization(b)
flatten_image_g, cumulative_sum_g, result_g, histogram_equalized_g, cum_sum_norm_g = histogram_equalization(g)
flatten_image_r, cumulative_sum_r, result_r, histogram_equalized_r, cum_sum_norm_r = histogram_equalization(r)

# Merge blue, green and red channels of the image
merged_result = cv2.merge([result_b,result_g,result_r])

fig, ax = plt.subplots(2, 7)
ax[0,0].imshow(img)
ax[0,0].set_title('Input')
ax[0,1].hist(flatten_image_b, bins=50, color='b')
ax[0,1].set_title('Histo(b)')
ax[0,2].hist(flatten_image_g, bins=50, color='g')
ax[0,2].set_title('Histo(g)')
ax[0,3].hist(flatten_image_r, bins=50, color='r')
ax[0,3].set_title('Histo(r)')
ax[0,4].plot(cumulative_sum_b, color='b')
ax[0,4].set_title('cumsum(b)')
ax[0,5].plot(cumulative_sum_g, color='g')
ax[0,5].set_title('cumsum(g)')
ax[0,6].plot(cumulative_sum_r, color='r')
ax[0,6].set_title('cumsum(r)')
ax[1,0].imshow(merged_result)
ax[1,0].set_title('Output')
ax[1,1].hist(histogram_equalized_b, bins=50, color='b')
ax[1,1].set_title('Histo_norm(b)')
ax[1,2].hist(histogram_equalized_g, bins=50, color='g')
ax[1,2].set_title('Histo_norm(g)')
ax[1,3].hist(histogram_equalized_r, bins=50, color='r')
ax[1,3].set_title('Histo_norm(r)')
ax[1,4].plot(cum_sum_norm_b, color='b')
ax[1,4].set_title('cumsum_norm(b)')
ax[1,5].plot(cum_sum_norm_g, color='g')
ax[1,5].set_title('cumsum_norm(g)')
ax[1,6].plot(cum_sum_norm_r, color='r')
ax[1,6].set_title('cumsum_norm(r)')
plt.show()

cv2.imwrite('prob1_output/histogram_equalization.png',merged_result)
cv2.imshow('input',img)
cv2.imshow('result', merged_result)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()