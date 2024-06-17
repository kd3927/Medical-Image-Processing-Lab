import numpy as np
import cv2
import matplotlib.pyplot as plt

def otsu_thresholding(image):
    # Calculate histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Calculate total number of pixels
    total_pixels = image.size

    # Calculate cumulative sum of the histogram
    cumulative_sum = np.cumsum(hist)
    
    # Calculate cumulative mean of the histogram
    cumulative_mean = np.cumsum(hist * np.arange(256))
    
    # Global mean
    global_mean = cumulative_mean[-1] / total_pixels

    # Calculate between-class variance for all thresholds
    between_class_variance = ((global_mean * cumulative_sum - cumulative_mean) ** 2) / (cumulative_sum * (total_pixels - cumulative_sum))
    
    # Ignore NaN values
    between_class_variance = np.nan_to_num(between_class_variance)

    # Get the threshold value that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    # Apply the threshold to the image
    otsu_image = (image >= optimal_threshold).astype(np.uint8) * 255

    return optimal_threshold, otsu_image

# Read the input image
image = cv2.imread('F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff', cv2.IMREAD_GRAYSCALE)

# Perform Otsu's thresholding
optimal_threshold, otsu_image = otsu_thresholding(image)

# Display the original and thresholded images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(otsu_image, cmap='gray')
plt.title(f'Otsu Thresholding (Threshold: {optimal_threshold})')
plt.axis('off')

plt.show()

# Plot the histogram and the optimal threshold
plt.figure(figsize=(10, 5))
plt.hist(image.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.5)
plt.axvline(optimal_threshold, color='red', linestyle='dashed', linewidth=2)
plt.title('Histogram and Otsu Threshold')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
