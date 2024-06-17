import numpy as np
import cv2
import matplotlib.pyplot as plt

def contrast_stretching(image):
    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Apply contrast stretching
    stretched_image = (image - min_val) * (255 / (max_val - min_val))
    
    return stretched_image.astype(np.uint8)

# Read the input image
image = cv2.imread('F:\gitHub\Medical-Image-Processing-Lab\Images\lena_dark.png', cv2.IMREAD_GRAYSCALE)

# Perform contrast stretching
stretched_image = contrast_stretching(image)

# Display the original and stretched images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.show()

# Plot the original and stretched histograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(image.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.5, density=True)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')

plt.subplot(1, 2, 2)
plt.hist(stretched_image.flatten(), bins=256, range=(0, 256), color='red', alpha=0.5, density=True)
plt.title('Stretched Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')

plt.tight_layout()
plt.show()
