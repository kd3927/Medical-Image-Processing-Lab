import numpy as np
import matplotlib.pyplot as plt
import cv2

def calculate_histogram(image):
    # Initialize histogram bins for each channel
    r_hist = np.zeros(256, dtype=int)
    g_hist = np.zeros(256, dtype=int)
    b_hist = np.zeros(256, dtype=int)
    
    # Iterate through each pixel in the image
    for row in image:
        for pixel in row:
            # Increment the corresponding histogram bin for each channel
            r_hist[pixel[0]] += 1  # Red channel
            g_hist[pixel[1]] += 1  # Green channel
            b_hist[pixel[2]] += 1  # Blue channel
    
    return r_hist, g_hist, b_hist

def plot_histogram(r_hist, g_hist, b_hist):
    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.title('RGB Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.bar(range(256),r_hist, color='red', label='Red', alpha=0.7)
    plt.bar(range(256),g_hist, color='green', label='Green', alpha=0.7)
    plt.bar(range(256),b_hist, color='blue', label='Blue', alpha=0.7)
    plt.legend()
    plt.show()

def main():
    # Load image
    image = cv2.imread(r'F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff')  # Load the image using matplotlib
    
    # Convert image to uint8 array if necessary
    if image.dtype != 'uint8':
        image = (image * 255).astype(np.uint8)
    
    # Calculate RGB histogram
    r_hist, g_hist, b_hist = calculate_histogram(image)
    
    # Plot histogram
    plot_histogram(r_hist, g_hist, b_hist)

if __name__ == "__main__":
    main()
