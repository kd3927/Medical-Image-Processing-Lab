

import numpy as np
import matplotlib.pyplot as plt
imgi = r'F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff'
def image_histogram(image):
    # Initialize histogram with zeros for 256 bins (for pixel intensities 0-255)
    histogram = np.zeros(256, dtype=int)
    
    # Iterate through each pixel in the image and update the histogram
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    
    return histogram


def plot_histogram(histogram):
    plt.figure()
    plt.bar(range(256), histogram) #, color='gray'
    plt.title('Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Example usage:
# Read an image (you need to have a grayscale image for simplicity)
image = plt.imread(imgi)
histogram = image_histogram(image)

if __name__ == '__main__':
    plot_histogram(histogram)   # Plot the histogram

# #Calculate and plot histogram using builtin function
#plt.hist(image.flatten(), 256)



