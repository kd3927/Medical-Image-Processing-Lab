import numpy as np
import matplotlib.pyplot as plt
import cv2

def histogram_equalization(image):
    # Calculate histogram
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Calculate cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized * 255).reshape(image.shape)

    return equalized_image.astype(np.uint8)

def image_histogram(image):
    # Initialize histogram with zeros for 256 bins (for pixel intensities 0-255)
    histogram = np.zeros(256, dtype=int)
    
    # Iterate through each pixel in the image and update the histogram
    for pixel_value in image.flatten():
        histogram[pixel_value] += 1
    
    return histogram

def plot_histogram(histogram,title_graph):
    plt.figure()
    plt.bar(range(256), histogram) #, color='gray'
    plt.title(title_graph)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

# Read the input image
image = cv2.imread(r'F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Perform histogram equalization
equalized_image = histogram_equalization(image)

histogram = image_histogram(image)
histogram1 = image_histogram(equalized_image)

cs = cumsum(histogram)
ceq= cumsum(histogram1)
# Display original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()

plot_histogram(histogram, "Original Histogram")
plot_histogram(histogram1, "Equalized Histogram")

#CDF plot for original and equalized image
cs = cumsum(histogram)
ceq= cumsum(histogram1)
# display the result
plt.plot(cs)
plt.plot(ceq)
plt.legend(['Original CDF','Equalized CDF'])
plt.show()