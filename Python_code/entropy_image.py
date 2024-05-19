import numpy as np
import cv2

def calculate_entropy(image):
    
    # Step 1: Calculate the histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Step 2: Normalize the histogram to get probabilities
    hist = hist / hist.sum()

    # Step 3: Calculate the entropy
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])

    return entropy

# Read the input image as Gray scale
image = cv2.imread(r'F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


entropy_value = calculate_entropy(image)
print(f'Entropy of the image: {entropy_value:.3f}')