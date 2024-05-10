""" 
Reading, writing and displaying images
opencv - reading and writing images
PIL - reading images
Matplotlib - displaying images

"""
#imgi = r'F:\gitHub\Medical-Image-Processing-Lab\Images\lena_color.tiff'

# #Reading images
import cv2
# # Reading image and converting it into an ndarray
# img = cv2.imread(imgi)
# # Converting img to grayscale
# img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# from PIL import Image
# import numpy as np
# # Reading image and converting it into grayscale.
# img = Image.open(imgi).convert('L')
# # #convert PIL Image object to numpy array
# img = np.array(img)

# # # # Converting ndarray to image for saving using PIL.
# im2 = Image.fromarray(img)

#Writing images
# import cv2
# img = cv2.imread(imgi)
# # cv2.imwrite will take an ndarray.
# cv2.write('file_name', img)

#Displaying images 
# import matplotlib.pyplot as plt

# # We import matplotlib.pyplot to display an image in grayscale.
# # If gray is not supplied the image will be displayed in color.
# # plt.imshow(img_gs, cmap='gray')
# # plt.show()
# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure ( figsize =(10, 3.6) )
# # first subplot
# plt.subplot (1,3,1)
# plt.imshow(img)
# # second subplot
# plt.subplot (1,3,2)
# plt.imshow(img_gs , cmap="gray")
# plt.axis('off' )
# # third subplot (zoom)
# plt.subplot (133)
# plt.imshow(img_gs[200:220, 200:220], cmap='gray', interpolation ='nearest' )
# plt.subplots_adjust(wspace=0, hspace =0., top =0.99, bottom=0.01, left =0.05, right =0.99)
# plt.show()


# #Color channel
imgi = r'F:\gitHub\Medical-Image-Processing-Lab\Images\retina.jpg'


import matplotlib.pyplot as plt

# Read a color image
image = cv2.imread(imgi)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#image=plt.imread(imgi)
plt.imshow(image)
plt.show()
# Split the color channels
red_channel = image[:,:,0]
green_channel = image[:,:,1]
blue_channel = image[:,:,2]

# Plot each color channel separately
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')

plt.show()
