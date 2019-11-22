import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import cmapy

######### Reading Images #########

# Opening an image from a file:
I = cv2.imread("Shark1.png")
I2 = cv2.imread("Shark2.png")


#Converting to different colour spaces:
def convertColourSpace(I):
    RGB = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
    XYZ = cv2.cvtColor(I, cv2.COLOR_BGR2XYZ)
    return RGB, YUV, XYZ


#calling function to convert colour spaces
RGB, YUV, XYZ = convertColourSpace(I)


#splitting colours channels
Y, U, V = cv2.split(YUV)
R, G, B = cv2.split(RGB)
X, Y1, Z = cv2.split(XYZ)


#merging colour channels 
merged_image_for_mask = cv2.merge([B, U, U])


#function to obtain detailed shark
def detail(Y1):
    #equalise the image to increase contrast
    I = cv2.equalizeHist(Y1)
    #apply color map function to image
    newI = cv2.applyColorMap(I, cmapy.cmap('inferno'))
    return newI


detailedI = detail(Y1)


#function to create an image suitable for masking
def threshold(I):
    image = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    #adaptive threshold to handle variable lighting problem in image
    th = cv2.adaptiveThreshold(image, 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 741, C =  3);
    return th
    
 
th = threshold(merged_image_for_mask)


#morphology function to get rid of the small black holes around the image background
def morph(I):
    kernel = np.ones((5,5),np.uint8)
    m = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel )
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
    return m
   
   
m = morph(th)

   
#masking the detailed image with the image created for the mask
res = cv2.bitwise_or(detailedI, m)


#getting height and width of image
h, w, channels = res.shape
print(h, w, channels)


#cropping image in relation to the height and width
C = res[100:500, 100:750]


fig = plt.figure()


#plotting pictures of process
plt.subplot(2, 2, 1)
plt.imshow(detailedI)
plt.xlabel('Detailed shark')
plt.subplot(2, 2, 2)
plt.imshow(m)
plt.xlabel('Mask with kernel')
plt.subplot(2, 2, 3)
plt.imshow(res)
plt.xlabel('Result before cropping')
plt.subplot(2, 2, 4)
plt.xlabel('Result after cropping')
plt.imshow(C)
plt.show()


#showing final image
cv2.imshow("Final Image", C)
Key = cv2.waitKey(0)


#References
#Article title:	docs/colorize_all_examples.md · master · Open Source / cmapy
#Website title:	GitLab
#URL:	https://gitlab.com/cvejarano-oss/cmapy/blob/master/docs/colorize_all_examples.md

#Article title:	OpenCV: Morphological Transformations
#Website title:	Docs.opencv.org
#URL:	https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html


