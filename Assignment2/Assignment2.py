'''
Image Processing Assignment 2
Lauren Byrne
C16452654

-------Description-------
This program will make a ball disappear from an image. It takes an image 
and will find the bounding box of a circular contour that is considered
a light shade and will remove it from the image, replacing it with the
bounding box that is shown to be grass making the ball disappear. 

-------Method-------
1. Convert image to grayscale.
2. Threshold image to isolate the ball in the image (usually a light colour)
3. Find the contours that exist on the new thresholded image. Two different methods
of thresholding are demonstrated below.
4. Find the largest contour and fit an ellipse to it along with a bounding box
to retrieve coordinates of the area
5. Create a thresholding for the grass
6. Using this coordinates of the bounding box found earlier, Using slicing, check
if bounding box of the same size ( but with a larger height to try to hi but located right next to it is completely white (255)
- if it is - that means it is grass
7. If it is, apply this slice ontop of the original bounding box to hide the ball with the grass
8. Use inpaint to change the edges of the rectangle to be more similar to neighbouring pixels
to create a more seamless transition between background and box

---------Alternative methods researched--------
Alternative methods had been researched and tested,
these techniques included hough circles and inpaint.
Hough circles is a function that is used to find circular objects in images.
The choice not to use hough circles was due to the inability to use the function
without hard coding in values.
This method has been shown in comments in the code. 

Inpaint is a function that removes small noises and holes from images.
It does this by replacing pixels with pixels that are similar to its neighbouring pixels 
However, as the holes needed to be filled were quite large in these images, 
inpaint did not give a good result when used to fill the hole.
However, it was useful in the final step of the project to blend the box with the image

---------Testing---------
Code has been tested on other images with similar features to three original tested images. 
- white ball with green surrounding the ball. 

---------Conclusion---------
Using the method described above (contours, thresholding and slicing)
A good result was found in the end.
The largest issue is with the snooker ball as the shadows of the ball
were not captured within the threshold of the ball. 
Works well with images where ball does not create a shadow and ball does not have a shadow on it.
Slightly struggles when image includes shadowing around and on ball.

---------References--------
Article title:	OpenCV: Image Inpainting
Website title:	Docs.opencv.org
URL:	https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html

Author	AI Consulting
Article title:	Convex Hull using OpenCV in C++ and Python | Learn OpenCV
Website title:	Learnopencv.com
URL:	https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/

Article title:	Creating Bounding boxes and circles for contours — OpenCV 2.4.13.7 documentation
Website title:	Docs.opencv.org
URL:	https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/bounding_rects_circles/bounding_rects_circles.html

Article title:	Hough Circle Transform — OpenCV 2.4.13.7 documentation
Website title:	Docs.opencv.org
URL:	https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html

'''


# importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys


# Opening an image from a file:
I = cv2.imread("spottheball.jpg")


# error checking to see if valid image has been inputted
# if invalid, program will exit
if I is None:
    print('Invalid input')
    sys.exit(0)
    
    
# creating a copy of the image
Original = I.copy()


# find height and width of image. 
height = np.size(I, 0)
width = np.size(I, 1)


# converting image to 2D grayscale image
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)


# remove noise using medianBlur
H = cv2.medianBlur(G, 5)


# thresholding for grayscale image
H[H < 155] = 0


# create new black image with height and width of original image
img = np.zeros((height, width, 1), dtype = "uint8")


def findCircle(H, img):
    # find the contours in an image
    # RETR_EXTERNAL finds only extreme outer flags meaning that child contours are left behind.
    # CHAIN_APPROX_SIMPLE gets only the end points - more efficient in terms of memory storage
    contours, hierarchy = cv2.findContours(H, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)


    # check to see if contours have been found
    if len(contours) != 0:


        # find only the biggest contour
        c = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(c)
        
        #code to draw contours to see where they lie on image
        #cv2.polylines(I, pts=hull, isClosed=True, color=(116,0,255))
        #contours2 = cv2.drawContours(I, c, -1, (116,100,255), 2)
           
        
        ellipse =cv2.fitEllipse(c)  
        #cv2.ellipse(I,ellipse, color=(115,0,255))
        x,y,w,h = cv2.boundingRect(c)
        
        # draw the bounding box
        # img is now a black image with only a white rectangle outlining where the ball was
        img = cv2.rectangle(img,(x,y),(x+w,y+h),255,5)
       
        return x,y,w,h,img
    
    #error checking for existance of contours
    else:
        print('No contours found')
        sys.exit(0)
        
'''
    ------ALTERNATIVE WAY--------
    # Apply Hough transform on the blurred image.
    # minRadius is set to 0, as this is a default which can be used when circle size is unknown
    # maxRadius is set to the width of the bounding box of the contour detected
    # Could not figure out how to make any of the other parameters automatic
    circles = cv2.HoughCircles(H,  
                       cv2.HOUGH_GRADIENT, 1.75, 50, param1 = 180, 
                   param2 = 40, minRadius = 0, maxRadius = w) 


    # Draw circles that are detected. 
    if circles is not None: 
      
        # Convert the circle parameters a, b and r to integers. 
        circles = np.uint16(np.around(circles)) 
        
        for i in circles[0,:]:
        
            # draw the circle mask onto new black image (setting thickness to -1 fills the circle)
            cv2.circle(img,(i[0],i[1]),i[2], (255,255,255),-1)
            circle_center = i[0]
            circleX = i[1]
            circleY = i[2]
          
        
        #cv2.imshow('detected circles1',H)
        cv2.waitKey(0)
    else:
        print('No circles detected to remove')
'''


#calling function to find circles in image 
x,y,w,h, img = findCircle(H, img)

    
# thresholding for grass
lowerRange = (20, 25, 25)
higherRange = (100, 255,255)


# convert image to HSV to extract the grass 
G = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)


mask = cv2.inRange(G, lowerRange, higherRange)


def performMorph(G, mask):


    # create an elliptical element
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


    # morphology - dilating to remove some noise from the mask. 
    mask= cv2.dilate(mask, shape)
    mask= cv2.dilate(mask, shape)
    
    
    return(mask)


#function to perform morphology on mask
mask = performMorph(G, mask)


G = cv2.bitwise_and(I, I, mask=mask)


#setting values for new bounding box beside existing bounding box from ball
x1 = x + w 
y1 = y 
x2 = x1 + w 
y2 = y1 + h
i = x1


# slicing a rectangle that is the same dimensions as the boundingRect that surrounds the ball
# from the mask created to retrieve the grass
slice = mask[y1:y2, x1:x2]


# checking if the sliced rectangle is purely white - meaning is only grass
# if it is, it will apply that rectangle on top of the original image where the original boundingRect is
#added more height to try to compensate for shadowing that may exist under a ball
while i < width:
    if (slice == 255).all():
        crop = G[y1:y2, x1:x2]
        I[y:y1 + h, x:x1] = crop
        break
    #move the box across the width of the image to find a pure white box    
    else:
        x1, y1, x2, y2, i = x1+1, y1+1, x2+1, y2+1, i+1
 

#convert to RGB to allow MatPlotLib to display images correctly
I = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
Original = cv2.cvtColor(Original,cv2.COLOR_BGR2RGB)


# slightly blurring image to blend new image by removing noise.
I = cv2.GaussianBlur(I,(1,1),1)


# using inPaint to create a better transition between the background and the bounding box.
# it works by replacing pixels by pixels similar to neighbouring pixels. 
# img is the binary image showing the rectangle where the ball was.
# I is the new image where the rectangle has been replaced with pixels similar to its neighbouring pixels showing
# better transition between rectangle and the surrounding pixels.
I = cv2.inpaint(I,img,3,cv2.INPAINT_TELEA)


#subplot function to show original image and new image side by side
fig = plt.figure(figsize=(25,25))
plt.subplot(2, 1, 1)
plt.imshow(I)
plt.subplot(2, 1, 2)
plt.imshow(Original)
plt.show()


#waitKey function waits for you to press a key before proceeding
cv2.waitKey(0)
