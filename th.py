# import cv2
# import numpy as np
# import skimage.morphology

# # read input
# img = cv2.imread('2.jpg')

# # convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # use thresholding
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# # get distance transform
# distance = thresh.copy()
# distance = cv2.distanceTransform(distance, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

# # get skeleton (medial axis)
# binary = thresh.copy()
# binary = binary.astype(np.float32)/255
# skeleton = skimage.morphology.skeletonize(binary).astype(np.float32)

# # apply skeleton to select center line of distance 
# thickness = cv2.multiply(distance, skeleton)

# # get average thickness for non-zero pixels
# average = np.mean(thickness[skeleton!=0])

# # thickness = 2*average
# thick = 2 * average
# print("thickness:", thick)
import cv2

image= cv2.imread('1.jpg')

#gray= cv2.cvtColor(image)

edged= cv2.Canny(image,30,200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)
