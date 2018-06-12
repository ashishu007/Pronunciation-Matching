import numpy as np
import cv2

KNOWN_DISTANCE = 24.0 	#DISTANCE OF OBJECT FROM CAMERA
KNOWN_WIDTH = 11.0 		#WIDTH OF OBJECT IN REALITY

PATH ="capture/"
IMAGE = ["<Image NAME 1>", "<Image NAME 2>"]
 
def find_marker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	c = max(cnts, key = cv2.contourArea)
 	return cv2.minAreaRect(c)

 def distance_to_camera(knownWidth, focalLength, perWidth):
	return (knownWidth * focalLength) / perWidth
 
for image in IMAGE:
	image = cv2.imread(PATH + imagePath + ".jpg")  
	marker = find_marker(image)
	dist = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
	box = np.int0(cv2.cv.BoxPoints(marker))
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2f" % (dist),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)