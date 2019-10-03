import cv2
import numpy as np
import time

import argparse

# parser = argparse.ArgumentParser(description='Path to video file')
# parser.add_argument('--path_to_file', type=str, help = 'Path to video file')

# args = parser.parse_args()

# If nno file path is provided, get video feed from webcam
# vid_capture = cv2.VideoCapture(args.path_to_file if args.path_to_file else 0)
vid_capture = cv2.VideoCapture(0)

# Generate a mask for a particular color range i.e the region of that color in a given image
def mask_for_range(hsv_frame,arr1, arr2):
	lower_bound = np.array(arr1)
	upper_bound = np.array(arr2)
	mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

	return mask
# Warm up the webcam
time.sleep(4)
count = 0
background = 0

# Capture and store background frame without any subject in the frame. This is needed to store the background which will be used later
for i in range(60):
	return_value, background = vid_capture.read()
	if return_value == False:
		continue

background = np.flip(background, axis = 1)

# Save the captured video feed in mp4 file
fourcc = cv2.cv.CV_FOURCC(*'avc1')
out = cv2.VideoWriter('invisible.mp4', fourcc, 20.0, (640, 480))

# Capture the camera feed
while (vid_capture.isOpened()):
	return_value, img = vid_capture.read()
	if not return_value:
		print("video capture error")

	count+=1
	img = np.flip(img, axis = 1)

# Convert captured image from BGR to HSV for better color detection
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Generate lower and upper masks for red color

	mask1 = mask_for_range(hsv_img, [100, 40, 40], [100, 255, 255])
	mask2 = mask_for_range(hsv_img, [155, 40, 40], [180, 255,255])

	final_mask = mask1 + mask2

	# Clear noise from mask
	final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 2)
	final_mask = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations = 1)

	# Segment out the mask from the image
	segmented_mask = cv2.bitwise_not(final_mask)

	# Remove everything but the cloth from the frame
	res1 = cv2.bitwise_and(img, img, mask = segmented_mask)

	# Create image showing static background on cloth region
	res2 = cv2.bitwise_and(background, background, mask= final_mask)

	# Generate final output
	cloaked_result = cv2.addWeighted(res1, 1, res2, 1, 0)
	out.write(cloaked_result)
	cv2.imshow("Cloak mode engaged", cloaked_result)

	# Quit if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break