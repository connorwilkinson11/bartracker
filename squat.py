
import cv2
import numpy as np
from matplotlib import pyplot as plt
  
cap = cv2.VideoCapture("ryanclean.mp4")

for _ in range(50):
	ret, frame = cap.read()
if not ret:
	print("Can't receive frame. Exiting")

print(frame.shape)


frame = frame[300:, 100:900]
width = int(frame.shape[1] * 0.6)
height = int(frame.shape[0] * 0.6)
dim = (width, height)
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
output = frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(7,7),1.5)
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp = 1.6, minDist = 500, minRadius=50, maxRadius=150)

if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('frame', frame)
start_point = (x, y)
end_point = (x, y)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (frame.shape[1],frame.shape[0]))
for _ in range(80):
	ret, frame = cap.read()
	frame = frame[300:, 100:900]
	width = int(frame.shape[1] * 0.6)
	height = int(frame.shape[0] * 0.6)
	dim = (width, height)
	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	if _ % 2 == 0:
		if not ret:
			print("Can't receive frame. Exiting")
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray,(7,7),1.5)
		circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp = 1.6, minDist = 500, minRadius=50, maxRadius=150)
		if circles is not None:
			circles = np.round(circles[0, :]).astype("int")
			for (x, y, r) in circles:
				cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
				cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
				end_point = (x,y)
				cv2.imshow('frame', frame)
				frame = cv2.line(frame, start_point, end_point, (255, 0, 0), 3)
		out.write(frame)
	else:
		frame = cv2.line(frame, start_point, end_point, (255, 0, 0), 3)
		cv2.imshow('frame', frame)
		out.write(frame)
# reading image
# converting image into grayscale image
output = cv2.resize(output, (1200, 1080))
cv2.imshow("Result", output)
print("End program")
print(cv2.__file__)
  
