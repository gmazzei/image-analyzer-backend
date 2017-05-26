from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


@csrf_exempt
def test(request):
	return HttpResponse("Test OK!")


@csrf_exempt
def index(request):
	uploaded_file = request.FILES["image"]

	image = convert_to_opencv_image(uploaded_file)
	detected_image = pedestrian_detection(image)
	cv2.imwrite('my_new_image_detected.png', detected_image)

	image_string = cv2.imencode('.png', detected_image)[1].tostring()

	return HttpResponse(image_string)


def convert_to_opencv_image(uploaded_file):
	np_image_array = np.fromstring(uploaded_file.read(), np.uint8)
	image = cv2.imdecode(np_image_array, cv2.IMREAD_COLOR)
	return image

def pedestrian_detection(image):
  
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
	
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
	padding=(8, 8), scale=1.05)
	 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	
	# show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
	#print("[INFO] {}: {} original boxes, {} after suppression".format(
	#	filename, len(rects), len(pick)))
	
	return image


