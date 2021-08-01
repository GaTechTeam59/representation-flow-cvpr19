import cv2 #Requires opencv-contrib-python==4.1.2.30
import os
import random
import numpy as np
def get_file_sizes(show=False):
	files = os.listdir("ssd/hmdb");
	#Iterate every file
	shapes = [];
	file_shape = dict()
	for i in range(len(files)):
		# print(len(files)-i);
		if show: print(len(files)-i,"\t",files[i]);
		if ".avi" in files[i] or ".mp4" in files[i]:
			#check size:
			vidcap_sample = cv2.VideoCapture("ssd/hmdb/"+files[i])
			success_sample,image_sample = vidcap_sample.read()
			shap = image_sample.shape;
			print(f"{shap} {files[i]}")
			file_shape[files[i]] = shap
			if shap not in shapes:
				shapes.append(shap);
	print(shapes)
	# print(file_shape)
	# return shapes, file_shape
get_file_sizes();