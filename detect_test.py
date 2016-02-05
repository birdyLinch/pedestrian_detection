import detector
import os
import params
import cv2


for root,dirs,files in os.walk('detect_test/'):
	for f in files:
		if f[0] !='.':
			img = cv2.imread(root+f,0)
			print f[:-4]+params.model_path[32:-6]
			detector.hunmen_detect_in_img(img, f[:-4]+params.model_path[32:-6])