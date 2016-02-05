import numpy as np
import cv2

from sklearn.externals import joblib
from sklearn import svm

import classifier_trainer
import gamma_normalize
import params

def hunmen_detect_in_img(origin_img,img_name):
	detector = joblib.load(params.model_path)
	
	img = np.copy(gamma_normalize.gamma_normalize(origin_img))
	
	pos_windows_scaled_coordinates = []
	pos_feature_vector_in_img = []
	pos_windows_scale = []
	
	v,h = img.shape

	scale = 0 #to record the resizing times

	while params.window_height <= v and params.window_width <= h:

		print img.shape

		pos_windows_in_single_scale,pos_feature_vector_in_scale = classifier_trainer.search_a_scale(detector, img)
		for i in xrange(pos_feature_vector_in_scale.__len__()):
			pos_windows_scale.append(scale)

		pos_feature_vector_in_img += pos_feature_vector_in_scale
		pos_windows_scaled_coordinates += pos_windows_in_single_scale

		img = cv2.resize(img,None,fx = params.scale_gap,fy = params.scale_gap)
		v,h = img.shape
		scale += 1

	mark(pos_windows_scaled_coordinates,pos_windows_scale,origin_img,img_name)
	
	return pos_windows_scaled_coordinates,pos_feature_vector_in_img

def mark(pos_windows_scaled_coordinates,pos_windows_scale,origin_img,img_name):
	rectangle = []

	for (endy,endx),scale in zip(pos_windows_scaled_coordinates,pos_windows_scale):

		propotion = np.power(params.scale_gap,(-1)*scale)
		origin_endy = np.int32(endy*propotion)
		origin_endx = np.int32(endx*propotion)
		origin_starty = np.int32((endy-params.window_height)*propotion)
		origin_startx = np.int32((endx-params.window_width)*propotion)
		point2 = (origin_endx,origin_endy)
		point1 = (origin_startx,origin_starty)
		rectangle.append((point1,point2))

	for rec in rectangle:
		cv2.rectangle(origin_img,rec[0],rec[1],255,2)
	cv2.imwrite('result/'+img_name+'.jpg',origin_img)

#p = np.power(params.scale_gap,(-1)*1)#

#a = cv2.imread('a.jpeg',0)
#hunmen_detect_in_img(a, '2')





