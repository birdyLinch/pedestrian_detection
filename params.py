# -*- coding: utf-8 -*-

"""
params for human detection programme

Jan 8,2016 0:05

@auther : Chen Lin
"""

import numpy as np
import os
import cv2

#####     params for R-HOG     #####
cells_block = (2,2)
pixels_cell = (8,8)
block_stride= (8,8) #which means the block movement


block_width = cells_block[1]*pixels_cell[1] #you should not change the para here
block_height = cells_block[0]*pixels_cell[0] #you should not change the para here

#####     params for window     #####
window_width = 64
window_height = 128


#####     params for histogram
bin_n = 9

#####      params for gradient     #####
dx_filter = np.array([[0.0,0.0,0.0],[-1.0,0.0,1.0],[0.0,0.0,0.0]])
dy_filter = dx_filter.T

#####      params for svm training     #####
C = 0.01
max_iter = 9000
tol = 1e-4

#####      params for detect     #####
window_search_pixelstep_h = 8
window_search_pixelstep_v = 8
scale_gap = 0.6


#####      pathlist to positive train set     #####
pos_train_dir = 'Dataset/INRIAPerson/96X160H96/Train/pos/'
margin_pixels = 16
pos_train_set = []
for root,dirs,files in os.walk(pos_train_dir):#these imgs are normalized
	for fn in files:
		if fn[0] != '.':
			pos_train_set.append(root+fn)

#####      pathlist to negitive train set     #####
neg_train_window_per_img = 5

neg_train_dir = 'Dataset/INRIAPerson/Train/neg/'
neg_train_set = []
for root,dirs,files in os.walk(neg_train_dir):
	for fn in files:
		if fn[0] != '.':
			neg_train_set.append(root+fn)


#####     params that infer from above     #####
#####  you should not change the para here #####
block_width = cells_block[1]*pixels_cell[1] 
block_height = cells_block[0]*pixels_cell[0] 



#####     detect model using     #####



#model_path = 'model/LinearSVM/C0.01_tol0.0001/model_iter0.model'
#model_path = 'model/LinearSVM/C0.01_tol0.0001/model_iter1.model'
#model_path = 'model/LinearSVM/C0.01_tol0.0001/model_iter2.model'
#model_path = 'model/LinearSVM/C0.01_tol0.0001/model_iter3.model'
model_path = 'model/LinearSVM/C0.01_tol0.0001/scale_gap0.7strideh_13_v13.model'

print 'current model --> '+model_path


#continue_train_time = 


#####     multithread     #####
thread_n = 3