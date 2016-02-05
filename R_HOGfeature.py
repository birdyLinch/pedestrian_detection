# -*- coding: utf-8 -*-

"""
R-HOGfeature Algorithm for human detection programme

Jan 8,2016 0:05

@auther : Chen Lin
"""

import cv2
import numpy as np
import params

def img_gred(img):
	dx = cv2.filter2D(img,-1,params.dx_filter)
	dy = cv2.filter2D(img,-1,params.dy_filter)

	magnitude,oriantation = cv2.cartToPolar(dx,dy) #which compute the gred(mag,ori)

	oriantation = np.int32((2*params.bin_n*oriantation/(2*np.pi))%params.bin_n)#cause we count from 0-180 degree with bin_n bins
	
	return magnitude,oriantation

def blocks_histogram(block_mag,block_ori):
	v,h = block_ori.shape
	cell_v,cell_h = params.cells_block
	pixel_v,pixel_h = params.pixels_cell
	
	#devide into cells using a search of non-overlaping cells
	cell_endy = pixel_v
	block_histogram = []
	while cell_endy <=v:
		cell_endx = pixel_h
		while cell_endx <=h:
			#every time going into this block represent a new cells with cell_endx and cell_endy

			
			cell_mag = block_mag[cell_endy-pixel_v:cell_endy,cell_endx-pixel_h:cell_endx]
			cell_ori = block_ori[cell_endy-pixel_v:cell_endy,cell_endx-pixel_h:cell_endx]
			
			#collecting cell histogram
			block_histogram.append(cells_histogram(cell_mag,cell_ori))

			cell_endx += pixel_h
		
		cell_endy += pixel_v

	#form the histogram of block
	block_histogram = np.float32(block_histogram).ravel()

	#nomalization
	norm = np.linalg.norm(block_histogram)
	
	if norm <= 0.0000001:
		norm = 1e-16

	nomalized_block_histogram = block_histogram/norm #using L2-norm %different from the essay's
	
	
	return nomalized_block_histogram
	



def cells_histogram(cell_mag,cell_ori):

	hist = np.bincount(cell_ori.ravel(),cell_mag.ravel(),params.bin_n)

	return np.vstack(hist)

#test# print cells_histogram(np.array([[1,1],[1,1]]),np.int32([[1,2],[2,0]]))



def R_HOG(window):
	
	v,h = window.shape

	gred_mag,gred_ori = img_gred(np.float32(window))

	featureVector = []
	
	#divided into blocks with movements with params.block_strides
	block_endy = params.block_height
	while block_endy <=v :
		block_endx = params.block_width
		while block_endx <=h:
			#every time going into this block represent a new block with block_endx and block_endy
			
			block_mag = gred_mag[block_endy-params.block_height:block_endy,block_endx-params.block_width:block_endx]
			block_ori = gred_ori[block_endy-params.block_height:block_endy,block_endx-params.block_width:block_endx]

			featureVector.append(blocks_histogram(block_mag,block_ori))
			
			block_endx += params.block_stride[1]

		block_endy += params.block_stride[0]


	return np.float32(featureVector).ravel()





#test# test_img = cv2.imread('Dataset/a.jpg',0)
#test# test_window = cv2.resize(test_img,(64,128))
#test# cv2.imwrite('Dataset/b.jpg',test_window)
#test# print test_window.shape
#test# print R_HOG(test_window).shape


