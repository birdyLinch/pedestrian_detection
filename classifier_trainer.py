# -*- coding: utf-8 -*-

"""
train classifier for human detection programme

Jan 8,2016 0:05

@auther : Chen Lin
"""
from threading import Thread
import R_HOGfeature
import params
import numpy as np
import cv2
from sklearn import svm
from sklearn.externals import joblib
import os
from gamma_normalize import gamma_normalize


#####     get positive training set     #####
def get_pos_train_vector_set():
	pos_train_feature_set = []
	
	window_startx = params.margin_pixels
	window_starty = params.margin_pixels
	window_endx = window_startx+params.window_width
	window_endy = window_starty+params.window_height
	
	for filepath in params.pos_train_set:
	
		img = cv2.imread(filepath,0)
	
		img = img[window_starty:window_endy,window_startx:window_endx]

		feature_vector = R_HOGfeature.R_HOG(img)

		##print feature_vector

		pos_train_feature_set.append(feature_vector)


	return pos_train_feature_set



#####     get negitive training set     #####
def get_neg_train_vector_set():

	neg_train_feature_set = []

	for filepath in params.neg_train_set:
		
		img = cv2.imread(filepath,0)
		v,h = img.shape
		window_startx = np.random.randint(0,h-params.window_width-1)
		window_starty = np.random.randint(0,v-params.window_height-1)

		window = img[window_starty:window_starty+params.window_height,window_startx:params.window_width+window_startx]

		feature_vector = R_HOGfeature.R_HOG(window)

		#print feature_vector
		neg_train_feature_set.append(feature_vector)

	return neg_train_feature_set 




#####     train the classifier     #####
def hard_sample_iter_train(train_set,response,itertime = 1):
	
	corrent_train_set = train_set
	corrent_response  = response

	print 'Origin train set feature amount -->'+str(train_set.__len__())

	for i in xrange(itertime+1): #times of compute models *hard_sample is use only if itertime >= 2
		
		model = train_my_model(corrent_train_set,corrent_response)
		
		if i < itertime:
			
			print 'start  get_hard_sample '+str(i+1)
			hard_sample = get_hard_sample_vector(model) #get_hard_sample return list of feature vector
			print 'finished  get_hard_sample '+str(i+1)

			corrent_train_set += hard_sample
			
			print 'current train set feature amount --> '+str(corrent_train_set.__len__())

			hard_sample_res = [-1 for j in xrange(hard_sample.__len__())]
			corrent_response += hard_sample_res
			
	return model

def train_my_model(train_set,response):
	model = svm.LinearSVC(C = params.C,max_iter=params.max_iter,tol = params.tol)
	model.fit(train_set, response)
	savepath = 'model/LinearSVM/'+'C'+str(params.C)+'_tol'+str(params.tol)+'/'+'scale_gap'+str(params.scale_gap)+'strideh_'+str(params.window_search_pixelstep_h)+'_v'+str(params.window_search_pixelstep_v)+'.model'
	joblib.dump(model, savepath)
	print 'model_trained      -->'+savepath
	return model



class hardsample_compute_thread(Thread):
	def __init__(self,model,thread_compute_set,feature_vector_set,thread_id):
		super(hardsample_compute_thread, self).__init__()
		self.compute_set = thread_compute_set
		self.feature_vector_set = feature_vector_set
		self.model = model
		self.id = thread_id

	def run(self):
		self.feature_vector_set.append(detect_in_test_set(self.model, self.compute_set,self.id))

def get_hard_sample_vector(model):
	#print 'start get_hard_sample'
	
	#####     prepare for multithread
	thread_n = params.thread_n #thread num can be changed
	
	thread_compute_set_len = int(params.neg_train_set.__len__()/thread_n)
	
	thread_compute_set = []
	
	thread_feature_vector_set = []
	
	threads = []

	#print 'set_length of each thread -->'+str(thread_compute_set_len)

	for i in xrange(thread_n):
		if i == thread_n-1:
			thread_compute_set.append(params.neg_train_set[i*thread_compute_set_len:])
		else:
			thread_compute_set.append(params.neg_train_set[i*thread_compute_set_len:(i+1)*thread_compute_set_len])
		
		thread_feature_vector_set.append([])

		threads.append(hardsample_compute_thread(model, thread_compute_set[i], thread_feature_vector_set[i],i))

	#print threads.__len__()
	print thread_feature_vector_set

	for thread in threads:
		thread.start()

	for thread in threads:
		thread.join()

	
	hard_sample_feature_vector = []
	
	for i in xrange(thread_n):

		hard_sample_feature_vector += thread_feature_vector_set[i][0] 
	#hard_sample_feature_vector = detect_in_test_set(model, params.neg_train_set)
	#print 'finished get_hard_sample'
	return hard_sample_feature_vector

def detect_in_test_set(model,test_set,thread_id = 0):	#test_set is the path of testset img
	
	pos_feature_vector_in_test_set = []
	tol_len = test_set.__len__()
	
	print 'thread_No: '+str(thread_id)+'    started with img_set: '+str(tol_len)

	n = 0
	#print 'start detect_in_test_set'
	for filepath in test_set:

		img = cv2.imread(filepath,0)
		
		img = gamma_normalize(img)  #-*- neg set is not normalized!!! -*-

		pos_feature_vector_in_test_set += search_img(model, img)
		n +=1
		print 'thread_No: '+str(thread_id)+'     '+'finished '+str(n)+'/'+str(tol_len)
	#print 'finished detect_in_test_set'

	print 'thread_No: '+str(thread_id)+' finished with '+str(pos_feature_vector_in_test_set.__len__())+' Hard Samples'

	return pos_feature_vector_in_test_set

def search_img(model,img):
	#print 'start search_img'
	pos_feature_vector_in_img = []
	
	v,h = img.shape

	while params.window_height <= v and params.window_width <= h:

		coord_noneed,pos_feature_vector_in_scale = search_a_scale(model,img)

		pos_feature_vector_in_img += pos_feature_vector_in_scale

		img = cv2.resize(img,None,fx = params.scale_gap,fy = params.scale_gap)

		v,h = img.shape


	#print 'finished search_img'
	return pos_feature_vector_in_img

def search_a_scale(model,img_scale):
	#print 'started search_a_scale'		
	pos_windows_coordinates_in_scale = []
	pos_feature_vector_in_scale = []

	v,h = img_scale.shape
	#print v,h
	window_endx = params.window_width
	#print window_endx
	while window_endx <= h:
		window_endy = params.window_height
		#print window_endy
		while window_endy <= v:
			#print 'start checking a window'

			window = img_scale[window_endy-params.window_height:window_endy,window_endx-params.window_width:window_endx ]
				
			feature_vector = R_HOGfeature.R_HOG(window) 

			res = model.predict(feature_vector.reshape(1,-1))

			if res == 1:
				pos_windows_coordinates_in_scale.append((window_endy,window_endx))
				pos_feature_vector_in_scale.append(feature_vector)

			#print 'finished checking a window'

			window_endy += params.window_search_pixelstep_v
		window_endx += params.window_search_pixelstep_h
	#print 'finished search_a_scale'
	return pos_windows_coordinates_in_scale,pos_feature_vector_in_scale







