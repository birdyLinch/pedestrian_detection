import numpy as np

def gamma_normalize(img):
	img = np.float32(img)/255
	img = img.__pow__(0.5)
	img = img*255
	return np.uint8(img)