import classifier_trainer
import params

#####     get train set     #####
print 'get_pos_train_vector_set'

pos_train_set = classifier_trainer.get_pos_train_vector_set()

print 'get_pos_train_vector_set           finished'
print 'pos_train_set number --> '+str(pos_train_set.__len__())

print 'get_neg_train_vector_set'

neg_train_set = []
for i in xrange(params.neg_train_window_per_img):
	neg_train_set += classifier_trainer.get_neg_train_vector_set()

print 'get_neg_train_vector_set           finished'
print 'neg_train_set number --> '+str(neg_train_set.__len__())

train_set = pos_train_set+neg_train_set

#####     get response     #####
print 'compute response'

pos_response = [1 for i in xrange(pos_train_set.__len__())]
neg_response = [-1 for i in xrange(neg_train_set.__len__())]

response = pos_response+neg_response

if train_set.__len__() != response.__len__():
	print 'the size of train_set and response are not corespondent' 

print 'compute response finished'


print 'start train'
classifier_trainer.hard_sample_iter_train(train_set, response)
#classifier_trainer.continue_hardsample_train(train_set, response, params.continue_train_time)


