'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 20
'''
from __future__ import division
import numpy as np




def perdict_label(train_topics_sort,test_topics_sort,train_label):
	'''
	calculate the perdict label of querying document
	'''

	print("----------------Predicting the Label of each Querying Document---------------------")

	predicts=[]
	cos=[]

	for i in xrange(test_topics_sort.shape[0]):
		# get cosine of the qureying document and the training data
		tmp=cosine(test_topics_sort[i],train_topics_sort)
		# sort the cosine return the sorted index of cosine
		tmp_sort=np.argsort(tmp)
		c_sorted=np.sort(tmp)
		# get the predicit label from the train files' label
		# by sorted index of cosine
		predict=np.array(train_label[tmp_sort])
		predicts.append(predict)
		cos.append(c_sorted)

	return np.array(predicts), np.array(cos)

def precision_recall(perdict_label,train_label,test_label,k):
	'''
		k is how many document would be retrieved
		calculate the precision and recall of querying document
	'''
	print("----------------Calculating the Precision and Recall---------------------")
	precision=np.zeros((perdict_label.shape[0],len(k)))
	recall=np.zeros((perdict_label.shape[0],len(k)))

	for i in xrange(len(k)):
		for j in xrange(perdict_label.shape[0]):
			# retrieve first k highest ranked document
			predict=perdict_label[j][perdict_label.shape[1]-k[i]:]
			# find out how many document is the same label with querying document 
			# precision= # of the same label / k
			precision[j][i]=((predict==test_label[j]).sum()/k[i])
			# recall= # of the same label / # of same label in training data
			recall[j][i]=((predict==test_label[j]).sum()/(train_label==test_label[j]).sum())

	print("++++++++++++++++++++Precision and Recall are Ready+++++++++++++++++++++")
	return precision,recall


def cosine(v,m):
	'''
	 v is a vector, m is a matrix
	 calculate the cosine of v and each row vector in m
	'''
	print("-----------Cosine Calculating----------------------")
	# make n a matrix with all row vectors are the v
	n=np.repeat(np.array(v).reshape(1,-1),m.shape[0],axis=0)
	# elements mutiple together of n and m, then add them by rows
	# divided by the norm of each row vectors in n and m
	c=np.sum(n*m,axis=1)/(np.linalg.norm(n,axis=1)*np.linalg.norm(m,axis=1))

	return c