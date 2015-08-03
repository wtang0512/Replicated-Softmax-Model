'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 13
'''
from __future__ import division
import numpy as np
import scipy as sp

def rsm_hidden(w_v,w_h,w_vh,testmatrix):

	testD=testmatrix.sum(axis=1)

	# compute hidden activations
	h = sigmoid(np.dot(testmatrix, w_vh) + np.outer(testD, w_h))

	print("rsm hidden vectors calculating............")

	return h

def rsmppl(w_v,w_h,w_vh,testmatrix):

	testD=testmatrix.sum(axis=1)

	# compute hidden activations
	h = sigmoid(np.dot(testmatrix, w_vh) + np.outer(testD, w_h))
	# compute visible activations
	v = np.dot(h, w_vh.T) + w_v
	# exp and normalize.
	tmp = np.exp(v)
	tsum = tmp.sum(axis=1)
	tsum = tsum.reshape(-1,1)
	pdf = tmp / tsum

	z = np.nansum(testmatrix * np.log(pdf))
	s = np.sum(testmatrix)
	ppl = np.exp(- z / s)
	print("PPL calculating..........")

	return ppl


def sigmoid(X):
    return (1 + sp.tanh(X/2))/2


