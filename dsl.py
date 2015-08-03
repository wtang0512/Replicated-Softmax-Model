'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 13
'''
import cPickle

def save(data,filename):
	fh = open(filename,'w')
	cPickle.dump(data, fh, 0)
	fh.close()

def load(filename):
	fh = open(filename, 'r')
	model = cPickle.load(fh)
	fh.close ()
	return model