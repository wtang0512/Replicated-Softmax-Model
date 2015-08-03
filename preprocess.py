'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 13
'''
from __future__ import division
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)


def data_perprocess(path):
	
	'''
		remove the stopwords and punctuations, and then get the stemmed tokens
	'''

	#set stopwords and punctuations
	english_stopwords=stopwords.words('english')
	english_punctuations=[',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%',\
						'<','>','`','``',"''",'--','|']

	print("---------------Start Reading Data and Perprocessing It---------------")

	text=[]
	label=[]
	i=1
	#read files from disks
	for filelist in os.listdir(path):
		for filename in os.listdir(path+'/'+filelist):

			#clear all parameters
			data=[];data_tokenized=[];data_fliter_stopwords=[];data_flitered=[];data_stemmed=[];

			#read from the documents
			data=[open(path+'/'+filelist+'/'+filename).read()]


			#tokenized the lower words
			data_tokenized=[[word.lower() for word in word_tokenize(document.decode('iso-8859-15'))] \
							for document in data]

			#remove stopwords and punctuations from the text
			data_fliter_stopwords=[[word for word in document if not word in english_stopwords]\
									 for document in data_tokenized]

			data_flitered=[[word for word in document if not word in english_punctuations]\
							 for document in data_fliter_stopwords]

			#stemming the text
			st=LancasterStemmer()
			data_stemmed=[[st.stem(word) for word in document] for document in data_flitered]


			text.append(data_stemmed[0])
			label.append(i)
		i=i+1
	print("++++++++++++++++Finished Perprocessing++++++++++++++++++++")
	return text, np.array(label)

def frequent_part(text,top_k):

	'''
			pick out the most k-th frequent tokens from oranginal data
	'''
	print("---------------Start Choosing The Most K Frequent Tokens ---------------")
	#find unique words in text
	all_words=sum(text,[])
	#count for the frequency of each token
	u_words=list(set(all_words))
	all_tokens_frequency=[all_words.count(word) for word in u_words]
	all_tokens_frequency=np.array(all_tokens_frequency)
	#sort the frequency of token and pick out the most k-th frequent word
	#k is set by users
	index=np.argsort(all_tokens_frequency)[all_tokens_frequency.shape[0]-top_k:]
	#frequency=all_tokens_f_sort[all_tokens_frequency.shape[0]+1-top_k]
	#index=np.where(all_tokens_frequency>=frequency)
	tokens_high=[u_words[i] for i in index]
	#tokens_high=set(token for token in set(all_words) if all_words.count(token)>=frequency)
	text_high=[[token for token in textword if token in tokens_high]\
						 for textword in text]

	print("++++++++++++++++Got the most K Frequent Tokens++++++++++++++++++++")
	return text_high


def dictionary_count(text_high):

	'''
		count the unique tokens in dictionary
		calculate the word-document matrix, according to the dictionary
	'''
	print("---------------Start Counting for the Dictionary ---------------")
	#count for the dictionary of the text
	dictionary=corpora.Dictionary(text_high)
	#get the id from the dictionary
	token_id=dictionary.token2id
	print("++++++++++++++++Got Dictionary++++++++++++++++++++")
	return dictionary,token_id

def corpus (dictionary,text):

	#transform the dictionary to bag of words
	corpus=[dictionary.doc2bow(textword) for textword in text]

	print("++++++++++++++++Calculated Bag of Words++++++++++++++++++++")

	return np.array(corpus)


def word_document(corpus,token_id):

	'''
		calculate the document-word matrix, according to the dictionary
	'''
	#create the sparse matrix of word-document
	datamatrix=np.zeros((len(corpus),len(token_id)))
	for i in xrange(len(corpus)):
		for obj in corpus[i]:
			datamatrix[i][obj[0]]=obj[1]

	print("++++++++++++++++Document-Word Matrix Made++++++++++++++++++++")
	return datamatrix

def target_value(path):

	target=[]
	i=1
	#read files from disks
	for filelist in os.listdir(path):
		for filename in os.listdir(path+'/'+filelist):
			target.append(i)
		i=i+1
	return np.array(target)


def tfidf(data):

	N=np.repeat(data.sum(axis=1).reshape(-1,1),data.shape[1],axis=1)
	tf=data/N
	idf_v=[]
	for i in xrange(data.shape[1]):
		c=np.count_nonzero(data.T[i])
		if (c==data.shape[0]):
			idf_v.append(0.0)
		else:
			idf_v.append(np.log(data.shape[0]/(1+c)))

	idf=np.repeat(np.array(idf_v).reshape(1,-1),data.shape[0],axis=0)
	print("++++++++++++++++TF-IDF Transformed++++++++++++++++++++")
	return tf*idf