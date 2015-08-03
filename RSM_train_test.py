'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 13
'''
#import the libraries you need
import rsm_numpy
import preprocess as ps
import ppl
import dsl

'''
	RSM model: train and test
'''

#path of trainning data
path_train='20news-bydate-train'

#perprocess the data
#tokenrize, stemming, remove stop words and punctuations
text,train_label=ps.data_perprocess(path_train)

#top_k is the first k highest frequency tokens which would be chosen as the word vectors
top_k=2000
#get the frequent part of the data
text_high=ps.frequent_part(text,top_k)

#get the document-word frequency matrix
dictionary,token_id=ps.dictionary_count(text_high)
corpus=ps.corpus(dictionary,text)
train=ps.word_document(corpus,token_id)

# hyperparameters
hiddens = 50
batch = 100
epochs = 1000
rate = 0.0001
iter=1

#train the RSM model by iter=1
RSM = rsm_numpy.RSM()
result = RSM.train(train, hiddens, epochs, iter, lr=rate, btsz=batch)
#save the result of the RSM_CD1
dsl.save(result,'result/rsm_result_1')


#set iterations=5, i.e., CD-5
iter=5

#train the RSM model by iter=5
RSM = rsm_numpy.RSM()
result = RSM.train(train, hiddens, epochs, iter, lr=rate, btsz=batch)
dsl.save(result,'result/rsm_result_5')

#path of test data
path_test='20news-bydate-test'

#perprocess the test data
test,test_label=ps.data_perprocess(path_test)
#get test word-document matrix 
corpus_test=ps.corpus(dictionary,test)
test=ps.word_document(corpus_test,token_id)

#calculate ppl for test
#load the rsm_1 model from disk
result=dsl.load('result/rsm_result_5')
w_vh=result['w_vh']
w_v=result['w_v']
w_h=result['w_h']
# return the perplexity which is to assess the topic model
Eppl_CD5=ppl.rsmppl(w_v,w_h,w_vh,test)
print("Eppl_CD5=",Eppl_CD5)

#load the rsm_1 model from disk
result=dsl.load('result/rsm_result_1')
w_vh=result['w_vh']
w_v=result['w_v']
w_h=result['w_h']
Eppl_CD1=ppl.rsmppl(w_v,w_h,w_vh,test)
print("Eppl_CD1=",Eppl_CD1)


dsl.save(text,'result/text')
dsl.save(train_label,'result/train_label')
dsl.save(dictionary,'result/dictionary')
dsl.save(token_id,'result/token_id')
dsl.save(train,'result/train')
dsl.save(test,'result/test')
dsl.save(test_label,'result/test_label')
dsl.save(corpus,'result/corpus')
dsl.save(corpus_test,'result/corpus_test')
