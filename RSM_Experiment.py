'''
	@author: Wen Tang(wtang6@ncsu.edu)
	@date: July 20
'''
#import the libraries you need
from __future__ import division
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
import dsl,ppl
import perprocess as ps
import matplotlib.pyplot as plt
import numpy as np
import lda
import analysis as ana

# When use LDA we need change the matrix type as int64
train=np.int64(train)
test=np.int64(test)

'''
Experiment1: perplexity of LDA and RSM
'''

#train the LDA model
print("-------------------LDA GET Training--------------------")
model=lda.LDA(n_topics=50,n_iter=2000,random_state=1)
model.fit(train)
#get the topic_word distribution and doc_topic distribution.
topic_word=model.components_
doc_topic=model.doc_topic_
#save the these data
dsl.save(topic_word,'result/topic_word')
dsl.save(model,'result/lda_model')
dsl.save(doc_topic,'result/doc_topic')
print("-------------------LDA Model Has Been Saved--------------------")

#sample the held document from the test data
sample=50
sample_id=np.random.randint(test.shape[0],size=(50,sample))
dsl.save(sample_id,'result/sample_id')


#Since the doc-topic distribution is different for each document, we need to 
#calculate it for each test document

#calculte the ppl of lda model
ppl_lda=[]
for i in xrange(sample):
    test_sample=test[sample_id[i]]
    doc_topic_test=model.transform(test_sample,max_iter=1000)
    pdf=np.dot(doc_topic_test,topic_word)
    z = np.nansum(test_sample * np.log(pdf))
    s = np.sum(test_sample)
    pplt = np.exp(- z / s)
    ppl_lda.append(pplt)

ppl_lda=np.array(ppl_lda)
dsl.save(ppl_lda,'result/ppl_lda')

#load the rsm_5 model from disk
result=dsl.load('result/rsm_result_5')
w_vh=result['w_vh']
w_v=result['w_v']
w_h=result['w_h']

#calculate the ppl of rsm_5 model
ppl_rsm_5=[]
for i in xrange(sample):
    test_sample=test[sample_id[i]]
    pplt=ppl.rsmppl(w_v,w_h,w_vh,test_sample)
    ppl_rsm_5.append(pplt)

ppl_rsm_5=np.array(ppl_rsm_5)
dsl.save(ppl_lda,'result/ppl_rsm_5')


#load the rsm_1 model from disk
result=dsl.load('result/rsm_result_1')
w_vh=result['w_vh']
w_v=result['w_v']
w_h=result['w_h']


#calculate the ppl of rsm_1 model
ppl_rsm_1=[]
for i in xrange(sample):
    test_sample=test[sample_id[i]]
    pplt=ppl.rsmppl(w_v,w_h,w_vh,test_sample)
    ppl_rsm_1.append(pplt)
ppl_rsm_1=np.array(ppl_rsm_1)
dsl.save(ppl_lda,'result/ppl_rsm_1')
  

#plot the figures of compare the ppl of rsm and lda
plt.plot(ppl_rsm_5,ppl_lda,'ro')
plt.plot(ppl_rsm_1,ppl_lda,'bo')
x_min=np.floor(np.min([ppl_rsm_5.min(),ppl_rsm_1.min(),ppl_lda.min()])*0.9)
x_max=np.ceil(np.max([ppl_rsm_5.max(),ppl_rsm_1.max(),ppl_lda.max()])*1.1)
x_axis=np.linspace(x_min,x_max,1000)
plt.plot(x_axis,x_axis,'k')
plt.axis([x_min,x_max,x_min,x_max])
plt.xlabel('RSM')
plt.ylabel('LDA')
plt.title('20 news perplexity')
plt.savefig('result/perplexity.png')



'''
Experiment2: query a document and retrieve the similar document
'''

#load the label for all the data
#train_label=dsl.load('train_label')
#test_label=dsl.load('test_label')

# retrieve K similar documents from database
# k starts from 1 ends in the length of training data, step=50
k=np.hstack((np.arange(1,100,10),np.arange(101,train_label.shape[0]-1,50)))
k=np.hstack((k,train_label.shape[0]-1))

#use topics to represent the documents
#here in LDA is 50 topics 
train_topics=doc_topic
test_topics=model.transform(test,max_iter=1000)
dsl.save(test_topics,'result/test_topics')
#train_topics=dsl.load('doc_topic')
#test_topics=dsl.load('test_topics')
train_topics_sort=np.argsort(train_topics)
test_topics_sort=np.argsort(test_topics)

#calculate the cosines between each query documents and training data
#get the predict label of the query documents
lda_perdict_label,lda_cosine=ana.perdict_label(train_topics_sort,test_topics_sort,train_label)

#return the precision and recalls for retrieving k similar documents
lda_p,lda_r=ana.precision_recall(lda_perdict_label,train_label,test_label,k)

#save the data
dsl.save(lda_cosine,'result/lda_cosine')
dsl.save(lda_p,'result/lda_precision')
dsl.save(lda_r,'result/lda_recall')


# use the hidden vectors to reperesent the documents
# here in RSM is 50 dimensions 
h_train=ppl.rsm_hidden(w_v,w_h,w_vh,train)
h_test=ppl.rsm_hidden(w_v,w_h,w_vh,test)

# calculate the similarity, precision and recall of RSM
rsm_perdict_label, rsm_cosine=ana.perdict_label(h_train,h_test,train_label)
rsm_p,rsm_r=ana.precision_recall(rsm_perdict_label,train_label,test_label,k)

dsl.save(rsm_cosine,'result/rsm_cosine')
dsl.save(rsm_p,'result/rsm_precision')
dsl.save(rsm_r,'result/rsm_recall')

# use tfidf to represent the documents
# here still keep 2000 verbs
# calculate the similarity, precision and recall of tfidf
train_tfidf=ps.tfidf(train)
test_tfidf=ps.tfidf(test)
tfidf_perdict_label, tfidf_cosine=ana.perdict_label(train_tfidf,test_tfidf,train_label)
tfidf_p,tfidf_r=ana.precision_recall(tfidf_perdict_label,train_label,test_label,k)

dsl.save(tfidf_cosine,'result/tfidf_cosine')
dsl.save(tfidf_p,'result/tfidf_precision')
dsl.save(tfidf_r,'result/tfidf_recall')


#Plot the trends of RSM, LDA and tfidf
#compare these three ways by precision and recalls
plt.figure(1)
plt.plot(rsm_r.mean(axis=0),rsm_p.mean(axis=0),'r')
plt.plot(lda_r.mean(axis=0),lda_p.mean(axis=0),'b')
plt.plot(tfidf_r.mean(axis=0),tfidf_p.mean(axis=0),'k')
plt.axis([0,1,0.048,0.053])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('20 news groups')
plt.savefig('result/precision_recall-mean.png')
plt.close()

plt.figure(2)
for i in xrange(test_label.shape[0]):
	plt.plot(rsm_r[i],rsm_p[i],'r')
	plt.plot(lda_r[i],lda_p[i],'b')
	plt.plot(tfidf_r[i],tfidf_p[i],'k')
plt.axis([0,1,0.0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('20 news groups')
plt.savefig('result/precision_recall-all.png')
plt.close()
