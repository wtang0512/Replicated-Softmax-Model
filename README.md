# Replicated-Softmax-Model
Repeat the experiments in the paper "Replicated softmax: an Undirected Topic Model".

1. Preprocess.py:
In this library, you could read the files and remove the stopwords, punctuations, and stem tokens. 
Then get the most K frequent tokens. And calculte the dictionary, corpus, doc-word matrix, also the tfidf weights.

2. rsm_numpy.py
This is the details of how the RSM works, which includes the CD-x.

3. ppl.py
This is to calculate the hidden vectors in RSM and the perplexity of RSM model.

4. analysis.py
In this one, cosine, precision and recall functions are included.

5. dsl.py
It is the way to save and load data.

6. RSM_train_test.py:
It is the examples to show how to train a RSM model, and then test it for the perplexity.

7. RSM_Experiment.py:
There are two experiments in it.
(1) Comparing the perplexity of LDA and RSM
(2) Comparing the precision and recall of LDA, RSM, TF-IDF.


