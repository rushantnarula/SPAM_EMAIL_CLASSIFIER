#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import recall_score, precision_score, f1_score

import pickle
import joblib


# In[2]:


DATA_JSON_FILE = 'E:/Users/Rushant Narula/Minor Project/Using Scikit Learn/SpamData/01_Processing/email-text-data.json'


# In[3]:


data = pd.read_json(DATA_JSON_FILE)


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.sort_index(inplace = True)


# In[7]:


data.tail()


# In[8]:


vectorizer = CountVectorizer(stop_words = 'english')


# In[9]:


all_features = vectorizer.fit_transform(data.MESSAGE)


# In[10]:


all_features.shape


# In[11]:


vectorizer.vocabulary_


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, 
                                                   test_size = 0.3, random_state=88)


# In[13]:


X_train.shape


# In[14]:


X_test.shape


# In[15]:


classifier = MultinomialNB()    
classifier.fit(X_train, y_train)


# In[ ]:





# In[16]:


nr_correct = (y_test == classifier.predict(X_test)).sum()


# In[17]:


print('No. of documents classified correctly', nr_correct)


# In[18]:


nr_incorrect = y_test.size - nr_correct


# In[19]:


print('No. of documents classified incorrectly', nr_incorrect)


# In[20]:


fraction_wrong = nr_incorrect/(nr_correct+nr_incorrect)
print('Accuracy is ', 1-fraction_wrong)


# In[21]:


classifier.score(X_test, y_test)


# In[22]:


recall_score(y_test, classifier.predict(X_test))


# In[23]:


precision_score(y_test, classifier.predict(X_test))


# In[24]:


f1_score(y_test, classifier.predict(X_test))


# In[25]:


pickle.dump(classifier, open("classifier.pkl", "wb"))


# In[26]:


example = ['Do you want free viagra ?']


# In[27]:


doc_term_matrix = vectorizer.transform(example)


# In[28]:


result = classifier.predict(doc_term_matrix)


# In[29]:


result[0]


# In[30]:


joblib.dump(vectorizer, 'cv.pkl')


# In[ ]:





# In[ ]:




