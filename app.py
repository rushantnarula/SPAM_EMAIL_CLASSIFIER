#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import flask
import pickle
import joblib
from flask import Flask, redirect, url_for, request, render_template


# In[2]:


app = Flask(__name__, template_folder = './templates',static_folder='./templates')


# In[3]:


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


# In[4]:


def ValuePredictor(data): 
    NB_spam_model = open('classifier.pkl','rb') 
    clf = joblib.load(NB_spam_model)

    cv_model = open('cv.pkl', 'rb')
    cv = joblib.load(cv_model)
    
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    
    return my_prediction[0]


# In[5]:


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict = request.form.values()
        result = ValuePredictor(to_predict)
        if result == 1:
            prediction = "Mail is Spam."
        else:
            prediction = "Mail is not Spam."

    return render_template('result.html',prediction = prediction)   


# In[ ]:


if __name__ == '__main__':
    app.run()


# In[ ]:




