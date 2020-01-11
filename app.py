import sys
import flask
from flask import render_template , url_for,request , redirect

app = flask.Flask(__name__)

#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import nltk
import pickle

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    nopunc =  [word.lower() for word in nopunc if word not in ['quot','author','book','novel']]
    return [stemmer.lemmatize(word) for word in nopunc]
    
with open('kmeans.pkl', 'rb') as picklefile:
    MODEL = pickle.load(picklefile)

with open('vectorizer.pkl', 'rb') as picklefile:
    VECT = pickle.load(picklefile)

with open('dataframe_full.pkl', 'rb') as picklefile:
    DF = pickle.load(picklefile)

#-------- ROUTES GO HERE -----------#


# This method takes input via an HTML page
@app.route('/page')
def page():
   return render_template('page.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

        inputs = flask.request.form

        words = inputs['words']
        X_sample2 = VECT.transform([words])
        predicted2 = MODEL.predict(X_sample2)
        #print(predicted2)
        result2 = DF.loc[DF['label'] == predicted2[0]].sample(5)
        #return flask.jsonify(result2['title'][0] , result2['authors'][0]  , result2['review'][0])
        return render_template('book_reco_result.html', results = result2,input_words = words)


# A welcome message to test our server
@app.route('/')
def index():
	return redirect(url_for('page'))

if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = 4000
    app.run(HOST , PORT , debug=True)


