import sys
from flask import Flask
from flask import render_template , url_for,request , redirect
import flask


#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import nltk
import pickle


app = flask.Flask(__name__)

with open('dataframe_full.pkl', 'rb') as picklefile:
    DF = pickle.load(picklefile)

print("start")

def create_soup(x):
    return ' '.join(x['authors']) + ' ' + ' '.join(x['title']) + ' ' + x['review'] 

X = DF.review
DF['soup'] = DF.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(DF['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

DF.reset_index(inplace=True)
indices = pd.Series(data=DF.index, index=DF.title)

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    print(title)
    idx = indices[title]
    print(idx)
    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))
    #print(sim_scores)
    # Sort the books based on the similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True  )
    sim_scores = sim_scores[1:6]
    # Get the books indices
    book_indices = [i[0] for i in sim_scores]
    t_h = DF['title'].iloc[book_indices]
    print(type(DF.loc[DF['title'] == t_h.values[0], ['authors']].values[0].tolist()[0]))
    print(DF.loc[DF['title'] == t_h.values[0], ['authors']].values[0].tolist()[0])
    # Return the top 5 most similar movies
    return t_h


    

@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'GET':
        words = flask.request.args['words']
        return render_template('book_reco_result.html', results = get_recommendations(words),input_words = words , df = DF)


@app.route('/')
def index():
    return render_template('page.html' , table_data_title = DF['title'],table_data_author = DF['authors'],table_data_review = DF['review'])

if __name__ == '__main__':
    app.run(debug=True)