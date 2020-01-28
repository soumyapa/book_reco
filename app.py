import sys
from flask import Flask
from flask import render_template , url_for,request , redirect
import flask
from flask_paginate import Pagination , get_page_parameter , get_page_args


#-------- MODEL GOES HERE -----------#
import pandas as pd
import numpy as np
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
    
    print(DF.loc[DF['title'] == t_h.values[0], ['authors']].values[0].tolist()[0])
    # Return the top 5 most similar movies
    return t_h

def get_books(df , offset=0, per_page=100) :
    return df[offset: offset + per_page]


@app.route('/')
def index():
    q = DF['title'].sort_values()
    page, per_page, offset = get_page_args(page_parameter='page',
                                        per_page_parameter='per_page')
    total = q.count()
    
    pagination_titles = get_books(q , offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
   
    # 'page' is the default name of the page parameter, it can be customized
    # e.g. Pagination(page_parameter='p', ...)
    # or set PAGE_PARAMETER in config file
    # also likes page_parameter, you can customize for per_page_parameter
    # you can set PER_PAGE_PARAMETER in config file
    # e.g. Pagination(per_page_parameter='pp')

    return render_template('title.html',
                           titles=pagination_titles,
                           page=page,
                           per_page=per_page,
                           pagination=pagination,
                           )
@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'GET':
        words = flask.request.args['words']
        return render_template('book_reco_result.html', results = get_recommendations(words), input_words = words , df = DF)

if __name__ == '__main__':
    app.run(debug=True)