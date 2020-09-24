# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:59:15 2020

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:45:19 2020

@author: Sai Ajay Vutukuri
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS, cross_origin

import json
import bs4 as bs
import urllib.request
import pickle
import requests
import pandas as pd
import numpy as np

import flasgger
from flasgger import Swagger

app = Flask(__name__)
cors = CORS(app)
Swagger(app)
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))
def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())    
@app.route('/')
@cross_origin()
def welcome():
    return "Welcome"

@app.route('/similarity')
@cross_origin()
def predict_note_authentication():
    
    """Lets Authenticate the Bank Note
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    """
    movie_title = request.args.get('name')
    rc = rcmd(movie_title)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str
   
@app.route("/recommend",methods=["GET","POST"])
@cross_origin(supports_credentials=True)
def recommend():
    # getting data from AJAX request
    
    data = json.loads(request.data)
    background_path = data['background_path']
    title = data['title']
    cast_ids = data['cast_ids']
    cast_names = data['cast_names']
    cast_chars = data['cast_chars']
    cast_bdays = data['cast_bdays']
    cast_bios = data['cast_bios']
    cast_places = data['cast_places']
    cast_profiles = data['cast_profiles']
    imdb_id = data['imdb_id']
    poster = data['poster']
    genres = data['genres']
    overview = data['overview']
    vote_average = data['rating']
    vote_count = data['vote_count']
    release_date = data['release_date']
    runtime = data['runtime']
    status = data['status']
    rec_movies = data['rec_movies']
    rec_posters = data['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = [{'poster':rec_posters[i],'id':i} for i in range(len(rec_posters))]
    
    casts = [{'name':cast_names[i],'id':cast_ids[i],'chars': cast_chars[i],'poster': cast_profiles[i]} for i in range(len(cast_profiles))]

    
    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = [{'id':i,'review':reviews_list[i],'status': reviews_status[i]} for i in range(len(reviews_list))]

    # passing all the data to the html file
    return {'title':title,'background_path':background_path,'poster':poster,'overview':overview,'vote_average':vote_average,
        'vote_count':vote_count,'release_date':release_date,'runtime':runtime,'status':status,'genres':genres,
        'movie_cards':movie_cards,'reviews':movie_reviews,'casts':casts}
if __name__ == '__main__':
    app.run()
