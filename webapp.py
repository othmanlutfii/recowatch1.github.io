from flask import Flask,render_template, redirect, url_for, request
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import pickle

app=Flask(__name__)

# model = pickle.load(open('./model/model.sav','rb'))

@app.route('/')
def index():
    return render_template("home.html", hasil="")

def make_reco(movie_user_likes):
    df = pd.read_csv("movies.csv")

    
    features = ['star','genre','director']
    
    def combine_features(row):
        return row['star']+" "+row['genre']+" "+row['director']

    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(combine_features,axis=1)
    
    cv = CountVectorizer() 
    count_matrix = cv.fit_transform(df["combined_features"])
    
    cosine_sim = cosine_similarity(count_matrix)
    
    def get_title_from_index(index):
        return df[df.index == index]["name"].values[0]

    def get_index_from_title(moviename):
        return df[df['name']==moviename].index.values[0]

    df["name"] = df["name"].str.lower()
    list_film = df["name"].values.tolist()

    movie_user_likes = movie_user_likes.lower()

    if movie_user_likes not in list_film:
        film = random.choices(df["name"],k=4)
        return film
    else:
        movie_index = get_index_from_title(movie_user_likes)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
        i = 0
        film = []
        print("Top 5 similar movies to " + movie_user_likes + " are:\n")
        for element in sorted_similar_movies:
            film.append(get_title_from_index(element[0]))
            i = i + 1
            if i > 4:
                break
        return film





@app.route('/reco', methods=['POST'])
def reco():

    movie_user_likes = request.form['var1']
    movie_user_likes = movie_user_likes.lower()

    df = pd.read_csv("movies.csv")
    df["name"] = df["name"].str.lower()
    list_film = df["name"].values.tolist()

    if movie_user_likes in list_film:
        hasil = make_reco(movie_user_likes)
        return render_template('code2.html', hasil=hasil )

    else:
        hasil = make_reco(movie_user_likes)
        return render_template('code3.html', hasil=hasil)




if __name__ == "__main__" :
    app.run(debug=True)
