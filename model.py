import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request



def create_sim():
    dataset=pd.read_csv('movie_dataset.csv')
    
    features = ['keywords','cast','genres','director']
    
    for feature in features:
        dataset[feature] = dataset[feature].fillna('')
        
    def combine_features(row):
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
    
    dataset["combined_features"] = dataset.apply(combine_features,axis=1)
    
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix=cv.fit_transform(dataset['combined_features'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return dataset,sim


# defining a function that recommends 10 most similar movies
def rcmd(m):
    
    dataset,sim = create_sim()
    # check if the movie is in our database or not
    if m not in dataset['title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = dataset.loc[dataset['title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(dataset['movie_title'][a])
        return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
