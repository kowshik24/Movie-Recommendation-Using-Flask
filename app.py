import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
app = Flask(__name__)

# Load data and create model
movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'], dtype={'movieId': 'int32', 
'title': 'str'})
rating_df=pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'], dtype={'userId': 
'int32', 'movieId': 'int32', 'rating': 'float32'})
df = pd.merge(rating_df,movies_df,on='movieId')
combine_movie_rating = df.dropna(axis=0,subset=['title'])
movie_ratingCount = (combine_movie_rating.groupby('title')['rating'].count().reset_index().rename(columns={'rating':'totalRatingCount'})
[['title','totalRatingCount']])
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount,on='title')
popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
movie_features_df = rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
model_knn.fit(movie_features_df_matrix)
  
# root api direct to index.html (home page)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    # Get user input
    if request.method == 'POST':
        movie_name = [str(x) for x in request.form.values()]
        if movie_name[0] not in movie_features_df.index:
            return render_template('movieNotFound.html', name=movie_name[0])
        query_index = movie_features_df.index.get_loc(movie_name[0])
        # Get recommendations from model
        distances , indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
        # Return recommendations to user
        recommendations = []
        for i in range(1,len(distances.flatten())):
            recommendation = movie_features_df.index[indices.flatten()[i]]
            recommendations.append(recommendation)
        return render_template('index.html', recommended_movie=recommendations)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)















