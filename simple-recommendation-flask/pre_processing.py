import pandas as pd 
import numpy as np 

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

ratings.drop(['timestamp'], axis=1, inplace=True)

M = ratings.pivot_table(index=['userId'], columns=['movieId'],values='rating')
M.to_csv('ml-latest-small/matrix_by_id.csv', index=False)