import pandas as pd 
import numpy as np
from math import floor

from flask import Flask, request, jsonify, abort
from flask_restful import Resource, Api 

app = Flask(__name__)
api = Api(app)

movies = pd.read_csv('ml-latest-small/movies.csv')

M = pd.read_csv('ml-latest-small/matrix_by_id.csv')
films = np.array(movies)

film_data = []
film_data_dict = {}

for film in films:
    film_data.append({'movie_id': film[0],'movie_title': film[1]})
    film_data_dict[film[0]] = film[1] 

def pearson(s1, s2):
    s1_c = s1-s1.mean()
    s2_c = s2-s2.mean()
    return np.sum(s1_c*s2_c)/np.sqrt(np.sum(s1_c**2)*np.sum(s2_c**2))

def get_recs(movie_id, M, num):
    reviews = []
    for id in M.columns:
        if id == movie_id:
            continue
        cor = pearson(M[movie_id], M[id])
        if np.isnan(cor):
            continue
        else:
            reviews.append((id, cor))
        reviews.sort(key=lambda tup: tup[1], reverse=True)
    return reviews[:num]

def get_paginated_result(url, list, args):
    page_size = int(args.get('page_size', 10))
    page = int(args.get('page', 1))
    page_count = int((len(list) + page_size - 1) / page_size)
    start = (page - 1) * page_size
    end = min(start + page_size, len(list))

    if page<page_count:
        next = f'{url}?page={page+1}&page_size={page_size}'
    else:
        next=None
    if page>1:
        prev = f'{url}?page={page-1}&page_size={page_size}'
    else:
        prev = None
    
    if page > page_count or page < 1:
        abort(404, description = f'Halaman {page} tidak ditemukan.') 
    return  {
        'page': page, 
        'page_size': page_size,
        'next': next, 
        'prev': prev,
        'Results': list[start:end]
    }

class Movie(Resource):
    def get(self):

        return get_paginated_result('movies/', film_data, request.args)


class Recommendation(Resource):
    def get(self, movie_id):
        length = request.args.get('length', 10)
        recommendations = get_recs(movie_id, M, int(length))
        results = [{'movie_id': int(rec[0]),'movie_title': film_data_dict[int(rec[0])], 'score': round(rec[1] * 100, 2)} for rec in recommendations]

        return jsonify({
            'movie_id': int(movie_id),
            'movie_title': film_data_dict[int(movie_id)],
            'recommendations': results
        })

api.add_resource(Movie, '/movies')
api.add_resource(Recommendation, '/recommendation/<movie_id>')

if __name__ == '__main__':
    app.run(port='5005', debug=True)