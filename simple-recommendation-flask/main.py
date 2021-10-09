import pandas as pd 
import numpy as np

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

def get_paginated_list(results, url, start, limit):
    start = int(start)
    limit = int(limit)
    count = len(results)
    if count < start or limit < 0:
        abort(404)
    # make response
    obj = {}
    obj['start'] = start
    obj['limit'] = limit
    obj['count'] = count
    # make URLs
    # make previous url
    if start == 1:
        obj['previous'] = ''
    else:
        start_copy = max(1, start - limit)
        limit_copy = start - 1
        obj['previous'] = url + '?start=%d&limit=%d' % (start_copy, limit_copy)
    # make next url
    if start + limit > count:
        obj['next'] = ''
    else:
        start_copy = start + limit
        obj['next'] = url + '?start=%d&limit=%d' % (start_copy, limit)
    # finally extract result according to bounds
    obj['results'] = results[(start - 1):(start - 1 + limit)]
    return obj

class Movie(Resource):
    def get(self):
        return jsonify(get_paginated_list(
        film_data, 
        '/movies', 
        start=request.args.get('start', 1), 
        limit=request.args.get('limit', 20)
    ))

class Recommendation(Resource):
    def get(self, movie_id):
        length = request.args.get('length', 10)
        recommendations = get_recs(movie_id, M, int(length))
        results = [{'movie_id': int(rec[0]),'movie_title': film_data_dict[int(rec[0])], 'score': round(rec[1] * 100, 2)} for rec in recommendations]
        return jsonify(results)

api.add_resource(Movie, '/movies')
api.add_resource(Recommendation, '/recommendation/<movie_id>')

if __name__ == '__main__':
    app.run(port='5005', debug=True)