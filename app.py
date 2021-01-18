# app.py
from flask import Flask
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import newspaper
import pandas as pd


APP = Flask(__name__)
API = Api(APP)

PAC_MODEL = joblib.load('pac.pkl')


class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('url')

        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array
        url = X_new[0]
        out = {'Prediction': PAC_MODEL.predict(process(url))[0]}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')

def process(url):
    article = newspaper.Article(url)
    article.download()
    article.parse()
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
    artext =  tfidf_vectorizer.transform(pd.Series(article.text))
    return artext


