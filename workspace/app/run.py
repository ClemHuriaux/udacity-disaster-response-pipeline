import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    all_categories = df.drop(columns=['id', 'message', 'original', 'genre', 'related'])
    all_categories_count = all_categories.sum().sort_values()[:10]
    all_categories_names = all_categories_count.index

    weather_disaster_distribution_count = df.iloc[:, -7:-2].sum().sort_values()
    weather_disaster_distribution_names = weather_disaster_distribution_count.index
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=all_categories_names,
                    y=all_categories_count
                )
            ],

            'layout': {
                'title': 'Distribution of all categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Names"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=weather_disaster_distribution_names,
                    y=weather_disaster_distribution_count
                )
            ],

            'layout': {
                'title': 'Distribution of Weather related disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Types"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()