import sys
import nltk

nltk.download(['punkt', 'wordnet', "stopwords"])
import pandas as pd
import numpy as np
import sqlalchemy as db
import re
import pickle


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """ Extract the information we need from the database

    INPUTS:
    -------
        database_filepath: str
            The name of the database we want to extract data

    RETURNS:
    --------
        X: Pandas Series
            The message column from the database
        y: DataFrame
            All targets from the DataFrame
        y.columns: Pandas Index (array)
            A list with the name of each columns of y
    """
    engine = db.create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM disaster_response", con=engine)
    X = df.message
    y = df.iloc[:, 4:]

    return X, y, y.columns


def tokenize(text):
    """ Tokenize a text to use it in a ML model and clean it (remove stop words, ponctuation, etc.)

    INPUTS:
    -------
        text: str
            The text you want to tokenize
    RETURNS:
    --------
        clean_tokens: lst(str)
            A list of the tokenized and cleaned txt
    """
    reg_tok = RegexpTokenizer(r'\w+')
    tokens = reg_tok.tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words("english")
    clean_tokens = [WordNetLemmatizer().lemmatize(word).lower() for word in tokens if word not in stop_words]
    return clean_tokens


def build_model():
    """ Create the pipeline we are going to use. Then select the best parameters with GridSearchCV

    RETURNS:
    --------
        pipeline: Pipeline Object
            The pipeline to train
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    params = {'clf__estimator__learning_rate': [0.5, 1],
              'clf__estimator__n_estimators': [25, 50]}

    cv = GridSearchCV(pipeline, param_grid=params)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Return f1 score, precision and recall for each cateegory of the dataset

    As we are using a python 3.6, we can't use the last versions of classification_report.

    So we can't have the result in the form a of a dict. The aim of this function is to loop through each category
    to print the f1 score, the precision and the recall. Additionaly, it takes the f1 score the calculate the
    mean f1 score of the dataset

    INPUTS:
    -------
        y_pred: DataFrame
            The prediction we did on y_test
        y_test: DataFrame
            The real values of y
    """

    def _get_each_f1_score(class_report):
        tot_avg = re.findall('accuracy.*', class_report)[0].split()
        return float(tot_avg[1])

    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    all_f1_score = []

    for col in category_names:
        print(f"Column: {col}")
        class_report = classification_report(Y_test[col], y_pred[col])
        print(class_report)
        print("-------------------------------------------------------")
        all_f1_score.append(_get_each_f1_score(class_report))

    print("The average f1 score is: ", np.mean(all_f1_score))


def save_model(model, model_filepath):
    """ Save the model we used to a pickle file

    INPUTS:
    -------
        model: Pipeline object
            The model we used
        model_filepath: str
            The name of the pickle file
    """

    with open(f"{model_filepath}", "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
