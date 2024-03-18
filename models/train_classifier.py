import sys

import nltk
import pandas as pd
import re
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    The function loads the data from the SQL database. It creates the feature X which contains
    the messages of the column "message". It also creates Y which contain the target variables.
    Finally, it creates the category_names which are extracted from the target variables.
    :param database_filepath:
    :return: X, Y, category_names
    X = feature column
    Y = target columns
    category_names = target column names
    """
    engine = create_engine('sqlite:///'+str(database_filepath))
    df = pd.read_sql('SELECT * FROM Messages_Categories_Table', engine)
    X = df["message"]
    Y = df.drop(columns=["id", "message", "original", "genre"])
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    """
    This function processes a given text by performing several steps:
    1. Text Tokenization
    2. Lemmatization

    Output:
    Processed tokens (tokenization, stopword removal, and lemmatization) are returned as a list.
    """
    # extraction of word tokens from text
    tokens = word_tokenize(text)
    # lemmatization (reducing words to their base/root form)
    lemmatizer = WordNetLemmatizer()
    # get list of processed tokens
    clean_tokens = []
    for word in tokens:
        clean_token = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    The function builds a pipeline

    :return:
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [0.2, 0.5, 1]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    The function takes the predicted y ("y_pred"), the true y values which are devined for testing ("y_test")
    and the column names of the target variables.
    :param model: model which is used for predicting the values
    :param X_test: feature (messages column) used in the model to predict the target columns
    :param Y_test: true values of the target columns
    :return:
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    for column in Y_test.columns:
        print('Accuracy of Models by Category: {}'.format(column))
        print(classification_report(Y_test[column], Y_pred[column]))


def save_model(model, model_filepath):
    """
    The function saves the model as pickle file
    :param model: Trained pipeline
    :param model_filepath: filepath to save the model
    :return: None
    """
    with open(model_filepath, 'wb') as file:
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
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    print("train_classifier.py executed successfully.")

    # run code with the following commands in the terminal:
    # cd models
    # python train_classifier.py ../data/DisasterResponse.db classifier.pkl