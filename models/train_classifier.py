  
import sys
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from workspace_utils import active_session
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pickle


def load_data(database_filepath):
    '''
    This function loads the database from the given filepath and creates 2 dataframe X and y
    Input : filepath
    output: Returns X and y
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table("ETL_message_categories", con=engine)
    
    X = df.message.values     
    Y = df.iloc[:, 4:]
    return (X, Y)


def tokenize(text):
    '''
    This function tokenizes the text
    Input : text
    Output: Cleaned tokens 
    '''
    text = re.sub(r"[^a-zA-Z0-9]"," ",  text.lower())
    words = word_tokenize(text)
    #stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    #lemmatizer
    lemmatizer = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmatizer
   


def build_model():
    """
    pipe line construction
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test):
    '''
    Evaluates the model
    Iterates across the various columns
    and generates the classification report
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table("ETL_message_categories", con=engine)
    Y = df.iloc[:, 4:]
    y_pred = model.predict(X_test)
    
    for i, column in enumerate(Y_test):
        print("------------------------------------")
        print("Category : ", Y.columns[i])
        print(classification_report(Y_test[column], y_pred[:, i]))
        print("Accuracy_score : ", accuracy_score(Y_test[column], y_pred[:, i]))
        print("      ")
    
    #print(classification_report(y_pred, Y_test.values, target_names=category_names))

    

def save_model(model, model_filepath):
    '''
    This function saves our model as a pickle file
    '''
    #model_filepath = 'Disaster_pipeline_model.sav'
    pickle.dump(model, open(model_filepath, 'wb'))
    pass

print(sys.argv[1:])
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
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
