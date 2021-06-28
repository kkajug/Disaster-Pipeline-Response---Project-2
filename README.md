# Disaster PipelineResponse Project-2

## Project Overview
#### This project is a part of the Udacity Data science Nanodegree. In this project, we have applied our skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages on a real time basis. The data set contains real messages that were sent durring disaster events. My machine learning pipeline categorises these events so that messages  ca be sent to appropriate disaster relief agency.

#### The funnel app inside the app folder can be used by an emergency worker to input a new messgae and get the classification results based on the message. The app also displays the visualisationsof the data.

### The project is divided into 3 sections -
#### Building an ETL pipeline
#### Building a ML pipeline
#### Running the web app where an emergency worker can input a new message and get classification results in several categories

## Pre Requisites
#### Python 3.5+
#### Libraries - Numpy, Pandas, Scikit-learn, NLTK, plotly, sqlalchemy, Pickle
#### Web app - Flask


## Files 

### -Data folder:
#### disaster_categories.csv - Data to process 
#### disaster_messages.csv - Data to process 
#### DisasterResponse.db: Created database from transformed and cleaned data
#### Process_data.py: Database to save clean data to

### -Models folder:
#### Train_classifier.py: Loads Data from dataset, tokenizes the text data to the model, builds pipeline, evaluates the model and saves the trained model to disk
#### Classifier.pkl: Saved trained classifier

### -App folder:
#### run.py: Flask file that runs app and displays the results and visualisations
#### templates: folder containing the html templates

## Instructions 
#### 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### 2. Run the following command in the app's directory to run your web app.
    - `python run.py`

#### 3. Go to http://0.0.0.0:3001/

## Author
#### Kajal Gupta

## Acknowledgment
#### Figure Eight dataset for disaster management
