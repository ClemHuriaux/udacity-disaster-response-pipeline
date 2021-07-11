# Udacity's disaster response pipeline project

## Summary
1. [Introduction](#Introduction)
2. [Description](#Description)
3. [Requirements](#Requirements)
4. [Running the project](#Running)

<a name="Introduction"></a>
## Introduction
This project is the second project from [Udacity Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) nano degree.
The goal of this project is to classify messages following a disaster. I had two csv files: one for messages and the other for the categories. The goal is to use 
Natural Processing Language (NLP) to classify messages in categories

<a name="Description"></a>
## Description
The project was separated into 3 parts:
  1. The first part is an ETL pipeline. Here I built all I needed to extract data, put these in a good shape and insert the final dataset into a database
  2. On this second part, I built an NLP model to predict messages categories with data I extracted from the Database. Then I saved this model as a pickle file
  3. The thid part is a visualisation one. The goal here is to deploy a small flask app to display a dashboard with visuals about data obtained with the ML step
 
 <a name="Requirements"></a>
 ## Requirements
 In order to run this project, you'll need several libraries. You can find the list below:
  * Python 3.6+
  * Pandas
  * Sklearn
  * Nltk
  * Flask
  * Plotly
In case you miss a package, you can install it using the following command:
```pip install <PackageName>```

<a name="Running"></a>
## Running the project
Follow the following steps to run project. The steps 2 to 4 are optionnal, this is just in case you want to run the ETL and ML parts yourself.
1. Clone the repo on your local system with: ```git clone https://github.com/ClemHuriaux/udacity-disaster-response-pipeline/```
2. Navigate to the workspace with: ```cd workspace```
3. Run the ETL pipeline: ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
4. Run the ML pipeline: ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
5. Navigate to the app folder: ```cd app```
6. Run the app: ```python run.py```
7. Navigate to the url: ```http://0.0.0.0:3001```
And here we go, you are on the website
