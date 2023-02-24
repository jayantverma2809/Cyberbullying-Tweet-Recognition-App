# Cyberbullying-Tweet-Recognition-App https://lnkd.in/d_beCUvK

This project involved analyzing tweets. 

By understanding the words involved in the tweet, we are going to predict whether a tweet is a cyberbullying tweet or not and if it is a cyberbullying tweet then predicting nature of the cyberbullying into 6 Categories:
* Age
* Ethnicity
* Gender
* Religion
* Other Cyberbullying

* Dataset Used : Cyberbullying Classification data from Kaggle ( https://lnkd.in/d7pfHGT8)

* Approach:-
1. Installing the required libraries and importing the data set using pandas was the first step.
2. Initial review of the data and checked the provided data set for any missing values.
3. Preformed Preprocessing of text which involved :

~ Removing emoji
~ Converting text to lowercase, removing (/r, /n characters), URLs,
non-utf characters, Numbers, punctuations, stopwords
~ Removing Contractions  
~ Cleaning Hashtags  
~ Filter Special characters
~ Removing Multi-space characters
~ Stemming
~ Lemmatization
4. Handling Duplicates and removing them
5. Performed Exploratory Data Analysis
6. Train and test split
7. tf-idf Vectorization
8. Trying different base models :-
~ Logistic Regression
~ Support Vector Classifier
~ Naive Bayes Classifier
~ Decision Tree Classifier
~ Random Forest Classifier
~ Ada Boost Classifier
9. Fine Tuning Support Vector Classifier
10. Model Evaluation and Saving the model
11. Created the Web App using Streamlit
12. Deployed Web App on Streamlit

* Libraries Used : pandas, numpy, matplotlib, seaborn, stats, scipy, re, pickle, string, image, collections, statsmodel, flask, nltk, emoji, wordcloud, streamlit

* Deployment Platform : Streamlit

* Kaggle : https://lnkd.in/dTe7PCqx
