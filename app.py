import pandas as pd
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordNetLemmatizer
from Prediction import *
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Page title
st.set_page_config(page_title="Jayant Verma", page_icon=Image.open('statics/me.png'),initial_sidebar_state="expanded")

# Add profile image
profile_image = Image.open("statics/me.png")
st.sidebar.image(profile_image, use_column_width=True)

# Add contact information
st.sidebar.title("Jayant Verma")
st.sidebar.write("Data Scientist")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("jayantverma9380@gmail.com")
st.sidebar.subheader("[LinkedIn](https://www.linkedin.com/in/jayantverma28)")
st.sidebar.subheader("[GitHub](https://github.com/jayantverma2809)")
st.sidebar.subheader("[Kaggle](https://www.kaggle.com/jayantverma9380)")

#Skills
st.sidebar.header("Skills")
st.sidebar.write("Here are some of my top skills:")
st.sidebar.write("- Python programming")
st.sidebar.write("- SQL")
st.sidebar.write("- Data analysis and visualization")
st.sidebar.write("- Feature Engineering & Feature Selection")
st.sidebar.write("- Machine learning")

#Projects
st.sidebar.title("Other Projects")
st.sidebar.write("Here are some of my projects:")
st.sidebar.header("Machine Learning Projects")
st.sidebar.subheader("[Used Phone Price Prediction](https://usedphonepriceprediction.azurewebsites.net/)")
st.sidebar.write("Description: Using unsupervised learning techniques to predict prices of used phones using their various features such as days used, camera, battery,etc.")
st.sidebar.header("Analysis Projects")
st.sidebar.subheader("[Stock Analysis Project]()")
st.sidebar.write("Description: Under this analysis project, the app does fundamental and technical analysis on the stock provided as input and provides various helpful insights which help investors to take better decisions")
st.sidebar.header("Data Preprocessing Projects")
st.sidebar.subheader("[EDA & Feature Engineering - Bike Sharing Data](https://lnkd.in/dzjAsajs)")
st.sidebar.write("Description: Under this data preprocessing project, I have performed time series analysis, exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, etc to convert raw data into model training ready data.")
st.sidebar.subheader("[EDA & Feature Engineering - Wine Quality Data](https://lnkd.in/dKRMT7Ym)")
st.sidebar.write("Under this data preprocessing project, I have performed exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, standardization to convert raw data into model training ready data.")


st.write('''
# Cyberbullying Tweet Recognition App

This app predicts the nature of the tweet into 6 Categories.
* Age
* Ethnicity
* Gender
* Religion
* Other Cyberbullying
* Not Cyberbullying

***
''')

image = Image.open('statics/twitter.png')
st.image(image, use_column_width= True)

# Text Box
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height= 150)
print(tweet_input)
st.write('''
***
''')

# print input on webpage
if tweet_input:
    st.header('''
    ***Predicting......
    ''')
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')
st.write('''
***
''')

# Output on the page
st.header("Prediction")
if tweet_input:
    prediction = prediction(tweet_input)
    if prediction == "age":
        st.image("statics/Age.png",use_column_width= True)
    elif prediction == "ethnicity":
        st.image("statics/Ethnicity.png",use_column_width= True)
    elif prediction == "gender":
        st.image("statics/Gender.png",use_column_width= True)
    elif prediction == "other_cyberbullying":
        st.image("statics/Other.png",use_column_width= True)
    elif prediction == "religion":
        st.image("statics/Religion.png",use_column_width= True)
    elif prediction == "not_cyberbullying":
        st.image("statics/not_cyber.png",use_column_width= True)
else:
    st.write('''
    ***No Text Entered!***
    ''')

st.write('''***''')
