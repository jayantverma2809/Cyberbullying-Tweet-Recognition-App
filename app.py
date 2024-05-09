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
st.set_page_config(page_title="Jayant Verma", page_icon=Image.open('statics/Me.jpg'),initial_sidebar_state="expanded")

# Add profile image
profile_image = Image.open('statics/Me.jpg')
st.sidebar.image(profile_image, use_column_width=True)

# Add contact information
st.sidebar.title("Jayant Verma")
st.sidebar.write("Data Scientist & AI Engineer")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("jayantverma9380@gmail.com")
st.sidebar.subheader("[LinkedIn](https://www.linkedin.com/in/jayantverma28)")
st.sidebar.subheader("[X](https://x.com/__kanhaiya__)")
st.sidebar.subheader("[Instagram](http://instagram.com/_._.kanhaiya)")
st.sidebar.subheader("[GitHub](https://github.com/jayantverma2809)")
st.sidebar.subheader("[Kaggle](https://www.kaggle.com/jayantverma9380)")

#Skills
st.sidebar.header("Skills")
st.sidebar.write("Here are some of my top skills:")
st.sidebar.write("- Python, SQL")
st.sidebar.write("- Databases : PostgreSQL, MongoDB, Pinecone, Qdrant")
st.sidebar.write("- Libraries & Frameworks : FastAPI, Openai, Llama-index, Langchain, Pinecone, LLM-Sherpa, Openpipe, PyMongo, Transformers, Huggingface, Boto3, Pandas, Numpy, Pandasai, Matplotlib, Seaborn, Scikit-Learn, Streamlit, BeautifulSoup")
st.sidebar.write("- Models : GPT-3.5, GPT-4, Text-da-vinci, Dall.E,Instructor Large, Llama-2, LLama-3, Stable Diffusion, Bart-Base, Claude Instant V1, Claude 3, Phi-3")
st.sidebar.write("- Data Science : Data Collection, Data Wrangling, Data Visualization, Exploratory Data Analysis, Feature Engineering, Feature Selection, Machine Learning(Regression, Classification, Clustering), Model Evaluation, Model Deployment")
st.sidebar.write("- AWS Services : Bedrock, Textract, Cognito")
st.sidebar.write("- Version Control & Deployment : Git & Github")

st.sidebar.title("Experience")
st.sidebar.header("- Data Scientist - Softsensor.ai [June, 2023 - Present]")
st.sidebar.write("Worked on Softsensor X, an AI-based application that allows users to perform QA on their data source, compare and contrast QA between multiple docs, image generation, and offers data visualization capabilities.")
st.sidebar.subheader("Key responsibilities and achievements include:")
st.sidebar.write("- Developed and managed the entire backend of the web app using FastAPI. - Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc. - Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation")
st.sidebar.write("- Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc.")
st.sidebar.write("- Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation")

st.sidebar.title("Open Source Contributions")
st.sidebar.subheader("Llama-index")
st.sidebar.write("An open-source framework for developing applications that use language models. It's a data framework that can connect custom data sources to large language models (LLMs).")

#Projects
st.sidebar.title("Projects")
st.sidebar.write("Here are some of my projects:")
st.sidebar.subheader("- Softsensor X")
st.sidebar.write("Developed and managed the entire backend of the web app using FastAPI.")
st.sidebar.write("Created pipelines for data ingestion, RAG (Retrieval Augmented Generation), RAG Fusion, Hybrid Search, Metadata filtering, Data Visualization, Image Generation, Agent for full automation, Query Transformation, Compare and contrast, etc.")
st.sidebar.write("Fine-tuned models such as GPT-3.5, Llama-2 for QA use case and Bart-base for query transformation.")
st.sidebar.subheader("- [Stock Analysis Project](https://jayantverma-stock-analysis-app.streamlit.app/)")
st.sidebar.write("Under this analysis project, the app does fundamental and technical analysis on the stock provided as input and provides various helpful insights which help investors to take better decisions")
st.sidebar.subheader("- [Used Phone Price Prediction](https://usedphonepriceprediction.azurewebsites.net/)")
st.sidebar.write("Using unsupervised learning techniques to predict prices of used phones using their various features such as days used, camera, battery,etc.")
st.sidebar.subheader("- [EDA & Feature Engineering - Bike Sharing Data](https://lnkd.in/dzjAsajs)")
st.sidebar.write("Under this data preprocessing project, I have performed time series analysis, exploratory data analysis and various feature engineering techniques such as transformations, handling outliers, etc to convert raw data into model training ready data.")
st.sidebar.subheader("- [EDA & Feature Engineering - Wine Quality Data](https://lnkd.in/dKRMT7Ym)")
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
