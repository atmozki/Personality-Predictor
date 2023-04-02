# imports ----------------------------------------------------------------------
from ast import alias
from ctypes import alignment
from email.mime import image
from textblob import TextBlob
import streamlit as st
import pandas as pd
from PIL import Image

# predict
import tensorflow
import numpy as np
import streamlit as st
import os
from tensorflow import keras
import re
import string
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

# visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sentiment analysis & subjectivity analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
s = SentimentIntensityAnalyzer()

# Initialisation ----------------------------------------------------------------------

# Predict Function Global declaration

def predict(model, input):
    print("User input : ", input)
    class_names = ['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ',
                   'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP']
    
    # GloVe
    # Load tokenizer
    # Tokenization is a process of converting a text into tokens(smaller chunks of text, strings into letters, words, or numbers)
    with open(picklepath, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
        seq = loaded_tokenizer.texts_to_sequences([input])

    padded = pad_sequences(seq, maxlen=max_sentence_length)
    pred = model.predict(padded)
    prediction = format(class_names[np.argmax(pred[0])])

    return prediction


st.set_page_config(
    page_title="PERSONALITY PREDICTOR",
    page_icon="🔮",
    layout="centered"
)

# Datasets
df = pd.read_csv("datasets/mbti_1_cleaned_all.csv")
df2 = pd.read_csv("datasets/data.csv")

# Sidebar ------------------------------------------------------------------------------

st.sidebar.subheader("Choose a Feature")
sections = ['Personality Prediction Tool', 'Data Visualization']
selected_sect = st.sidebar.selectbox("Predict or Visualize:", sections)

# Section one: Prediction --------------------------------------------------------------

image = Image.open('logo.png')
col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
col2.image(image, use_column_width=True)

if selected_sect == 'Personality Prediction Tool':
    st.subheader("Let's guess your MBTI...")
    user_input = st.text_input(value="", label="Enter text here:",
                               help="Type something here, then press the Enter!")

    st.info("Meaningful Sentences only! No emojis or symbols!")
    
    # simple text cleaning
    user_input = user_input.lower()
    user_input = re.sub(r'[^\w\s]', '', user_input)
    user_input = re.sub(r"\d", '', user_input)

    print(user_input)

    # sidebar
    mdls = ['Kaggle Data', 'Reddit Data']
    selected_mod = st.radio(
        label='Choose a model for Prediction', options=mdls)

    # load models
    # 1. Model trained with Kaggle Data
    if selected_mod == 'Kaggle Data':
        filepath = "model/model_kaggle.h5"
        picklepath = "model/tokenizer_kaggle.pickle"
        max_sentence_length = 764

        with st.spinner("Predicting..."):
            model = keras.models.load_model(filepath)

    # 2. Model trained with Reddit Data
    else:
        filepath = "model/model_reddit.h5"
        picklepath = "model/tokenizer_reddit.pickle"
        max_sentence_length = 881

        with st.spinner("Predicting..."):
            model = keras.models.load_model(filepath)

    # MBTI Prediction Tool
    if (user_input != ""):
        if len(user_input) < 10:
            st.error("Invalid text! Enter text with more than 10 letters")
        else:
            # MBTI
            with st.spinner("Result"):
                prediction = predict(model, user_input)

            print("Predicted : ", prediction)

            # Subjectivity
            sc = TextBlob(user_input).sentiment.subjectivity
            if sc > 0.5:
                subjectivity = 'Opinionated'

            elif sc < 0.5:
                subjectivity = 'Subjective'

            else:
                subjectivity = 'Somewhat Subjective and Objecive'

            # Result
            score = s.polarity_scores(user_input)
            if score['compound'] > 0:
                st.subheader("Result")
                st.write("The Predictor thinks you're an ", prediction,
                         "+ the words are ", subjectivity, "and Positive😃")

            elif score['compound'] == 0:
                st.subheader("Result")
                st.write("The Predictor thinks you're an ", prediction,
                         "+ the words are ", subjectivity, "and Neutral😶")

            elif score['compound'] < 0:
                st.subheader("Result")
                st.write("The Predictor thinks you're an ", prediction,
                         "+ the words are ", subjectivity, "and Negative😢")
        
            #Explanation(Basic Template explaination)

            st.subheader("Explanation")

            if prediction == 'ENFJ':
                st.write("ENFJs are natural-born leaders who exude charisma and warmth. They are the life of the party and the first to volunteer for a cause. They are also deeply empathetic and intuitive, and they use these traits to help others. ENFJs are often described as the most charismatic of all the types, and they are often found in positions of leadership. They are also known for their ability to inspire others and bring out the best in people. ENFJs are often described as the most charismatic of all the types, and they are often found in positions of leadership. They are also known for their ability to inspire others and bring out the best in people.")
                st.write("For More Details Click here: [ENFJ](https://www.16personalities.com/enfj-personality)")
            elif prediction == 'ENFP':
                st.write("ENFPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ENFPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ENFP](https://www.16personalities.com/enfp-personality)")
            elif prediction == 'ENTJ':
                st.write("ENTJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ENTJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ENTJ](https://www.16personalities.com/entj-personality)")
            elif prediction == 'ENTP':
                st.write("ENTPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ENTPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ENTP](https://www.16personalities.com/entp-personality)")
            elif prediction == 'ESFJ':
                st.write("ESFJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ESFJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ESFJ](https://www.16personalities.com/esfj-personality)")
            elif prediction == 'ESFP':
                st.write("ESFPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ESFPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ESFP](https://www.16personalities.com/esfp-personality)")
            elif prediction == 'ESTJ':
                st.write("ESTJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ESTJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ESTJ](https://www.16personalities.com/estj-personality)")
            elif prediction == 'ESTP':
                st.write("ESTPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ESTPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ESTP](https://www.16personalities.com/estp-personality)")
            elif prediction == 'INFJ':
                st.write("INFJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. INFJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [INFJ](https://www.16personalities.com/infj-personality)")
            elif prediction == 'INFP':
                st.write("INFPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. INFPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [INFP](https://www.16personalities.com/infp-personality)")
            elif prediction == 'INTJ':
                st.write("INTJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. INTJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [INTJ](https://www.16personalities.com/intj-personality)")
            elif prediction == 'INTP':
                st.write("INTPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. INTPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [INTP](https://www.16personalities.com/intp-personality)")
            elif prediction == 'ISFJ':
                st.write("ISFJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ISFJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ISFJ](https://www.16personalities.com/isfj-personality)")
            elif prediction == 'ISFP':
                st.write("ISFPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ISFPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ISFP](https://www.16personalities.com/isfp-personality)")
            elif prediction == 'ISTJ':
                st.write("ISTJs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ISTJs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ISTJ](https://www.16personalities.com/istj-personality)")
            elif prediction == 'ISTP':
                st.write("ISTPs are natural-born leaders who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause. ISTPs are warm, empathetic, and creative individuals who are always looking for new ways to help others. They are often described as the life of the party and the first to volunteer for a cause.")
                st.write("For More Details Click here: [ISTP](https://www.16personalities.com/istp-personality)")
            else:
                st.write("Please enter a valid input")
            
            st.subheader("Thank You ")


# Section two: Visualisation ------------------------------------------------------------

elif selected_sect == 'Data Visualization':
    st.title("Data Visualization")

    # sidebar
    st.sidebar.markdown("***")
    st.sidebar.caption("What do they mean?")

    with st.sidebar.expander("16 MBTI Types"):
        st.write('**Analysts**: INTJ, INTP, ENTJ, ENTP')
        st.write('**Diplomats**: INFJ, INFP, ENFJ, ENFP')
        st.write('**Sentinels**: ISTJ, ISFJ, ESTJ, ESFJ')
        st.write('**Explorers**: ISTP, ISFP, ESTP, ESFP')

    with st.sidebar.expander("4 Dimensions"):
        st.write('**IE**: Introvert, Extrovert')
        st.write('**NS**: Intuition, Sensing')
        st.write('**TF**: Thinking, Feeling')
        st.write('**JP**: Judging, Perceiving')

    # Selection Dropdown
    sections = ['Text Analysis', 'Personality Types']
    selected_viz = st.selectbox("Explore the Kaggle Data:", sections)

    # Selection one
    if selected_viz == 'Personality Types':
        st.subheader("4 Dimensions")
        # Donut Charts
        fig1 = {
            "data": [
                {"values": [6675, 1999], "labels": ["I", "E"], "domain": {"x": [0.2, 0.5], "y": [0.5, .95]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [7477, 1197], "labels": ["N", "S"], "domain": {"x": [0.51, 0.8], "y": [0.5, .95]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [4693, 3981], "labels": ["T", "F"], "domain": {"x": [0.2, 0.5], "y": [0, 0.45]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"},
                {"values": [5240, 3434], "labels": ["J", "P"], "domain": {"x": [0.51, 0.8], "y": [0, 0.45]},
                 "hoverinfo":"label+percent", "hole": .4, "type": "pie"}],
            "layout": {"piecolorway": px.colors.qualitative.Pastel2}
        }

        st.plotly_chart(fig1)

        st.subheader("16 Personality Types")
        # MBTI value counts
        df3 = pd.DataFrame({
            "mbti": ["INFP", "INFJ", "INTP", "INTJ", "ENTP", "ENFP", "ISTP", "ISFP", "ENTJ", "ISTJ", "ENFJ", "ISFJ", "ESTP", "ESFP", "ESFJ", "ESTJ"],
            "value": [1831, 1470, 1304, 1091, 685, 675, 337, 271, 231, 205, 190, 166, 89, 48, 42, 39],
        })

        fig2 = px.bar(df3, x="mbti", y="value", height=400,
                      color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig2.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig2)

    # Selection two
    elif selected_viz == 'Text Analysis':
        st.subheader("Sentiment & Subjectivity Analysis")
        col1, col2 = st.columns(2)
        # Sentiment
        with col1:
            df3 = pd.DataFrame({
                "sentiment": ["Positive", "Negative", "Neutral"],
                "value": [7530, 1127, 17],
            })

            fig3 = px.bar(df3, x="sentiment", y="value", height=400, width=400,
                          color_discrete_sequence=px.colors.qualitative.Pastel1)
            fig3.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig3)

        # Subjectivity
        with col2:
            df3 = pd.DataFrame({
                "subjectivity": ["Subjective", "Objective", "Neutral"],
                "value": [7285, 1388, 1],
            })

            fig4 = px.bar(df3, x="subjectivity", y="value", height=400,
                          width=400, color_discrete_sequence=px.colors.qualitative.Plotly)
            fig4.update_layout(legend_title_text='', showlegend=False)
            st.plotly_chart(fig4)

        # Top 30 words
        st.subheader("Top 30 Words Used")
        df_wc = pd.read_csv("datasets/wordcloud.csv")
        df_wc = df_wc.iloc[:30]

        fig5 = px.bar(df_wc, x="Word", y="WordCount", height=400,
                      color_discrete_sequence=px.colors.qualitative.Pastel2)
        fig5.update_layout(legend_title_text='', showlegend=False)
        st.plotly_chart(fig5)

        if st.checkbox("Show wordcloud"):
            # Word Cloud
            df_wc = pd.read_csv("datasets/wordcloud.csv")

            words = " ".join(df_wc.Word)
            cloud = WordCloud(width=800, height=400, max_words=200,
                              background_color='White', colormap='tab10').generate(words)

            plt.imshow(cloud, interpolation='gaussian')
            plt.axis("off")
            st.pyplot(plt)
