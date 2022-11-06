# Imports
import requests
import streamlit as st
from transformers import pipeline

from constants import API_KEY


# Use google API to translate uzbek text to english
def translate_to_english(text):
    r = requests.get(f'https://translation.googleapis.com/language/translate/v2?key={API_KEY}&q={text}&souce=uz&target=en')
    return r.json()['data']['translations'][0]['translatedText']


# Use google API to translate english text to uzbek
def translate_to_uzbek(text):
    r = requests.get(f'https://translation.googleapis.com/language/translate/v2?key={API_KEY}&q={text}&souce=en&target=uz')
    return r.json()['data']['translations'][0]['translatedText']


# Load NLP models
@st.cache(allow_output_mutation=True)
def load_models():
    # Load summarization model
    summarizer = pipeline("summarization")

    # Load semantic classification model
    semantic = pipeline("text-classification", model='cardiffnlp/twitter-roberta-base-sentiment')

    # Load toxicity model
    toxicity = pipeline("text-classification", model='unitary/toxic-bert')

    # Load generation model
    generation = pipeline("text-generation")

    # Load image captioning model
    qa = pipeline("question-answering")

    return {
        'summarizer': summarizer,
        'semantic': semantic,
        'toxicity': toxicity,
        'generation': generation,
        'qa': qa,
    }