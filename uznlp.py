# Imports
import streamlit as st
from skimage import io
import requests

# Model related imports
import torch
from transformers import pipeline

API_KEY = 'AIzaSyDzNkKPC8uNyx-3lkjz7sh7MaF5XWqXHhA'


def translate_to_english(text):
    r = requests.get(f'https://translation.googleapis.com/language/translate/v2?key={API_KEY}&q={text}&souce=uz&target=en')
    return r.json()['data']['translations'][0]['translatedText']


def translate_to_uzbek(text):
    r = requests.get(f'https://translation.googleapis.com/language/translate/v2?key={API_KEY}&q={text}&souce=en&target=uz')
    return r.json()['data']['translations'][0]['translatedText']


# Load main models
@st.cache(allow_output_mutation=True,
          hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()})
def load_models():
    # Load summarization model
    summarizer = pipeline("summarization")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)

    return summarizer


# Variables for storing app persistent data
@st.cache(persist=True, allow_output_mutation=True)
def cached_variables():
    return {
        'img_idx': 0,
        'your_answer': '',
        'xray_idx': 0
    }


# Global variables



if __name__ == '__main__':
    # Load models
    summarizer = load_models()

    # Sidebar menu
    st.sidebar.header('UzNLP')
    option = st.sidebar.selectbox('Select model', (' ', 
                                                  'Summarization', 
                                                  'Sentiment', 
                                                  'Generation',
                                                  'Translation', 
                                                  ))

    # Landing page
    if option == ' ':
        title = 'UzNLP'
        intro_message_uzb = "NLP modellar o'zbek tillida."
        intro_message_rus = "НЛП модели на узбекском языке."
        intro_message_eng = "NLP models in Uzbek language."
        st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: white;'>{intro_message_uzb}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: white;'>{intro_message_rus}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: white;'>{intro_message_eng}</h3>", unsafe_allow_html=True)
        landing_img_path = 'C:\\Users\\Otabek Nazarov\\Desktop\\ML\\kaggle\\medtrain\\data\\landing.jpg'
        landing_image = io.imread(landing_img_path, as_gray=False)
        col1, col2, col3 = st.columns([0.1, 7, 0.1])
        col2.image(landing_image)

    # Breast cancer classifier page
    elif option == 'Summarization':
        # General UI
        app_cache = cached_variables()
        st.title('Text summarization. Суммаризация текста. Tekst summariziyasi.')
        st.markdown(' ')
        st.markdown(' ')
        
        col1, col2 = st.columns([1, 1])

        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="question",
            height=200,
            )
        query_uzb = st.session_state.question
        query_eng = translate_to_english(query_uzb)

        
        if query_eng == '':
            answer = ''
        else:
            answer_eng = summarizer(query_eng, min_length=5, max_length=150)[0]['summary_text']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        # print(f'{question_uzb}-{question_eng}-{answer_eng}-{answer_uzb}')

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb}", 
            key="answer",
            height=200,
            )

    # Chest X-ray diagnosis page
    elif option == 'Sentiment':
        app_cache = cached_variables()
        st.title('Sentiment analysis. Анализ сентиментов. Sentiment analizi.')
        st.markdown(' ')
        st.markdown(' ')

    elif option == 'Generation':
        app_cache = cached_variables()
        st.title('Text generation. Генерация текста. Tekst generatsiyasi.')
        st.markdown(' ')
        st.markdown(' ')

    elif option == 'Translation':
        app_cache = cached_variables()
        st.title('Text translation. Перевод текста. Tekst tarjimasi.')
        st.markdown(' ')
        st.markdown(' ')