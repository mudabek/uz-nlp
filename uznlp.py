# Imports
import streamlit as st
from skimage import io
import requests

# Model related imports
import torch
from transformers import pipeline

API_KEY = 'AIzaSyDzNkKPC8uNyx-3lkjz7sh7MaF5XWqXHhA'

SEMANTIC_LABELS = {
    'LABEL_0': 'Negative. Негативный. Negativ.',
    'LABEL_1': 'Neutral. Нейтральный. Neytral.',
    'LABEL_2': 'Positive. Позитивнвый. Pozitiv.'
}


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

    # Load semantic classification model
    semantic = pipeline("text-classification", model='cardiffnlp/twitter-roberta-base-sentiment')

    # Load toxicity model
    # toxicity = pipeline("text-classification", model='unitary/toxic-bert')

    # Load generation model
    # generation = pipeline("text-generation")

    # Load image captioning model
    qa = pipeline("question-answering")

    return {
        'summarizer': summarizer,
        'semantic': semantic,
        # 'toxicity': toxicity,
        # 'generation': generation,
        'qa': qa,
    }


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
    models_dict = load_models()

    # Sidebar menu
    st.sidebar.header('UzNLP')
    option = st.sidebar.selectbox('Select model', (' ', 
                                                  'Summarization', 
                                                  'Sentiment Classification',
                                                  'Toxicity Classification', 
                                                  'Generation',
                                                  'Translation',
                                                  'Question Answering', 
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

    # Text summarization
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
            answer_uzb = ''
        else:
            answer_eng = models_dict['summarizer'](query_eng, min_length=5, max_length=150)[0]['summary_text']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb}", 
            key="answer",
            height=200,
            )

    # Sentiment classification   
    elif option == 'Sentiment Classification':
        app_cache = cached_variables()
        st.title('Sentiment classifier. Классификация сентиментов. Sentiment klassifikaciyasi.')
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
            score= ''
        else:
            result = models_dict['semantic'](query_eng)[0]
            answer = SEMANTIC_LABELS[result['label']]
            score = f"({result['score']*100:.1f} %)"

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer} {score}", 
            key="answer",
            height=200,
            )
    
    # Toxicity classification   
    elif option == 'Toxicity Classification':
        app_cache = cached_variables()
        st.title('Toxicity classifier. Классификация токсичности. Yomon soz klassifikaciyasi.')
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
            toxicity_score = models_dict['toxicity'](query_eng)[0]['score']

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"Toxicity - {toxicity_score*100:.1f}", 
            key="answer",
            height=200,
            )

    # Text generation
    elif option == 'Generation':
        app_cache = cached_variables()
        st.title('Text generation. Генерация текста. Tekst generatsiyasi.')
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
            answer_eng = models_dict['generation'](query_eng,max_length=180)[0]['generated_text']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb}", 
            key="answer",
            height=200,
            )
        

    # Text translation
    elif option == 'Translation':
        app_cache = cached_variables()
        st.title('Text translation. Перевод текста. Tekst tarjimasi.')
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

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{query_eng}", 
            key="answer",
            height=200,
            )

    # Question answering
    elif option == 'Question Answering':
        app_cache = cached_variables()
        st.title('Question Answering. Вопросы-ответы. Savol-javoblar.')
        st.markdown(' ')
        st.markdown(' ')

        col1, col2 = st.columns([1, 1])

        col1.text_area(
            "Tekst. Text. Текст.", 
            key="context",
            height=200,
            )

        col1.text_area(
            "Savol. Question. Вопрос.", 
            key="question",
            height=100,
            )

        context_uzb = st.session_state.context
        query_uzb = st.session_state.question
        context_eng = translate_to_english(context_uzb)
        query_eng = translate_to_english(query_uzb)

        
        if query_eng == '':
            answer_uzb = ''
            score = ''
        else:
            result = models_dict['qa'](context=context_eng, question=query_eng)
            score = f"({result['score']*100:.1f} %)"
            answer_eng = result['answer']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb} {score}", 
            key="answer",
            height=300,
            )