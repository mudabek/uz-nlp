# Imports
import streamlit as st
from skimage import io

from constants import SEMANTIC_LABELS
from utils import translate_to_english, translate_to_uzbek, load_models


if __name__ == '__main__':
    # Load models
    models_dict = load_models()

    # Sidebar menu
    st.sidebar.header('UzNLP')
    option = st.sidebar.selectbox('Select model. Model tanlang. Выберите Модель.', (' ', 
                                                  'Summarization', 
                                                  'Sentiment Classification',
                                                  'Toxicity Classification', 
                                                  'Generation',
                                                  'Translation',
                                                  'Question Answering', 
                                                  ))

    # Landing page
    if option == ' ':
        landing_img_path = 'landing.jpg'
        landing_image = io.imread(landing_img_path, as_gray=False)
        col1, col2, col3 = st.columns([0.1, 7, 0.1])
        col2.image(landing_image)

    # Text summarization
    elif option == 'Summarization':
        # General UI
        st.title('Text summarization. Суммаризация текста. Tekst summariziyasi.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get text to summarize
        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="question",
            height=200,
            )
        query_uzb = st.session_state.question
        query_eng = translate_to_english(query_uzb)

        # Get summary from model
        if query_eng == '':
            answer_uzb = ''
        else:
            answer_eng = models_dict['summarizer'](query_eng, min_length=5, max_length=150)[0]['summary_text']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        # Display the results
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb}", 
            key="answer",
            height=200,
            )

    # Sentiment classification   
    elif option == 'Sentiment Classification':
        st.title('Sentiment classifier. Классификация сентиментов. Sentiment klassifikaciyasi.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get text for semantic classification
        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="sent_question",
            height=200,
            )
        query_uzb = st.session_state.sent_question
        query_eng = translate_to_english(query_uzb)

        # Get classification result
        if query_eng == '':
            answer = ''
            score = ''
        else:
            result = models_dict['semantic'](query_eng)[0]
            answer = SEMANTIC_LABELS[result['label']]
            score = f"({result['score']*100:.1f} %)"

        # Display the result
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer} {score}", 
            key="answer",
            height=200,
            )
    
    # Toxicity classification   
    elif option == 'Toxicity Classification':
        # General UI
        st.title('Toxicity classifier. Классификация токсичности. Yomon soz klassifikaciyasi.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get text for toxicity classification
        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="question",
            height=200,
            )
        query_uzb = st.session_state.question
        query_eng = translate_to_english(query_uzb)

        # Get classification results
        if query_eng == '':
            toxicity_score = ''
        else:
            toxicity_score = f"Toxicity - ({models_dict['toxicity'](query_eng)[0]['score']*100:.1f} %)"

        # Display the result
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{toxicity_score}", 
            key="answer",
            height=200,
            )

    # Text generation
    elif option == 'Generation':
        # General UI
        st.title('Text generation. Генерация текста. Tekst generatsiyasi.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get initial prompt for text generation
        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="question",
            height=200,
            )
        query_uzb = st.session_state.question
        query_eng = translate_to_english(query_uzb)

        # Get model's generation result
        if query_eng == '':
            answer_uzb = ''
        else:
            answer_eng = models_dict['generation'](query_eng,max_length=180)[0]['generated_text']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        # Display the result
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb}", 
            key="answer",
            height=200,
            )
        

    # Text translation
    elif option == 'Translation':
        st.title('Text translation. Перевод текста. Tekst tarjimasi.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get text in Uzbek and translate it to English
        col1.text_area(
            "Kiritish. Input. Ввод.", 
            key="gen_question",
            height=200,
            )
        query_uzb = st.session_state.gen_question
        query_eng = translate_to_english(query_uzb).replace("&#39;", "'")

        # Display translation result
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{query_eng}", 
            key="answer",
            height=200,
            )

    # Question answering
    elif option == 'Question Answering':
        # General UI
        st.title('Question Answering. Вопросы-ответы. Savol-javoblar.')
        st.markdown(' ')
        st.markdown(' ')
        col1, col2 = st.columns([1, 1])

        # Get context for QA model
        col1.text_area(
            "Tekst. Text. Текст.", 
            key="context",
            height=150,
            )

        # Get question for QA model
        col1.text_area(
            "Savol. Question. Вопрос.", 
            key="question",
            height=50,
            )

        # Translate into Enlgish
        context_uzb = st.session_state.context
        query_uzb = st.session_state.question
        context_eng = translate_to_english(context_uzb)
        query_eng = translate_to_english(query_uzb)

        # Get model results
        if query_eng == '':
            answer_uzb = ''
            score = ''
        else:
            result = models_dict['qa'](context=context_eng, question=query_eng)
            score = f"({result['score']*100:.1f} %)"
            answer_eng = result['answer']
            answer_uzb = translate_to_uzbek(answer_eng)
            answer_uzb = answer_uzb.replace("&#39;", "'")

        # Display results
        col2.text_area(
            label="Natija. Результат. Result.",
            value=f"{answer_uzb} {score}", 
            key="answer",
            height=295,
            )