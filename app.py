import streamlit as st
from time import sleep
from stqdm import stqdm  # For animation after submit event
from transformers import pipeline
import json
import spacy
import spacy_streamlit

def draw_all():
    st.markdown(
        """
        <style>
        /* General Page Styling */
        body {
            background-color: #f7f9fc;
            font-family: 'Arial', sans-serif;
        }

        /* Title Styling */
        .title-text {
            font-size: 40px;
            font-weight: bold;
            color: rgb(250 250 250);
            text-align: center;
            margin-bottom: 10px;
        }

        /* Subtitle Styling */
        .sub-text {
            font-size: 18px;
            color: rgb(230 234 241 / 65%);
            text-align: center;
            margin-bottom: 6px;
        }

        /* Features List Styling */
        .features-container {
            display: flex;
            justify-content: center;
        }
        .features-list {
            margin: 0;
            padding: 11px 28px;
            max-width: 600px;
            background: linear-gradient(135deg,#2e8cc4,#ff8e61);

            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            font-size: 20px;
            color: #ffffff;
            list-style-type: none;
            line-height: 1.5;
        }
        .features-list li {
            position: relative;
            padding: 10px 0;
        }
        .features-list li:before {
            content: '';
            margin-right: 10px;
            color: #264653;
        }
        </style>

        <div class="title-text">Welcome to NLP Genie </div>
        <p class="sub-text">
            Unleash NLP Genie! 🚀 Perfect for researchers,<br>
            businesses, and creators—turn complex texts into actionable insights instantly!
        </p>
        <div class="features-container">
            <ul class="features-list">
                <li>✨Smart Text Summarizer</li>
                <li>📍Named Entity Recognition</li>
                <li>💬Sentiment Analysis</li>
                <li>❓Question Answering</li>
                <li>📝Text Completion</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    draw_all()

def main():
    st.title("NLP Genie")
    menu = [
        "--Select--", "Summarizer", "Named Entity Recognition",
        "Sentiment Analysis", "Question Answering", "Text Completion"
    ]
    choice = st.sidebar.selectbox("Choose What you want to do!!", menu)

    if choice == "--Select--":
        st.write("""
            This is a Natural Language Processing Based Web App that can do anything you can imagine with text.
        """)

        st.write("""
            Natural Language Processing (NLP) is a computational technique
            to understand the human language in the way they speak and write.
        """)

        st.write("""
            NLP is a subfield of Artificial Intelligence (AI) designed to understand
            the context of text just like humans.
        """)

        st.image('banner_image.jpg')

    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        st.write("Enter the Text you want to summarize!")
        raw_text = st.text_area("Your Text", "")
        num_words = st.number_input("Enter the Number of Words in Summary")

        if raw_text != "" and num_words is not None:
            num_words = int(num_words)
            summarizer = pipeline('summarization', framework='pt')
            summary = summarizer(raw_text, min_length=num_words, max_length=50)
            s1 = json.dumps(summary[0])
            d2 = json.loads(s1)
            result_summary = d2['summary_text']
            result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(), result_summary.split('.'))))
            st.write(f"Here's your Summary: {result_summary}")

    elif choice == "Named Entity Recognition":
        nlp = spacy.load("en_core_web_trf")
        st.subheader("Text Based Named Entity Recognition")
        st.write("Enter the Text below to extract Named Entities!")
        raw_text = st.text_area("Your Text", "Enter Text Here")

        if raw_text != "Enter Text Here":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results!!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title="List of Entities")

    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_analysis = pipeline("sentiment-analysis",framework='pt')
        st.write("Enter the Text below to find out its Sentiment!")
        raw_text = st.text_area("Your Text", "Enter Text Here")

        if raw_text != "Enter Text Here":
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results!!"):
                sleep(0.1)
            if sentiment == "POSITIVE":
                st.write("""# This text has a Positive Sentiment. 🤗""")
            elif sentiment == "NEGATIVE":
                st.write("""# This text has a Negative Sentiment. 😤""")
            elif sentiment == "NEUTRAL":
                st.write("""# This text seems Neutral... 😐""")

    elif choice == "Question Answering":
        st.subheader("Question Answering")
        st.write("Enter the Context and ask the Question to find out the Answer!")
        question_answering = pipeline("question-answering")


        context = st.text_area("Context", "Enter the Context Here")
        question = st.text_area("Your Question", "Enter your Question Here")

        if context != "Enter Text Here" and question != "Enter your Question Here":
            result = question_answering(question=question, context=context)
            s1 = json.dumps(result)
            d2 = json.loads(s1)
            generated_text = d2['answer']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f"Here's your Answer:\n{generated_text}")

    elif choice == "Text Completion":
        st.subheader("Text Completion")
        st.write("Enter the incomplete text to complete it automatically using AI!")
        text_generation = pipeline("text-generation",framework='pt')
        message = st.text_area("Your Text", "Enter the Text to complete")

        if message != "Enter the Text to complete":
            generator = text_generation(message)
            s1 = json.dumps(generator[0])
            d2 = json.loads(s1)
            generated_text = d2['generated_text']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f"Here's your Generated Text:\n{generated_text}")


if __name__ == '__main__':
    main()
