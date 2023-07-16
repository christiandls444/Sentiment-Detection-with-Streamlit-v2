import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from textblob import TextBlob
from wordcloud import WordCloud
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

pd.set_option('display.max_colwidth', None)

# Text cleaning and preprocessing
def clean_text(text):
    if text and isinstance(text, str):
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+|[^a-zA-Z]', ' ', text.lower())
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if len(word) > 1 and word not in stopwords.words('english')])
        text = ' '.join(list(dict.fromkeys(text.split())))
    else:
        text = ''
    return text

# Load the trained model
with open('./model/classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer
with open('./model/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def show_sentiment_detection():
    st.title("Sentiment Detection")
    new_sentence = st.text_area("Enter the text for sentiment analysis", "")
    
    analyze_clicked = st.button("Analyze")
       
    if analyze_clicked:
        if new_sentence:
            # Preprocess the user input
            cleaned_sentence = clean_text(new_sentence)
            new_sentence_features = vectorizer.transform([cleaned_sentence])
            probabilities = model.predict_proba(new_sentence_features)[0]

            sentiment_categories = ['Negative', 'Moderately Negative', 'Neutral', 'Moderately Positive', 'Positive']
            sorted_sentiments = sorted(zip(sentiment_categories, probabilities), key=lambda x: x[1], reverse=True)

            # Print the user input and predicted sentiment
            st.markdown(
                f'<div style="background-color: #262730; padding: 10px; border-radius: 5px;">'
                f'<span style="color: #fff;">Your sentence: {new_sentence}</span> '
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="background-color: #262730; margin-top: 10px; padding: 10px; border-radius: 5px;">'
                f'<span style="color: #fff;">Predicted sentiment: {sorted_sentiments[0][0]}</span> '
                '</div>',
                unsafe_allow_html=True
            )

            # Prepare data for bar chart
            sentiment_labels = [sentiment for sentiment, _ in sorted_sentiments]
            probabilities = [probability * 100 for _, probability in sorted_sentiments]

            # Display bar chart
            st.subheader("Sentiment Probabilities")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sentiment_labels, probabilities)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Probability (%)")
            ax.set_title("Sentiment Probabilities")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            for i, v in enumerate(probabilities):
                ax.text(i, v, f"{v:.2f}%", color='black', ha='center')
            plt.xticks()
            st.pyplot(fig)
            
             # Save as image button
            save_as_image(new_sentence, sorted_sentiments, fig)
            
        else:
            st.error("Please enter a sentence for sentiment analysis.")
            
def save_as_image(new_sentence, sorted_sentiments, fig):
    # Create a new PIL image with the text
    image = Image.new("RGB", (1200, 1200), "white")
    st_image = ImageDraw.Draw(image)
    font = ImageFont.truetype("./fonts/Arial.ttf", 18)
    # Split the new_sentence into a list of words
    words = new_sentence.split()

    # Group the words into lines of 12 words each
    lines = [words[i:i+14] for i in range(0, len(words), 14)]

    # Create a string with new lines for every 12 words
    sentence_lines = '\n'.join([' '.join(line) for line in lines])

    # Display the sentence with new lines every 12 words
    st_image.text((100, 50), f"Your sentence:\n{sentence_lines}", fill="black", font=font)

    # Calculate the height for each sentiment line
    sentiment_height = 30
    sentiment_start_y = 180
    for i, sentiment in enumerate(sorted_sentiments):
        sentiment_y = sentiment_start_y + (i * sentiment_height)
        st_image.text((100, sentiment_y), f"Predicted sentiment: {sentiment}", fill="black", font=font)
 
    # Save the chart as bytes
    chart_bytes = BytesIO()
    fig.savefig(chart_bytes, format='png')
    chart_bytes.seek(0)

    # Open the chart as an image
    chart_image = Image.open(chart_bytes)

    # Combine the text and chart images
    combined_image = Image.new("RGB", (950, 1200), "white")
    combined_image.paste(image, (0, 0))
    combined_image.paste(chart_image, (0, 300))

    # Save the combined image as bytes
    image_bytes = BytesIO()
    combined_image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Encode the image bytes to base64
    encoded_image = base64.b64encode(image_bytes.read()).decode()

    # Create the HTML for downloading the image
    href = f'<a href="data:image/png;base64,{encoded_image}" download="sentiment_analysis.png">Download</a>'

    # Display the download link
    st.markdown(href, unsafe_allow_html=True)
    
    
def show_home():
    st.title("Home")
    st.image("asset/sentiments.png", use_column_width=True)
    st.subheader("Welcome to the Sentiment Analysis App")
    st.write("Welcome to the Sentiment Analysis App! This application provides you with the tools to conduct sentiment analysis on text data. Whether you have a sentence or a paragraph, simply input your text, and the app will predict the sentiment associated with it. The sentiment categories include Positive, Moderately Positive, Negative, Moderately Negative, and Neutral. Please note that this app is based on an experimental methodology, and while we strive for accuracy, it's important to consider the insights gained and the iterative improvements made throughout the experiment. Feel free to explore and analyze your text data using this app. We hope it enhances your understanding of sentiment analysis and contributes to your research and learning journey.")
    
def show_exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    tweets = pd.read_csv('./data/Tweets.csv')
    st.subheader("Dataset")
    st.write("This dataset is sourced from my instructor, Amusa Abdulahi Tomisin, who is a Nigerian and teaches the course 'Mastering Sentiment Analysis Building a Powerful-Web Application' at Omdena School. I have obtained permission to use this dataset.")
    st.dataframe(tweets.sample(10))
    
    df = pd.read_csv('./data/output.csv')
    st.subheader("Text Cleaning and Preprocessing")
    st.dataframe(df[['text', 'clean_text']].sample(10))
    
    st.subheader("Sentiment Labeling with TextBlob Analysis")
    st.dataframe(df[['clean_text', 'textblob_polarity', 'sentiment_textblob']].sample(10))
    
    # It's important to note that I have decided to save the processed images to avoid the need for repetitive processing,
    # which can be time-consuming. This approach allows for quicker access to the results and facilitates further analysis
    
    st.subheader("Sentiment Counts")
    st.image("asset/sentiment_counts.png", use_column_width=True)
    
    st.subheader("Distribution of Sentiment by Sentence Length")
    st.image("asset/distribution_of_sentiment_by_sentences_length.png", use_column_width=True)
    
    st.subheader("Top 10 Frequently used Words")
    st.image("asset/top_frequently_used_words.png", use_column_width=True)
    
    st.subheader("Positive Word Cloud")
    st.image("asset/positive.png", use_column_width=True)
    
    st.subheader("Moderately Positive Word Cloud")
    st.image("asset/moderately_positive.png", use_column_width=True)
    
    st.subheader("Neutral Word Cloud")
    st.image("asset/neutral.png", use_column_width=True)
    
    st.subheader("Moderately Negative Word Cloud")
    st.image("asset/moderately_negative.png", use_column_width=True)
    
    st.subheader("Negative Word Cloud")
    st.image("asset/negative.png", use_column_width=True)

def show_author():
    st.title("Author")
    st.image("asset/profile.png", width=200)
    
    st.write("Hey everyone! I'm Christian M. De Los Santos, from the Philippines. I have over 2 years of experience in the field of data analytics, with a special focus on machine learning. I firmly believe that AI and ML have the power to bring about positive change in our communities, which is why I'm here, eager to make an impact. Learning from all of you brilliant minds is something I'm truly looking forward to. Let's collaborate and create something amazing together!")
    st.write("christiandelossantos444@gmail.com")
    st.markdown('<a href="https://www.linkedin.com/in/christiandls444/" target="_blank">LinkedIn Profile</a>', unsafe_allow_html=True)
    
def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Home", "Sentiment Detection", "Exploratory Data Analysis", "Author"])

    if selected_page == "Home":
        show_home()
    elif selected_page == "Sentiment Detection":
        show_sentiment_detection()
    elif selected_page == "Exploratory Data Analysis":
        show_exploratory_data_analysis()
    elif selected_page == "Author":
        show_author()


if __name__ == "__main__":
    main()
