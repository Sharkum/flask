import re
import string
import numpy as np
from flask import Flask, render_template, request
import pickle
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import sequence
from loading import *

model = init()

with open('tokenizer.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

with open('logistic_cv.pkl', 'rb') as f:  # Loading the model
    count_vect, lr = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')  # The homepage


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        Text = request.form.get("Review")  # Takes input from the user
        # call preprocessDataAndPredict and pass inputs
        try:
            feed = text_cleaning(Text)
            # pass prediction to template
            prediction = predict_sentiment(feed)  # assigns the sentiment to Prediction
            score = predict_score(feed)  # assigns rating to score
            return render_template('predict.html', prediction=prediction, score=score)  # The output page

        except ValueError:  # To handle invalid inputs
            return "Please Enter Valid Review"

        pass
    pass


def text_cleaning(text):  # Removes the redundant words and cleans the text
    STOPWORDS = set(stopwords.words('english'))
    PUNCT_TO_REMOVE = string.punctuation
    # Lower casing the string
    text = text.lower()
    # Removing the punctuation marks
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    # Removing the Stopwords
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    # Removing any Web Based symbols
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    return text


def predict_sentiment(feed):  # Predicting the sentiment of the review
    feed = count_vect.transform([feed])  # Convert the text into vector for the model
    prediction = lr.predict(feed)  # Calling the logistic regression model
    if prediction == 1:  # Converting the {0,1} output into {Negative, Positive}
        return 'Positive Review'
    else:
        return 'Negative Review'


def predict_score(feed):  # Predicting the sentiment of the review
    feed = tokenizer.texts_to_sequences([feed])  # Convert the text into vector for the model
    feed = sequence.pad_sequences(feed, maxlen=512)  # Makes the length of input 512
    array = model.predict(feed)  # Calling the LSTM model
    score = np.argmax(array) + 1  # finding the score associated with max probability
    return score

if __name__ == "__main__" :
    app.run()