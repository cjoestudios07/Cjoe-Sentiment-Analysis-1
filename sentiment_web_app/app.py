from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Dataset
data = {
    'text': [
        'I love this product',
        'This is amazing',
        'I hate this',
        'This is terrible',
        'Not bad at all',
        'I am very happy',
        'I am sad'
    ],
    'label': [
        'Positive',
        'Positive',
        'Negative',
        'Negative',
        'Positive',
        'Positive',
        'Negative' 
    ]
}

df = pd.read_csv('dataset.csv') 

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

model = MultinomialNB()
model.fit(X, df['label'])

def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ['happy', 'love', 'great', 'amazing']):
        return 'Happy 😊'
    elif any(word in text for word in ['hate', 'angry', 'annoying']):
        return 'Angry 😡'
    elif any(word in text for word in ['sad', 'bad', 'terrible']):
        return 'Sad 😢'
    elif any(word in text for word in ['scared', 'hurt', 'afraid', 'terrified']):
        return 'Afraid 😟'
    else:
        return 'Neutral 😐'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']

    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)[0]
    emotion = detect_emotion(user_input)

    return render_template('index.html',
                           prediction=prediction,
                           emotion=emotion,
                           text=user_input)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)