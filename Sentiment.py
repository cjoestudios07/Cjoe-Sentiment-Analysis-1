import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    'text': [
        'I love this product',
        'This is amazing',
        'I hate this',
        'This is terrible',
        'Not bad at all',
        'I am very happy',
        'I am sad',
        'I am not happy',
        'This looks bad'
    ],
    'label': [
        'positive',
        'positive',
        'negative',
        'negative',
        'positive',
        'positive',
        'negative',
        'negative',
        'negative',
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

model = MultinomialNB()
model.fit(X, df['label'])

while True:
    user_input = input("Enter a sentence: ")
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)
    print("Sentiment:", prediction[0])