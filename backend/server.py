from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pickle
import pandas as pd
import re

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

with open('./model/email_spam_predictor_new.sav', 'rb') as file:
    model = pickle.load(file)

def extract_features(subject, body):
    def words_in_texts(words, texts):
        indicator_array = []
        for i in texts:
            arr = []
            for j in words:
                if j in i:
                    arr.append(1)
                else:
                    arr.append(0)
            indicator_array.append(arr)
        return np.asarray(indicator_array)

    def num_words(text):
        return len(text.split())

    def re_or_fw(text):
        match = re.search(r"(fw :|re :)", text)
        return int(match is not None)

    def special_char(text):
        match = re.findall(r"([^\w ])", text)
        return len(match)

    input = pd.DataFrame({'id': [0], 'subject': [subject], 'body': [body]})

    tfidf_vectorizer_subject = TfidfVectorizer()
    tfidf_vectorizer_body = TfidfVectorizer()

    X_subject_tfidf = tfidf_vectorizer_subject.fit_transform(input['subject'])
    X_body_tfidf = tfidf_vectorizer_body.fit_transform(input['body'])

    X_combined = hstack([X_subject_tfidf, X_body_tfidf])

    body_num_words = np.array(input['body'].apply(num_words)).reshape(-1, 1)
    subject_num_words = np.array(input['subject'].apply(num_words)).reshape(-1, 1)
    re_or_fw_feature = np.array(input['subject'].astype(str).apply(re_or_fw)).reshape(-1, 1)
    subject_char = np.array(input['subject'].apply(len)).reshape(-1, 1)
    body_char = np.array(input['body'].apply(len)).reshape(-1, 1)
    subject_special = np.array(input['subject'].astype(str).apply(special_char)).reshape(-1, 1)
    body_special = np.array(input['body'].apply(special_char)).reshape(-1, 1)

    words = ['offer', 'help', 'win', 'price', 'card']
    subject_words_in_texts = words_in_texts(words, input['subject'])
    body_words_in_texts = words_in_texts(words, input['body'])

    subject_words_in_texts_sparse = csr_matrix(subject_words_in_texts)
    body_words_in_texts_sparse = csr_matrix(body_words_in_texts)
    
    X_combined_new = hstack([X_combined, body_num_words, subject_num_words, re_or_fw_feature, subject_char, body_char, subject_special, body_special, subject_words_in_texts_sparse, body_words_in_texts_sparse])

    return X_combined_new


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        subject = data.get('subject', '')
        body = data.get('email', '')

        features = extract_features(subject, body)

        prediction = model.predict(features)[0]
        result = "SPAM" if prediction == 1 else "NOT SPAM"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)