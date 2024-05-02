from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pickle
import pandas as pd


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

with open('./model/email_spam_predictor.sav', 'rb') as file:
    model = pickle.load(file)

def extract_features(subject, email):
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

    def cap_prop(text):
        count = 0
        for c in text:
            if c.isupper():
                count += 1
        if count > 0:
            return count/len(text)
        else:
            return 0

    def re_or_fw(text):
        match = text.str.extract(r"(Fw:|Re:)")
        return match[0].notna().astype(int)

    def special_char(text):
        match = text.str.extractall(r"([^\w ])")
        counts = match.groupby(level=0).size()
        return counts.reindex(text.index, fill_value=0)

    input = pd.DataFrame({'id': [0], 'subject': [subject], 'email': [email]})
    input['e_num_words'] = input['email'].astype(str).apply(num_words)
    input['s_num_words'] = input['subject'].astype(str).apply(num_words)
    input['e_cap_prop'] = input['email'].astype(str).apply(cap_prop)
    input['s_cap_prop'] = input['subject'].astype(str).apply(cap_prop)
    input['re_or_fw'] = re_or_fw(input['subject'])
    input['s_char'] = input['subject'].astype(str).apply(len)
    input['e_char'] = input['email'].astype(str).apply(len)
    input['e_special'] = special_char(input['email'])
    input['s_special'] = special_char(input['subject'])

    words = ['html', 'body', 'font', 'click', 'href', 'please', 'offer']
    word = pd.DataFrame(words_in_texts(words, input['email']))
    word['id'] = input['id']
    input = input.merge(word, on='id')

    input.columns = input.columns.astype(str)
    input = input.iloc[:, 3:]

    return input


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        subject = data.get('subject', '')
        email = data.get('email', '')

        features = extract_features(subject, email)

        prediction = model.predict(features)[0]
        result = "SPAM" if prediction == 1 else "NOT SPAM"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)