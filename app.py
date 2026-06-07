from flask import Flask, request, jsonify

import pickle

import numpy as np

import re

import joblib

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification



app = Flask(__name__)



# --- 1. Models & Loaders ---

email_model = pickle.load(open('final_linear_svm_model.pkl', 'rb'))

email_scaler = pickle.load(open('final_fitted_scaler.pkl', 'rb'))



url_model = joblib.load('url_svm_model.pkl')

url_scaler = joblib.load('url_scaler.pkl')

url_feature_names = joblib.load('url_feature_names.pkl')



device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('./') 

sms_model = AutoModelForSequenceClassification.from_pretrained('./').to(device)



# --- Feature Extraction Functions ---



def extract_email_features(text):

    words = text.split()

    return np.array([[float(len(words)), float(len(set(words))), float(sum(1 for w in words if w.lower() in {"the", "a", "is", "and"})), 

                      float(len(re.findall(r'http[s]?://\S+', text))), float(len(set(re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)))), 

                      float(len(re.findall(r'\S+@\S+', text))), float(len(re.findall(r'[^\w\s]', text))), 

                      float(sum(1 for w in words if w.lower() in ['urgent', 'password', 'verify', 'account']))]])



def extract_url_features(url):

    feat_dict = {name: 0.0 for name in url_feature_names}

    url_str = url.lower()

    

    feat_dict['length_url'] = float(len(url))

    feat_dict['nb_dots'] = float(url.count('.'))

    feat_dict['nb_slash'] = float(url.count('/'))

    feat_dict['nb_hyphens'] = float(url.count('-'))

    feat_dict['nb_at'] = float(url.count('@'))

    feat_dict['nb_qm'] = float(url.count('?'))

    feat_dict['nb_and'] = float(url.count('&'))

    feat_dict['nb_eq'] = float(url.count('='))

    feat_dict['nb_underscore'] = float(url.count('_'))

    feat_dict['nb_www'] = 1.0 if "www." in url_str else 0.0

    feat_dict['nb_digits'] = float(sum(c.isdigit() for c in url_str))

    feat_dict['https_token'] = 1.0 if "https" in url_str else 0.0

    

    return np.array([feat_dict.get(col, 0.0) for col in url_feature_names]).reshape(1, -1)



# --- Routes ---



@app.route('/predict/email', methods=['POST'])

def predict_email():

    try:

        text = request.json.get('text', '')

        features = email_scaler.transform(extract_email_features(text))

        score = email_model.decision_function(features)[0]

        # Agar score 0.5 se zyada hai to Phishing

        pred = 1 if score > 0.5 else 0

        return jsonify({'result': "Phishing" if pred == 1 else "Safe", 'score': float(score)})

    except Exception as e:

        return jsonify({'error': str(e)})



@app.route('/predict/url', methods=['POST'])

def predict_url():

    try:

        url = request.json.get('text', '')

        features = extract_url_features(url)

        features_scaled = url_scaler.transform(features)

        score = url_model.decision_function(features_scaled)[0]

        # Agar score -1.5 se zyada hai to Phishing

        pred = 1 if score > -1.5 else 0

        return jsonify({'result': "Phishing" if pred == 1 else "Safe", 'score': float(score)})

    except Exception as e:

        return jsonify({'error': str(e)})



@app.route('/predict/sms', methods=['POST'])

def predict_sms():

    try:

        msg = request.json.get('text', '')

        inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        with torch.no_grad():

            outputs = sms_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        pred = torch.argmax(probs).item()

        return jsonify({'result': "Phishing" if pred == 1 else "Safe", 'confidence': float(probs[pred])})

    except Exception as e:

        return jsonify({'error': str(e)})



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)