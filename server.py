import pickle
import torch
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
from transformers import BertTokenizer, BertModel
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load the classification models (Pickle files)
with open('xgb_clf_L1.pkl', 'rb') as f:
    clf_F1 = pickle.load(f)
with open('xgb_clf_L2.pkl', 'rb') as f:
    clf_F2 = pickle.load(f)
with open('mlp_clf_L3.pkl', 'rb') as f:
    clf_F3 = pickle.load(f)
with open('xgb_clf_L4.pkl', 'rb') as f:
    clf_F4 = pickle.load(f)
with open('xgb_clf_L5.pkl', 'rb') as f:
    clf_F5 = pickle.load(f)
with open('lr_clf_L6.pkl', 'rb') as f:
    clf_F6 = pickle.load(f)

# Define the function to create BERT embeddings for a single sentence
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean of the last hidden states
    return embedding.numpy().flatten()  # Flatten to match XGBoost input format

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Assuming index.html is in the "templates" folder

# API endpoint for classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    sentence = data.get('sentence')

    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    # Get the BERT embedding for the sentence
    embedding = get_embedding(sentence)

    # Perform classification with the models
    prediction_F1 = clf_F1.predict([embedding])[0]
    prediction_F2 = clf_F2.predict([embedding])[0]
    prediction_F3 = clf_F3.predict([embedding])[0]
    prediction_F4 = clf_F4.predict([embedding])[0]
    prediction_F5 = clf_F5.predict([embedding])[0]
    prediction_F6 = clf_F6.predict([embedding])[0]

    # Prepare the result in the required format
    if prediction_F1 == 0:
        result = {
            'prediction': f'F1: {prediction_F1}'
        }
    else:
        result = {
            'prediction': f'F1: {prediction_F1}, F2: {prediction_F2}, F3: {prediction_F3}, F4: {prediction_F4}, F5: {prediction_F5}, F6: {prediction_F6}'
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
