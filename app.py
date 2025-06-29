from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import nltk
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

svm_model = pickle.load(open('svm_model.pkl', 'rb'))

nn_model = load_model('nn_model.h5') 


vect = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english')) 

def preprocess_text(text):
    text = BeautifulSoup(text).get_text() 
    text = text.lower()
    text = re.sub("[^a-zA-Z]",' ',text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE) 
    text = re.sub(r'\@w+|\#', '', text) 
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    input_text = request.form['review']
    
    processed_text = preprocess_text(input_text)
    
    X_new = vect.transform([processed_text])
    
    svm_prediction = svm_model.predict(X_new)
    nn_prediction = nn_model.predict(X_new.toarray())
    nn_prediction = (nn_prediction > 0.5).astype(int)
    
    
    label_map = {1: 'Positive', 0: 'Negative'}
    svm_result = label_map[svm_prediction[0]]
    nn_result = label_map[nn_prediction[0][0]]
    
   
    return render_template('index.html', 
                           svm_result=f"SVM Prediction: {svm_result}",
                           nn_result=f"NN Prediction: {nn_result}")

if __name__ == "__main__":
    app.run(debug=True)