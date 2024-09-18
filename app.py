from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline
# from database.database import Database
import random
app = Flask(__name__)


# Load the NLP model
nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data['query']
    
    # Example context (you should replace this with your actual data)
    context = """
    Perfect Home Finder helps you find the best homes available in your area. 
    We provide expert advice and a wide range of properties to choose from.
    """
    
    # Use the NLP model to get the answer
    result = nlp(question=query, context=context)
    
    return jsonify({'results': [result['answer']]})

if __name__ == '__main__':
	app.run(host='localhost', port=5000, debug=True)