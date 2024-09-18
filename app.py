from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import re
import random


app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')






# Load the NLP model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Set clean_up_tokenization_spaces to True in the tokenizer configuration
tokenizer.clean_up_tokenization_spaces = True

# Load the NLP pipeline with the configured tokenizer and model
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
def handle_math_question(question):
    match = re.match(r'(\d+)\s*([+\-*/])\s*(\d+)', question.lower())
    if match:
        num1, operator, num2 = match.groups()
        num1, num2 = int(num1), int(num2)
        if operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == '*':
            return num1 * num2
        elif operator == '/':
            return num1 / num2
    return None

def ask_nlp(question, context):
    return nlp(question=question, context=context, clean_up_tokenization_spaces=True)



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']
    
    # Example context (you should replace this with your actual data)
    context = """
    Perfect Home Finder helps you find the best homes available in various regions. 
    We provide expert advice and a wide range of properties to choose from, including urban, suburban, and rural areas. 
    Our services include property valuation, neighborhood analysis, and personalized home recommendations. 
    We cover a broad spectrum of home prices, from affordable starter homes to luxurious estates. 
    In addition to property details, we offer insights into local amenities such as schools, parks, and shopping centers. 
    We also provide information on crime rates, ensuring you can find a safe and secure neighborhood. 
    For example, cities like Denver have diverse climates with cold winters and hot summers, and are known for their safety and vibrant communities. 
    Our goal is to help you find the perfect home that meets all your needs and preferences.
    Perfect Home Finder is your ultimate resource for discovering the best homes available across various regions. We specialize in assisting homebuyers with tailored advice and a comprehensive selection of properties, ensuring you find the perfect fit for your lifestyle and budget. 

Our team of real estate experts provides in-depth property valuation services, helping you understand the true worth of a home based on market trends and neighborhood dynamics. We also conduct thorough neighborhood analyses, giving you insights into the quality of life, safety, and community features in the areas you are considering.

We cater to a broad spectrum of home prices, ranging from affordable starter homes ideal for first-time buyers to luxurious estates equipped with high-end amenities. No matter your financial situation, we strive to present options that align with your needs and preferences.

In addition to helping you find the right home, we offer valuable insights into local amenities. Our database includes information on schools—both K-12 and colleges—allowing you to assess the educational opportunities available in the vicinity. We provide ratings and reviews for schools, so you can make informed decisions based on quality of education, extracurricular activities, and overall student performance.

Moreover, we understand the importance of community features, so we include details about nearby parks, recreational areas, and shopping centers. Whether you're looking for family-friendly activities, outdoor spaces for leisure, or convenient access to retail options, we have you covered.

Safety is a top priority when selecting a neighborhood, and we equip you with comprehensive information on crime rates in different areas. By analyzing local crime statistics, we help you make informed choices about the safety of your potential new home, allowing you to prioritize peace of mind.

For instance, cities like Denver are renowned for their diverse climates, which include cold winters and warm summers. They are also celebrated for their vibrant communities and overall safety, making them desirable locations for families and individuals alike. Whether you prefer urban settings with bustling nightlife or quiet suburban environments, we assist you in navigating your options.

Our mission at Perfect Home Finder is to facilitate a seamless home-buying experience. We pride ourselves on our customer-centric approach, ensuring that every interaction is tailored to your specific needs. From your initial inquiry to the final closing process, we are dedicated to providing support and guidance every step of the way.

    """
    
    # Check if the question is a math question
    math_result = handle_math_question(query)
    if math_result is not None:
        answer = str(math_result)
    else:
        # Use the NLP model to get the answer
        result = ask_nlp(query, context)
        answer = result['answer']
    
    return jsonify({'results': [answer]})
# results = []  
# for question in questions:
#     math_result = handle_math_question(question)
#     if math_result is not None:
#         results.append((question, math_result))
#     else:
#         result = ask_nlp(question, context)
#         results.append((question, result['answer']))

# # Print the results
# for q, answer in results:
#     print(f"Question: {q}")
#     print(f"Answer: {answer}\n")
if __name__ == '__main__':
	app.run(host='localhost', port=5000, debug=True)
