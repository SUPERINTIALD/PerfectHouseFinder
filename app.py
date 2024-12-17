from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import re
import random
import sympy as sp
import matplotlib.pyplot as plt
import base64
import io
import numpy as np
import os
import pandas as pd
from appFunc import (
    extract_combined_school_info,
)
from handleMathQuestions import handle_math_question
app = Flask(__name__)


@app.route('/')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
    images = [
        './static/img/Yuri.jpg',
        './static/img/Henry.jpg',
    ]
    return render_template('about.html', images=images)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat')
def chatRoom():
    return render_template('index.html')
@app.route('/api/data', methods=['GET'])
def api_data():
    # Example static JSON response
    data = {
        "status": "success",
        "message": "This is the data you requested.",
        "example_list": [1, 2, 3, 4, 5]
    }
    return jsonify(data), 200

# Load the NLP model
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    # model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

# Set clean_up_tokenization_spaces to True in the tokenizer configuration
tokenizer.clean_up_tokenization_spaces = True

# Load the NLP pipeline with the configured tokenizer and model
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
nlp_gpt2 = pipeline("text-generation", model="gpt2")
#Get school data
public_schools = pd.read_csv('./datasets/schools/Public_Schools/Public_Schools.csv')
private_schools = pd.read_csv('./datasets/schools/Private_Schools/Private_Schools.csv')
public_schools['Type'] = 'Public'
private_schools['Type'] = 'Private'
print("Public Schools Columns:", public_schools.columns)
print("Private Schools Columns:", private_schools.columns)

combined_schools = pd.concat([public_schools, private_schools], ignore_index=True)


# school_data = load_dataset('mw4/schools')
# crime_data = pd.read_csv('./database/datasetsCrime/crime.csv/crime.csv')


# print("First 10 lines of the school dataset:")
# for i, item in enumerate(school_data['train']):
#     if i >= 10:
#         break
#     print(item)

plt.switch_backend('Agg')




#General Info about our company
general_info = """
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
    We pride ourselves on our extensive database of properties, which is regularly updated to reflect the latest market trends and availability. Our user-friendly platform allows you to filter properties based on various criteria such as price range, number of bedrooms, property type, and more. This ensures that you can quickly find homes that match your specific requirements.
    Our team is committed to providing exceptional customer service. We offer personalized consultations to understand your unique needs and preferences, guiding you through every step of the home-buying process. From initial property searches to final negotiations, we are here to support you.
    In addition to our property listings, we provide valuable resources such as mortgage calculators, home-buying guides, and market analysis reports. These tools are designed to empower you with the knowledge needed to make informed decisions.
    At Perfect Home Finder, we believe that finding your dream home should be an enjoyable and stress-free experience. Our mission is to simplify the process and help you discover a place you can truly call home.
    
    """ 

# def extract_crime_info(df):
#     info = {}
#     for _, row in df.iterrows():
#         location = str(row['NEIGHBORHOOD_ID']).strip().capitalize()  # Convert to str and handle missing values
#         offense_type = row['OFFENSE_TYPE_ID']
#         if location in info:
#             info[location].append(offense_type)
#         else:
#             info[location] = [offense_type]
    
#     # Calculate the percentage of each offense type
#     for location, offenses in info.items():
#         total_offenses = len(offenses)
#         offense_counts = pd.Series(offenses).value_counts(normalize=True) * 100
#         info[location] = offense_counts.to_dict()
    
#     return info

# Add Named Entity Recognition (NER) to Extract Location and Topic
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_location_and_topic(question):
    entities = ner_pipeline(question)
    location = None
    topic = None

    for entity in entities:
        if entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC':
            location = entity['word'].replace("##", "") if not location else location + entity['word'].replace("##", "")
    if "school" in question.lower():
        topic = "schools"
    return location, topic


def extract_school_info(dataset):
    info = {}
    for item in dataset:
        if 'name' in item:
            parts = item['name'].split(',')
            if len(parts) > 1:
                location = parts[-1].strip().capitalize()
                school_name = parts[0].strip()
                if location in info:
                    info[location].append(school_name)
                else:
                    info[location] = [school_name]

                # info[location] = school_name #Only gets 1 school name
    return info

# Extract  information
# crime_info = extract_crime_info(crime_data)
# school_info = extract_school_info(school_data['train'])
school_info = extract_combined_school_info(combined_schools)


def get_relevant_context(question):
    location_match = re.search(r'in (\w+)', question.lower())
    if location_match:
        location = location_match.group(1).strip().capitalize()
        # crime_context = f"Crime rate in {location}: {crime_info.get(location, 'No data available')}"
        # school_context = f"School rating in {location}: {school_info.get(location, 'No data available')}"
        if location in school_info:
            schools = format_list_response(school_info[location], header=f"Schools in {location}:")
        else:
            schools = "No schools found in this location."
        school_context = schools

        return f"{general_info}\n{school_context}"
        # return f"{general_info}\n"

    return general_info   



# def get_relevant_context(question):
#     locations = re.findall(r'in (\w+)', question.lower())
#     contexts = []
#     for loc in locations:
#         location = loc.strip().capitalize()
#         crime_data = crime_info.get(location, {})
#         if crime_data:
#             crime_context = f"Crime types in {location}: {', '.join([f'{k}: {v:.2f}%' for k, v in crime_data.items()])}"
#         else:
#             crime_context = f"Crime types in {location}: No data available"
#         school_context = f"Schools in {location}: {', '.join(school_info.get(location, ['No data available']))}"
#         contexts.append(f"{school_context}\n")
#     return f"{general_info}\n" + "\n".join(contexts)



def format_list_response(items, header="Here are the results:"):
    if not items:
        return "No data available."
    return f"{header}\n" + "\n".join([f"- {item}" for item in items])

def ask_nlp(question, context):
    return nlp(question=question, context=context, clean_up_tokenization_spaces=True)


# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data['query']
#     # Check if the question is a math question
#     math_result = handle_math_question(query)
#     if math_result is not None:
#         answer = str(math_result)
#     else:
#         relevant_context = get_relevant_context(query)
#         result = ask_nlp(query, relevant_context)
#         answer = result['answer']

#     return jsonify({'results': [answer]})





#Updated Chat system using NER
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data['query']

    # Extract location and topic
    location, topic = extract_location_and_topic(query)

    # Check if the question is a math question
    math_result = handle_math_question(query)
    if math_result is not None:
        answer = str(math_result)
    elif topic == "schools" and location:
        if location in school_info:
            answer = format_list_response(school_info[location], header=f"Schools in {location}:")
        else:
            answer = "No schools found in this location."
    else:
        relevant_context = get_relevant_context(query)
        result = ask_nlp(query, relevant_context)
        answer = result['answer']

    return jsonify({'results': [answer]})


    # relevant_context = get_relevant_context(query)
    # result = ask_nlp(query, relevant_context)
    # answer = result['answer']
        # Use the NLP model to get the answer
        # result = ask_nlp(query, context)
        # answer = result['answer']
        # if re.search(r'\b(what|who|where|when|why|how)\b', query.lower()):
            # Use the NLP model to get the answer
            # result = ask_nlp(query, context)
            # answer = result['answer']
        # else:
            # Use the GPT-2 model to generate text
            # gpt2_input = context + "\n" + query
            # gpt2_output = nlp_gpt2(query, max_length=150, truncation = True, clean_up_tokenization_spaces=True)
            # answer = gpt2_output[0]['generated_text']
    
    # return jsonify({'results': [answer]})
    # return jsonify({'results': [answer]})
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
