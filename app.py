# from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
# from datasets import load_dataset
import re
# import random
# import sympy as sp
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# import base64
# import io
# import numpy as np
import os
# import pandas as pd
import threading

# Handle the schools information
from appFunc import (
    extract_combined_school_info,
)

# Handle Math problems
from handleMathQuestions import handle_math_question

# Handle Context 
from context import general_info

# Handle House Prices
from HandleHousePrice import (
    load_and_merge_housing_data,
    # filter_houses_by_price,
    process_price_query
)
# from HandleHousePrediction import load_and_merge_data, forecast_with_monte_carlo, generate_forecast_plot
from HandleHousePrediction import load_and_merge_data, process_forecast_query

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login/login.html')


@app.route('/create_account')
def create_account():
    return render_template('login/createAccount.html')

@app.route('/home')
def home():
	return render_template('index.html')

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
    return render_template('chat.html')


# @app.route('/api/data', methods=['GET'])
# def api_data():
#     # Example static JSON response
#     data = {
#         "status": "success",
#         "message": "This is the data you requested.",
#         "example_list": [1, 2, 3, 4, 5]
#     }
#     return jsonify(data), 200

'''
LOAD NLP MODEL
'''


#This is another NLP that doesnt work anymore
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
# model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

'''
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

# Set clean_up_tokenization_spaces to True in the tokenizer configuration
tokenizer.clean_up_tokenization_spaces = True

# Load the NLP pipeline with the configured tokenizer and model
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
nlp_gpt2 = pipeline("text-generation", model="gpt2")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
'''

model_lock = threading.Lock()

# Lazy-loaded models (initialized as None)
nlp = None
# nlp_gpt2 = None
ner_pipeline = None

def load_qa_pipeline():
    global nlp
    with model_lock:
        if nlp is None:  # Only load if not already initialized
            tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
            model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
            nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
            tokenizer.clean_up_tokenization_spaces = True

    return nlp

# def load_text_generation_pipeline():
def load_text_generation_pipeline():
    global nlp_gpt2
    with model_lock:
        if nlp_gpt2 is None:  # Only load if not already initialized
            nlp_gpt2 = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1")
    return nlp_gpt2
#     global nlp_gpt2
#     with model_lock:
#         if nlp_gpt2 is None:  # Only load if not already initialized
#             nlp_gpt2 = pipeline("text-generation", model="distilgpt2")  # Switch to smaller GPT-2 model
#     return nlp_gpt2

def load_ner_pipeline():
    global ner_pipeline
    with model_lock:
        if ner_pipeline is None:  # Only load if not already initialized
            # ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")  # Switch to smaller NER model
            ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return ner_pipeline






#Get school data
# public_schools = pd.read_csv('./datasets/schools/Public_Schools/Public_Schools.csv')
# private_schools = pd.read_csv('./datasets/schools/Private_Schools/Private_Schools.csv')
# public_schools['Type'] = 'Public'
# private_schools['Type'] = 'Private'
# print("Public Schools Columns:", public_schools.columns)
# print("Private Schools Columns:", private_schools.columns)

# combined_schools = pd.concat([public_schools, private_schools], ignore_index=True)


# school_data = load_dataset('mw4/schools')
# crime_data = pd.read_csv('./database/datasetsCrime/crime.csv/crime.csv')


# print("First 10 lines of the school dataset:")
# for i, item in enumerate(school_data['train']):
#     if i >= 10:
#         break
#     print(item)




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

def extract_location_and_topic(question):
    # entities = ner_pipeline(question)
    ner = load_ner_pipeline()  # Load dynamically
    entities = ner(question)
    location = None
    topic = None

    for entity in entities:
        if entity['entity'] == 'B-LOC' or entity['entity'] == 'I-LOC':
            location = entity['word'].replace("##", "") if not location else location + entity['word'].replace("##", "")
    if "school" in question.lower():
        topic = "schools"
    return location, topic


# def extract_school_info(dataset):
#     info = {}
#     for item in dataset:
#         if 'name' in item:
#             parts = item['name'].split(',')
#             if len(parts) > 1:
#                 location = parts[-1].strip().capitalize()
#                 school_name = parts[0].strip()
#                 if location in info:
#                     info[location].append(school_name)
#                 else:
#                     info[location] = [school_name]

#                 # info[location] = school_name #Only gets 1 school name
#     return info

# Extract  information
# crime_info = extract_crime_info(crime_data)
# school_info = extract_school_info(school_data['train'])
# school_info = extract_combined_school_info(combined_schools)





# Add preloading: Load the school and crime data once

# school_info = extract_combined_school_info()

def get_relevant_context(question):
    location_match = re.search(r'in (\w+)', question.lower())
    if location_match:
        location = location_match.group(1).strip().capitalize()
        # crime_context = f"Crime rate in {location}: {crime_info.get(location, 'No data available')}"
        # school_context = f"School rating in {location}: {school_info.get(location, 'No data available')}"

        school_info = extract_combined_school_info()

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
    nlp = load_qa_pipeline()

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





# #Updated Chat system using NER
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data['query']

#     # Extract location and topic
#     location, topic = extract_location_and_topic(query)

#     # Check if the question is a math question
#     math_result = handle_math_question(query)
#     if math_result is not None:
#         answer = str(math_result)
#     elif topic == "schools" and location:
#         if location in school_info:
#             answer = format_list_response(school_info[location], header=f"Schools in {location}:")
#         else:
#             answer = "No schools found in this location."
#     else:
#         relevant_context = get_relevant_context(query)
#         result = ask_nlp(query, relevant_context)
#         answer = result['answer']

#     return jsonify({'results': [answer]})


# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data['query']

#     # Extract location and topic
#     location, topic = extract_location_and_topic(query)

#     # Check if the question is a math question
#     math_result = handle_math_question(query)
#     if math_result is not None:
#         answer = str(math_result)

#     # Check if the query is about house prices or predictions
#     elif "$" in query or "price" in query.lower() or "forecast" in query.lower():
#         # Load house price data
#         bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
#         top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"

#         # Merge data into a single dataset
#         merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

#         # Process forecasting queries
#         if "forecast" in query.lower():
#             try:
#                 # Filter location-specific data
#                 location_data = merged_data[merged_data['RegionName'].str.contains(location, case=False)]
#                 if location_data.empty:
#                     answer = f"No data available for {location}."
#                 else:
#                     # Prepare time-series data
#                     date_columns = [col for col in map(str, location_data.columns) if re.match(r'\d{4}-\d{2}-\d{2}', col)]
#                     data = location_data[date_columns].iloc[0]
#                     data.index = pd.to_datetime([col.split('_')[0] for col in data.index])

#                     # Fill missing values
#                     data = data.interpolate(method='linear').fillna(method='ffill')

#                     # Forecast using ARIMA/SARIMA with Monte Carlo simulation
#                     mean_forecast, _ = forecast_with_monte_carlo(
#                         data, query, start_date=data.index[0]
#                     )
#                     answer = f"Forecast for {location}: {mean_forecast[-1]:.2f}"
#             except Exception as e:
#                 answer = f"Error processing forecast query: {str(e)}"
#         # Handle price-related queries
#         else:
#             answer = process_price_query(query, merged_data)

#     # Handle school-related queries
#     elif topic == "schools" and location:
#         if location in school_info:
#             answer = format_list_response(school_info[location], header=f"Schools in {location}:")
#         else:
#             answer = "No schools found in this location."

#     # Default NLP processing for other questions
#     else:
#         relevant_context = get_relevant_context(query)
#         result = ask_nlp(query, relevant_context)
#         answer = result['answer']

#     return jsonify({'results': [answer]})






# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data['query']

#     # Extract location and topic
#     location, topic = extract_location_and_topic(query)

#     # Check if the question is a math question
#     math_result = handle_math_question(query)
#     if math_result is not None:
#         answer = str(math_result)

#     # Check if the query is about house prices or predictions
#     elif "$" in query or "price" in query.lower() or "forecast" in query.lower():
#         # Load house price data
#         bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
#         top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"

#         # Merge data into a single dataset
#         merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

#         # Process forecasting queries
#         if "forecast" in query.lower():
#             try:
#                 # Filter location-specific data
#                 location_data = merged_data[merged_data['RegionName'].str.contains(location, case=False)]
#                 if location_data.empty:
#                     answer = f"No data available for {location}."
#                 else:
#                     # Prepare time-series data
#                     date_columns = [col for col in map(str, location_data.columns) if re.match(r'\d{4}-\d{2}-\d{2}', col)]
#                     data = location_data[date_columns].iloc[0]
#                     data.index = pd.to_datetime([col.split('_')[0] for col in data.index])

#                     # Fill missing values
#                     data = data.interpolate(method='linear').ffill()

#                     # Forecast using ARIMA/SARIMA with Monte Carlo simulation
#                     mean_forecast, simulated_paths = forecast_with_monte_carlo(
#                         data, query, start_date=data.index[0]
#                     )

#                     # Generate and encode plot
#                     forecast_dates = pd.date_range(
#                         start=data.index[-1] + pd.DateOffset(months=1),
#                         periods=len(mean_forecast), freq='ME'
#                     )

#                     plot_image = generate_forecast_plot(
#                         data, mean_forecast, simulated_paths, forecast_dates, location, "SARIMA"
#                     )

#                     # Return forecast and plot
#                     answer = f"Forecast for {location}: {mean_forecast.iloc[-1]:.2f}"
#                     return jsonify({'results': [answer], 'plot': f"data:image/png;base64,{plot_image}"})
#             except Exception as e:
#                 answer = f"Error processing forecast query: {str(e)}"
#         # Handle price-related queries
#         else:
#             answer = process_price_query(query, merged_data)

#     # Handle school-related queries
#     elif topic == "schools" and location:
#         if location in school_info:
#             answer = format_list_response(school_info[location], header=f"Schools in {location}:")
#         else:
#             answer = "No schools found in this location."

#     # Default NLP processing for other questions
#     else:
#         relevant_context = get_relevant_context(query)
#         result = ask_nlp(query, relevant_context)
#         answer = result['answer']

#     return jsonify({'results': [answer]})














query_lock = threading.Lock()

@app.route('/chat', methods=['POST'])
def chat():
    with query_lock:
        try:
            data = request.get_json()
            query = data['query']

            # Extract location and topic
            location, topic = extract_location_and_topic(query)

            # Check if the question is a math question
            math_result = handle_math_question(query)
            if math_result is not None:
                answer = str(math_result)

            # Check if the query is about house prices or predictions
            elif "$" in query or "price" in query.lower() or "forecast" in query.lower():
                # Load house price data
                bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
                top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"

                # Merge data into a single dataset
                merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

                # Process forecasting queries
                if "forecast" in query.lower():
                    try:
                        # Call process_forecast_query directly
                        answer, plot_image = process_forecast_query(query, merged_data, location)
                        response = f"{answer}<br><img src='data:image/png;base64,{plot_image}'/>"
                        return jsonify({'results': [response]})
                    except Exception as e:
                        answer = f"Error processing forecast query: {str(e)}"

                # Handle price-related queries
                else:
                    answer = process_price_query(query, merged_data)

            # Handle school-related queries
            elif topic == "schools" and location:
                school_info = extract_combined_school_info() #Dynamically load school data

                if location in school_info:
                    answer = format_list_response(school_info[location], header=f"Schools in {location}:")
                else:
                    answer = "No schools found in this location."

            # Default NLP processing for other questions
            else:
                # try:
                relevant_context = get_relevant_context(query)
                result = ask_nlp(query, relevant_context)
                answer = result['answer']
         
            return jsonify({'results': [answer]})
            
        except Exception as e:
            # Return an error if anything fails
            return jsonify({'results': [f"Error processing query: {str(e)}"]})






















# Updated Chat system with House Price integration
# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     query = data['query']

#     # Extract location and topic
#     location, topic = extract_location_and_topic(query)

#     # Check if the question is a math question
#     math_result = handle_math_question(query)
#     if math_result is not None:
#         answer = str(math_result)

#     # Check if the query is about house prices
#     elif "$" in query or "price" in query.lower():
#         # Load house price data (optimize this step by preloading if needed)
#         bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
#         top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"
#         merged_data = load_and_merge_housing_data(bottom_tier_path, top_tier_path)

#         # Process house price query
#         answer = process_price_query(query, merged_data)

#     # Handle school-related queries
#     elif topic == "schools" and location:
#         if location in school_info:
#             answer = format_list_response(school_info[location], header=f"Schools in {location}:")
#         else:
#             answer = "No schools found in this location."

#     # Default NLP processing for other questions
#     else:
#         relevant_context = get_relevant_context(query)
#         result = ask_nlp(query, relevant_context)
#         answer = result['answer']

#     return jsonify({'results': [answer]})
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
	# app.run(host='localhost', port=5000, debug=True)
    port  = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)