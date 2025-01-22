# from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

import traceback

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


'''
LOAD NLP MODEL
'''

model_lock = threading.Lock()

# Lazy-loaded models (initialized as None)
nlp = None
nlp_gpt2 = None
ner_pipeline = None
llama3_pipeline = None

def load_qa_pipeline():
    global nlp
    with model_lock:
        if nlp is None:  # Only load if not already initialized
            tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
            model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
            nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
            tokenizer.clean_up_tokenization_spaces = True

    return nlp

def load_ner_pipeline():
    global ner_pipeline
    with model_lock:
        if ner_pipeline is None:  # Only load if not already initialized
            # ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")  # Switch to smaller NER model
            ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return ner_pipeline



def load_llama3_pipeline():
    global llama3_pipeline
    with model_lock:
        if llama3_pipeline is None:  # Only load if not already initialized
            llama3_pipeline = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-2.7B",
    device=0,  # Use GPU if available
)
    return llama3_pipeline

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

def format_list_response(items, header="Here are the results:"):
    if not items:
        return "No data available."
    return f"{header}\n" + "\n".join([f"- {item}" for item in items])

def ask_nlp(question, context):
    nlp = load_qa_pipeline()

    return nlp(question=question, context=context, clean_up_tokenization_spaces=True)

#########################################################################################################
#LANGCHAIN TEST
#########################################################################################################
def math_tool(query):
    math_result = handle_math_question(query)
    return str(math_result) if math_result is not None else "No math solution found."

math_tool = Tool(
    name="Math Tool",
    func=math_tool,
    description="Handles math-related queries."
)

# Tool: Handle School Queries
def school_tool(query):
    location, topic = extract_location_and_topic(query)
    if topic == "schools" and location:
        school_info = extract_combined_school_info()
        if location in school_info:
            return format_list_response(school_info[location], header=f"Schools in {location}:")
        else:
            return "No schools found in this location."
    return "School-related query could not be processed."

school_tool = Tool(
    name="School Tool",
    func=school_tool,
    description="Handles queries related to schools and educational institutions."
)

# Tool: Handle Forecast Queries
# def forecast_tool(query):
#     location, _ = extract_location_and_topic(query)
#     bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
#     top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"
#     merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

#     if "forecast" in query.lower():
#         try:
#             answer, plot_image = process_forecast_query(query, merged_data, location)
#             return f"{answer}<br><img src='data:image/png;base64,{plot_image}'/>"
#         except Exception as e:
#             return f"Error processing forecast query: {str(e)}"
#     return process_price_query(query, merged_data)
# def forecast_tool(query):
#     location, _ = extract_location_and_topic(query)
#     bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
#     top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"
#     merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

#     if "forecast" in query.lower():
#         try:
#             answer, plot_image = process_forecast_query(query, merged_data, location)
#             # Ensure JSON-friendly response
#             return {
#                 "answer": answer,
#                 "plot": f"data:image/png;base64,{plot_image}"
#             }
#         except Exception as e:
#             return f"Error processing forecast query: {str(e)}"
#     return process_price_query(query, merged_data)

def forecast_tool(query):
    location, _ = extract_location_and_topic(query)
    bottom_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Bottom_tier_time_series.csv"
    top_tier_path = "./datasets/ZHVI/City_ZHVI_All_Homes_Top_tier_time_series.csv"
    merged_data = load_and_merge_data(bottom_tier_path, top_tier_path)

    if "forecast" in query.lower():
        try:
            answer, plot_image = process_forecast_query(query, merged_data, location)
            # Return structured data with Base64-encoded image
            return f"Forecast completed. Plot available separately."

        except Exception as e:
            return {"error": f"Error processing forecast query: {str(e)}"}
    return process_price_query(query, merged_data)
forecast_tool = Tool(
    name="Forecast Tool",
    func=forecast_tool,
    description="Handles queries related to housing prices and forecasts."
)

# Tool: General NLP Queries
def general_nlp_tool(query):
    relevant_context = get_relevant_context(query)
    result = ask_nlp(query, relevant_context)
    return result.get("answer", "I couldn't find a precise answer.")

general_nlp_tool = Tool(
    name="General NLP Tool",
    func=general_nlp_tool,
    description="Handles general NLP-based queries."
)

# Initialize LangChain Agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
tools = [math_tool, school_tool, forecast_tool, general_nlp_tool]
memory = ConversationBufferMemory(k=3)  # Retain only the last 3 interactions

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iteration=5,
    memory=memory  # Add memory for truncation

)




query_lock = threading.Lock()
@app.route('/chat', methods=['POST'])
def chat():
    with query_lock:
        try:
            data = request.get_json()
            query = data['query']

            # Use LangChain agent
            response = agent.invoke({"input": query})

            # Serialize response to ensure JSON compatibility
            # if isinstance(response, dict):
            #     response = {key: str(value) for key, value in response.items()}
            # elif not isinstance(response, str):
            #     response = str(response)
 # Check if the response contains image data
            if isinstance(response, dict) or "plot_image" in response:
                if "plot_image" in response:
                    # Include Base64-encoded image in the JSON response
                     return jsonify({
                    "answer": response["answer"],
                    "plot_image": f"data:image/png;base64,{response['plot_image']}"
                    })
            return jsonify({"results": [response]})
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error: {error_trace}")
            return jsonify({'results': [f"Error processing query: {str(e)}"]})

if __name__ == '__main__':
	app.run(host='localhost', port=10000, debug=True)
    # port  = int(os.environ.get('PORT', 10000))
    # app.run(host='0.0.0.0', port=port)