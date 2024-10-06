from flask import Flask, request, jsonify, render_template
import subprocess
import re
import pandas as pd
from datasets import load_dataset

app = Flask(__name__)

# Load the CSV data
crime_data = pd.read_csv('./database/datasetsCrime/crime.csv/crime.csv')

# Load the Hugging Face dataset
school_data = load_dataset('mw4/schools')

def extract_crime_info(df):
    info = {}
    for _, row in df.iterrows():
        location = str(row['NEIGHBORHOOD_ID']).strip().capitalize()
        offense_type = row['OFFENSE_TYPE_ID']
        if location in info:
            info[location].append(offense_type)
        else:
            info[location] = [offense_type]
    
    for location, offenses in info.items():
        total_offenses = len(offenses)
        offense_counts = pd.Series(offenses).value_counts(normalize=True) * 100
        info[location] = offense_counts.to_dict()
    
    return info

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
    return info

crime_info = extract_crime_info(crime_data)
school_info = extract_school_info(school_data['train'])

general_info = """
Perfect Home Finder helps you find the best homes available in various regions. 
We provide expert advice and a wide range of properties to choose from, including urban, suburban, and rural areas. 
Our services include property valuation, neighborhood analysis, and personalized home recommendations. 
We cover a broad spectrum of home prices, from affordable starter homes to luxurious estates. 
In addition to property details, we offer insights into local amenities such as schools, parks, and shopping centers. 
We also provide information on crime rates, ensuring you can find a safe and secure neighborhood. 
For example, cities like Denver have diverse climates with cold winters and hot summers, and are known for their safety and vibrant communities. 
Our goal is to help you find the perfect home that meets all your needs and preferences.
"""

@app.route('/')
def index():
    return render_template('indexgpt4all.html')

@app.route('/get_context', methods=['POST'])
def get_context():
    question = request.json.get('question')
    context = get_relevant_context(question)
    return jsonify({'context': context})

def get_relevant_context(question):
    location_match = re.search(r'in (\w+)', question.lower())
    if location_match:
        location = location_match.group(1).strip().capitalize()
        crime_context = f"Crime rate in {location}: {crime_info.get(location, 'No data available')}"
        school_context = f"School rating in {location}: {school_info.get(location, 'No data available')}"
        return f"{general_info}\n{crime_context}\n{school_context}"
    return general_info

@app.route('/ask_gpt4all', methods=['POST'])
def ask_gpt4all():
    question = request.json.get('question')
    context = request.json.get('context')
    response = ask_gpt4all_via_cli(question, context)
    return jsonify({'response': response})

def ask_gpt4all_via_cli(question, context):
    input_text = context + "\n" + question
    # Call the GPT-4All command-line interface
    gpt4all_cli_path = r'C:\Users\fungy\gpt4all'  # Update this path

    result = subprocess.run([gpt4all_cli_path, '--input', input_text], capture_output=True, text=True)
    return result.stdout.strip()

if __name__ == '__main__':
    app.run(debug=True)