from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering 
from datasets import load_dataset
import re

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Set clean_up_tokenization_spaces to True in the tokenizer configuration
tokenizer.clean_up_tokenization_spaces = True

# Load the NLP pipeline with the configured tokenizer and model
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Load datasets from Hugging Face
# crime_data = load_dataset("community-datasets/crime_and_punish")
school_data = load_dataset('mw4/schools')

# Print the first 10 lines of the raw datasets for inspection
print("First 10 lines of the crime dataset:")
for i, item in enumerate(crime_data['train']):
    if i >= 10:
        break
    print(item)

print("First 10 lines of the school dataset:")
for i, item in enumerate(school_data['train']):
    if i >= 10:
        break
    print(item)

def extract_info(dataset, key, location_key='location'):
    info = {}
    for item in dataset:
        if key in item and location_key in item:
            location = item[location_key].strip().capitalize()
            info[location] = item[key]
    return info

# Adjust the extraction logic based on the dataset structure
def extract_school_info(dataset):
    info = {}
    for item in dataset:
        if 'name' in item:
            parts = item['name'].split(',')
            if len(parts) > 1:
                location = parts[-1].strip().capitalize()
                school_name = parts[0].strip()
                info[location] = school_name
    return info

crime_info = extract_info(crime_data['train'], 'crime')
school_info = extract_school_info(school_data['train'])

# Print extracted data for debugging
print("Crime Info:", crime_info)
print("School Info:", school_info)

# General information section
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

# Function to dynamically select relevant context based on the question
def get_relevant_context(question):
    location_match = re.search(r'in (\w+)', question.lower())
    if location_match:
        location = location_match.group(1).strip().capitalize()
        crime_context = f"Crime rate in {location}: {crime_info.get(location, 'No data available')}"
        school_context = f"School rating in {location}: {school_info.get(location, 'No data available')}"
        return f"{general_info}\n{crime_context}\n{school_context}"
    return general_info

questions = [
    "What does Perfect Home Finder do?",
    "What is 1+1?",
    "How does Perfect Home Finder assist you?",
    "What services does Perfect Home Finder offer?",
    "Can Perfect Home Finder help with property advice?",
    "What kind of properties can you find with Perfect Home Finder?",
    "Is Perfect Home Finder available in all areas?",
    "What is the price range of homes?",
    "What is 10 * 5?",
    "What environment is Denver? Is it cold? or hot? is it safe?",
    "What are the crime rates in the area?",
    "What amenities are available near the homes?",
    "What is the climate like in different regions?",
    "What are the school ratings in Denver?",
    "What amenities are available in New York?",
    "How bad are crimes in Washington?",
    "What is the crime rate in Los Angeles?",
    "What is the school rating in Chicago?",
    "What is the crime rate in Chicago?",
    "What is the crime rate in Denver?",
    "What is the school rating in Denver?",
    "What is the crime rate in New York?",
    "What is the school rating in New York?",
    "What is the crime rate in Miami?",
    "What is the school rating in Miami?",
    "What is the crime rate in Houston?",
    "What is the school rating in Houston?",
    "What is the crime rate in Seattle?",
    "What is the school rating in Seattle?",
    "What is the crime rate in San Francisco?",
    "What is the school rating in San Francisco?",
    "What is the crime rate in Boston?",
    "What is the school rating in Boston?",
    "What is the crime rate in Philadelphia?",
    "What is the school rating in Philadelphia?",
]

def handle_math_question(question):
    match = re.match(r'what is (\d+)\s*([+\-*/])\s*(\d+)', question.lower())
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

max_tokens = tokenizer.model_max_length

sample_text = "Perfect Home Finder helps you find the best homes available in various regions."
tokens = tokenizer.tokenize(sample_text)
num_tokens = len(tokens)
tokens_left = max_tokens - num_tokens
print(f"Number of tokens: {num_tokens}")
print(f"Maximum tokens allowed: {max_tokens}")
print(f"Tokens left: {tokens_left}")

results = []
for question in questions:
    math_result = handle_math_question(question)
    if math_result is not None:
        results.append((question, math_result))
    else:
        relevant_context = get_relevant_context(question)
        question_tokens = tokenizer.tokenize(question)
        context_tokens = tokenizer.tokenize(relevant_context)
        total_tokens = len(question_tokens) + len(context_tokens)
        tokens_left = max_tokens - total_tokens
        print(f"Number of tokens in question: {len(question_tokens)}")
        print(f"Total tokens (context + question): {total_tokens}")
        print(f"Tokens left: {tokens_left}\n")
        
        if total_tokens <= max_tokens:
            result = ask_nlp(question, relevant_context)
            results.append((question, result['answer']))
        else:
            results.append((question, "Context too long to process"))

for q, answer in results:
    print(f"Question: {q}")
    print(f"Answer: {answer}\n")