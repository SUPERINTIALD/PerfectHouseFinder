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
# school_data = load_dataset('mw4/schools')
# amenities_data = load_dataset('amenities', split='train[:10]')

# Print the structure of the dataset items
# print("Crime Data Sample:", crime_data['train'][0])
# print("School Data Sample:", school_data['train'][0])
# print("Amenities Data Sample:", amenities_data[0])

# Function to extract relevant information from datasets
# def extract_info(dataset, key):
#     info = []
#     for item in dataset:
#         # Print item to understand its structure
#         print("Item:", item)
#         info.append(f"{item['location']}: {item[key]}")
#     return "\n".join(info)

# Extract information from datasets
# crime_info = extract_info(crime_data['train'], 'crime')
# school_info = extract_info(school_data['train'], 'school')
# amenities_info = extract_info(amenities_data['train'], 'amenities')

# Combine all contexts into a single context string
context = """
Perfect Home Finder helps you find the best homes available in various regions. 
We provide expert advice and a wide range of properties to choose from, including urban, suburban, and rural areas. 
Our services include property valuation, neighborhood analysis, and personalized home recommendations. 
We cover a broad spectrum of home prices, from affordable starter homes to luxurious estates. 
In addition to property details, we offer insights into local amenities such as schools, parks, and shopping centers. 
We also provide information on crime rates, ensuring you can find a safe and secure neighborhood. 
For example, cities like Denver have diverse climates with cold winters and hot summers, and are known for their safety and vibrant communities. 
Our goal is to help you find the perfect home that meets all your needs and preferences.
"""

# Crime rates in different regions:
# {crime_info}

# School ratings in different regions:
# {school_info}
# # Amenities available in different regions:
# {amenities_info}

# Example questions
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
    "What amenities are available in New York?"
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


    # Tokenize a sample text and count the number of tokens
context_tokens = tokenizer.tokenize(context)
num_context_tokens = len(context_tokens)
max_tokens = tokenizer.model_max_length


sample_text = "Perfect Home Finder helps you find the best homes available in various regions."
tokens = tokenizer.tokenize(sample_text)
num_tokens = len(tokens)
max_tokens = tokenizer.model_max_length
tokens_left = max_tokens - num_tokens
print(f"Number of tokens: {num_tokens}")
print(f"Maximum tokens allowed: {max_tokens}")
print(f"Tokens left: {tokens_left}")

for question in questions:
    question_tokens = tokenizer.tokenize(question)
    num_question_tokens = len(question_tokens)
    total_tokens = num_context_tokens + num_question_tokens
    tokens_left = max_tokens - total_tokens
    # print(f"Question: {question}")
    print(f"Number of tokens in question: {num_question_tokens}")
    print(f"Total tokens (context + question): {total_tokens}")
    print(f"Tokens left: {tokens_left}\n")



# Use the NLP model to get the answer
results = []
for question in questions:
    math_result = handle_math_question(question)
    if math_result is not None:
        results.append((question, math_result))
    else:
        result = ask_nlp(question, context)
        results.append((question, result['answer']))

# Print the results
for q, answer in results:
    print(f"Question: {q}")
    print(f"Answer: {answer}\n")