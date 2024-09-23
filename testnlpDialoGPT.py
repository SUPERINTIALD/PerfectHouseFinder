import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
# Load the DialoGPT model for text generation
tokenizer_gpt = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model_gpt = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# def handle_math_question(question):
#     # Match basic arithmetic operations
#     match = re.match(r'(\d+)\s*([+\-*/])\s*(\d+)', question.lower())
#     if match:
#         num1, operator, num2 = match.groups()
#         num1, num2 = int(num1), int(num2)
#         if operator == '+':
#             return num1 + num2
#         elif operator == '-':
#             return num1 - num2
#         elif operator == '*':
#             return num1 * num2
#         elif operator == '/':
#             return num1 / num2
#     return None
# Example context and questions


crime_data = pd.read_csv('./database/datasetsCrime/crime.csv/crime.csv')
school_data = load_dataset('mw4/schools')



def extract_crime_info(df):
    info = {}
    for _, row in df.iterrows():
        location = str(row['NEIGHBORHOOD_ID']).strip().capitalize()  # Convert to str and handle missing values
        offense_type = row['OFFENSE_TYPE_ID']
        if location in info:
            info[location].append(offense_type)
        else:
            info[location] = [offense_type]
    
    # Calculate the percentage of each offense type
    for location, offenses in info.items():
        total_offenses = len(offenses)
        offense_counts = pd.Series(offenses).value_counts(normalize=True) * 100
        info[location] = offense_counts.to_dict()
    
    return info
# Adjust the extraction logic based on the dataset structure
# def extract_school_info(dataset):
#     info = {}
#     for item in dataset:
#         if 'name' in item:
#             parts = item['name'].split(',')
#             if len(parts) > 1:
#                 location = parts[-1].strip().capitalize()
#                 school_name = parts[0].strip()
#                 info[location] = school_name
#     return info

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

questions = [
    "Does money buy happiness?",
    "What is the best way to buy happiness ?",
    "How can I find a good house?",
    "This is so difficult !",
 

]
#    "What are the best schools in the area?",
#     "I want to buy a nice home",
#     "What is safety?",
#     "How can I find a good house?",
#     "Tell me about the crime rates in Denver.",
#     "What are the best schools in the area?"
def get_relevant_context(question):
    location_match = re.search(r'in (\w+)', question.lower())
    if location_match:
        location = location_match.group(1).strip().capitalize()
        crime_context = f"Crime rate in {location}: {crime_info.get(location, 'No data available')}"
        school_context = f"School rating in {location}: {school_info.get(location, 'No data available')}"
        return f"{context}\n{crime_context}\n{school_context}"
    return context

def sectionize_input(input_ids, max_length):
    """Split input_ids into sections of max_length."""
    sections = []
    for i in range(0, input_ids.shape[-1], max_length):
        sections.append(input_ids[:, i:i + max_length])
    return sections

def test_dialo_gpt(questions, context):
    results = []
    chat_history_ids = None  # Initialize chat history
    max_length = model_gpt.config.n_positions  # Maximum length for the model

    # for question in questions:
    for step, question in enumerate(questions):
        relevant_context = get_relevant_context(question)

        # Combine context and question
        input_text = relevant_context + "\n" + question
        
        # Encode the input text and add the eos_token
        new_user_input_ids = tokenizer_gpt.encode(input_text + tokenizer_gpt.eos_token, return_tensors='pt')
        

        sections = sectionize_input(new_user_input_ids, max_length)

        for section in sections:

            #Append the new user input tokens to the chat history
            bot_input_ids = torch.cat([chat_history_ids, section], dim=-1) if chat_history_ids is not None else section


            attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

            # Generate a response while limiting the total chat history to 1000 tokens
            chat_history_ids = model_gpt.generate(bot_input_ids, max_new_tokens=50, pad_token_id=tokenizer_gpt.eos_token_id, attention_mask=attention_mask)
            
            # Decode the response
        response = tokenizer_gpt.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Append the question and response to the results
        results.append((question, response))
    
    return results
results = test_dialo_gpt(questions, context)

        # math_result = handle_math_question(question)
        # if math_result is not None:
        #     results.append((question, math_result))
        # else:
        # try:
            # gpt_input = context + "\n\n" + question
            # input_ids = tokenizer_gpt.encode(gpt_input + tokenizer_gpt.eos_token, return_tensors='pt', truncation=True, max_length=1024)
            # gpt_input = context + "\n\n" + question
            
            

#TEST TEST TEST TEST TEST FOR DIALO GPT COPILOT
        #     gpt_input = context + "\n\nQuestion: " + question + "\nAnswer:"
        #     inputs = tokenizer_gpt(gpt_input, return_tensors='pt', truncation=True, max_length=1024)
        #     input_ids = inputs['input_ids']
        #     attention_mask = inputs['attention_mask']
        #     print(f"Input IDs: {input_ids}")
        #     chat_history_ids = model_gpt.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer_gpt.eos_token_id)
        #     # Debugging information
        #     print(f"Chat History IDs: {chat_history_ids}")
        #     answer = tokenizer_gpt.decode(chat_history_ids[0], skip_special_tokens=True)
        #     results.append((question, answer))
        # except Exception as e:
        #     print(f"Error during generation: {e}")
        #     results.append((question, "Sorry, I couldn't generate a response."))

        # Print the results
for question, answer in results:
    print(f"Question: {question}")
    print(f"DialoGPT: {answer}\n")
# print(f"Question: {q}")
    # print(f"Answer: {answer}\n")



# Run the test
# test_dialo_gpt(questions, context)

