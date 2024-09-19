import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
results = []

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
    "I want to buy a nice home",
    "What is safety?",
    "How can I find a good house?",
    "Tell me about the crime rates in Denver.",
    "What are the best schools in the area?"
]
def test_dialo_gpt(questions, context):
    for question in questions:
        # math_result = handle_math_question(question)
        # if math_result is not None:
        #     results.append((question, math_result))
        # else:
        try:
            # gpt_input = context + "\n\n" + question
            # input_ids = tokenizer_gpt.encode(gpt_input + tokenizer_gpt.eos_token, return_tensors='pt', truncation=True, max_length=1024)
            # gpt_input = context + "\n\n" + question
            gpt_input = context + "\n\nQuestion: " + question + "\nAnswer:"

            inputs = tokenizer_gpt(gpt_input, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Debugging information
            # # Debugging information
            print(f"Input IDs: {input_ids}")
            
            # chat_history_ids = model_gpt.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer_gpt.eos_token_id)
            chat_history_ids = model_gpt.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer_gpt.eos_token_id)

            # Debugging information
            print(f"Chat History IDs: {chat_history_ids}")
            
            answer = tokenizer_gpt.decode(chat_history_ids[0], skip_special_tokens=True)
            results.append((question, answer))
        except Exception as e:
            print(f"Error during generation: {e}")
            results.append((question, "Sorry, I couldn't generate a response."))

        # Print the results
    for q, answer in results:
        print(f"Question: {q}")
        print(f"Answer: {answer}\n")
# print(f"Question: {q}")
    # print(f"Answer: {answer}\n")



# Run the test
test_dialo_gpt(questions, context)