from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # If you have a compatible GPU

# Specify the Hugging Face model name
model_name = "deepseek-ai/DeepSeek-V3"

# Download configuration and tokenizer
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Download model weights
model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
# Input text
input_text = "What is the purpose of life?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Decode output
print(tokenizer.decode(outputs.logits.argmax(-1), skip_special_tokens=True))
