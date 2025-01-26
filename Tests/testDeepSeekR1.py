import requests
import json

url = "http://localhost:11434/api/chat"
payload = {
    "model": "deepseek-r1:14b",
    "messages": [
        {"role": "user", "content": "What is the purpose of life?"}
    ]
}

try:
    # Send the request
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)

    # Process raw response text line by line
    raw_response = response.text.strip()

    # Initialize variables for tags
    inside_think_tag = False
    think_content = ""

    for line in raw_response.splitlines():
        try:
            # Parse each line as a JSON object
            data = json.loads(line)
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]

                # Check for <think> tags and capture their content
                if "<think>" in content:
                    inside_think_tag = True
                    print("\n--- Start of <think> ---")
                elif "</think>" in content:
                    inside_think_tag = False
                    print("--- End of <think> ---\n")
                elif inside_think_tag:
                    think_content += content + " "
                
                # Print content regardless of tags
                print(content, end="")

        except json.JSONDecodeError:
            print(f"Invalid JSON Line: {line}")

    # Optionally display captured <think> content separately
    # if think_content.strip():
        # print("\n\nCaptured <think> Content:")
        # print(think_content.strip())

except requests.exceptions.RequestException as e:
    print("Request Exception:", e)
