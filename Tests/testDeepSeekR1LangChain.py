import requests
from typing import Any, Dict, List
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.llms.base import BaseLLM
from langchain.agents import initialize_agent, Tool

from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult
from langchain.memory import ConversationBufferMemory

from typing import List, Optional
import requests
import json

from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult
from typing import List, Optional
from pydantic import Field
import requests
import json

class DeepSeekLLM(BaseLLM):
    """
    Custom LangChain wrapper for DeepSeek R1 API.
    """
    api_url: str = Field(..., description="The URL of the DeepSeek API.")
    model_name: str = Field(..., description="The name of the model to use.")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Calls the DeepSeek API and returns the response as a string.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code}, {response.text}")

        # Process streamed JSON response
        raw_response = response.text.strip()
        # print("Raw Response:", raw_response)

        assistant_response = ""
        for line in raw_response.splitlines():
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    # Remove <think> tags if present
                    if "<think>" in content or "</think>" in content:
                        content = content.replace("<think>", "").replace("</think>", "").strip()
                    assistant_response += content
            except json.JSONDecodeError:
                continue

        return assistant_response


    async def _agenerate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        Asynchronous method to handle prompts in a batch.
        """
        return self._generate(prompts, stop, **kwargs)

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        Handles batch processing of prompts and formats the output as an LLMResult.
        """
        generations = []
        for prompt in prompts:
            output = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=output)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "deepseek"

# Step 1: Replace ChatOpenAI with DeepSeekLLM
api_url = "http://localhost:11434/api/chat"
deepseek_model = "deepseek-r1:14b"

# Initialize DeepSeek LLM
llm = DeepSeekLLM(api_url=api_url, model_name=deepseek_model)

# Step 2: Add Memory
memory = ConversationBufferMemory(k=3)

# Step 3: Define a Tool
def llm_query_tool(input_text: str) -> str:
    """A simple tool that queries the LLM."""
    return llm(input_text)

# Create a tool object
query_tool = Tool(
    name="DeepSeek Query Tool",
    func=llm_query_tool,
    description="Use this tool to query the DeepSeek LLM with any question or prompt."
)

# Step 4: Initialize the Agent
# agent = initialize_agent(
#     tools=[query_tool],
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=True,
#     max_iteration=5,
#     memory=memory  # Add memory for truncation
# )
agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iteration=5,
    memory=memory,  # Add memory for truncation
    handle_parsing_errors=True  # Retry on parsing errors
)

# Step 5: Use the Agent
if __name__ == "__main__":
    print("Welcome to the LangChain DeepSeek Agent!")
    while True:
        user_input = input("Ask me anything (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")
