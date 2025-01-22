from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import json  # To handle JSON-style inputs if needed

# Tool: Mock NLP Function
def mock_ask_nlp_tool(input_text):
    # Parse the input string into a dictionary
    try:
        input_data = eval(input_text)  # Unsafe in production, use `json.loads` for proper JSON input
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        return f"Processed with NLP: '{query}' in context: '{context}'"
    except Exception as e:
        return f"Error processing input for NLP Tool: {str(e)}"

ask_nlp_tool = Tool(
    name="NLP Tool",
    func=mock_ask_nlp_tool,
    description="Handles precise question-answering tasks using NLP. Input should be a dictionary with 'query' and 'context' keys."
)

# Tool: Mock Forecast Function
def mock_forecast_tool(input_text):
    # Parse the input string into a dictionary
    try:
        input_data = eval(input_text)  # Unsafe in production, use `json.loads` for proper JSON input
        query = input_data.get("query", "")
        location = input_data.get("location", "")
        return f"Forecast generated for '{location}' based on '{query}'"
    except Exception as e:
        return f"Error processing input for Forecast Tool: {str(e)}"

forecast_tool = Tool(
    name="Forecast Tool",
    func=mock_forecast_tool,
    description="Handles forecast queries for housing prices. Input should be a dictionary with 'query' and 'location' keys."
)

# Initialize LangChain Agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Replace with HuggingFace model if needed

tools = [ask_nlp_tool, forecast_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Simulate LangChain Execution
def main():
    print("Starting LangChain Test...")

    # Test Query 1: Ask NLP
    query1 = "What are the best schools in Denver?"
    context1 = "Schools in Denver include Denver High School and East Denver Middle School."
    agent_input1 = str({"query": query1, "context": context1})  # Pass as a string
    response1 = agent.run(agent_input1)
    print("\nLangChain Response 1:")
    print(response1)

    # Test Query 2: Forecast
    query2 = "Forecast housing prices in Denver for the next 10 years."
    location2 = "Denver"
    agent_input2 = str({"query": query2, "location": location2})  # Pass as a string
    response2 = agent.run(agent_input2)
    print("\nLangChain Response 2:")
    print(response2)

if __name__ == "__main__":
    main()
