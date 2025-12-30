from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv() 

# 1. Simple State: No complex 'Annotated' lists. Just a dictionary.
class State(TypedDict):
    query: str
    tool_output: str
    final_answer: str

# 2. Setup Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 3. Define the "Nodes" (The steps)
def researcher(state: State):
    # Imagine this node calls a tool or does a search
    print(f"--- Researching: {state['query']} ---")
    return {"tool_output": "The capital of France is Paris."}

def writer(state: State):
    # This node takes the tool output and writes a nice sentence
    print("--- Writing final answer ---")
    prompt = f"Based on this info: {state['tool_output']}, answer the user: {state['query']}"
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# 4. Connect the dots
workflow = StateGraph(State)

workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# 5. Run it
result = app.invoke({"query": "What is the capital of France?"})
print("\nFinal Result:", result["final_answer"])