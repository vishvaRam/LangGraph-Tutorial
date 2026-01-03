import os
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- 1. SETUP: FILES & DATA ---
os.makedirs("Data", exist_ok=True)
if not os.path.exists("Data/Book.txt"):
    with open("Data/Book.txt", "w") as f:
        f.write("Marley was dead: to begin with. Scrooge was his partner. Scrooge was a tight-fisted hand at the grindstone.")
if not os.path.exists("Data/HR-Policies-Manuals.txt"):
    with open("Data/HR-Policies-Manuals.txt", "w") as f:
        f.write("Work from Home Policy: Employees can work remotely on Fridays.\nExpense Policy: Dinner is reimbursed up to $30 if working past 8 PM.")

# --- 2. SETUP EMBEDDINGS (Local & Fast) ---
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_or_load_vector_store(file_path, collection_name, persist_dir="./chroma_db_storage"):
    """Loads existing collection if available, otherwise creates new one."""
    
    # Try to load existing collection first
    if os.path.exists(persist_dir):
        try:
            existing_db = Chroma(
                collection_name=collection_name,
                embedding_function=hf_embeddings,
                persist_directory=persist_dir
            )
            # Verify collection has data
            if existing_db._collection.count() > 0:
                print(f"✅ Loaded existing collection '{collection_name}' with {existing_db._collection.count()} documents")
                return existing_db
        except Exception as e:
            print(f"⚠️  Could not load collection '{collection_name}': {e}")
    
    # Create new collection if loading failed or doesn't exist
    print(f"🔄 Creating new collection '{collection_name}'...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None

    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    
    # Create and persist
    new_db = Chroma.from_documents(
        documents=docs,
        embedding=hf_embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    print(f"✅ Created collection '{collection_name}' with {len(docs)} documents")
    return new_db

# Create/Load DBs
print("--- ⚙️ Loading Knowledge Bases... ---")
book_db = create_or_load_vector_store("Data/Book.txt", "christmas_carol_db")
policy_db = create_or_load_vector_store("Data/HR-Policies-Manuals.txt", "company_policy_db")

# --- 3. DEFINE TOOLS ---
tools = []
if book_db:
    tools.append(create_retriever_tool(
        book_db.as_retriever(),
        "search_christmas_carol",
        "Use ONLY for queries about the book 'A Christmas Carol', Scrooge, or Marley."
    ))
if policy_db:
    tools.append(create_retriever_tool(
        policy_db.as_retriever(),
        "search_company_policy",
        "Use ONLY for queries about HR, work from home, expenses, or leave."
    ))

# --- 4. DEFINE STATE (The LangGraph Way) ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- 5. DEFINE NODES ---

def agent_node(state: AgentState):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    sys_msg = ("system", "You are a helpful assistant. "
               "Route queries carefully: HR queries -> Policy Tool. "
               "Book queries -> Book Tool. General queries -> Answer directly.")
    
    messages = [sys_msg] + state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}

tool_node = ToolNode(tools)

# --- 6. DEFINE GRAPH (The Application) ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# --- 7. RUNNER ---
def run_query(query: str):
    print(f"\nUser: {query}")
    inputs = {"messages": [("user", query)]}
    
    for chunk in app.stream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            t_name = last_msg.tool_calls[0]['name']
            print(f"   🤖 Agent decides: Need more info -> Calling [{t_name}]")
            
        if last_msg.type == "tool":
             print(f"   💾 Tool Output: {last_msg.content[:50]}...")
             
    final_msg = chunk["messages"][-1]
    if final_msg.content:
        print(f"Agent: {final_msg.content}")

# Execute Tests
run_query("Who was Scrooge's partner?")
run_query("Can I work from home on Friday?")
run_query("Write a short haiku about AI.")
