import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import pickle

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyDZIg2weE4Wa1I5FGnoUUCnAYw6FQsjOKA"))

# Set up Gemini embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)

# List of PDFs to process
pdf_files = [
    {"path": r"C:\Users\Frhnx\Downloads\13-Human-Psychology-principles.pdf", "index": "textbook_faiss_index"},
    {"path": r"C:\Users\Frhnx\Downloads\Sample-Adult-History-And-Physical-By-M2-Student (1)111.pdf", "index": "medical_faiss_index"},
    {"path": r"C:\Users\Frhnx\Downloads\kepy101.pdf","index": "textbook2_faiss_index"}
]

# Memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to process PDFs and store embeddings
def process_pdf(pdf_path, index_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist. Please check the path.")
    
    if os.path.exists(index_path) and os.path.isdir(index_path):
        print(f"Loading existing FAISS index from {index_path}...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"Processing {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    
    os.makedirs(index_path, exist_ok=True)  # Ensure directory exists
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(index_path)
    return vector_db

# Process all PDFs and store FAISS databases
vector_dbs = {}
for pdf in pdf_files:
    vector_dbs[pdf["index"]] = process_pdf(pdf["path"], pdf["index"])

# Query function
def query_agent(query):
    all_docs = []
    for db in vector_dbs.values():
        all_docs.extend(db.similarity_search(query, k=2))
    
    context = "\n".join([doc.page_content for doc in all_docs])
    
    prompt = f"""
    You are a highly advanced AI assistant. 
    Using the provided textbooks and documents, generate a comprehensive and insightful response to the query.
    Provide a structured and well-articulated answer with supporting details where relevant.
    
    Context:
    {context}
    
    Question: {query}
    Answer:
    """
    
    response = llm.invoke(prompt)
    refined_response = response.content.strip()

    
    # Store conversation history
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(refined_response)
    
    return refined_response

# Chat loop for user interaction
print("Welcome! Ask me anything related to the textbooks and documents. Type 'exit' to quit.")
while True:
    user_query = input("User: ")
    if user_query.lower() == "exit":
        break
    answer = query_agent(user_query)
    print("Bot:", answer)
