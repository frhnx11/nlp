import os
import gradio as gr
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory

# Configure Gemini API
genai.configure(api_key=os.getenv("AIzaSyA88ahtGMDBoT8TryOkmSNP4XIRG7muv00"))

# Set up Gemini embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# List of PDFs to process
pdf_files = [
    {"path": r"C:/Users/Frhnx/Downloads/13-Human-Psychology-principles.pdf", "index": "textbook_faiss_index"},
    {"path": r"C:\Users\Frhnx\Downloads\Sample-Adult-History-And-Physical-By-M2-Student (1)111.pdf", "index": "medical_faiss_index"},
    {"path": r"C:\Users\Frhnx\Downloads\1976_eysenck_wilson_-_textbook_of_human_psychology_lancaster_mtp_press111.pdf", "index": "medical_faiss_index"},
    
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    os.makedirs(index_path, exist_ok=True)
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
        all_docs.extend(db.similarity_search(query, k=3))  # Increase k for better context
    
    if not all_docs:
        return "I couldn't find relevant information. Could you try rephrasing?"
    
    context = "\n".join([doc.page_content for doc in all_docs])
    
    prompt = f"""
    You are a highly advanced AI assistant specializing in medical and psychological analysis.
    Using the provided textbooks and medical history, ask relevant follow-up questions to build a personality profile.
    Your responses should be insightful and structured, ensuring a comprehensive understanding of the user. Ask only one question at a time.

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

# Function to dynamically ask relevant questions
def chatbot_interface(history, user_input):
    if user_input.lower() == "exit":
        summary_prompt = """
        Based on the conversation so far, provide a summary of the user's personality.
        """
        summary = query_agent(summary_prompt)
        history.append((user_input, summary))
        return history
    
    history.append((user_input, "Thinking..."))
    response = query_agent(user_input)
    history[-1] = (user_input, response)
    
    return history

# Create Gradio UI
def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 🏥 Medical Personality Analysis AI
        Answer questions to help build a profile based on your medical history.
        """)
        
        chatbot = gr.Chatbot(elem_id="chatbot", height=400)
        user_input = gr.Textbox(placeholder="Type your response here...", elem_id="user_input")
        submit_btn = gr.Button("Submit")
        
        submit_btn.click(chatbot_interface, inputs=[chatbot, user_input], outputs=chatbot)
        submit_btn.click(lambda: "", inputs=[], outputs=user_input)  # Clear input box after submission
    
    demo.launch()

# Run the Gradio app
if __name__ == "__main__":
    launch_gradio()
