# backend/app.py
import os
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY not found. Please set it as an environment variable.")
    print(f"Giá trị của OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    # Consider exiting the application here if the API key is crucial
    # exit()

llm_name = "gpt-4o-mini"

# In-memory storage for the QA chain and memory (for simplicity)
qa_chain = None
chat_history = []

def load_db_from_json_faiss(json_file, chain_type, k, chunk_size=1000, chunk_overlap=150, start_sample=0, end_sample=100):
    global qa_chain
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    seen_contexts = set()
    for i, item in enumerate(data):
        if i >= end_sample:
            break
        if i >= start_sample:
            context = item['true_context']
            if context not in seen_contexts:
                doc = Document(page_content=context, metadata={'chunk': str(i+1)})
                docs.append(doc)
                seen_contexts.add(context)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    template = """As an expert assistant specializing in UIT course information, use the provided context to answer the user's question. You must summarize, list, code course or analyze the information as directly instructed by the user. If the relevant information cannot be found within the provided context, respond with "Tôi không thể trả lời câu hỏi này vì thông tin không có sẵn."

    Context:
    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key='answer', return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0.5, openai_api_key=openai_api_key),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa

def initialize_chatbot():
    global qa_chain
    # Make sure the path to your JSON data file is correct
    qa_chain = load_db_from_json_faiss('D:/IR_front-end/filtered_course_data.json', 'stuff', 4)
    print("Chatbot initialized!") # Optional: Add a print statement for confirmation

@app.route('/api/chat', methods=['POST'])
def chat():
    global qa_chain, chat_history
    data = request.get_json()
    user_message = data['message']

    if not qa_chain:
        return jsonify({"response": "Chatbot is not initialized yet. Please try again in a moment."}), 500

    try:
        result = qa_chain({"question": user_message})
        response = result["answer"]

        chat_history.append({"sender": "user", "text": user_message})
        chat_history.append({"sender": "bot", "text": response})

        return jsonify({"response": response})
    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    global chat_history
    return jsonify(chat_history)

if __name__ == '__main__':
    initialize_chatbot()  # Gọi hàm khởi tạo ở đây
    app.run(debug=True, port=5000)