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

# Load file .env để lấy API key
load_dotenv()

# Khởi tạo Flask
app = Flask(__name__)
CORS(app)

# Lấy OpenAI API key từ biến môi trường
openai_api_key = os.getenv("OPENAI_API_KEY")

# Kiểm tra nếu không tìm thấy API key
if not openai_api_key:
    print("Lỗi: OPENAI_API_KEY không tồn tại. Vui lòng thiết lập biến môi trường này.")
    print(f"Giá trị của OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    # Nếu cần thiết, có thể thoát chương trình ở đây
    # exit()

# Tên mô hình GPT
llm_name = "gpt-4o-mini"

# Lưu trữ trong bộ nhớ (tạm thời, cho đơn giản)
qa_chain = None  # Chuỗi xử lý câu hỏi và trả lời
chat_history = []  # Lịch sử chat

# Hàm load dữ liệu từ file JSON để tạo FAISS database
def load_db_from_json_faiss(json_file, chain_type, k, chunk_size=1000, chunk_overlap=150, start_sample=0, end_sample=100):
    global qa_chain
    # Đọc dữ liệu từ file JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []  # Lưu danh sách các tài liệu
    seen_contexts = set()  # Để tránh trùng lặp context
    for i, item in enumerate(data):
        if i >= end_sample:
            break
        if i >= start_sample:
            context = item['true_context']
            if context not in seen_contexts:
                doc = Document(page_content=context, metadata={'chunk': str(i+1)})
                docs.append(doc)
                seen_contexts.add(context)

    # Chia nhỏ tài liệu thành các đoạn nhỏ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)

    # Tạo vector embeddings từ dữ liệu
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # Tạo retriever để tìm kiếm tài liệu tương tự
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Prompt mẫu để chatbot trả lời
    template = """As an expert assistant specializing in UIT course information, use the provided context to answer the user's question. You must summarize, list, code course or analyze the information as directly instructed by the user. If the relevant information cannot be found within the provided context, respond with "Tôi không thể trả lời câu hỏi này vì thông tin không có sẵn."

    Context:
    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Lưu lại lịch sử hội thoại
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key='answer', return_messages=True)

    # Tạo QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0.5, openai_api_key=openai_api_key),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=False,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return qa

# Hàm khởi tạo chatbot
def initialize_chatbot():
    global qa_chain
    # Đường dẫn tới file JSON (chỉnh lại cho đúng với file của bạn)
    qa_chain = load_db_from_json_faiss('D:/IR_front-end/filtered_course_data.json', 'stuff', 4)
    print("Chatbot đã được khởi tạo!")  # Thông báo khi khởi tạo xong

# API xử lý chat
@app.route('/api/chat', methods=['POST'])
def chat():
    global qa_chain, chat_history
    data = request.get_json()
    user_message = data['message']

    # Kiểm tra nếu chatbot chưa khởi tạo
    if not qa_chain:
        return jsonify({"response": "Chatbot chưa được khởi tạo. Vui lòng thử lại sau."}), 500

    try:
        # Xử lý câu hỏi của người dùng
        result = qa_chain({"question": user_message})
        response = result["answer"]

        # Lưu lịch sử chat
        chat_history.append({"sender": "user", "text": user_message})
        chat_history.append({"sender": "bot", "text": response})

        return jsonify({"response": response})
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu chat: {e}")
        return jsonify({"response": "Đã xảy ra lỗi khi xử lý yêu cầu của bạn."}), 500

# API lấy lịch sử hội thoại
@app.route('/api/history', methods=['GET'])
def get_history():
    global chat_history
    return jsonify(chat_history)

# Chạy ứng dụng Flask
if __name__ == '__main__':
    initialize_chatbot()  # Gọi hàm khởi tạo chatbot
    app.run(debug=True, port=5000)
