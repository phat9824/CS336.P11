import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ChatHistory from './components/ChatHistory';
import ChatInput from './components/ChatInput';

function App() {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchChatHistory();
    }, []);

    const fetchChatHistory = async () => {
        try {
            const response = await axios.get('http://localhost:5000/api/history');
            setMessages(response.data);
        } catch (error) {
            console.error('Error fetching chat history:', error);
        }
    };

    const handleSendMessage = async (newMessage) => {
        if (newMessage.trim()) {
            const userMessage = { sender: 'user', text: newMessage };
            setMessages(prevMessages => [...prevMessages, userMessage]);

            setLoading(true);
            try {
                const response = await axios.post('http://localhost:5000/api/chat', { message: newMessage });
                const botMessage = { sender: 'bot', text: response.data.response };
                setMessages(prevMessages => [...prevMessages, botMessage]);
            } catch (error) {
                console.error('Error sending message:', error);
                const errorMessage = { sender: 'bot', text: 'Đã có lỗi xảy ra khi gửi tin nhắn.' };
                setMessages(prevMessages => [...prevMessages, errorMessage]);
            } finally {
                setLoading(false);
            }
        }
    };

    return (
        <div className="App">
            <h1>Chatbot thông tin môn học</h1>
            <ChatHistory messages={messages} loading={loading} />
            <ChatInput onSendMessage={handleSendMessage} />
        </div>
    );
}

export default App;
