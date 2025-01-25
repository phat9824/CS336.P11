import React from 'react';

function ChatHistory({ messages, loading }) {
    return (
        <div className="chat-history">
            {messages.map((message, index) => (
                <div key={index} className={`message ${message.sender}`}>
                    {message.text}
                </div>
            ))}
            {loading && <div className="message bot loading">Đang trả lời...</div>}
        </div>
    );
}

export default ChatHistory;