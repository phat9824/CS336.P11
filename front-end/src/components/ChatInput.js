import React, { useState } from 'react';

function ChatInput({ onSendMessage }) {
    const [input, setInput] = useState('');

    const handleSubmit = (event) => {
        event.preventDefault();
        onSendMessage(input);
        setInput('');
    };

    return (
        <form className="chat-input" onSubmit={handleSubmit}>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Nhập tin nhắn..."
            />
            <button type="submit">Gửi</button>
        </form>
    );
}

export default ChatInput;