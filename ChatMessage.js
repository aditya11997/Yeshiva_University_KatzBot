import React from 'react';
import '../styles/ChatMessage.css';  // For styling

const ChatMessage = ({ message }) => {
  return (
    <div className={`chat-message ${message.sender}`}>
      <div className="chat-bubble">
        <p>{message.text}</p>
      </div>
    </div>
  );
};

export default ChatMessage;
