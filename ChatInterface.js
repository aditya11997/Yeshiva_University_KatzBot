import React, { useState, useEffect, useRef } from 'react';
import '../styles/ChatInterface.css';  // Ensure you have a CSS file for custom styles
import ReactMarkdown from 'react-markdown';


const ChatInterface = ({ selectedLanguage }) => {
  const [messages, setMessages] = useState([
    { origin: 'bot', message: 'Hello! Welcome to KatzBot, how can I assist you today?\n\n**Disclaimer: This chatbot service is not intended for private or confidential information.**' }
  ]);
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recognition, setRecognition] = useState(null);

  // Reference for chat container to scroll
  const chatMessagesRef = useRef(null);

  // Scroll to bottom of chat container
  const scrollToBottom = () => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight;
    }
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize the Web Speech API on component mount
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      recognitionInstance.continuous = false;  // Stop after one result
      recognitionInstance.interimResults = false; // Only final results
      recognitionInstance.lang = selectedLanguage;  // Set the language for recognition
      setRecognition(recognitionInstance);
    } else {
      console.error("Speech Recognition API is not supported in this browser.");
    }
  }, [selectedLanguage]);

  // Function to stop recording after 10 seconds or silence
  const stopRecording = () => {
    if (recognition) {
      recognition.stop();
      setIsRecording(false);
      console.log("Stopped recording due to silence or timer.");
    }
  };


  const handleVoiceInput = () => {
    if (recognition) {
      setIsRecording(true);
      recognition.start();

      // Timer: Stop recording after 10 seconds
      const timer = setTimeout(() => {
        stopRecording();
      }, 10000);  // 10 seconds

      recognition.onresult = (event) => {
        clearTimeout(timer);
        const transcript = event.results[0][0].transcript;
        setUserInput(transcript); // Set recognized speech in the input field
        setIsRecording(false);
      };

      // Stop recording on silence
      recognition.onspeechend = () => {
        stopRecording();  // Stop recording when user stops speaking
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error", event);
        setIsRecording(false);
      };
    } else {
      console.error("Speech recognition not initialized.");
    }
  };

  // Function to send the message to the Flask API
  const sendMessageToAPI = async (userMessage) => {
    setLoading(true);  // Start spinner
    try {
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          language: selectedLanguage
        }),
      });

      const data = await response.json();
      setLoading(false);  // Stop spinner
      return data.response;
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);  // Stop spinner
      return "Error reaching the server!";
    }
  };


// Function to format text using ReactMarkdown
const formatText = (text) => {
  return (
    <ReactMarkdown components={{
      // Customize any specific markdown elements if needed
      p: ({ node, ...props }) => <span {...props} />, // Replace <p> with <span> for inline text handling
      br: () => <br />, // Handle line breaks explicitly
      b: ({ node, ...props }) => <strong {...props} />, // Ensure **bold** becomes <strong>
      ol: ({ node, ...props }) => <ol {...props} />, // Ensure ordered list is properly rendered
      li: ({ node, ...props }) => <li {...props} />, // Ensure list items are properly rendered
    }}>
      {text}
    </ReactMarkdown>
  );
};


  // Handles when the user submits a message
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (userInput.trim()) {
      const newMessages = [...messages, { origin: 'human', message: userInput }];
      setMessages(newMessages);
      setUserInput("");

      // Send the message to the API and get the bot's response
      const botResponse = await sendMessageToAPI(userInput);

      // Add the bot's response to the chat
      setMessages([...newMessages, { origin: 'bot', message: botResponse }]);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages" ref={chatMessagesRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`chat-row ${msg.origin}`}>
            <div className={`chat-bubble ${msg.origin}`}>
              {msg.origin === 'bot' ? 
                <img src={`${process.env.PUBLIC_URL}/bot-icon.png`} alt="Bot Icon" className="chat-icon" /> 
                : 
                <img src={`${process.env.PUBLIC_URL}/user-icon.png`} alt="User Icon" className="chat-icon" />
              }
              <div className="chat-text">
                {msg.origin === 'bot' ? (
                  <div className="chat-text">
                    {formatText(msg.message)} {/* Apply the formatText function here */}
                  </div>
                ) : (
                  <div className="chat-text">
                    {msg.message}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {loading && <div className="spinner">⌛ Waiting for response...</div>}
      </div>
      <div className='inputForm'>
        <form onSubmit={handleSubmit}>
          <button type="button"
                  onClick={handleVoiceInput}
                  className={`mic-button ${isRecording ? 'recording' : ''}`}
                  disabled={isRecording}>
            <i className="fas fa-microphone"></i>
          </button>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Type your message here"
          />
          <button type="submit" className="send-button">▶️</button>
        </form>
      </div>
      <p className="disclaimer">KatzBot can generate inaccurate answers. Check important info.</p>
    </div>
  );
};

export default ChatInterface;
