import React, { useState, useRef } from 'react';
import { Upload, Send, Trash, Loader2 } from 'lucide-react';

const RAGInterface = () => {
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [uploadState, setUploadState] = useState({
    isUploading: false,
    isProcessing: false,
    uploadProgress: 0,
    error: null
  });
  const fileInputRef = useRef(null);

  const handleFileUpload = async (event) => {
    const uploadedFile = event.target.files[0];
    
    // Reset upload state
    setUploadState({
      isUploading: true,
      isProcessing: false,
      uploadProgress: 0,
      error: null
    });
    
    setFile(uploadedFile);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      // Simulate upload progress (optional)
      const simulateProgress = () => {
        let progress = 0;
        const interval = setInterval(() => {
          progress += 20;
          setUploadState(prev => ({
            ...prev,
            uploadProgress: Math.min(progress, 100)
          }));

          if (progress >= 100) {
            clearInterval(interval);
            // Transition to processing
            setUploadState(prev => ({
              ...prev,
              isUploading: false,
              isProcessing: true
            }));
          }
        }, 200);
      };

      simulateProgress();

      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      // Update state on successful upload
      setUploadState(prev => ({
        ...prev,
        isProcessing: false,
        error: null
      }));

      console.log(data.message);
    } catch (error) {
      // Handle upload error
      setUploadState({
        isUploading: false,
        isProcessing: false,
        uploadProgress: 0,
        error: error.message
      });
      console.error('Error uploading file:', error);
    }
  };

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return;

    const newMessage = { text: inputMessage, sender: 'user' };
    setMessages([...messages, newMessage]);
    setInputMessage('');

    try {
      const response = await fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputMessage }),
      });
      const data = await response.json();
      simulateTypingEffect(data.response);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  const simulateTypingEffect = (botResponse) => {
    setIsTyping(true);
    let currentIndex = 0;
    const typingInterval = setInterval(() => {
      if (currentIndex < botResponse.length) {
        const botText = botResponse.slice(0, currentIndex + 15);
        const typingMessage = { text: botText, sender: 'bot' };
        setMessages((prevMessages) => {
          // Replace the last bot message with the "typing" one
          const newMessages = [...prevMessages];
          if (newMessages.length > 0 && newMessages[newMessages.length - 1].sender === 'bot') {
            newMessages[newMessages.length - 1] = typingMessage;
          } else {
            newMessages.push(typingMessage);
          }
          return newMessages;
        });
        currentIndex++;
      } else {
        clearInterval(typingInterval);
        setIsTyping(false);
      }
    }, 10);
  };

  const handleNewCV = async () => {
    // Clear messages
    setMessages([]);
    setFile(null);

    // Make an API call to clear ChromaDB data
    try {
      const response = await fetch('http://localhost:5000/clear_cv_data', {
        method: 'POST',
      });
      const data = await response.json();
      console.log(data.message);
    } catch (error) {
      console.error('Error clearing CV data:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4">
      <div className="rounded-lg border border-gray-300 bg-white text-card-foreground shadow-sm mb-4">
        <div className="flex flex-col space-y-1.5 p-6">
          <h3 className="text-2xl font-semibold leading-none tracking-tight">File Upload</h3>
        </div>
        <div className="p-6 pt-0">
          <div className="flex items-center">
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileUpload}
            />
            <button 
              onClick={() => fileInputRef.current.click()}
              disabled={uploadState.isUploading || uploadState.isProcessing}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
            >
              {uploadState.isUploading || uploadState.isProcessing ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Upload className="mr-2 h-4 w-4" />
              )}
              Upload File
            </button>
            
            {file && <span className="ml-2">{file.name}</span>}
            
            {uploadState.isUploading && (
              <div className="ml-4 w-full bg-gray-200 rounded-full h-2.5">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{width: `${uploadState.uploadProgress}%`}}
                ></div>
              </div>
            )}
            
            {uploadState.isProcessing && (
              <div className="ml-4 flex items-center text-blue-600">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing PDF...
              </div>
            )}
            
            {uploadState.error && (
              <div className="ml-4 text-red-600">
                {uploadState.error}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-gray-300 bg-white text-card-foreground shadow-sm flex flex-col flex-grow">
        <div className="flex flex-col space-y-1.5 p-6">
          <h3 className="text-2xl font-semibold leading-none tracking-tight">Chat</h3>
        </div>
        <div className="p-6 pt-0 flex-grow overflow-y-auto">
          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} mb-2`}>
              <div className={`p-2 rounded-lg ${message.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
                {message.text}
              </div>
            </div>
          ))}
        </div>
        <div className="p-4 border-t">
          <div className="flex items-center">
            <input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-grow mr-2 flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button 
              onClick={handleSendMessage}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2 ml-2"
            >
              <Send className="h-4 w-4" />
            </button>
            <button 
              onClick={handleNewCV}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-destructive text-destructive-foreground hover:bg-destructive/90 h-10 px-4 py-2 ml-2"
            >
              <Trash className="h-4 w-4" /> 
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;