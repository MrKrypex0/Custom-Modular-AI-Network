"use client";

import React, { useState, useEffect } from 'react';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [users, setUsers] = useState([]);
  const [chatRooms, setChatRooms] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchUsers();
    fetchChatRooms();
    fetchMessages();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await fetch('/api/users');
      const data = await response.json();
      setUsers(data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchChatRooms = async () => {
    try {
      const response = await fetch('/api/chatrooms');
      const data = await response.json();
      setChatRooms(data);
    } catch (error) {
      console.error('Error fetching chat rooms:', error);
    }
  };

  const fetchMessages = async () => {
    try {
      const response = await fetch('/api/messages');
      const data = await response.json();
      setMessages(data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const sendMessage = async (type = 'text') => {
    if (!currentMessage.trim() && type === 'text') return;

    setLoading(true);
    try {
      const response = await fetch('/api/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: currentMessage,
          type: type,
        }),
      });

      const data = await response.json();
      setMessages([...messages, data]);
      setCurrentMessage('');
    } catch (error) {
      console.error('Error sending message:', error);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
      <div className="w-full flex">
        {/* Sidebar */}
        <div className="w-64 bg-gray-800 h-screen p-4 border-r border-gray-700">
          <div className="mb-6">
            <h2 className="text-lg font-bold mb-4 text-white">Accounts</h2>
            <div className="space-y-2">
              {users.map((user) => (
                <div key={user.id} className="flex items-center p-2 hover:bg-gray-700 rounded">
                  <div className="w-8 h-8 rounded-full bg-gray-500"></div>
                  <span className="ml-2 text-white">{user.name}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="border-t border-gray-700 pt-4">
            <h2 className="text-lg font-bold mb-4 text-white">Chat Rooms</h2>
            <div className="space-y-2">
              {chatRooms.map((room) => (
                <div key={room.id} className="p-2 hover:bg-gray-700 rounded text-white">
                  {room.name}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col h-screen bg-white">
          <div className="p-4 border-b border-gray-200 bg-gray-50">
            <h1 className="text-xl font-bold text-gray-900">General Chat</h1>
          </div>

          {/* Messages */}
          <div className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-4">
              {messages.map((message) => (
                <div key={message.id} className={`flex items-start ${message.isCurrentUser ? 'justify-end' : ''}`}>
                  {!message.isCurrentUser && <div className="w-8 h-8 rounded-full bg-gray-600"></div>}
                  <div className={`${message.isCurrentUser ? 'mr-4 bg-blue-600 text-white' : 'ml-4 bg-gray-100'} p-3 rounded-lg`}>
                    <p className="font-semibold">{message.isCurrentUser ? 'You' : message.user.name}</p>
                    <p>{message.content}</p>
                  </div>
                  {message.isCurrentUser && <div className="w-8 h-8 rounded-full bg-gray-600"></div>}
                </div>
              ))}
            </div>
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="flex space-x-4">
              <input
                type="text"
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
              />
              <button 
                onClick={() => sendMessage('text')}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                Send
              </button>
              <button 
                onClick={() => sendMessage('image')}
                disabled={loading}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                Image
              </button>
              <button 
                onClick={() => sendMessage('video')}
                disabled={loading}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50"
              >
                Video
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>  
  );
}