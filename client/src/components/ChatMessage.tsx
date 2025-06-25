
import React from 'react';
import { Bot, User, Leaf } from 'lucide-react';
import type { Message } from './ChatBot';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (message.isBot) {
    return (
      <div className="flex items-start space-x-4 animate-fade-in">
        <div className="w-10 h-10 bg-gradient-to-r from-green-600 to-emerald-600 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg ring-2 ring-green-100">
          <Leaf className="w-5 h-5 text-white" />
        </div>
        <div className="max-w-md lg:max-w-lg xl:max-w-xl">
          <div className="bg-white/90 backdrop-blur-sm border border-green-200 rounded-2xl px-5 py-4 shadow-lg hover:shadow-xl transition-all duration-300">
            <p className="text-gray-800 text-sm leading-relaxed font-medium">{message.text}</p>
          </div>
          <p className="text-xs text-green-600 mt-2 ml-3 font-medium">{formatTime(message.timestamp)}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start space-x-4 justify-end animate-fade-in">
      <div className="max-w-md lg:max-w-lg xl:max-w-xl">
        <div className="bg-gradient-to-r from-green-600 to-emerald-600 rounded-2xl px-5 py-4 shadow-lg hover:shadow-xl transition-all duration-300">
          <p className="text-white text-sm leading-relaxed font-medium">{message.text}</p>
        </div>
        <p className="text-xs text-green-600 mt-2 mr-3 text-right font-medium">{formatTime(message.timestamp)}</p>
      </div>
      <div className="w-10 h-10 bg-gradient-to-r from-slate-600 to-slate-700 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg ring-2 ring-slate-100">
        <User className="w-5 h-5 text-white" />
      </div>
    </div>
  );
};
