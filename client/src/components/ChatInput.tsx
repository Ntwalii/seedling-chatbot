
import React, { KeyboardEvent } from 'react';
import { Send, Sprout } from 'lucide-react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: (message: string) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSend,
  disabled = false,
}) => {
  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (value.trim() && !disabled) {
      onSend(value);
    }
  };

  return (
    <div className="bg-white/95 backdrop-blur-md border-t border-green-200 p-6 shadow-2xl">
      <div className="flex items-end space-x-4 max-w-4xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={disabled ? "Seedling AI is thinking..." : "Ask me about crops, soil, weather, or farming techniques..."}
            disabled={disabled}
            rows={1}
            className="w-full resize-none border-2 border-green-200 rounded-2xl px-5 py-4 pr-12 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500 disabled:bg-green-50 disabled:cursor-not-allowed text-sm leading-relaxed placeholder-green-400 shadow-lg transition-all duration-200"
            style={{ minHeight: '52px', maxHeight: '120px' }}
          />
          {disabled && (
            <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
              <Sprout className="w-4 h-4 text-green-500 animate-pulse" />
            </div>
          )}
        </div>
        <button
          onClick={handleSend}
          disabled={!value.trim() || disabled}
          className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-green-300 disabled:to-emerald-300 text-white rounded-2xl p-4 transition-all duration-200 hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed shadow-lg hover:shadow-xl group"
        >
          <Send className="w-5 h-5 group-hover:translate-x-0.5 transition-transform duration-200" />
        </button>
      </div>
      <div className="text-center mt-3">
        <p className="text-xs text-green-600 font-medium">Press Enter to send • Shift + Enter for new line</p>
      </div>
    </div>
  );
};
