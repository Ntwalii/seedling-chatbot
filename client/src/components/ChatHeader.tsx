
import React from 'react';
import { Sprout } from 'lucide-react';

export const ChatHeader = () => {
  return (
    <div className="bg-white/80 backdrop-blur-sm border-b border-green-200 p-4 shadow-sm">
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 bg-gradient-to-r from-green-600 to-emerald-600 rounded-full flex items-center justify-center">
          <Sprout className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-green-800">Seedling AI</h1>
        </div>
      </div>
    </div>
  );
};
