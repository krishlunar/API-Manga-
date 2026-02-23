import React, { useRef } from 'react';
import { LibraryItem } from '../types';

interface LibraryProps {
  items: LibraryItem[];
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
}

export const Library: React.FC<LibraryProps> = ({ items, onSelect, onUpload }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-background-dark text-white font-display">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-6 bg-background-dark/80 backdrop-blur-md border-b border-white/5">
        <h1 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-[#a78bfa] to-[#4b2bee] bg-clip-text text-transparent">
          Library
        </h1>
      </header>

      {/* Manga Grid Section */}
      <main className="px-4 pb-24 pt-4">
        {items.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[60vh] text-center text-white/50">
             <span className="material-symbols-outlined text-7xl mb-6 text-white/10">library_books</span>
             <h2 className="text-xl font-medium text-white mb-2">Your library is empty</h2>
             <p className="text-sm max-w-md mx-auto mb-8">Upload a local .cbz or .zip file to start reading your manga collection.</p>
             <button
                onClick={() => fileInputRef.current?.click()}
                className="px-8 py-3 bg-primary hover:bg-primary/90 text-white rounded-full font-medium transition-all shadow-[0_0_20px_rgba(75,43,238,0.3)] hover:shadow-[0_0_25px_rgba(75,43,238,0.5)] active:scale-95"
             >
                Upload Manga
             </button>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {items.map((item) => (
              <div 
                key={item.id}
                onClick={() => onSelect(item.id)}
                className="group relative aspect-[2/3] overflow-hidden rounded-lg bg-primary/10 transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] cursor-pointer ring-1 ring-white/10 hover:ring-primary/50"
              >
                <div 
                  className="h-full w-full bg-cover bg-center transition-transform duration-500 group-hover:scale-110" 
                  style={{ backgroundImage: `url('${item.coverUrl}')` }}
                >
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-4">
                      <p className="text-sm font-medium text-white line-clamp-2">{item.title}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Upload Button (FAB) - Only show if items exist */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept=".zip,.cbz,.rar" 
        className="hidden" 
      />
      {items.length > 0 && (
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="fixed bottom-8 right-8 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-primary text-white shadow-[0_0_15px_rgba(75,43,238,0.4)] transition-transform hover:scale-105 active:scale-95 hover:bg-primary/90"
          aria-label="Upload Manga"
        >
          <span className="material-symbols-outlined text-3xl">add</span>
        </button>
      )}
    </div>
  );
};