import React, { useState } from 'react';
import { Library } from './components/Library';
import { Reader } from './components/Reader';
import { processMangaFile } from './utils/zipUtils';
import { MangaItem, LibraryItem } from './types';

function App() {
  const [view, setView] = useState<'library' | 'reader'>('library');
  const [currentManga, setCurrentManga] = useState<MangaItem | null>(null);
  const [libraryItems, setLibraryItems] = useState<LibraryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    try {
      const manga = await processMangaFile(file);
      setCurrentManga(manga);
      
      // Add to library list (persisting just metadata for this session)
      setLibraryItems(prev => [
        ...prev, 
        { id: manga.id, title: manga.title, coverUrl: manga.coverUrl }
      ]);
      
      setView('reader');
    } catch (error) {
      console.error("Failed to process file", error);
      alert("Failed to process file. Please ensure it is a valid .zip or .cbz file containing images.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectManga = (id: string) => {
    // In a real app with persistence, we would load the manga pages from storage here.
    // For this memory-only version, we check if it's the currently loaded one.
    if (currentManga && currentManga.id === id) {
        setView('reader');
    } else {
        alert("In-memory storage limit: Only the most recently uploaded manga is readable.");
    }
  };

  const handleCloseReader = () => {
    setView('library');
  };

  if (view === 'reader' && currentManga) {
    return <Reader manga={currentManga} onClose={handleCloseReader} />;
  }

  return (
    <>
      <Library 
        items={libraryItems} 
        onSelect={handleSelectManga} 
        onUpload={handleUpload} 
      />
      
      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm">
           <div className="flex flex-col items-center gap-4">
              <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
              <p className="text-white font-medium animate-pulse">Processing Manga...</p>
           </div>
        </div>
      )}
    </>
  );
}

export default App;