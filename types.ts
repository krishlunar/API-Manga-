export interface MangaItem {
  id: string;
  title: string;
  coverUrl: string;
  pages: string[]; // Blob URLs
}

export type ViewMode = 'library' | 'reader';

export interface LibraryItem {
  id: string;
  title: string;
  coverUrl: string;
  isDemo?: boolean;
}

export interface SpeechBubble {
  text: string;
  box_2d: [number, number, number, number]; // ymin, xmin, ymax, xmax (0-1000 scale)
}

export interface PageAnalysis {
  bubbles: SpeechBubble[];
  status: 'loading' | 'complete' | 'error';
}

export interface OCRCache {
  [pageIndex: number]: PageAnalysis;
}