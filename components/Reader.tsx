import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MangaItem, OCRCache, SpeechBubble } from '../types';
import { analyzeMangaPages, blobUrlToBase64, performBatchedOCR } from '../utils/ocr';

interface ReaderProps {
  manga: MangaItem;
  onClose: () => void;
}

const BATCH_SIZE = 4;

export const Reader: React.FC<ReaderProps> = ({ manga, onClose }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showControls, setShowControls] = useState(true);
  const [isFullScreen, setIsFullScreen] = useState(false);
  
  // Analysis State
  const [analysisCache, setAnalysisCache] = useState<OCRCache>({});
  const [ocrTextCache, setOcrTextCache] = useState<Record<number, string>>({});
  const [debugError, setDebugError] = useState<string | null>(null);
  const [isAnalysisEnabled, setIsAnalysisEnabled] = useState(false);
  const processingQueue = useRef<Set<number>>(new Set());
  const pendingOCRBatches = useRef<Set<number>>(new Set());
  const preloadedImages = useRef<Record<number, HTMLImageElement>>({});
  
  // TTS State
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const synth = useRef<SpeechSynthesis>(window.speechSynthesis);
  const [activeBubbleIndex, setActiveBubbleIndex] = useState<number | null>(null);
  const lastSpokenPageIndexRef = useRef<number | null>(null);

  // --- Analysis Logic (Batch Queue) ---

  // Preload images for a batch into browser memory only — DO NOT call Gemini here
  const preloadBatch = useCallback(async (batchIndex: number) => {
    if (processingQueue.current.has(batchIndex)) return;

    const startPage = batchIndex * BATCH_SIZE;
    const endPage = Math.min(startPage + BATCH_SIZE, manga.pages.length);

    // If all pages already have preloaded images, skip
    let allPreloadedOrDone = true;
    for (let i = startPage; i < endPage; i++) {
      if (!preloadedImages.current[i]) {
        allPreloadedOrDone = false;
        break;
      }
    }
    if (allPreloadedOrDone) return;

    processingQueue.current.add(batchIndex);

    // Mark entries as preloading so UI can reflect intent (but not call API)
    setAnalysisCache(prev => {
      const nextState = { ...prev };
      for (let i = startPage; i < endPage; i++) {
        if (nextState[i]?.status !== 'complete') {
          nextState[i] = { bubbles: [], status: 'preloaded' };
        }
      }
      return nextState;
    });

    try {
      // Create Image objects and cache them in memory. This is the ONLY background work allowed.
      const promises: Promise<void>[] = [];
      for (let i = startPage; i < endPage; i++) {
        if (!manga.pages[i]) continue;
        if (preloadedImages.current[i]) continue;

        promises.push(new Promise<void>((resolve) => {
          const img = new Image();
          img.crossOrigin = 'anonymous';
          img.onload = () => {
            preloadedImages.current[i] = img;
            resolve();
          };
          img.onerror = () => { resolve(); };
          img.src = manga.pages[i];
        }));
      }

      await Promise.all(promises);
    } catch (err) {
      console.warn('Preload batch failed', err);
    } finally {
      processingQueue.current.delete(batchIndex);
    }
  }, [manga.pages]);

  // Explicit analysis function that MUST be triggered by an explicit user action.
  const analyzeBatch = useCallback(async (batchIndex: number) => {
    if (processingQueue.current.has(batchIndex)) return;

    const startPage = batchIndex * BATCH_SIZE;
    const endPage = Math.min(startPage + BATCH_SIZE, manga.pages.length);

    processingQueue.current.add(batchIndex);

    // Set loading state
    setAnalysisCache(prev => {
      const nextState = { ...prev };
      for (let i = startPage; i < endPage; i++) {
        if (nextState[i]?.status !== 'complete') nextState[i] = { bubbles: [], status: 'loading' };
      }
      return nextState;
    });

    try {
      const imagePromises: Promise<string>[] = [];
      for (let i = startPage; i < endPage; i++) {
        imagePromises.push(blobUrlToBase64(manga.pages[i]));
      }
      const base64Images = await Promise.all(imagePromises);

      // THIS is the only place that calls the Gemini API (analyzeMangaPages)
      const batchResults = await analyzeMangaPages(base64Images);

      setAnalysisCache(prev => {
        const nextState = { ...prev };
        batchResults.forEach((bubbles, relativeIndex) => {
          const absoluteIndex = startPage + relativeIndex;
          bubbles.sort((a, b) => {
            const yDiff = a.box_2d[0] - b.box_2d[0];
            if (Math.abs(yDiff) > 50) return yDiff;
            return b.box_2d[1] - a.box_2d[1];
          });
          nextState[absoluteIndex] = { bubbles, status: 'complete' };
        });
        return nextState;
      });
    } catch (err) {
      console.error('Batch analysis failed', err);
      setAnalysisCache(prev => {
        const nextState = { ...prev };
        for (let i = startPage; i < endPage; i++) {
          nextState[i] = { bubbles: [], status: 'error' };
        }
        return nextState;
      });
    } finally {
      processingQueue.current.delete(batchIndex);
    }
  }, [manga.pages]);

  // Trigger Strategy
  useEffect(() => {
    // Always preload current and next batch images into memory — but do NOT call the API.
    const currentBatch = Math.floor(currentIndex / BATCH_SIZE);
    preloadBatch(currentBatch);

    const nextBatch = currentBatch + 1;
    if (nextBatch * BATCH_SIZE < manga.pages.length) {
      const timer = setTimeout(() => {
        preloadBatch(nextBatch);
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [currentIndex, preloadBatch, manga.pages.length]);

  // --- Background Preloading of next pages (n+1, n+2, n+3) ---
  useEffect(() => {
    const preload = (idx: number) => {
      if (!manga.pages[idx]) return;
      if (preloadedImages.current[idx]) return;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = manga.pages[idx];
      img.onload = () => { preloadedImages.current[idx] = img; };
      img.onerror = () => { /* ignore */ };
    };

    for (let i = 1; i <= 3; i++) {
      preload(currentIndex + i);
    }
  }, [currentIndex, manga.pages]);

  // --- Batched OCR (per-page text) ---
  const requestOCRForPage = useCallback(async (index: number) => {
    try {
      setDebugError(null); // clear previous errors
      
      // Check if all pages in this batch are already cached
      const batchStart = index;
      const batchEnd = Math.min(batchStart + 3, manga.pages.length);
      let allCached = true;
      for (let i = batchStart; i < batchEnd; i++) {
        if (!ocrTextCache[i]) {
          allCached = false;
          break;
        }
      }
      if (allCached) {
        console.log(`[OCR] Pages ${batchStart}-${batchEnd - 1} already cached`);
        return;
      }

      // ===== AUDIO UNLOCK TRICK FOR MOBILE =====
      // Immediately after user interaction (button click), unlock the speech engine
      // with a silent utterance. This bypasses mobile autoplay restrictions.
      const unlockMsg = new SpeechSynthesisUtterance('');
      unlockMsg.volume = 0;
      window.speechSynthesis.speak(unlockMsg);
      // ==========================================

      // Prevent duplicate batch requests
      if (pendingOCRBatches.current.has(batchStart)) {
        console.log(`[OCR] Batch starting at page ${batchStart} already pending`);
        return;
      }
      pendingOCRBatches.current.add(batchStart);
      console.log(`[OCR] Starting batch from page ${batchStart}`);

      // Grab next 3 pages starting from currentIndex
      const urls = manga.pages.slice(batchStart, batchStart + 3);
      
      if (urls.length === 0) {
        console.log(`[OCR] No pages to process`);
        return;
      }

      console.log(`[OCR] Requesting ${urls.length} pages (indices ${batchStart}-${batchStart + urls.length - 1})`);
      const texts = await performBatchedOCR(urls);
      
      // Map texts to absolute page indexes
      setOcrTextCache(prev => {
        const next = { ...prev };
        texts.forEach((t, i) => {
          const absoluteIndex = batchStart + i;
          next[absoluteIndex] = t || '';
          console.log(`[OCR] Cached page ${absoluteIndex}`);
        });
        return next;
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('OCR request failed', err);
      setDebugError(errorMsg);
    } finally {
      pendingOCRBatches.current.delete(index);
    }
  }, [manga.pages]);

  // --- TTS Logic ---

  // Audio unlock effect: triggered when TTS is enabled
  useEffect(() => {
    if (!ttsEnabled) return;

    // Unlock speech engine with silent utterance
    const unlockMsg = new SpeechSynthesisUtterance('');
    unlockMsg.volume = 0;
    window.speechSynthesis.speak(unlockMsg);
    
    console.log('[TTS] Audio unlocked on enable');
  }, [ttsEnabled]);

  // Separate effect to handle Page Turn or Disable
  useEffect(() => {
    if (!ttsEnabled || !isAnalysisEnabled) {
      synth.current.cancel();
      setActiveBubbleIndex(null);
      lastSpokenPageIndexRef.current = null;
    }
  }, [ttsEnabled, isAnalysisEnabled]);

  // Effect to queue speech when data becomes available
  const speakCurrentPage = useCallback(async () => {
    if (!ttsEnabled || !isAnalysisEnabled) return;

    const pageIndex = currentIndex;
    try {
      const cachedText = ocrTextCache[pageIndex];
      if (cachedText && cachedText.trim().length > 0) {
        if (lastSpokenPageIndexRef.current === pageIndex && synth.current.speaking) return;
        if (lastSpokenPageIndexRef.current !== pageIndex) {
          synth.current.cancel();
          setActiveBubbleIndex(null);
        }
        lastSpokenPageIndexRef.current = pageIndex;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(cachedText);

        let voices = window.speechSynthesis.getVoices();
        if (voices.length === 0) {
          await new Promise(resolve => {
            const handleVoices = () => { voices = window.speechSynthesis.getVoices(); window.speechSynthesis.removeEventListener('voiceschanged', handleVoices); resolve(null); };
            window.speechSynthesis.addEventListener('voiceschanged', handleVoices);
            setTimeout(resolve, 1000);
          });
        }
        const voice = voices.find(v => v.name.includes('Google US English')) || voices.find(v => v.lang.startsWith('en'));
        if (voice) utterance.voice = voice;
        utterance.rate = 1.0; utterance.pitch = 1.0; utterance.volume = 1;
        utterance.onstart = () => setActiveBubbleIndex(null);
        utterance.onend = () => setActiveBubbleIndex(null);
        utterance.onerror = (e) => setDebugError(`TTS Error: ${e instanceof Error ? e.message : String(e)}`);
        synth.current.speak(utterance);
        return;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('[TTS] OCR text error:', errorMsg);
      setDebugError(`TTS OCR text error: ${errorMsg}`);
      return;
    }

    // Fallback to bubble-based TTS
    try {
      const currentPageData = analysisCache[pageIndex];
      if (!currentPageData || currentPageData.status !== 'complete') return;
      if (currentPageData.bubbles.length === 0) return;
      if (lastSpokenPageIndexRef.current === pageIndex && synth.current.speaking) return;
      if (lastSpokenPageIndexRef.current !== pageIndex) {
        synth.current.cancel();
        setActiveBubbleIndex(null);
      }
      lastSpokenPageIndexRef.current = pageIndex;
      window.speechSynthesis.cancel();

      let voices = window.speechSynthesis.getVoices();
      if (voices.length === 0) {
        await new Promise(resolve => {
          const handleVoices = () => { voices = window.speechSynthesis.getVoices(); window.speechSynthesis.removeEventListener('voiceschanged', handleVoices); resolve(null); };
          window.speechSynthesis.addEventListener('voiceschanged', handleVoices);
          setTimeout(resolve, 1000);
        });
      }

      currentPageData.bubbles.forEach((bubble, i) => {
        const utterance = new SpeechSynthesisUtterance(bubble.text);
        const voice = voices.find(v => v.name.includes('Google US English')) || voices.find(v => v.lang.startsWith('en'));
        if (voice) utterance.voice = voice;
        utterance.rate = 1.0; utterance.pitch = 1.0; utterance.volume = 1;
        utterance.onstart = () => setActiveBubbleIndex(i);
        utterance.onend = () => { if (i === currentPageData.bubbles.length - 1) setActiveBubbleIndex(null); };
        utterance.onerror = () => { if (i === currentPageData.bubbles.length - 1) setActiveBubbleIndex(null); };
        synth.current.speak(utterance);
      });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('[TTS] Bubbles error:', errorMsg);
      setDebugError(`TTS Bubbles error: ${errorMsg}`);
    }
  }, [currentIndex, ttsEnabled, isAnalysisEnabled, analysisCache, ocrTextCache]);

  useEffect(() => {
    // call speakCurrentPage whenever dependencies change
    speakCurrentPage();
  }, [speakCurrentPage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      synth.current.cancel();
    };
  }, []);

  // --- Navigation & UI ---

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ') {
        setCurrentIndex((prev) => Math.min(prev + 1, manga.pages.length - 1));
      } else if (e.key === 'ArrowLeft') {
        setCurrentIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Escape') {
        if (document.fullscreenElement) {
           document.exitFullscreen();
        } else {
           onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [manga.pages.length, onClose]);

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullScreen(true);
    } else {
      document.exitFullscreen();
      setIsFullScreen(false);
    }
  };

  const handleNext = () => setCurrentIndex((prev) => Math.min(prev + 1, manga.pages.length - 1));
  const handlePrev = () => setCurrentIndex((prev) => Math.max(prev - 1, 0));
  const toggleControls = () => setShowControls(!showControls);

  const currentAnalysis = analysisCache[currentIndex];
  const isLoading = currentAnalysis?.status === 'loading';

  return (
    <div className="relative h-screen w-full flex items-center justify-center bg-black overflow-hidden select-none font-sans">
      
      {/* Debug Error Box */}
      {debugError && (
        <div className="absolute top-0 left-0 right-0 z-[999] bg-red-600 text-white p-4 text-sm font-bold border-b-2 border-red-800 max-h-32 overflow-auto break-words">
          <div className="flex justify-between items-start">
            <span>ERROR: {debugError}</span>
            <button onClick={() => setDebugError(null)} className="ml-4 font-bold text-lg">×</button>
          </div>
        </div>
      )}
      
      {/* Container for Image + Overlays */}
      <div 
        className="relative h-full w-full max-w-5xl flex items-center justify-center"
        onClick={toggleControls}
      >
        <img 
            src={manga.pages[currentIndex]} 
            className="max-h-full max-w-full object-contain shadow-2xl"
            alt={`Page ${currentIndex + 1}`}
        />

        {/* Bounding Box Overlays */}
        {isAnalysisEnabled && currentAnalysis?.status === 'complete' && (
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <OverlayLayer 
                    bubbles={currentAnalysis.bubbles} 
                    activeBubbleIndex={activeBubbleIndex}
                />
            </div>
        )}
      </div>

      {/* Loading Indicator */}
      {isAnalysisEnabled && isLoading && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 flex flex-col items-center gap-2">
            <div className="size-12 rounded-full border-4 border-primary border-t-transparent animate-spin drop-shadow-lg"></div>
            <div className="bg-black/60 backdrop-blur-md px-4 py-1 rounded-full text-white text-xs font-bold uppercase tracking-wider">
                Analyzing Batch...
            </div>
        </div>
      )}

      {/* Top Overlay */}
      <div className={`absolute top-0 left-0 right-0 p-6 flex justify-between items-start transition-opacity duration-300 z-30 ${showControls ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <button 
          onClick={onClose}
          className="size-12 flex items-center justify-center rounded-lg bg-reader-dark/40 backdrop-blur-md border border-white/10 hover:bg-reader-dark/60 transition-colors text-white"
        >
          <span className="material-symbols-outlined text-2xl">home</span>
        </button>

        {/* TTS Toggle Pill */}
        {isAnalysisEnabled && (
             <button 
                onClick={(e) => { e.stopPropagation(); setTtsEnabled(!ttsEnabled); }}
                className={`flex items-center gap-2 px-4 h-12 rounded-full backdrop-blur-md border transition-all ${ttsEnabled ? 'bg-primary text-white border-primary/50' : 'bg-reader-dark/40 text-white/70 border-white/10 hover:bg-reader-dark/60'}`}
             >
                <span className="material-symbols-outlined text-xl">
                    {ttsEnabled ? 'volume_up' : 'volume_off'}
                </span>
                <span className="text-sm font-medium hidden sm:inline">
                    {ttsEnabled ? 'Audio On' : 'Audio Off'}
                </span>
             </button>
        )}

        <button 
          onClick={toggleFullScreen}
          className="size-12 flex items-center justify-center rounded-lg bg-reader-dark/40 backdrop-blur-md border border-white/10 hover:bg-reader-dark/60 transition-colors text-white"
        >
          <span className="material-symbols-outlined text-2xl">
            {isFullScreen ? 'close_fullscreen' : 'fullscreen'}
          </span>
        </button>
      </div>

      {/* Bottom Controls */}
      <div className={`fixed bottom-10 left-0 right-0 px-6 flex flex-col items-center gap-6 transition-all duration-300 z-30 ${showControls ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0 pointer-events-none'}`}>
        
        {/* Slider */}
        <div className="w-full max-w-xl flex items-center gap-4">
            <span className="text-xs text-white/50 w-8 text-right">{currentIndex + 1}</span>
            <input 
              type="range" 
              min="0" 
              max={manga.pages.length - 1} 
              value={currentIndex} 
              onChange={(e) => setCurrentIndex(Number(e.target.value))}
              className="flex-1 h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-white transition-all hover:bg-white/20 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white"
            />
            <span className="text-xs text-white/50 w-8">{manga.pages.length}</span>
        </div>

        {/* Main Buttons */}
        <div className="flex items-center gap-4 p-2 rounded-xl bg-reader-dark/60 backdrop-blur-xl border border-white/10 shadow-2xl">
          <button 
            onClick={handlePrev}
            disabled={currentIndex === 0}
            className="size-14 flex items-center justify-center rounded-lg hover:bg-white/10 transition-colors disabled:opacity-30 disabled:hover:bg-transparent"
          >
            <span className="material-symbols-outlined text-white text-3xl">chevron_left</span>
          </button>

          {/* Central Action: Analysis Toggle */}
          <div className="relative group">
             <button 
               onClick={() => {
                 const batch = Math.floor(currentIndex / BATCH_SIZE);
                 setIsAnalysisEnabled(true);
                 analyzeBatch(batch);
               }}
               className={`size-16 flex items-center justify-center rounded-lg transition-all border ${isAnalysisEnabled ? 'bg-primary text-white border-primary shadow-[0_0_20px_rgba(75,43,238,0.5)]' : 'bg-white/5 text-white/80 border-white/5 hover:bg-white/10'}`}
               title="Analyze/Scan current batch"
             >
                 <span className="material-symbols-outlined text-3xl">psychology</span>
             </button>
          </div>

          {/* OCR Extract Button - Show if current page not cached */}
          {!ocrTextCache[currentIndex] && (
            <button
              onClick={() => requestOCRForPage(currentIndex)}
              className="size-14 flex items-center justify-center rounded-lg bg-blue-600/30 hover:bg-blue-600/50 border border-blue-500 transition-colors shadow-lg"
              title="Extract text from current page and next 2 pages"
            >
              <span className="material-symbols-outlined text-blue-300 text-2xl">text_snippet</span>
            </button>
          )}
          {ocrTextCache[currentIndex] && (
            <button
              onClick={() => requestOCRForPage(currentIndex)}
              className="size-14 flex items-center justify-center rounded-lg hover:bg-white/10 transition-colors opacity-50"
              title="Extract text (current page already cached)"
              disabled
            >
              <span className="material-symbols-outlined text-white text-2xl">check_circle</span>
            </button>
          )}

          <button 
            onClick={handleNext}
            disabled={currentIndex === manga.pages.length - 1}
            className="size-14 flex items-center justify-center rounded-lg hover:bg-white/10 transition-colors disabled:opacity-30 disabled:hover:bg-transparent"
          >
            <span className="material-symbols-outlined text-white text-3xl">chevron_right</span>
          </button>
        </div>
      </div>
    </div>
  );
};

// Helper component to handle positioning
const OverlayLayer: React.FC<{ bubbles: SpeechBubble[], activeBubbleIndex: number | null }> = ({ bubbles, activeBubbleIndex }) => {
    const [imgRect, setImgRect] = useState<{width: number, height: number} | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const updateRect = () => {
            if (containerRef.current) {
                const img = containerRef.current.parentElement?.querySelector('img');
                if (img) {
                    setImgRect({
                        width: img.offsetWidth,
                        height: img.offsetHeight
                    });
                }
            }
        };
        updateRect();
        window.addEventListener('resize', updateRect);
        const interval = setInterval(updateRect, 500);
        return () => {
            window.removeEventListener('resize', updateRect);
            clearInterval(interval);
        }
    }, []);

    if (!imgRect) return <div ref={containerRef} />;

    return (
        <div 
            ref={containerRef}
            style={{ 
                width: imgRect.width, 
                height: imgRect.height, 
                position: 'absolute' 
            }}
        >
            {bubbles.map((bubble, i) => {
                const [ymin, xmin, ymax, xmax] = bubble.box_2d;
                const top = (ymin / 1000) * 100;
                const left = (xmin / 1000) * 100;
                const height = ((ymax - ymin) / 1000) * 100;
                const width = ((xmax - xmin) / 1000) * 100;

                const isActive = i === activeBubbleIndex;

                return (
                    <div
                        key={i}
                        className={`absolute transition-all duration-300 group ${isActive ? 'border-4 border-green-400 bg-green-400/20 z-50 shadow-[0_0_15px_rgba(74,222,128,0.5)]' : 'border-2 border-red-500 bg-red-500/10 hover:bg-red-500/20'}`}
                        style={{
                            top: `${top}%`,
                            left: `${left}%`,
                            width: `${width}%`,
                            height: `${height}%`,
                        }}
                        title={bubble.text}
                    >
                        {/* Number Badge (Hide if active to reduce clutter, or change color) */}
                        <div className={`absolute -top-3 -left-3 size-6 rounded-full text-white text-xs font-bold flex items-center justify-center shadow-md z-10 ${isActive ? 'bg-green-600 scale-110' : 'bg-red-600'}`}>
                            {i + 1}
                        </div>
                        
                        {/* Tooltip */}
                        <div className={`opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-black/90 text-white text-xs p-2 rounded w-48 pointer-events-none transition-opacity z-20 ${isActive ? 'hidden' : ''}`}>
                            {bubble.text}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};