import { GoogleGenAI } from "@google/genai";
import { SpeechBubble } from "../types";

// Initialize Gemini API Client
// The key is expected to be in process.env.API_KEY when running in an environment
// that supports it. Do NOT embed or hardcode API keys in source.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const BUBBLE_SYSTEM_PROMPT = `Analyze the provided manga pages in the exact order they are given.
For each page, detect all speech bubbles.
Return an array of results, where the first item corresponds to the first image, the second item to the second image, and so on.
For each bubble, extract the English text and the bounding box (ymin, xmin, ymax, xmax) on a 0-1000 scale relative to the image dimensions.
Ignore sound effects and narration boxes if they don't contain dialogue.`;

const OCR_STACKED_PROMPT = `This image contains 3 manga pages stacked vertically. Read the text and return it formatted strictly as a JSON array of 3 strings, where each string is the text for one page.`;

// Existing bubble analysis (keeps old behavior)
export const analyzeMangaPages = async (
  base64Images: string[]
): Promise<SpeechBubble[][]> => {
  try {
    const parts: any[] = [];

    base64Images.forEach(b64 => {
      parts.push({
        inlineData: { mimeType: "image/png", data: b64 }
      });
    });

    parts.push({ text: BUBBLE_SYSTEM_PROMPT });

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: { parts }
    });

    const text = response.text;
    if (!text) return base64Images.map(() => []);

    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed.map((pageResult: any) => pageResult.bubbles || []);
      }
      return base64Images.map(() => []);
    } catch (e) {
      console.error("Failed to parse JSON response", e);
      return base64Images.map(() => []);
    }
  } catch (error) {
    console.error("OCR Batch Error:", error);
    return base64Images.map(() => []);
  }
};

// Convert a blob URL (or any image URL) to base64 (data part only)
export const blobUrlToBase64 = async (url: string): Promise<string> => {
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result?.toString();
      if (result) resolve(result.split(',')[1]); else reject(new Error("Failed to convert blob to base64"));
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// Merge up to 3 images vertically onto a single canvas and return a base64 JPEG (data part only)
// Images are scaled down to max width of 800px and aggressively compressed
export const mergeImagesToBase64 = async (imageUrls: string[]): Promise<string> => {
  // Load images
  const imgs: HTMLImageElement[] = await Promise.all(imageUrls.map(url => new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error('Failed to load image ' + url));
    img.src = url;
  })));

  // Limit canvas width to 800px for faster uploads
  const MAX_WIDTH = 800;
  const maxNaturalWidth = imgs.reduce((w, img) => Math.max(w, img.naturalWidth || img.width), 0);
  const scale = Math.min(1, MAX_WIDTH / maxNaturalWidth);
  
  const canvasWidth = Math.round(maxNaturalWidth * scale);
  const scaledHeights = imgs.map(img => Math.round((img.naturalHeight || img.height) * scale));
  const totalHeight = scaledHeights.reduce((s, h) => s + h, 0);

  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = totalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas context');

  console.log(`[OCR] Merging ${imgs.length} images: ${canvasWidth}x${totalHeight}px (scale: ${scale.toFixed(2)})`);

  let y = 0;
  for (let i = 0; i < imgs.length; i++) {
    const img = imgs[i];
    const scaledH = scaledHeights[i];
    ctx.drawImage(img, 0, y, canvas.width, scaledH);
    y += scaledH;
  }

  // Export as JPEG with aggressive compression (0.4) for faster upload
  const dataUrl = canvas.toDataURL('image/jpeg', 0.4);
  const base64 = dataUrl.split(',')[1];
  const sizeKB = Math.round((base64.length * 0.75) / 1024);
  console.log(`[OCR] Compressed to ${sizeKB}KB`);
  return base64;
};

// Perform a batched OCR request: merge up to 3 images and send a single request.
// Returns an array of strings (one per page requested).
export const performBatchedOCR = async (imageUrls: string[]): Promise<string[]> => {
  try {
    const mergedBase64 = await mergeImagesToBase64(imageUrls);

    const parts: any[] = [];
    parts.push({ inlineData: { mimeType: 'image/jpeg', data: mergedBase64 } });
    parts.push({ text: OCR_STACKED_PROMPT });

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: { parts }
    });

    const text = response.text;
    if (!text) return imageUrls.map(() => '');

    try {
      // Strip markdown formatting that Gemini might wrap the JSON in
      const cleanedResponse = text.replace(/```json/gi, '').replace(/```/g, '').trim();
      const parsed = JSON.parse(cleanedResponse);
      if (Array.isArray(parsed)) {
        // Ensure length matches requested pages
        const results = imageUrls.map((_, i) => parsed[i] || '');
        return results;
      }
      return imageUrls.map(() => '');
    } catch (e) {
      console.error('Failed to parse OCR JSON response', e, 'Raw response:', text);
      throw new Error(`JSON parse failed: ${e instanceof Error ? e.message : 'unknown error'}. Raw: ${text.substring(0, 100)}`);
    }
  } catch (err) {
    console.error('performBatchedOCR error', err);
    throw err;
  }
};