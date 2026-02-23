import { GoogleGenAI, Type } from "@google/genai";
import { SpeechBubble } from "../types";

// Initialize Gemini API Client
// The key is expected to be in process.env.API_KEY
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const SYSTEM_PROMPT = `Analyze the provided manga pages in the exact order they are given.
For each page, detect all speech bubbles.
Return an array of results, where the first item corresponds to the first image, the second item to the second image, and so on.
For each bubble, extract the English text and the bounding box (ymin, xmin, ymax, xmax) on a 0-1000 scale relative to the image dimensions.
Ignore sound effects and narration boxes if they don't contain dialogue.`;

// Now accepts an array of images and returns an array of bubble arrays
export const analyzeMangaPages = async (
  base64Images: string[]
): Promise<SpeechBubble[][]> => {
  try {
    const parts = [];
    
    // Add all images to the request
    base64Images.forEach(b64 => {
        parts.push({
            inlineData: {
                mimeType: "image/png",
                data: b64
            }
        });
    });

    parts.push({ text: SYSTEM_PROMPT });

    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: { parts },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          description: "List of analysis results corresponding to the input images",
          items: {
            type: Type.OBJECT,
            properties: {
              bubbles: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    text: { type: Type.STRING },
                    box_2d: {
                      type: Type.ARRAY,
                      items: { type: Type.INTEGER },
                      description: "The bounding box [ymin, xmin, ymax, xmax] of the bubble."
                    }
                  },
                  required: ["text", "box_2d"]
                }
              }
            },
            required: ["bubbles"]
          }
        }
      }
    });

    const text = response.text;
    if (!text) return base64Images.map(() => []);
    
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        // Map the response to ensure we return just the bubble arrays
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

export const blobUrlToBase64 = async (url: string): Promise<string> => {
  const response = await fetch(url);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result?.toString();
      if (result) {
        // Remove "data:image/xxx;base64," prefix
        resolve(result.split(',')[1]); 
      } else {
        reject(new Error("Failed to convert blob to base64"));
      }
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};