import JSZip from 'jszip';
import { MangaItem } from '../types';

export const processMangaFile = async (file: File): Promise<MangaItem> => {
  const zip = new JSZip();
  const zipContent = await zip.loadAsync(file);
  
  const imageFiles: { name: string; blob: Blob }[] = [];

  // Iterate over files in the zip
  const entries = Object.keys(zipContent.files).filter((filename) => {
    // Filter out directories and non-image files (basic check)
    return !zipContent.files[filename].dir && /\.(jpg|jpeg|png|webp|gif)$/i.test(filename);
  });

  // Sort alphanumerically to ensure correct page order
  // Using 'numeric' collation handles "1.jpg", "2.jpg", "10.jpg" correctly
  entries.sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

  for (const filename of entries) {
    const fileData = zipContent.files[filename];
    const blob = await fileData.async('blob');
    imageFiles.push({ name: filename, blob });
  }

  if (imageFiles.length === 0) {
    throw new Error("No valid images found in the archive.");
  }

  // Create Object URLs
  const pages = imageFiles.map(img => URL.createObjectURL(img.blob));

  return {
    id: crypto.randomUUID(),
    title: file.name.replace(/\.(cbz|zip)$/i, ''),
    coverUrl: pages[0], // Use first page as cover
    pages: pages,
  };
};