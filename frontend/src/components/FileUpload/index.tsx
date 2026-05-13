import { useCallback, useState, useRef } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isUploading: boolean;
}

export default function FileUpload({ onFileSelect, isUploading }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
        if (file.type.startsWith('image/')) {
          const url = URL.createObjectURL(file);
          setPreview(url);
        }
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300 ${
        isDragOver
          ? 'border-primary-400 bg-primary-500/10 scale-[1.02]'
          : 'border-dark-300 bg-dark-700/50 hover:border-primary-500/50 hover:bg-dark-600/50'
      } ${isUploading ? 'pointer-events-none opacity-60' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        onChange={handleChange}
        className="hidden"
        id="file-upload-input"
      />

      {preview ? (
        <div className="flex flex-col items-center gap-4">
          <img src={preview} alt="Preview" className="w-48 h-48 object-cover rounded-xl shadow-2xl" />
          <p className="text-dark-100 text-sm">Click or drop another file to replace</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-2xl bg-dark-500/50 flex items-center justify-center">
            <svg className="w-10 h-10 text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div>
            <p className="text-lg font-semibold text-white">Drop your file here</p>
            <p className="text-sm text-dark-100 mt-1">or click to browse • Images & Videos up to 100MB</p>
          </div>
          <div className="flex gap-2 mt-2">
            {['JPG', 'PNG', 'WebP', 'MP4'].map((fmt) => (
              <span key={fmt} className="px-3 py-1 rounded-full bg-dark-500/50 text-xs text-dark-50 font-medium">
                {fmt}
              </span>
            ))}
          </div>
        </div>
      )}

      {isUploading && (
        <div className="absolute inset-0 flex items-center justify-center bg-dark-800/80 rounded-2xl">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin" />
            <p className="text-primary-400 font-medium">Uploading...</p>
          </div>
        </div>
      )}
    </div>
  );
}
