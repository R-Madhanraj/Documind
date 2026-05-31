import { useState, useRef } from "react";
import { uploadPDF } from "../services/api";

export const PDFUploader = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging]   = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress]       = useState(0);
  const [error, setError]             = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    if (file.type !== "application/pdf") { setError("Only PDF files supported"); return; }
    if (file.size > 50 * 1024 * 1024)   { setError("Max file size is 50MB"); return; }

    setError(null);
    setIsUploading(true);
    setProgress(0);

    try {
      const result = await uploadPDF(file, setProgress);
      onUploadSuccess(result);
    } catch (err) {
      setError(err.response?.data?.detail || "Upload failed. Is the backend running?");
    } finally {
      setIsUploading(false);
      setProgress(0);
    }
  };

  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = ()  => setIsDragging(false);
  const onDrop      = (e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); };

  return (
    <div>
      <div
        onClick={() => !isUploading && fileInputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`border border-dashed rounded-xl p-7 text-center transition-all
          ${isUploading ? "opacity-60 cursor-default" : "cursor-pointer"}
          ${isDragging  ? "border-accent bg-accent-dim" : "border-bdr-hi bg-bg hover:border-muted"}
        `}
      >
        {isUploading ? (
          <div>
            <div className="text-xs text-muted font-mono mb-3">Processing chunks...</div>
            <div className="h-0.5 bg-bdr rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${progress}%`,
                  background: "linear-gradient(90deg, #00e5a0, #00b3ff)",
                }}
              />
            </div>
            <div className="text-[11px] text-accent font-mono mt-2">{progress}%</div>
          </div>
        ) : (
          <div>
            <div className={`w-10 h-10 mx-auto mb-3 bg-bdr border border-bdr-hi rounded-xl flex items-center justify-center text-lg transition-transform
              ${isDragging ? "scale-110" : "scale-100"}`}
            >📄</div>
            <div className="text-sm font-semibold mb-1">
              {isDragging ? "Drop it" : "Drop PDF here"}
            </div>
            <div className="text-[11px] text-muted font-mono">or click to browse</div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-2.5 px-3 py-2.5 bg-danger/10 border border-danger/30 rounded-lg text-xs text-danger font-mono">
          {error}
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  );
};