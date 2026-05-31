import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
});

export const uploadPDF = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/ingest/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (progressEvent) => {
      if (onUploadProgress && progressEvent.total) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onUploadProgress(percent);
      }
    },
  });
  return response.data;
};

export const askQuestion = async (question, model = "gemini-2.5-flash", filterSource = null) => {
  const response = await api.post("/chat/ask", {
    question,
    model,
    filter_source: filterSource,  // null = search all docs
  });
  return response.data;
};

// Fetch all documents stored in ChromaDB
export const getDocuments = async () => {
  const response = await api.get("/ingest/documents");
  return response.data.documents; // [{ filename, chunks }]
};

export const deleteDocument = async (filename) => {
  const response = await api.delete(`/ingest/delete/${encodeURIComponent(filename)}`);
  return response.data;
};

export const checkHealth = async () => {
  const response = await api.get("/health", { baseURL: "http://localhost:8000" });
  return response.data;
};