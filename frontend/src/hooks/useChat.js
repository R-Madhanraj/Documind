import { useState, useCallback } from "react";
import { askQuestion } from "../services/api";

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (question, model = "gemini-2.5-flash", filterSource = null) => {
    if (!question.trim() || isLoading) return;

    setError(null);

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: question,
      sources: [],
    };

    const loadingMessage = {
      id: Date.now() + 1,
      role: "assistant",
      content: "",
      sources: [],
      loading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setIsLoading(true);

    try {
      const response = await askQuestion(question, model, filterSource);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.loading
            ? { ...msg, content: response.answer, sources: response.sources, loading: false }
            : msg
        )
      );
    } catch (err) {
      let errorMessage = "Something went wrong. Please try again.";

      if (err.response?.status === 400) {
        errorMessage = err.response.data.detail;
      } else if (err.response?.status === 422) {
        errorMessage = "Question is too short — please add more detail.";
      } else if (err.response?.status === 429) {
        errorMessage = "Gemini API quota exceeded. Try again later.";
      } else if (err.response?.status === 503) {
        errorMessage = "Gemini API unavailable. Check your API key.";
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (!err.response) {
        errorMessage = "Cannot reach backend. Is it running on port 8000?";
      }

      setMessages((prev) =>
        prev.map((msg) =>
          msg.loading
            ? { ...msg, content: errorMessage, sources: [], loading: false, isError: true }
            : msg
        )
      );
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, isLoading, error, sendMessage, clearMessages };
};