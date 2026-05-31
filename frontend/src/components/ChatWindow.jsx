import { useState, useEffect, useRef } from "react";
import { CitationBadge } from "./CitationBadge";

export const ChatWindow = ({ messages, isLoading, onSendMessage, chatModel, activeDoc }) => {
  const [input, setInput]           = useState("");
  const [inputError, setInputError] = useState(null);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed)           { setInputError("Type something first"); return; }
    if (trimmed.length < 2) { setInputError("Too short — add a bit more detail"); return; }
    setInputError(null);
    onSendMessage(trimmed, chatModel, activeDoc);
    setInput("");
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const suggestions = [
    "Summarise this document",
    "What are the main topics?",
    "List the key points",
    "What does page 1 say?",
  ];

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* Scope banner */}
      <div className="flex items-center gap-1.5 px-8 py-2 border-b border-bdr bg-surface text-[11px] text-muted font-mono">
        <span className="text-accent">⌖</span>
        {activeDoc
          ? <><span>Searching in</span>&nbsp;<span className="text-white">{activeDoc}</span></>
          : <><span>Searching across</span>&nbsp;<span className="text-white">all documents</span></>
        }
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-8">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center gap-5">
            <div className="text-xs text-muted">Try asking something:</div>
            <div className="flex gap-2 flex-wrap justify-center max-w-[500px]">
              {suggestions.map(s => (
                <button
                  key={s}
                  onClick={() => { setInput(s); setInputError(null); inputRef.current?.focus(); }}
                  className="px-4 py-2 rounded-full border border-bdr-hi bg-transparent text-muted text-xs font-sans cursor-pointer hover:border-accent/50 hover:text-white transition-all"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="max-w-[720px] mx-auto flex flex-col gap-6">
          {messages.map((msg, i) => (
            <div
              key={msg.id}
              className="fade-up flex flex-col"
              style={{
                alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                animationDelay: `${i * 0.05}s`,
              }}
            >
              <div className="text-[10px] tracking-[1.5px] text-muted mb-1.5 font-mono uppercase">
                {msg.role === "user" ? "You" : "DocuMind"}
              </div>

              <div className={`max-w-[85%] px-4 py-3 text-sm leading-relaxed
                ${msg.role === "user"
                  ? "rounded-2xl rounded-br-sm border border-accent/15"
                  : "bg-surface border border-bdr rounded-2xl rounded-bl-sm"
                }
                ${msg.isError ? "text-danger" : "text-white"}`}
                style={msg.role === "user"
                  ? { background: "linear-gradient(135deg, rgba(0,229,160,0.08), rgba(0,179,255,0.08))" }
                  : {}
                }
              >
                {msg.loading ? (
                  <div className="flex gap-1.5 items-center py-1">
                    <span className="dot w-1.5 h-1.5 rounded-full bg-accent inline-block" />
                    <span className="dot w-1.5 h-1.5 rounded-full bg-accent inline-block" />
                    <span className="dot w-1.5 h-1.5 rounded-full bg-accent inline-block" />
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                )}
              </div>

              {msg.role === "assistant" && msg.sources?.length > 0 && !msg.loading && (
                <div className="flex flex-wrap gap-1.5 mt-2 pl-1">
                  {msg.sources.map((src, j) => <CitationBadge key={j} source={src} />)}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="px-8 pb-6 pt-4 border-t border-bdr bg-bg/90 backdrop-blur-md">
        <div className="max-w-[720px] mx-auto">
          <div className={`flex gap-2.5 items-end bg-surface rounded-2xl px-3.5 py-2.5 border transition-colors
            ${inputError ? "border-danger/50" : "border-bdr-hi"}`}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (inputError) setInputError(null);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
              onKeyDown={handleKey}
              placeholder="Ask anything about the document..."
              disabled={isLoading}
              className="flex-1 bg-transparent border-none outline-none text-white text-sm leading-relaxed font-sans resize-none overflow-hidden min-h-6 placeholder:text-muted disabled:opacity-50"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || isLoading}
              className={`w-[34px] h-[34px] rounded-xl shrink-0 border-none flex items-center justify-center transition-all
                ${input.trim() && !isLoading
                  ? "cursor-pointer hover:opacity-90"
                  : "bg-bdr cursor-not-allowed opacity-60"
                }`}
              style={input.trim() && !isLoading
                ? { background: "linear-gradient(135deg, #00e5a0, #00b3ff)" }
                : {}
              }
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                stroke={input.trim() && !isLoading ? "#000" : "#4a5058"}
                strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>

          <div className={`mt-2 text-[11px] font-mono text-center transition-colors
            ${inputError ? "text-danger" : "text-muted"}`}
          >
            {inputError || `Enter to send · Shift+Enter for new line · ${chatModel}`}
          </div>
        </div>
      </div>
    </div>
  );
};