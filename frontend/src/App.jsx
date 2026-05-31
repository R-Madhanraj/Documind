import { useState, useEffect } from "react";
import { PDFUploader } from "./components/PDFUploader";
import { ChatWindow } from "./components/ChatWindow";
import { useChat } from "./hooks/useChat";
import { getDocuments, deleteDocument } from "./services/api";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  const [chatModel, setChatModel] = useState("gemini-2.5-flash");
  const { messages, isLoading, sendMessage, clearMessages } = useChat();

  useEffect(() => { fetchDocuments(); }, []);

  const fetchDocuments = async () => {
    try { setDocuments(await getDocuments()); }
    catch (e) { console.error("Could not fetch documents:", e); }
  };

  const handleUploadSuccess = (result) => {
    fetchDocuments();
    setActiveDoc(result.filename);
    clearMessages();
  };

  const handleDelete = async (filename) => {
    try {
      await deleteDocument(filename);
      setDocuments(prev => prev.filter(d => d.filename !== filename));
      if (activeDoc === filename) { setActiveDoc(null); clearMessages(); }
    } catch (e) { console.error("Delete failed:", e); }
  };

  const handleSelectDoc = (filename) => {
    setActiveDoc(prev => prev === filename ? null : filename);
    clearMessages();
  };

  return (
    <div className="h-screen flex flex-col relative overflow-hidden bg-bg text-white">
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />

      {/* Header */}
      <header className="fade-up flex items-center justify-between px-8 h-[60px] shrink-0 border-b border-bdr bg-bg/80 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-[13px] font-bold text-black"
            style={{ background: "linear-gradient(135deg, #00e5a0, #00b3ff)" }}
          >
            D
          </div>
          <span className="text-[15px] font-bold tracking-tight">DocuMind</span>
          <span className="text-[10px] text-muted font-mono bg-bdr px-1.5 py-0.5 rounded-full border border-bdr-hi">
            BETA
          </span>
        </div>

        <div className="flex items-center gap-2 bg-accent-dim border border-bdr-hi px-3 py-1.5 rounded-full text-xs text-accent font-mono">
          <div className={`w-1.5 h-1.5 rounded-full ${documents.length > 0 ? "bg-accent" : "bg-muted"}`}
            style={{ animation: documents.length > 0 ? "pulse-ring 2s infinite" : "none" }}
          />
          {activeDoc
            ? activeDoc
            : documents.length > 0
              ? `All documents (${documents.length})`
              : "No documents loaded"
          }
        </div>

        <div className="text-xs text-muted font-mono">100% local</div>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden relative z-[1]">

        {/* Sidebar */}
        <aside className="fade-up-1 w-[300px] shrink-0 border-r border-bdr flex flex-col bg-surface">

          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">

            {/* Upload */}
            <div>
              <p className="text-[10px] tracking-[2px] text-muted uppercase font-mono mb-3">Upload</p>
              <PDFUploader onUploadSuccess={handleUploadSuccess} />
            </div>

            {/* Documents */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <p className="text-[10px] tracking-[2px] text-muted uppercase font-mono">
                  Documents ({documents.length})
                </p>
                {documents.length > 1 && (
                  <button
                    onClick={() => { setActiveDoc(null); clearMessages(); }}
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition-all cursor-pointer
                      ${activeDoc === null
                        ? "text-accent bg-accent-dim border-bdr-hi"
                        : "text-muted bg-transparent border-bdr hover:text-white"
                      }`}
                  >
                    all
                  </button>
                )}
              </div>

              {documents.length === 0 ? (
                <div className="text-xs text-muted font-mono text-center py-5 border border-dashed border-bdr-hi rounded-xl">
                  No documents yet
                </div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  {documents.map(doc => (
                    <DocumentRow
                      key={doc.filename}
                      doc={doc}
                      active={activeDoc === doc.filename}
                      onSelect={() => handleSelectDoc(doc.filename)}
                      onDelete={() => handleDelete(doc.filename)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar footer */}
          <div className="p-6 border-t border-bdr flex flex-col gap-2.5">
            <p className="text-[10px] tracking-[2px] text-muted uppercase font-mono">Models</p>

            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted font-mono">Embed</span>
              <span className="text-[11px] text-white font-mono bg-bdr border border-bdr-hi px-2 py-0.5 rounded-md">
                gemini-embedding-001
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted font-mono">Chat</span>
              <ModelSwitcher value={chatModel} onChange={setChatModel} />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[11px] text-muted font-mono">Store</span>
              <span className="text-[11px] text-white font-mono bg-bdr border border-bdr-hi px-2 py-0.5 rounded-md">
                ChromaDB
              </span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="fade-up-2 flex-1 flex flex-col overflow-hidden">
          {documents.length > 0 ? (
            <ChatWindow
              messages={messages}
              isLoading={isLoading}
              onSendMessage={sendMessage}
              chatModel={chatModel}
              activeDoc={activeDoc}
            />
          ) : (
            <EmptyState />
          )}
        </main>
      </div>
    </div>
  );
}

/* ── Document Row ─────────────────────────────────────────────────────────── */
function DocumentRow({ doc, active, onSelect, onDelete }) {
  const [hovered, setHovered]             = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const displayName = doc.filename.length > 28
    ? doc.filename.slice(0, 25) + "..."
    : doc.filename;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setConfirmDelete(false); }}
      className={`flex items-center justify-between px-2.5 py-2 rounded-lg cursor-pointer transition-all gap-2 border
        ${active
          ? "bg-accent-dim border-bdr-hi"
          : hovered
            ? "bg-bdr border-bdr"
            : "border-transparent"
        }`}
    >
      <div onClick={onSelect} className="flex-1 min-w-0">
        <div className={`text-xs font-semibold truncate ${active ? "text-accent" : "text-white"}`}>
          {displayName}
        </div>
        <div className="text-[10px] text-muted font-mono mt-0.5">{doc.chunks} chunks</div>
      </div>

      {hovered && (
        confirmDelete ? (
          <div className="flex gap-1 shrink-0">
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              className="text-[10px] px-1.5 py-0.5 rounded font-mono text-danger bg-danger/10 border border-danger/30 cursor-pointer hover:bg-danger/20 transition-colors"
            >yes</button>
            <button
              onClick={(e) => { e.stopPropagation(); setConfirmDelete(false); }}
              className="text-[10px] px-1.5 py-0.5 rounded font-mono text-muted bg-bdr border border-bdr-hi cursor-pointer hover:text-white transition-colors"
            >no</button>
          </div>
        ) : (
          <button
            onClick={(e) => { e.stopPropagation(); setConfirmDelete(true); }}
            className="w-5 h-5 rounded shrink-0 bg-transparent border-none text-muted cursor-pointer flex items-center justify-center text-xs hover:text-danger transition-colors"
          >✕</button>
        )
      )}
    </div>
  );
}

/* ── Model Switcher ───────────────────────────────────────────────────────── */
const MODELS = [
  { value: "gemini-2.5-flash", label: "2.5 Flash",   provider: "Gemini", available: true  },
  { value: "gemini-2.5-pro",   label: "2.5 Pro",     provider: "Gemini", available: true  },
  { value: "gemini-2.0-flash", label: "2.0 Flash",   provider: "Gemini", available: true  },
  { value: "llama3",           label: "Llama 3",     provider: "Ollama", available: false },
  { value: "mistral",          label: "Mistral",     provider: "Ollama", available: false },
  { value: "deepseek-r1",      label: "DeepSeek R1", provider: "Ollama", available: false },
];

function ModelSwitcher({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const current = MODELS.find(m => m.value === value) || MODELS[0];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 bg-bdr border border-bdr-hi rounded-md px-2 py-0.5 text-accent text-[11px] font-mono cursor-pointer outline-none hover:border-accent/30 transition-colors"
      >
        {current.label}
        <svg width="8" height="8" viewBox="0 0 10 6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
          <path d={open ? "M1 5L5 1L9 5" : "M1 1L5 5L9 1"} />
        </svg>
      </button>

      {open && (
        <div
          className="absolute bottom-[calc(100%+8px)] right-0 w-[200px] bg-surface border border-bdr-hi rounded-xl overflow-hidden z-50"
          style={{ boxShadow: "0 16px 40px rgba(0,0,0,0.6)", animation: "fadeUp 0.15s ease" }}
        >
          <div className="fixed inset-0 -z-10" onClick={() => setOpen(false)} />

          <div className="px-2.5 pt-2 pb-1 text-[9px] tracking-[1.5px] text-muted font-mono uppercase">
            ☁ Gemini (Cloud)
          </div>
          {MODELS.filter(m => m.provider === "Gemini").map(m => (
            <ModelOption key={m.value} model={m} selected={value === m.value}
              onSelect={(v) => { onChange(v); setOpen(false); }} />
          ))}

          <div className="h-px bg-bdr my-1.5" />

          <div className="px-2.5 py-1 text-[9px] tracking-[1.5px] text-muted font-mono uppercase">
            ⬡ Ollama (Local · coming soon)
          </div>
          {MODELS.filter(m => m.provider === "Ollama").map(m => (
            <ModelOption key={m.value} model={m} selected={value === m.value}
              onSelect={(v) => { onChange(v); setOpen(false); }} />
          ))}
          <div className="h-2" />
        </div>
      )}
    </div>
  );
}

function ModelOption({ model, selected, onSelect }) {
  return (
    <button
      disabled={!model.available}
      onClick={() => model.available && onSelect(model.value)}
      className={`w-full flex items-center justify-between px-2.5 py-1.5 text-left text-xs font-mono transition-colors border-none outline-none
        ${!model.available
          ? "opacity-40 cursor-not-allowed text-muted bg-transparent"
          : selected
            ? "text-accent cursor-pointer hover:bg-accent-dim bg-transparent"
            : "text-white cursor-pointer hover:bg-accent-dim bg-transparent"
        }`}
    >
      <span>{model.label}</span>
      <span className="flex items-center gap-1.5">
        {!model.available && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-bdr text-muted border border-bdr-hi">SOON</span>
        )}
        {selected && model.available && (
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
            <path d="M1 6l4 4 6-8" stroke="#00e5a0" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </span>
    </button>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-10">
      <div
        className="w-16 h-16 rounded-2xl border border-bdr-hi flex items-center justify-center text-3xl"
        style={{ background: "linear-gradient(135deg, rgba(0,229,160,0.1), #1e3a2f)" }}
      >📄</div>
      <div className="text-center">
        <div className="text-lg font-bold mb-1.5">No documents loaded</div>
        <div className="text-sm text-muted max-w-[280px] leading-relaxed">
          Upload a PDF on the left to start asking questions about its content.
        </div>
      </div>
    </div>
  );
}