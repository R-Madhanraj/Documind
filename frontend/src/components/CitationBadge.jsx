import { useState } from "react";

export const CitationBadge = ({ source }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-mono border cursor-pointer outline-none transition-all
          ${hovered
            ? "bg-accent-dim border-accent/50 text-accent"
            : "bg-bdr border-bdr-hi text-muted"
          }`}
      >
        <span className="opacity-60">p.</span>
        {source.page_number}
        <span className="opacity-60 border-l border-current pl-1.5">
          {Math.round(source.score * 100)}%
        </span>
      </button>

      {hovered && (
        <div
          className="absolute bottom-[calc(100%+8px)] left-0 w-[280px] p-3 bg-surface border border-bdr-hi rounded-xl z-[100]"
          style={{ boxShadow: "0 16px 40px rgba(0,0,0,0.6)", animation: "fadeUp 0.15s ease" }}
        >
          <div className="text-[10px] text-accent font-mono mb-1.5 tracking-wider">
            PAGE {source.page_number} · {Math.round(source.score * 100)}% MATCH
          </div>
          <div className="text-xs text-muted font-mono leading-relaxed line-clamp-4">
            {source.text}
          </div>
        </div>
      )}
    </div>
  );
};