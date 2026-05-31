export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg:           "#080909",
        surface:      "#0e1011",
        bdr:          "#1e3a2f",
        "bdr-hi":     "#00e5a035",
        accent:       "#00e5a0",
        "accent-dim": "#00e5a020",
        muted:        "#4a5058",
        danger:       "#ff4d6d",
      },
      fontFamily: {
        sans: ["Syne", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};