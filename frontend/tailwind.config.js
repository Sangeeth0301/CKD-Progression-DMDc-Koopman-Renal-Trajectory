/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#070A11",
        surface: "#0D131F",
        surfaceHover: "#111A2B",
        cardBorder: "rgba(255, 255, 255, 0.08)",
        primaryTeal: "#06B6D4",
        primaryTealHover: "#22D3EE",
        emeraldPreserve: "#10B981",
        alertRed: "#EF4444",
        alertAmber: "#F59E0B",
        textPrimary: "#F8FAFC",
        textMuted: "#94A3B8",
        textDim: "#64748B",
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
