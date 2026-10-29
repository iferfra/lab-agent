import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        brand: {
          50: "#eef4ff",
          100: "#dae6ff",
          200: "#bdd3ff",
          300: "#90b6ff",
          400: "#5c8dfd",
          500: "#3766f8",
          600: "#2147ed",
          700: "#1a35d9",
          800: "#1c2db0",
          900: "#1d2c8b",
        },
      },
      keyframes: {
        blink: { "0%, 80%, 100%": { opacity: "0.2" }, "40%": { opacity: "1" } },
        fadeIn: { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
      },
      animation: {
        blink: "blink 1.4s infinite both",
        fadeIn: "fadeIn 0.25s ease-out",
      },
    },
  },
  plugins: [typography],
};
