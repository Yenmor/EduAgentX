import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        circuit: {
          bg: "#0F111A",
          panel: "#151925",
          line: "#2A3142",
          text: "#E5E7EB",
          muted: "#9CA3AF",
          blue: "#3B82F6",
          green: "#22C55E",
          violet: "#A78BFA"
        }
      },
      boxShadow: {
        glow: "0 0 28px rgba(59, 130, 246, 0.16)"
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;
