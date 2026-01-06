module.exports = {
  plugins: {
    "@tailwindcss/postcss": {
      base: ".",
      content: [
        "./index.html",
        "./*.{js,ts,jsx,tsx}",
        "./**/*.{js,ts,jsx,tsx}",
        "./utils/**/*.{js,ts,jsx,tsx}",
      ],
    },
    autoprefixer: {},
  },
};
