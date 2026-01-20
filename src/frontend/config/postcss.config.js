module.exports = {
  plugins: {
    "@tailwindcss/postcss": {
      base: "..",
      content: [
        "../index.html",
        "../*.{js,ts,jsx,tsx}",
        "../src/**/*.{js,ts,jsx,tsx}",
      ],
    },
    autoprefixer: {},
  },
};
