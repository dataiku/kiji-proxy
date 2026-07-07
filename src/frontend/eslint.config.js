const tsPlugin = require("@typescript-eslint/eslint-plugin");
const tsParser = require("@typescript-eslint/parser");
const reactPlugin = require("eslint-plugin-react");
const reactHooksPlugin = require("eslint-plugin-react-hooks");
const i18nextPlugin = require("eslint-plugin-i18next");

module.exports = [
  // Global ignores (replaces .eslintignore)
  {
    ignores: [
      "node_modules/",
      "dist/",
      "release/",
      "config/webpack.config.js",
      "config/postcss.config.js",
      "config/tailwind.config.js",
      "server.js",
      "**/*.d.ts",
    ],
  },

  // Base config for all JS/TS files
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
      parser: tsParser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        // browser
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        fetch: "readonly",
        console: "readonly",
        setTimeout: "readonly",
        clearTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        URL: "readonly",
        HTMLElement: "readonly",
        HTMLInputElement: "readonly",
        Event: "readonly",
        MouseEvent: "readonly",
        KeyboardEvent: "readonly",
        RequestInit: "readonly",
        Response: "readonly",
        AbortController: "readonly",
        // node
        require: "readonly",
        module: "readonly",
        process: "readonly",
        __dirname: "readonly",
        __filename: "readonly",
        exports: "readonly",
        Buffer: "readonly",
        global: "readonly",
      },
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
    },
    settings: {
      react: { version: "detect" },
    },
    rules: {
      // eslint:recommended (subset relevant for TS projects)
      "no-const-assign": "error",
      "no-dupe-args": "error",
      "no-dupe-keys": "error",
      "no-duplicate-case": "error",
      "no-extra-boolean-cast": "error",
      "no-func-assign": "error",
      "no-irregular-whitespace": "error",
      "no-unreachable": "error",
      "no-unsafe-finally": "error",
      "no-unused-labels": "error",
      "no-useless-catch": "error",
      "no-useless-escape": "error",
      "no-var": "error",
      "prefer-const": "error",
      "use-isnan": "error",
      "valid-typeof": "error",

      // @typescript-eslint/recommended
      ...tsPlugin.configs.recommended.rules,

      // react/recommended
      ...reactPlugin.configs.recommended.rules,

      // react-hooks/recommended
      ...reactHooksPlugin.configs.recommended.rules,

      // Project overrides
      "react/react-in-jsx-scope": "off",
      "react/no-unescaped-entities": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/no-explicit-any": "warn",
    },
  },

  // JS file overrides
  {
    files: ["**/*.js"],
    rules: {
      "@typescript-eslint/no-var-requires": "off",
      "@typescript-eslint/no-require-imports": "off",
    },
  },

  // i18n: flag un-extracted UI copy in React components. Scoped to the
  // renderer's JSX/TSX and limited to `jsx-text-only` (visible text between
  // tags) so it does not flag brand names, class names, or technical literals
  // in attributes/expressions. Advisory (warn) so it guides incremental
  // extraction without gating CI — the hard i18n gate is the plural-aware
  // parity check (scripts/check-i18n-parity.js, run as `npm run i18n:check`).
  {
    files: ["src/**/*.{jsx,tsx}"],
    plugins: { i18next: i18nextPlugin },
    rules: {
      "i18next/no-literal-string": ["warn", { mode: "jsx-text-only" }],
    },
  },
];
