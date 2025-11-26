# Electron App Setup

This directory contains the Electron version of the Privacy Proxy UI.

## Prerequisites

1. Install dependencies:
```bash
npm install
```

2. Make sure the Go backend server is running on `http://localhost:8080` (or configure a different URL).

## Development

### Option 1: Development with Webpack Dev Server
1. Start the webpack dev server in one terminal:
```bash
npm run dev
```

2. In another terminal, start Electron in dev mode:
```bash
npm run electron:dev
```

This will use the webpack dev server at `http://localhost:3000` with hot reloading.

### Option 2: Development with Built Files
1. Build the app for Electron:
```bash
npm run build:electron
```

2. Start Electron:
```bash
npm run electron
```

## Production Build

To build and package the Electron app:

```bash
npm run electron:pack
```

This will:
1. Build the React app for Electron
2. Package it into a distributable format using electron-builder

## Configuration

### Forward Endpoint

The Electron app connects to `http://localhost:8080` by default. You can change this via the Settings menu (Preferences... or Cmd+, / Ctrl+,).

## Project Structure

- `electron-main.js` - Main Electron process (window management, app lifecycle)
- `electron-preload.js` - Preload script (bridge between main and renderer processes)
- `electron.d.ts` - TypeScript declarations for Electron API
- `privacy-proxy-ui.tsx` - React component (automatically detects Electron environment)
- `webpack.config.js` - Webpack configuration (supports both web and Electron builds)

## Features

- ✅ Automatic detection of Electron vs web environment
- ✅ Direct API calls to backend (no proxy needed in Electron)
- ✅ Native window controls and menu
- ✅ DevTools support in development mode
- ✅ Cross-platform support (Windows, macOS, Linux)

## Troubleshooting

### App won't start
- Make sure you've run `npm install` to install all dependencies including Electron
- Check that the backend server is running on the configured port

### API calls failing
- Verify the forward endpoint is correct (default: `http://localhost:8080`)
- Check that CORS is properly configured on the backend if needed
- In Electron, the app makes direct HTTP requests (no proxy), so ensure the forward endpoint is accessible

### Build errors
- Make sure all dependencies are installed: `npm install`
- Try deleting `node_modules` and `dist` folders, then reinstall: `rm -rf node_modules dist && npm install`

