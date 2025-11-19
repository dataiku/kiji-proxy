const { app, BrowserWindow, Menu, ipcMain, safeStorage } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const isDev = process.env.NODE_ENV === 'development';

let mainWindow;
let goProcess = null;

// Storage for API key (using safeStorage when available, fallback to encrypted file)
const getStoragePath = () => {
  return path.join(app.getPath('userData'), 'config.json');
};

// Check if safeStorage is available
const isEncryptionAvailable = () => {
  return safeStorage.isEncryptionAvailable();
};

// Get the path to the Go binary in the app bundle
const getGoBinaryPath = () => {
  if (isDev) {
    // In development, look for the binary in the project root
    const devPath = path.join(__dirname, '..', 'build', 'yaak-proxy');
    if (fs.existsSync(devPath)) {
      return devPath;
    }
    // Fallback: assume it's running separately
    return null;
  }
  
  // In production, the binary is in the app's resources directory
  // For macOS app bundles: Contents/Resources/
  if (process.platform === 'darwin') {
    // app.getAppPath() returns the path to the app bundle's Contents/Resources/app.asar or Contents/Resources/app
    const resourcesPath = process.resourcesPath || app.getAppPath();
    const binaryPath = path.join(resourcesPath, 'resources', 'yaak-proxy');
    
    // If not found, try alternative paths
    if (fs.existsSync(binaryPath)) {
      return binaryPath;
    }
    
    // Try without 'resources' subdirectory (if resources are at root)
    const altPath = path.join(resourcesPath, 'yaak-proxy');
    if (fs.existsSync(altPath)) {
      return altPath;
    }
  }
  
  // For other platforms or if not found
  const resourcesPath = process.resourcesPath || app.getAppPath();
  return path.join(resourcesPath, 'resources', 'yaak-proxy');
};

// Get the path to resources directory
const getResourcesPath = () => {
  if (isDev) {
    return path.join(__dirname, '..');
  }
  
  if (process.platform === 'darwin') {
    return process.resourcesPath || app.getAppPath();
  }
  
  return process.resourcesPath || app.getAppPath();
};

// Launch the Go binary backend
const launchGoBinary = () => {
  const binaryPath = getGoBinaryPath();
  
  if (!binaryPath || !fs.existsSync(binaryPath)) {
    console.warn('Go binary not found at:', binaryPath);
    console.warn('The app will try to connect to an existing backend server.');
    return;
  }
  
  // Get resources path for ONNX library and model files
  const resourcesPath = getResourcesPath();
  const onnxLibPath = path.join(resourcesPath, 'resources', 'libonnxruntime.1.23.1.dylib');
  const modelPath = path.join(resourcesPath, 'resources', 'pii_onnx_model');
  
  // Set up environment variables
  const env = { ...process.env };
  
  // Set ONNX Runtime library path if it exists
  if (fs.existsSync(onnxLibPath)) {
    env.ONNXRUNTIME_SHARED_LIBRARY_PATH = onnxLibPath;
  } else {
    // Try alternative location
    const altOnnxPath = path.join(resourcesPath, 'libonnxruntime.1.23.1.dylib');
    if (fs.existsSync(altOnnxPath)) {
      env.ONNXRUNTIME_SHARED_LIBRARY_PATH = altOnnxPath;
    }
  }
  
  // Set working directory to resources so model files can be found
  const workingDir = fs.existsSync(modelPath) ? path.join(resourcesPath, 'resources') : resourcesPath;
  
  console.log('Launching Go binary:', binaryPath);
  console.log('Working directory:', workingDir);
  console.log('ONNX library path:', env.ONNXRUNTIME_SHARED_LIBRARY_PATH || 'not set');
  
  // Spawn the Go process
  goProcess = spawn(binaryPath, [], {
    cwd: workingDir,
    env: env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  
  // Handle stdout
  goProcess.stdout.on('data', (data) => {
    console.log(`[Go Backend] ${data.toString().trim()}`);
  });
  
  // Handle stderr
  goProcess.stderr.on('data', (data) => {
    console.error(`[Go Backend Error] ${data.toString().trim()}`);
  });
  
  // Handle process exit
  goProcess.on('exit', (code, signal) => {
    console.log(`Go binary exited with code ${code} and signal ${signal}`);
    goProcess = null;
    
    // If the process exited unexpectedly and we're not shutting down, show an error
    if (code !== 0 && code !== null && !app.isQuitting) {
      if (mainWindow) {
        mainWindow.webContents.send('backend-error', {
          message: 'Backend server exited unexpectedly',
          code: code
        });
      }
    }
  });
  
  // Handle process errors
  goProcess.on('error', (error) => {
    console.error('Failed to start Go binary:', error);
    goProcess = null;
    
    if (mainWindow) {
      mainWindow.webContents.send('backend-error', {
        message: 'Failed to start backend server',
        error: error.message
      });
    }
  });
};

// Stop the Go binary
const stopGoBinary = () => {
  if (goProcess) {
    console.log('Stopping Go binary...');
    goProcess.kill('SIGTERM');
    
    // Force kill after 3 seconds if still running
    setTimeout(() => {
      if (goProcess && !goProcess.killed) {
        console.log('Force killing Go binary...');
        goProcess.kill('SIGKILL');
      }
      goProcess = null;
    }, 3000);
  }
};

function createWindow() {
  // Get icon path (works in both dev and production)
  const iconPath = path.join(__dirname, 'assets', 'icon.png');
  const iconExists = fs.existsSync(iconPath);
  
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      webSecurity: true,
      preload: path.join(__dirname, 'electron-preload.js')
    },
    ...(iconExists && { icon: iconPath }), // App icon (only set if file exists)
    show: false // Don't show until ready
  });

  // Load the app
  const startUrl = isDev
    ? 'http://localhost:3000' // Use webpack dev server in dev mode (run 'npm run dev' first)
    : `file://${path.join(__dirname, 'dist', 'index.html')}`; // Use built files in production

  mainWindow.loadURL(startUrl).catch((err) => {
    console.error('Failed to load URL:', err);
    // If dev server is not running, fall back to built files
    if (isDev) {
      console.log('Dev server not available, loading from dist...');
      mainWindow.loadURL(`file://${path.join(__dirname, 'dist', 'index.html')}`);
    }
  });

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Open DevTools in development mode
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    require('electron').shell.openExternal(url);
    return { action: 'deny' };
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo', label: 'Undo' },
        { role: 'redo', label: 'Redo' },
        { type: 'separator' },
        { role: 'cut', label: 'Cut' },
        { role: 'copy', label: 'Copy' },
        { role: 'paste', label: 'Paste' },
        { role: 'selectAll', label: 'Select All' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload', label: 'Reload' },
        { role: 'forceReload', label: 'Force Reload' },
        { role: 'toggleDevTools', label: 'Toggle Developer Tools' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Actual Size' },
        { role: 'zoomIn', label: 'Zoom In' },
        { role: 'zoomOut', label: 'Zoom Out' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Toggle Fullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize', label: 'Minimize' },
        { role: 'close', label: 'Close' }
      ]
    },
    {
      label: 'Settings',
      submenu: [
        {
          label: 'Preferences...',
          accelerator: process.platform === 'darwin' ? 'Cmd+,' : 'Ctrl+,',
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send('open-settings');
            }
          }
        }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About',
          click: () => {
            // You can add an about dialog here
          }
        }
      ]
    }
  ];

  // macOS specific menu adjustments
  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about', label: 'About ' + app.getName() },
        { type: 'separator' },
        { role: 'services', label: 'Services' },
        { type: 'separator' },
        { role: 'hide', label: 'Hide ' + app.getName() },
        { role: 'hideOthers', label: 'Hide Others' },
        { role: 'unhide', label: 'Show All' },
        { type: 'separator' },
        { role: 'quit', label: 'Quit ' + app.getName() }
      ]
    });

    // Window menu
    template[4].submenu = [
      { role: 'close', label: 'Close' },
      { role: 'minimize', label: 'Minimize' },
      { role: 'zoom', label: 'Zoom' },
      { type: 'separator' },
      { role: 'front', label: 'Bring All to Front' }
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Launch the Go binary backend first
  launchGoBinary();
  
  // Wait a moment for the backend to start, then create the window
  setTimeout(() => {
    createWindow();
    createMenu();
  }, 1000);

  app.on('activate', () => {
    // On macOS, re-create a window when the dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      // Ensure backend is running
      if (!goProcess) {
        launchGoBinary();
        setTimeout(() => {
          createWindow();
        }, 1000);
      } else {
        createWindow();
      }
    }
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    stopGoBinary();
    app.quit();
  }
});

// Handle app quitting
app.on('before-quit', () => {
  app.isQuitting = true;
  stopGoBinary();
});

// Handle app will quit (macOS)
app.on('will-quit', () => {
  stopGoBinary();
});

// IPC handlers for secure storage
ipcMain.handle('get-api-key', async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return null;
    }

    const data = fs.readFileSync(storagePath, 'utf8');
    const config = JSON.parse(data);

    if (config.apiKey && config.encrypted && isEncryptionAvailable()) {
      // Decrypt the API key
      const buffer = Buffer.from(config.apiKey, 'base64');
      const decrypted = safeStorage.decryptString(buffer);
      return decrypted;
    } else if (config.apiKey && !config.encrypted) {
      // Legacy unencrypted storage (migrate on next save)
      return config.apiKey;
    }

    return null;
  } catch (error) {
    console.error('Error reading API key:', error);
    return null;
  }
});

ipcMain.handle('set-api-key', async (event, apiKey) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, 'utf8');
      config = JSON.parse(data);
    }

    if (apiKey && apiKey.trim()) {
      if (isEncryptionAvailable()) {
        // Encrypt the API key
        const encrypted = safeStorage.encryptString(apiKey);
        config.apiKey = encrypted.toString('base64');
        config.encrypted = true;
      } else {
        // Fallback to unencrypted storage (not ideal, but works)
        console.warn('Encryption not available, storing API key unencrypted');
        config.apiKey = apiKey;
        config.encrypted = false;
      }
    } else {
      // Remove API key
      delete config.apiKey;
      delete config.encrypted;
    }

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), 'utf8');
    return { success: true };
  } catch (error) {
    console.error('Error saving API key:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-forward-endpoint', async () => {
  try {
    const storagePath = getStoragePath();
    const defaultEndpoint = 'https://api.openai.com/v1';
    
    if (!fs.existsSync(storagePath)) {
      return defaultEndpoint;
    }

    const data = fs.readFileSync(storagePath, 'utf8');
    const config = JSON.parse(data);
    
    // Migrate old default value to new default
    if (config.forwardEndpoint === 'http://localhost:8080') {
      config.forwardEndpoint = defaultEndpoint;
      // Save the updated config
      fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), 'utf8');
    }
    
    return config.forwardEndpoint || defaultEndpoint;
  } catch (error) {
    console.error('Error reading forward endpoint:', error);
    return 'https://api.openai.com/v1';
  }
});

ipcMain.handle('set-forward-endpoint', async (event, url) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, 'utf8');
      config = JSON.parse(data);
    }

    if (url && url.trim()) {
      config.forwardEndpoint = url.trim();
    } else {
      delete config.forwardEndpoint;
    }

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), 'utf8');
    return { success: true };
  } catch (error) {
    console.error('Error saving forward endpoint:', error);
    return { success: false, error: error.message };
  }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    require('electron').shell.openExternal(navigationUrl);
  });
});

