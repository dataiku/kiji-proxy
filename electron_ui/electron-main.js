const { app, BrowserWindow, Menu, ipcMain, safeStorage } = require('electron');
const path = require('path');
const fs = require('fs');
const isDev = process.env.NODE_ENV === 'development';

let mainWindow;

// Storage for API key (using safeStorage when available, fallback to encrypted file)
const getStoragePath = () => {
  return path.join(app.getPath('userData'), 'config.json');
};

// Check if safeStorage is available
const isEncryptionAvailable = () => {
  return safeStorage.isEncryptionAvailable();
};

function createWindow() {
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
    icon: path.join(__dirname, 'assets', 'icon.png'), // Optional: add an icon
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
  createWindow();
  createMenu();

  app.on('activate', () => {
    // On macOS, re-create a window when the dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
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

