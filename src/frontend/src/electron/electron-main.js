const {
  app,
  BrowserWindow,
  Menu,
  Tray,
  nativeImage,
  ipcMain,
  safeStorage,
} = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const isDev = process.env.NODE_ENV === "development";

// Initialize Sentry for error tracking
const Sentry = require("@sentry/electron/main");
Sentry.init({
  dsn: "https://d7ad4213601549253c0d313b271f83cf@o4510660510679040.ingest.de.sentry.io/4510660556095568",
  environment: isDev ? "development" : "production",
  tracesSampleRate: 1.0,
});

let mainWindow;
let goProcess = null;
let tray = null;

// Storage for API key (using safeStorage when available, fallback to encrypted file)
const getStoragePath = () => {
  return path.join(app.getPath("userData"), "config.json");
};

// Check if safeStorage is available
const isEncryptionAvailable = () => {
  return safeStorage.isEncryptionAvailable();
};

// Get the path to the Go binary in the app bundle
const getGoBinaryPath = () => {
  if (isDev) {
    // In development, look for the binary in the project root
    // __dirname is src/frontend/src/electron, so we need to go up three levels to reach project root
    const devPath = path.join(
      __dirname,
      "..",
      "..",
      "..",
      "..",
      "build",
      "yaak-proxy"
    );
    console.log("[DEBUG] Development mode - checking for binary at:", devPath);
    if (fs.existsSync(devPath)) {
      console.log("[DEBUG] ✅ Binary found at:", devPath);
      return devPath;
    }
    console.log("[DEBUG] ⚠️ Binary not found in development mode");
    // Fallback: assume it's running separately
    return null;
  }

  // In production, the binary is in the app's resources directory
  // For macOS app bundles: Contents/Resources/
  console.log("[DEBUG] Production mode - looking for binary");
  console.log("[DEBUG] process.resourcesPath:", process.resourcesPath);
  console.log("[DEBUG] app.getAppPath():", app.getAppPath());

  if (process.platform === "darwin") {
    // app.getAppPath() returns the path to the app bundle's Contents/Resources/app.asar or Contents/Resources/app
    const resourcesPath = process.resourcesPath || app.getAppPath();
    const binaryPath = path.join(resourcesPath, "resources", "yaak-proxy");

    console.log("[DEBUG] Checking primary path:", binaryPath);
    // If not found, try alternative paths
    if (fs.existsSync(binaryPath)) {
      console.log("[DEBUG] ✅ Binary found at:", binaryPath);
      return binaryPath;
    }

    // Try without 'resources' subdirectory (if resources are at root)
    const altPath = path.join(resourcesPath, "yaak-proxy");
    console.log("[DEBUG] Checking alternative path:", altPath);
    if (fs.existsSync(altPath)) {
      console.log("[DEBUG] ✅ Binary found at:", altPath);
      return altPath;
    }

    // List what's actually in the resources directory
    try {
      const resDir = path.join(resourcesPath, "resources");
      console.log("[DEBUG] Contents of resources directory:", resDir);
      if (fs.existsSync(resDir)) {
        const files = fs.readdirSync(resDir);
        console.log("[DEBUG] Files:", files.slice(0, 20)); // First 20 files
      } else {
        console.log("[DEBUG] ⚠️ Resources directory does not exist");
      }
    } catch (err) {
      console.error("[DEBUG] Error listing resources:", err);
    }
  }

  // For other platforms or if not found
  const resourcesPath = process.resourcesPath || app.getAppPath();
  const finalPath = path.join(resourcesPath, "resources", "yaak-proxy");
  console.log(
    "[DEBUG] ⚠️ Binary not found, returning default path:",
    finalPath
  );
  return finalPath;
};

// Get the path to resources directory
const getResourcesPath = () => {
  if (isDev) {
    // In development, __dirname is src/frontend/src/electron, so go up three levels to project root
    return path.join(__dirname, "..", "..", "..", "..");
  }

  if (process.platform === "darwin") {
    return process.resourcesPath || app.getAppPath();
  }

  return process.resourcesPath || app.getAppPath();
};

// Launch the Go binary backend
const launchGoBinary = () => {
  // Skip launching backend if EXTERNAL_BACKEND is set (e.g., running in debugger)
  if (
    process.env.EXTERNAL_BACKEND === "true" ||
    process.env.SKIP_BACKEND_LAUNCH === "true"
  ) {
    console.log(
      "Skipping backend launch (EXTERNAL_BACKEND=true). Connecting to existing backend server."
    );
    return;
  }

  const binaryPath = getGoBinaryPath();

  console.log("[DEBUG] launchGoBinary - binary path:", binaryPath);
  if (!binaryPath || !fs.existsSync(binaryPath)) {
    console.error("[DEBUG] ❌ Go binary not found at:", binaryPath);
    console.warn("Go binary not found at:", binaryPath);
    console.warn("The app will try to connect to an existing backend server.");
    return;
  }
  console.log("[DEBUG] ✅ Go binary exists, proceeding to launch");

  // Get project root path (resources path in dev mode)
  const projectRoot = getResourcesPath();
  console.log("[DEBUG] Project root / resources path:", projectRoot);

  // Set up environment variables
  const env = { ...process.env };

  // In development mode, set ONNX Runtime library path
  // Try multiple locations relative to project root
  const onnxPaths = [
    path.join(projectRoot, "build", "libonnxruntime.1.23.1.dylib"), // build/libonnxruntime.1.23.1.dylib
    path.join(
      projectRoot,
      "src",
      "frontend",
      "resources",
      "libonnxruntime.1.23.1.dylib"
    ), // src/frontend/resources/libonnxruntime.1.23.1.dylib
    path.join(projectRoot, "libonnxruntime.1.23.1.dylib"), // root/libonnxruntime.1.23.1.dylib
  ];

  // Also try to find in Python venv
  if (fs.existsSync(path.join(projectRoot, ".venv"))) {
    const venvLib = path.join(
      projectRoot,
      ".venv",
      "lib",
      "python3.13",
      "site-packages",
      "onnxruntime",
      "capi",
      "libonnxruntime.1.23.2.dylib"
    );
    if (fs.existsSync(venvLib)) {
      onnxPaths.unshift(venvLib); // Check venv first
    }
  }

  let foundOnnxLib = null;
  for (const libPath of onnxPaths) {
    if (fs.existsSync(libPath)) {
      foundOnnxLib = libPath;
      env.ONNXRUNTIME_SHARED_LIBRARY_PATH = libPath;
      break;
    }
  }

  if (!foundOnnxLib) {
    console.warn(
      "ONNX Runtime library not found in any of these locations:",
      onnxPaths
    );
  }

  // Set working directory to project root so model files can be found
  const workingDir = projectRoot;

  // Prepare command line arguments
  const args = [];
  if (isDev) {
    // In development mode, use config file for file system access
    const configPath = path.join(
      projectRoot,
      "src",
      "backend",
      "config",
      "config.development.json"
    );
    if (fs.existsSync(configPath)) {
      args.push("--config", configPath);
    }
  }

  console.log("[DEBUG] Spawning Go process:");
  console.log("[DEBUG]   - Binary:", binaryPath);
  console.log("[DEBUG]   - Args:", args);
  console.log("[DEBUG]   - CWD:", workingDir);
  console.log(
    "[DEBUG]   - ONNXRUNTIME_SHARED_LIBRARY_PATH:",
    env.ONNXRUNTIME_SHARED_LIBRARY_PATH
  );

  // Spawn the Go process
  goProcess = spawn(binaryPath, args, {
    cwd: workingDir,
    env: env,
    stdio: ["ignore", "pipe", "pipe"],
  });

  console.log("[DEBUG] Go process spawned with PID:", goProcess.pid);

  // Handle stdout
  goProcess.stdout.on("data", (data) => {
    console.log(`[Go Backend] ${data.toString().trim()}`);
  });

  // Handle stderr
  // Note: Go's log package writes to stderr by default, so not all stderr is errors
  goProcess.stderr.on("data", (data) => {
    const output = data.toString().trim();
    // Only mark as error if it contains error keywords
    if (
      output.toLowerCase().includes("error") ||
      output.toLowerCase().includes("fatal") ||
      output.toLowerCase().includes("panic") ||
      output.toLowerCase().includes("failed")
    ) {
      console.error(`[Go Backend Error] ${output}`);
    } else {
      // Regular log output (Go's log.Printf writes to stderr)
      console.log(`[Go Backend] ${output}`);
    }
  });

  // Handle process exit
  goProcess.on("exit", (code, signal) => {
    console.log(`Go binary exited with code ${code} and signal ${signal}`);
    goProcess = null;

    // If the process exited unexpectedly and we're not shutting down, show an error
    if (code !== 0 && code !== null && !app.isQuitting) {
      if (mainWindow) {
        mainWindow.webContents.send("backend-error", {
          message: "Backend server exited unexpectedly",
          code: code,
        });
      }
    }
  });

  // Handle process errors
  goProcess.on("error", (error) => {
    console.error("Failed to start Go binary:", error);
    goProcess = null;

    if (mainWindow) {
      mainWindow.webContents.send("backend-error", {
        message: "Failed to start backend server",
        error: error.message,
      });
    }
  });
};

// Stop the Go binary
const stopGoBinary = () => {
  if (goProcess) {
    console.log("Stopping Go binary...");
    goProcess.kill("SIGTERM");

    // Force kill after 3 seconds if still running
    setTimeout(() => {
      if (goProcess && !goProcess.killed) {
        console.log("Force killing Go binary...");
        goProcess.kill("SIGKILL");
      }
      goProcess = null;
    }, 3000);
  }
};

// Show or create main window
function showMainWindow() {
  if (mainWindow) {
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.show();
    mainWindow.focus();
  } else {
    createWindow();
  }
}

// Create system tray icon
function createTray() {
  // Use icon-16.png for the menu bar
  const iconPath = path.join(__dirname, "..", "..", "assets", "icon-16.png");

  if (!fs.existsSync(iconPath)) {
    console.warn("Tray icon not found at:", iconPath);
    return;
  }

  const icon = nativeImage.createFromPath(iconPath);

  // For macOS, resize to 16x16 and mark as template image for dark mode support
  if (process.platform === "darwin") {
    tray = new Tray(icon.resize({ width: 16, height: 16 }));
    tray.setToolTip("Yaak Privacy Proxy");
  } else {
    tray = new Tray(icon);
    tray.setToolTip("Yaak Privacy Proxy");
  }

  // Build context menu
  const contextMenu = Menu.buildFromTemplate([
    {
      label: "Open Yaak Proxy",
      click: () => {
        showMainWindow();
      },
    },
    {
      label: "About Yaak Proxy",
      click: () => {
        showMainWindow();
        // Send IPC to open about dialog after a short delay to ensure window is ready
        setTimeout(() => {
          if (mainWindow) {
            mainWindow.webContents.send("open-about");
          }
        }, 100);
      },
    },
    {
      label: "Settings",
      click: () => {
        showMainWindow();
        // Send IPC to open settings after a short delay to ensure window is ready
        setTimeout(() => {
          if (mainWindow) {
            mainWindow.webContents.send("open-settings");
          }
        }, 100);
      },
    },
    { type: "separator" },
    {
      label: "Terms & Conditions",
      click: () => {
        showMainWindow();
        // Send IPC to open terms after a short delay to ensure window is ready
        setTimeout(() => {
          if (mainWindow) {
            mainWindow.webContents.send("open-terms");
          }
        }, 100);
      },
    },
    {
      label: "Documentation",
      click: () => {
        require("electron").shell.openExternal(
          "https://github.com/hanneshapke/yaak-proxy/blob/main/docs/README.md"
        );
      },
    },
    {
      label: "File a Bug Report",
      click: () => {
        require("electron").shell.openExternal(
          "https://github.com/hanneshapke/yaak-proxy/issues/new?template=10_bug_report.yml"
        );
      },
    },
    {
      label: "Request a Feature",
      click: () => {
        require("electron").shell.openExternal(
          "https://github.com/hanneshapke/yaak-proxy/discussions/new/choose"
        );
      },
    },
    {
      label: "Email us",
      click: () => {
        require("electron").shell.openExternal(
          "mailto:opensource@dataiku.com?subject=[Yaak Proxy User]"
        );
      },
    },
    { type: "separator" },
    {
      label: "Quit Yaak Proxy",
      click: () => {
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);

  // Optional: Click tray icon to show window (single click on macOS)
  tray.on("click", () => {
    showMainWindow();
  });
}

function createWindow() {
  // Get icon path (works in both dev and production)
  const iconPath = path.join(__dirname, "..", "..", "assets", "icon.png");
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
      webSecurity: false, // Disable webSecurity for local development
      allowRunningInsecureContent: true,
      preload: path.join(__dirname, "electron-preload.js"),
    },
    ...(iconExists && { icon: iconPath }), // App icon (only set if file exists)
    show: false, // Don't show until ready
  });

  // Load the app
  // In both dev and production, the Go backend serves the UI on port 8080
  // The backend embeds the UI files when built with the 'embed' tag
  let startUrl = "http://localhost:8080";

  console.log("[DEBUG] Mode:", isDev ? "development" : "production");
  console.log("[DEBUG] Loading UI from Go backend at:", startUrl);
  console.log("[DEBUG] __dirname:", __dirname);

  console.log("[DEBUG] Attempting to load URL:", startUrl);
  mainWindow.loadURL(startUrl).catch((err) => {
    console.error("[DEBUG] ❌ Failed to load URL:", startUrl);
    console.error("Failed to load URL:", err);
    console.error("Make sure the Go backend is running on port 8080");
  });

  // Show window when ready to prevent visual flash
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();

    // Open DevTools in development mode
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Inject CSS workaround when DOM is ready
  mainWindow.webContents.on("dom-ready", () => {
    // WORKAROUND: Remove existing link tag and create a new one with proper attributes
    // This forces the browser to load the CSS properly
    mainWindow.webContents
      .executeJavaScript(
        `
      (function() {
        const existingLink = document.querySelector('link[rel="stylesheet"]');
        if (existingLink) {
          const cssUrl = existingLink.href;

          // Remove the existing link
          existingLink.remove();

          // Create a new link with explicit attributes
          const newLink = document.createElement('link');
          newLink.rel = 'stylesheet';
          newLink.type = 'text/css';
          newLink.href = cssUrl + '?t=' + Date.now(); // Add timestamp to bust cache

          newLink.onerror = function(err) {
            // Fallback: Use XMLHttpRequest instead of fetch
            const xhr = new XMLHttpRequest();
            xhr.open('GET', cssUrl, true);
            xhr.onload = function() {
              if (xhr.status === 200) {
                const styleTag = document.createElement('style');
                styleTag.textContent = xhr.responseText;
                styleTag.id = 'injected-css';
                document.head.appendChild(styleTag);
              }
            };
            xhr.send();
          };

          document.head.appendChild(newLink);
        }
      })();
    `
      )
      .catch((err) =>
        console.error("Failed to execute CSS loading script:", err)
      );
  });

  // Hide window on close (don't quit app) - allows background running
  mainWindow.on("close", (event) => {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
      return false;
    }
  });

  // Handle window closed
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    require("electron").shell.openExternal(url);
    return { action: "deny" };
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: "File",
      submenu: [
        {
          label: "Quit",
          accelerator: process.platform === "darwin" ? "Cmd+Q" : "Ctrl+Q",
          click: () => {
            app.quit();
          },
        },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo", label: "Undo" },
        { role: "redo", label: "Redo" },
        { type: "separator" },
        { role: "cut", label: "Cut" },
        { role: "copy", label: "Copy" },
        { role: "paste", label: "Paste" },
        { role: "selectAll", label: "Select All" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload", label: "Reload" },
        { role: "forceReload", label: "Force Reload" },
        { role: "toggleDevTools", label: "Toggle Developer Tools" },
        { type: "separator" },
        { role: "resetZoom", label: "Actual Size" },
        { role: "zoomIn", label: "Zoom In" },
        { role: "zoomOut", label: "Zoom Out" },
        { type: "separator" },
        { role: "togglefullscreen", label: "Toggle Fullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize", label: "Minimize" },
        { role: "close", label: "Close" },
      ],
    },
    {
      label: "Settings",
      submenu: [
        {
          label: "Preferences...",
          accelerator: process.platform === "darwin" ? "Cmd+," : "Ctrl+,",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-settings");
            }
          },
        },
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "About Yaak Proxy",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-about");
            }
          },
        },
        {
          label: "Terms & Conditions",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-terms");
            }
          },
        },
      ],
    },
  ];

  // macOS specific menu adjustments
  if (process.platform === "darwin") {
    template.unshift({
      label: app.getName(),
      submenu: [
        {
          label: "About " + app.getName(),
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-about");
            }
          },
        },
        { type: "separator" },
        { role: "services", label: "Services" },
        { type: "separator" },
        { role: "hide", label: "Hide " + app.getName() },
        { role: "hideOthers", label: "Hide Others" },
        { role: "unhide", label: "Show All" },
        { type: "separator" },
        { role: "quit", label: "Quit " + app.getName() },
      ],
    });

    // Window menu
    template[4].submenu = [
      { role: "close", label: "Close" },
      { role: "minimize", label: "Minimize" },
      { role: "zoom", label: "Zoom" },
      { type: "separator" },
      { role: "front", label: "Bring All to Front" },
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Launch the Go binary backend first
  launchGoBinary();

  // Create the system tray icon
  createTray();

  // Wait a moment for the backend to start, then create the window
  setTimeout(() => {
    createWindow();
    createMenu();
  }, 1000);

  app.on("activate", () => {
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
    } else if (mainWindow) {
      // If window exists but is hidden, show it
      showMainWindow();
    }
  });
});

// Keep app running in menu bar even when all windows are closed
app.on("window-all-closed", () => {
  // Don't quit - the tray icon keeps the app running
  // Users must explicitly choose "Quit Yaak Proxy" from the tray menu
});

// Handle app quitting
app.on("before-quit", () => {
  app.isQuitting = true;
  stopGoBinary();

  // Cleanup tray icon
  if (tray) {
    tray.destroy();
    tray = null;
  }
});

// Handle app will quit (macOS)
app.on("will-quit", () => {
  stopGoBinary();
});

// IPC handlers for secure storage
ipcMain.handle("get-api-key", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return null;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);

    if (config.apiKey && config.encrypted && isEncryptionAvailable()) {
      // Decrypt the API key
      const buffer = Buffer.from(config.apiKey, "base64");
      const decrypted = safeStorage.decryptString(buffer);
      return decrypted;
    } else if (config.apiKey && !config.encrypted) {
      // Legacy unencrypted storage (migrate on next save)
      return config.apiKey;
    }

    return null;
  } catch (error) {
    console.error("Error reading API key:", error);
    return null;
  }
});

ipcMain.handle("set-api-key", async (event, apiKey) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    if (apiKey && apiKey.trim()) {
      if (isEncryptionAvailable()) {
        // Encrypt the API key
        const encrypted = safeStorage.encryptString(apiKey);
        config.apiKey = encrypted.toString("base64");
        config.encrypted = true;
      } else {
        // Fallback to unencrypted storage (not ideal, but works)
        console.warn("Encryption not available, storing API key unencrypted");
        config.apiKey = apiKey;
        config.encrypted = false;
      }
    } else {
      // Remove API key
      delete config.apiKey;
      delete config.encrypted;
    }

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving API key:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-forward-endpoint", async () => {
  try {
    const storagePath = getStoragePath();
    const defaultEndpoint = "https://api.openai.com/v1";

    if (!fs.existsSync(storagePath)) {
      return defaultEndpoint;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);

    // Migrate old default value to new default
    if (config.forwardEndpoint === "http://localhost:8080") {
      config.forwardEndpoint = defaultEndpoint;
      // Save the updated config
      fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    }

    return config.forwardEndpoint || defaultEndpoint;
  } catch (error) {
    console.error("Error reading forward endpoint:", error);
    return "https://api.openai.com/v1";
  }
});

ipcMain.handle("set-forward-endpoint", async (event, url) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    if (url && url.trim()) {
      config.forwardEndpoint = url.trim();
    } else {
      delete config.forwardEndpoint;
    }

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving forward endpoint:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-ca-cert-setup-dismissed", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return false;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.caCertSetupDismissed || false;
  } catch (error) {
    console.error("Error reading CA cert setup dismissed flag:", error);
    return false;
  }
});

ipcMain.handle("set-ca-cert-setup-dismissed", async (event, dismissed) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.caCertSetupDismissed = !!dismissed;

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving CA cert setup dismissed flag:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-terms-accepted", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return false;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.termsAccepted || false;
  } catch (error) {
    console.error("Error reading terms accepted flag:", error);
    return false;
  }
});

ipcMain.handle("set-terms-accepted", async (event, accepted) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    // Read existing config if it exists
    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.termsAccepted = !!accepted;

    // Save config
    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving terms accepted flag:", error);
    return { success: false, error: error.message };
  }
});

// Security: Prevent new window creation
app.on("web-contents-created", (event, contents) => {
  contents.on("new-window", (event, navigationUrl) => {
    event.preventDefault();
    require("electron").shell.openExternal(navigationUrl);
  });
});
