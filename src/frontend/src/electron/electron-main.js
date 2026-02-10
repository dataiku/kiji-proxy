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
let splashWindow = null;
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
      "kiji-proxy"
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
    const binaryPath = path.join(resourcesPath, "resources", "kiji-proxy");

    console.log("[DEBUG] Checking primary path:", binaryPath);
    // If not found, try alternative paths
    if (fs.existsSync(binaryPath)) {
      console.log("[DEBUG] ✅ Binary found at:", binaryPath);
      return binaryPath;
    }

    // Try without 'resources' subdirectory (if resources are at root)
    const altPath = path.join(resourcesPath, "kiji-proxy");
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
  const finalPath = path.join(resourcesPath, "resources", "kiji-proxy");
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

// Wait for the Go backend to be ready by polling the health endpoint
const waitForBackend = async (maxRetries = 30, retryInterval = 500) => {
  const { net } = require("electron");
  const healthUrl = "http://localhost:8080/health";

  console.log("[DEBUG] Waiting for backend to be ready...");

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await net.fetch(healthUrl);
      if (response.status === 200) {
        console.log(
          `[DEBUG] ✅ Backend ready after ${attempt} attempt(s) (~${
            attempt * retryInterval
          }ms)`
        );
        return true;
      }
      console.log(
        `[DEBUG] Backend responded with status ${response.status}, attempt ${attempt}/${maxRetries}`
      );
    } catch (error) {
      console.log(
        `[DEBUG] Backend not reachable (attempt ${attempt}/${maxRetries}): ${error.message}`
      );
    }

    if (attempt < maxRetries) {
      await new Promise((resolve) => setTimeout(resolve, retryInterval));
    }
  }

  console.error(
    `[DEBUG] ❌ Backend failed to become ready after ${maxRetries} attempts (~${
      maxRetries * retryInterval
    }ms)`
  );
  return false;
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

// Create splash window shown during backend startup
function createSplashWindow() {
  const iconPath = path.join(__dirname, "..", "..", "assets", "yaak.png");
  let imgSrc = "";
  try {
    const imgData = fs.readFileSync(iconPath);
    imgSrc = `data:image/png;base64,${imgData.toString("base64")}`;
  } catch {
    // Fallback: no image, just show spinner
  }

  const splashHtml = `
    <html>
    <head>
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
          background: transparent;
          display: flex;
          justify-content: center;
          align-items: center;
          height: 100vh;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          -webkit-app-region: drag;
        }
        .container {
          background: rgba(15, 23, 42, 0.92);
          backdrop-filter: blur(12px);
          border-radius: 20px;
          padding: 40px 50px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 20px;
        }
        .mascot {
          width: 100px;
          height: 100px;
          animation: bounce 1.5s ease-in-out infinite;
          filter: drop-shadow(0 10px 15px rgba(0, 0, 0, 0.3));
        }
        .status {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .spinner {
          width: 18px;
          height: 18px;
          border: 2.5px solid rgba(148, 163, 184, 0.3);
          border-top-color: #60a5fa;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        .text {
          color: #cbd5e1;
          font-size: 14px;
          font-weight: 500;
          letter-spacing: 0.02em;
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-20px); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      </style>
    </head>
    <body>
      <div class="container">
        ${imgSrc ? `<img class="mascot" src="${imgSrc}" alt="" />` : ""}
        <div class="status">
          <div class="spinner"></div>
          <span class="text">Starting up...</span>
        </div>
      </div>
    </body>
    </html>
  `;

  splashWindow = new BrowserWindow({
    width: 300,
    height: 280,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    center: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splashWindow.loadURL(
    `data:text/html;charset=utf-8,${encodeURIComponent(splashHtml)}`
  );

  splashWindow.on("closed", () => {
    splashWindow = null;
  });
}

// Close and destroy the splash window
function closeSplashWindow() {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.close();
    splashWindow = null;
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
    const resizedIcon = icon.resize({ width: 16, height: 16 });
    // Mark as template image for automatic dark mode adaptation
    resizedIcon.setTemplateImage(true);
    tray = new Tray(resizedIcon);
    tray.setToolTip("Kiji Privacy Proxy");
  } else {
    tray = new Tray(icon);
    tray.setToolTip("Kiji Privacy Proxy");
  }

  // Build context menu
  const contextMenu = Menu.buildFromTemplate([
    {
      label: "Open Kiji Privacy Proxy",
      click: () => {
        showMainWindow();
      },
    },
    {
      label: "About Kiji Privacy Proxy",
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
      label: "Terms && Conditions",
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
          "https://github.com/hanneshapke/kiji-proxy/blob/main/docs/README.md"
        );
      },
    },
    {
      label: "File a Bug Report",
      click: () => {
        require("electron").shell.openExternal(
          "https://github.com/hanneshapke/kiji-proxy/issues/new?template=10_bug_report.yml"
        );
      },
    },
    {
      label: "Request a Feature",
      click: () => {
        require("electron").shell.openExternal(
          "https://github.com/hanneshapke/kiji-proxy/discussions/new/choose"
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
      label: "Quit Kiji Privacy Proxy",
      click: () => {
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);

  // On macOS, left-click shows the context menu (default behavior)
  // On Windows/Linux, we can add a click handler if needed
  if (process.platform !== "darwin") {
    tray.on("click", () => {
      showMainWindow();
    });
  }
}

function createWindow() {
  // Get icon path (works in both dev and production)
  const iconPath = path.join(__dirname, "..", "..", "assets", "icon.png");
  const iconExists = fs.existsSync(iconPath);

  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 1000,
    minWidth: 800,
    minHeight: 700,
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
  const startUrl = "http://localhost:8080";

  console.log("[DEBUG] Mode:", isDev ? "development" : "production");
  console.log("[DEBUG] Loading UI from Go backend at:", startUrl);
  console.log("[DEBUG] __dirname:", __dirname);

  // Retry loading the page if it fails (safety net in case backend becomes
  // temporarily unreachable after the initial waitForBackend() check)
  let loadRetries = 0;
  const MAX_LOAD_RETRIES = 3;

  mainWindow.webContents.on(
    "did-fail-load",
    (_event, errorCode, errorDescription) => {
      console.error(
        `[DEBUG] ❌ Page failed to load: ${errorDescription} (code: ${errorCode})`
      );

      if (loadRetries < MAX_LOAD_RETRIES) {
        loadRetries++;
        const retryDelay = 1000 * loadRetries;
        console.log(
          `[DEBUG] Retrying load in ${retryDelay}ms (attempt ${loadRetries}/${MAX_LOAD_RETRIES})...`
        );
        setTimeout(() => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.loadURL(startUrl).catch((err) => {
              console.error("[DEBUG] Retry loadURL failed:", err.message);
            });
          }
        }, retryDelay);
      } else {
        console.error(
          "[DEBUG] Max retries reached. Backend may not be running."
        );
      }
    }
  );

  console.log("[DEBUG] Attempting to load URL:", startUrl);
  mainWindow.loadURL(startUrl).catch((err) => {
    console.error("[DEBUG] ❌ Failed to load URL:", startUrl);
    console.error("Failed to load URL:", err);
    console.error("Make sure the Go backend is running on port 8080");
  });

  // Show window when ready to prevent visual flash
  mainWindow.once("ready-to-show", () => {
    // Create menu before showing window to ensure it's ready
    createMenu();

    mainWindow.show();
    closeSplashWindow();

    // On macOS, focus the app to ensure menu bar is visible
    if (process.platform === "darwin") {
      app.focus({ steal: true });
    }

    // Open DevTools in development mode
    if (isDev) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Inject CSS workaround when DOM is ready
  mainWindow.webContents.on("dom-ready", () => {
    // WORKAROUND: Reload stylesheet with cache-busting to ensure CSS loads properly.
    // Important: only remove the old stylesheet AFTER the new one has loaded.
    mainWindow.webContents
      .executeJavaScript(
        `
      (function() {
        const existingLink = document.querySelector('link[rel="stylesheet"]');
        if (existingLink) {
          const cssUrl = existingLink.href;

          const newLink = document.createElement('link');
          newLink.rel = 'stylesheet';
          newLink.type = 'text/css';
          newLink.href = cssUrl + '?t=' + Date.now();

          newLink.onload = function() {
            existingLink.remove();
          };

          newLink.onerror = function() {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', cssUrl, true);
            xhr.onload = function() {
              if (xhr.status === 200) {
                const styleTag = document.createElement('style');
                styleTag.textContent = xhr.responseText;
                styleTag.id = 'injected-css';
                document.head.appendChild(styleTag);
                existingLink.remove();
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
          label: "Start Tour",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-tour");
            }
          },
        },
        { type: "separator" },
        {
          label: "About Kiji Privacy Proxy",
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-about");
            }
          },
        },
        {
          label: "Terms && Conditions",
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
app.whenReady().then(async () => {
  // Launch the Go binary backend first
  launchGoBinary();

  // Create the system tray icon
  createTray();

  // Show splash screen while backend starts up
  createSplashWindow();

  // Wait for backend to be ready before creating window
  await waitForBackend();
  createWindow();

  app.on("activate", async () => {
    // On macOS, re-create a window when the dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) {
      // Ensure backend is running
      if (!goProcess) {
        launchGoBinary();
        await waitForBackend();
      } else {
        // Process exists but might not be listening yet
        await waitForBackend(10, 500);
      }
      createWindow();
    } else if (mainWindow) {
      // If window exists but is hidden, show it
      showMainWindow();
    }
  });
});

// Keep app running in menu bar even when all windows are closed
app.on("window-all-closed", () => {
  // Don't quit - the tray icon keeps the app running
  // Users must explicitly choose "Quit Kiji Privacy Proxy" from the tray menu
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

// Valid provider types
const VALID_PROVIDERS = ["openai", "anthropic", "gemini", "mistral"];

// Migrate old single-key config format to new multi-provider format
const migrateConfig = (config) => {
  // If already migrated (has providers object), return as-is
  if (config.providers) {
    return config;
  }

  console.log("[DEBUG] Migrating config to multi-provider format");

  // Initialize providers object
  config.providers = {
    openai: { apiKey: "", encrypted: false, model: "" },
    anthropic: { apiKey: "", encrypted: false, model: "" },
    gemini: { apiKey: "", encrypted: false, model: "" },
    mistral: { apiKey: "", encrypted: false, model: "" },
  };

  // Migrate old apiKey to openai provider
  if (config.apiKey) {
    config.providers.openai.apiKey = config.apiKey;
    config.providers.openai.encrypted = config.encrypted || false;
    delete config.apiKey;
    delete config.encrypted;
  }

  // Set default active provider
  if (!config.activeProvider) {
    config.activeProvider = "openai";
  }

  return config;
};

// Read and migrate config file
const readConfig = () => {
  const storagePath = getStoragePath();
  let config = {};

  if (fs.existsSync(storagePath)) {
    const data = fs.readFileSync(storagePath, "utf8");
    config = JSON.parse(data);
  }

  // Migrate if needed
  const migratedConfig = migrateConfig(config);

  // Save if migrated
  if (!config.providers) {
    fs.writeFileSync(
      storagePath,
      JSON.stringify(migratedConfig, null, 2),
      "utf8"
    );
  }

  return migratedConfig;
};

// Save config file
const saveConfig = (config) => {
  const storagePath = getStoragePath();
  fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
};

// Decrypt an API key
const decryptApiKey = (providerConfig) => {
  if (!providerConfig || !providerConfig.apiKey) {
    return null;
  }

  if (providerConfig.encrypted && isEncryptionAvailable()) {
    const buffer = Buffer.from(providerConfig.apiKey, "base64");
    return safeStorage.decryptString(buffer);
  } else if (!providerConfig.encrypted) {
    return providerConfig.apiKey;
  }

  return null;
};

// Encrypt an API key
const encryptApiKey = (apiKey) => {
  if (!apiKey || !apiKey.trim()) {
    return { apiKey: "", encrypted: false };
  }

  if (isEncryptionAvailable()) {
    const encrypted = safeStorage.encryptString(apiKey.trim());
    return { apiKey: encrypted.toString("base64"), encrypted: true };
  } else {
    console.warn("Encryption not available, storing API key unencrypted");
    return { apiKey: apiKey.trim(), encrypted: false };
  }
};

// IPC handlers for secure storage

// Legacy handler - delegates to active provider
ipcMain.handle("get-api-key", async () => {
  try {
    const config = readConfig();
    const activeProvider = config.activeProvider || "openai";
    const providerConfig = config.providers?.[activeProvider];

    const decrypted = decryptApiKey(providerConfig);
    if (decrypted) {
      console.log(
        `[DEBUG] API key decrypted for ${activeProvider} (length: ${decrypted.length})`
      );
    }
    return decrypted;
  } catch (error) {
    console.error("[ERROR] Error reading API key:", error);
    return null;
  }
});

// Legacy handler - delegates to active provider
ipcMain.handle("set-api-key", async (event, apiKey) => {
  try {
    const config = readConfig();
    const activeProvider = config.activeProvider || "openai";

    if (!config.providers) {
      config.providers = {};
    }
    if (!config.providers[activeProvider]) {
      config.providers[activeProvider] = { model: "" };
    }

    const { apiKey: encryptedKey, encrypted } = encryptApiKey(apiKey);
    config.providers[activeProvider].apiKey = encryptedKey;
    config.providers[activeProvider].encrypted = encrypted;

    saveConfig(config);
    return { success: true };
  } catch (error) {
    console.error("Error saving API key:", error);
    return { success: false, error: error.message };
  }
});

// Get active provider
ipcMain.handle("get-active-provider", async () => {
  try {
    const config = readConfig();
    return config.activeProvider || "openai";
  } catch (error) {
    console.error("Error reading active provider:", error);
    return "openai";
  }
});

// Set active provider
ipcMain.handle("set-active-provider", async (event, provider) => {
  try {
    if (!VALID_PROVIDERS.includes(provider)) {
      return { success: false, error: `Invalid provider: ${provider}` };
    }

    const config = readConfig();
    config.activeProvider = provider;
    saveConfig(config);
    return { success: true };
  } catch (error) {
    console.error("Error setting active provider:", error);
    return { success: false, error: error.message };
  }
});

// Get API key for specific provider
ipcMain.handle("get-provider-api-key", async (event, provider) => {
  try {
    if (!VALID_PROVIDERS.includes(provider)) {
      console.error(`Invalid provider: ${provider}`);
      return null;
    }

    const config = readConfig();
    const providerConfig = config.providers?.[provider];
    return decryptApiKey(providerConfig);
  } catch (error) {
    console.error(`Error reading API key for ${provider}:`, error);
    return null;
  }
});

// Set API key for specific provider
ipcMain.handle("set-provider-api-key", async (event, provider, apiKey) => {
  try {
    if (!VALID_PROVIDERS.includes(provider)) {
      return { success: false, error: `Invalid provider: ${provider}` };
    }

    const config = readConfig();
    if (!config.providers) {
      config.providers = {};
    }
    if (!config.providers[provider]) {
      config.providers[provider] = { model: "" };
    }

    const { apiKey: encryptedKey, encrypted } = encryptApiKey(apiKey);
    config.providers[provider].apiKey = encryptedKey;
    config.providers[provider].encrypted = encrypted;

    saveConfig(config);
    return { success: true };
  } catch (error) {
    console.error(`Error saving API key for ${provider}:`, error);
    return { success: false, error: error.message };
  }
});

// Get custom model for specific provider
ipcMain.handle("get-provider-model", async (event, provider) => {
  try {
    if (!VALID_PROVIDERS.includes(provider)) {
      console.error(`Invalid provider: ${provider}`);
      return "";
    }

    const config = readConfig();
    return config.providers?.[provider]?.model || "";
  } catch (error) {
    console.error(`Error reading model for ${provider}:`, error);
    return "";
  }
});

// Set custom model for specific provider
ipcMain.handle("set-provider-model", async (event, provider, model) => {
  try {
    if (!VALID_PROVIDERS.includes(provider)) {
      return { success: false, error: `Invalid provider: ${provider}` };
    }

    const config = readConfig();
    if (!config.providers) {
      config.providers = {};
    }
    if (!config.providers[provider]) {
      config.providers[provider] = { apiKey: "", encrypted: false };
    }

    config.providers[provider].model = model || "";

    saveConfig(config);
    return { success: true };
  } catch (error) {
    console.error(`Error saving model for ${provider}:`, error);
    return { success: false, error: error.message };
  }
});

// Get full providers config
ipcMain.handle("get-providers-config", async () => {
  try {
    const config = readConfig();
    const activeProvider = config.activeProvider || "openai";

    // Build response with hasApiKey (boolean) and model for each provider
    const providers = {};
    for (const provider of VALID_PROVIDERS) {
      const providerConfig = config.providers?.[provider] || {};
      providers[provider] = {
        hasApiKey: !!(
          providerConfig.apiKey && providerConfig.apiKey.length > 0
        ),
        model: providerConfig.model || "",
      };
    }

    return {
      activeProvider,
      providers,
    };
  } catch (error) {
    console.error("Error reading providers config:", error);
    return {
      activeProvider: "openai",
      providers: {
        openai: { hasApiKey: false, model: "" },
        anthropic: { hasApiKey: false, model: "" },
        gemini: { hasApiKey: false, model: "" },
        mistral: { hasApiKey: false, model: "" },
      },
    };
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

ipcMain.handle("get-welcome-dismissed", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return false;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.welcomeDismissed || false;
  } catch (error) {
    console.error("Error reading welcome dismissed flag:", error);
    return false;
  }
});

ipcMain.handle("set-welcome-dismissed", async (event, dismissed) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.welcomeDismissed = !!dismissed;

    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving welcome dismissed flag:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-tour-completed", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return false;
    }

    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.tourCompleted || false;
  } catch (error) {
    console.error("Error reading tour completed flag:", error);
    return false;
  }
});

ipcMain.handle("set-tour-completed", async (event, completed) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.tourCompleted = !!completed;

    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving tour completed flag:", error);
    return { success: false, error: error.message };
  }
});

// Model directory management
ipcMain.handle("select-model-directory", async () => {
  const { dialog } = require("electron");
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"],
    title: "Select Model Directory",
    message: "Choose the directory containing your PII model files",
  });

  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }

  return result.filePaths[0];
});

ipcMain.handle("get-model-directory", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return null;
    }
    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.modelDirectory || null;
  } catch (error) {
    console.error("Error reading model directory:", error);
    return null;
  }
});

ipcMain.handle("set-model-directory", async (event, directory) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    if (directory && directory.trim()) {
      config.modelDirectory = directory.trim();
    } else {
      delete config.modelDirectory;
    }

    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");
    return { success: true };
  } catch (error) {
    console.error("Error saving model directory:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("reload-model", async (event, directory) => {
  try {
    const { net } = require("electron");
    const request = net.request({
      method: "POST",
      url: "http://localhost:8080/api/model/reload",
    });

    request.setHeader("Content-Type", "application/json");

    return new Promise((resolve, reject) => {
      let responseData = "";

      request.on("response", (response) => {
        response.on("data", (chunk) => {
          responseData += chunk.toString();
        });

        response.on("end", () => {
          try {
            const data = JSON.parse(responseData);
            resolve(data);
          } catch (error) {
            reject(error);
          }
        });
      });

      request.on("error", (error) => {
        console.error("Error reloading model:", error);
        resolve({ success: false, error: error.message });
      });

      request.write(JSON.stringify({ directory }));
      request.end();
    });
  } catch (error) {
    console.error("Error reloading model:", error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle("get-model-info", async () => {
  try {
    const { net } = require("electron");
    const request = net.request({
      method: "GET",
      url: "http://localhost:8080/api/model/info",
    });

    return new Promise((resolve, reject) => {
      let responseData = "";

      request.on("response", (response) => {
        response.on("data", (chunk) => {
          responseData += chunk.toString();
        });

        response.on("end", () => {
          try {
            const data = JSON.parse(responseData);
            resolve(data);
          } catch (error) {
            reject(error);
          }
        });
      });

      request.on("error", (error) => {
        console.error("Error getting model info:", error);
        resolve({ error: error.message });
      });

      request.end();
    });
  } catch (error) {
    console.error("Error getting model info:", error);
    return { error: error.message };
  }
});

// Transparent Proxy Settings
ipcMain.handle("get-transparent-proxy-enabled", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return false;
    }
    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.transparentProxyEnabled || false;
  } catch (error) {
    console.error("Error reading transparent proxy setting:", error);
    return false;
  }
});

ipcMain.handle("set-transparent-proxy-enabled", async (event, enabled) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.transparentProxyEnabled = !!enabled;

    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");

    // Notify the backend about the change
    const { net } = require("electron");
    const request = net.request({
      method: "POST",
      url: "http://localhost:8080/api/proxy/transparent/toggle",
    });

    request.setHeader("Content-Type", "application/json");

    return new Promise((resolve) => {
      let responseData = "";

      request.on("response", (response) => {
        response.on("data", (chunk) => {
          responseData += chunk.toString();
        });

        response.on("end", () => {
          try {
            const data = JSON.parse(responseData);
            resolve(data);
          } catch (error) {
            // Config was saved, but backend notification failed - still success
            console.warn("Error parsing backend response:", error);
            resolve({ success: true });
          }
        });
      });

      request.on("error", (error) => {
        console.error("Error notifying backend:", error);
        // Config was saved, but backend notification failed - still success
        resolve({ success: true });
      });

      request.write(JSON.stringify({ enabled: !!enabled }));
      request.end();
    });
  } catch (error) {
    console.error("Error saving transparent proxy setting:", error);
    return { success: false, error: error.message };
  }
});

// PII Detection Confidence Threshold
ipcMain.handle("get-entity-confidence", async () => {
  try {
    const storagePath = getStoragePath();
    if (!fs.existsSync(storagePath)) {
      return 0.25;
    }
    const data = fs.readFileSync(storagePath, "utf8");
    const config = JSON.parse(data);
    return config.entityConfidence ?? 0.25;
  } catch (error) {
    console.error("Error reading entity confidence:", error);
    return 0.25;
  }
});

ipcMain.handle("set-entity-confidence", async (event, confidence) => {
  try {
    const storagePath = getStoragePath();
    let config = {};

    if (fs.existsSync(storagePath)) {
      const data = fs.readFileSync(storagePath, "utf8");
      config = JSON.parse(data);
    }

    config.entityConfidence = confidence;

    fs.writeFileSync(storagePath, JSON.stringify(config, null, 2), "utf8");

    // Notify the backend about the change
    const { net } = require("electron");
    const request = net.request({
      method: "POST",
      url: "http://localhost:8080/api/pii/confidence",
    });

    request.setHeader("Content-Type", "application/json");

    return new Promise((resolve) => {
      let responseData = "";

      request.on("response", (response) => {
        response.on("data", (chunk) => {
          responseData += chunk.toString();
        });

        response.on("end", () => {
          try {
            const data = JSON.parse(responseData);
            resolve(data);
          } catch (error) {
            console.warn("Error parsing backend response:", error);
            resolve({ success: true });
          }
        });
      });

      request.on("error", (error) => {
        console.error("Error notifying backend:", error);
        resolve({ success: true });
      });

      request.write(JSON.stringify({ confidence }));
      request.end();
    });
  } catch (error) {
    console.error("Error saving entity confidence:", error);
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
