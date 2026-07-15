const {
  app,
  BrowserWindow,
  Menu,
  Tray,
  nativeImage,
  safeStorage,
  shell,
} = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const { registerIpcHandlers } = require("./ipc-handlers");
const { setMenuLanguage, mt } = require("./menu-i18n");
const isDev = process.env.NODE_ENV === "development";

// Telemetry (Sentry error reporting) is OPT-IN. It is only initialized when the
// user turned on "Crash & error reporting" in Settings (persisted as
// `telemetryEnabled` in the Electron config). See initTelemetryMain().
const Sentry = require("@sentry/electron/main");
const {
  SENTRY_DSN,
  SENTRY_TRACES_SAMPLE_RATE,
} = require("../telemetry/sentry");

// Whether the user has opted into telemetry. Read from the persisted config;
// defaults to false (opt-in) and fails closed on any read error.
const isTelemetryEnabled = () => {
  try {
    return readConfig().telemetryEnabled === true;
  } catch (error) {
    console.error("[Telemetry] Failed to read telemetry preference:", error);
    return false;
  }
};

// Initialize Sentry in the main process, but only if the user opted in. Called
// once at startup (after readConfig is available). No-op when disabled, so no
// data ever leaves the machine unless telemetry is on.
const initTelemetryMain = () => {
  if (!isTelemetryEnabled()) {
    console.log(
      "[Telemetry] Disabled (opt-in). Enable it in Settings to send crash/error reports."
    );
    return;
  }
  Sentry.init({
    dsn: SENTRY_DSN,
    environment: isDev ? "development" : "production",
    tracesSampleRate: SENTRY_TRACES_SAMPLE_RATE,
  });
  console.log("[Telemetry] Enabled: Sentry initialized (main process).");
};

// Configure auto-updater. In dev, reads dev-app-update.yml from app.getAppPath().
const autoUpdater = require("electron-updater").autoUpdater;
autoUpdater.autoDownload = true;
autoUpdater.autoInstallOnAppQuit = true;
if (isDev) {
  autoUpdater.forceDevUpdateConfig = true;
  autoUpdater.autoInstallOnAppQuit = false;
}

autoUpdater.on("update-available", (info) => {
  console.log(`[AutoUpdater] Update available: v${info.version}`);
});

autoUpdater.on("update-downloaded", (info) => {
  console.log(`[AutoUpdater] Update downloaded: v${info.version}`);
  updateDownloaded = true;
  if (mainWindow) {
    createMenu();
  }
  applyTrayIcon();
  updateTrayMenu();
});

autoUpdater.on("error", (err) => {
  console.error("[AutoUpdater] Error:", err);
});

let mainWindow;
let splashWindow = null;
let goProcess = null;
let tray = null;
let updateDownloaded = false;

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
    // In dev, __dirname is src/frontend/src/electron — go up four levels to project root
    return path.join(__dirname, "..", "..", "..", "..");
  }
  return process.resourcesPath || app.getAppPath();
};

// Launch the Go binary backend
// Map of provider type → env var names understood by the Go backend.
// Keep in sync with src/backend/main.go loadApplicationConfig().
const PROVIDER_ENV_NAMES = {
  openai: { apiKey: "OPENAI_API_KEY" },
  anthropic: { apiKey: "ANTHROPIC_API_KEY", baseUrl: "ANTHROPIC_BASE_URL" },
  gemini: { apiKey: "GEMINI_API_KEY", baseUrl: "GEMINI_BASE_URL" },
  mistral: { apiKey: "MISTRAL_API_KEY", baseUrl: "MISTRAL_BASE_URL" },
  custom: { apiKey: "CUSTOM_API_KEY", baseUrl: "CUSTOM_BASE_URL" },
};

// Build env var pairs from the persisted Electron config so the Go backend
// picks up the user's saved API keys and custom endpoint URLs at spawn time.
// Values from the saved config take precedence over inherited process.env
// because they were explicitly set by the user via Settings.
const buildProviderEnvFromConfig = () => {
  const env = {};
  try {
    const cfg = readConfig();
    const providers = cfg.providers || {};

    for (const [provider, names] of Object.entries(PROVIDER_ENV_NAMES)) {
      const providerCfg = providers[provider];
      if (!providerCfg) continue;

      const decryptedKey = decryptApiKey(providerCfg);
      if (decryptedKey) {
        env[names.apiKey] = decryptedKey;
      }

      const baseUrl = (providerCfg.baseUrl || "").trim();
      if (baseUrl && names.baseUrl) {
        env[names.baseUrl] = baseUrl;
      }
    }
  } catch (error) {
    console.error("Error building provider env from saved config:", error);
  }
  return env;
};

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

  // Set up environment variables.
  // Order matters: the saved provider config wins over inherited process.env
  // because the user explicitly set those values via the Settings UI.
  const env = { ...process.env, ...buildProviderEnvFromConfig() };

  // Forward the user's telemetry opt-in to the Go backend so it only initializes
  // Sentry when the user consented. Set explicitly (true/false) rather than
  // inheriting any stray value from process.env.
  env.KIJI_TELEMETRY_ENABLED = isTelemetryEnabled() ? "true" : "false";

  // In development mode, set ONNX Runtime library path
  // Try multiple locations relative to project root
  const onnxPaths = [
    path.join(projectRoot, "build", "libonnxruntime.1.24.2.dylib"), // build/libonnxruntime.1.24.2.dylib
    path.join(
      projectRoot,
      "src",
      "frontend",
      "resources",
      "libonnxruntime.1.24.2.dylib"
    ), // src/frontend/resources/libonnxruntime.1.24.2.dylib
    path.join(projectRoot, "libonnxruntime.1.24.2.dylib"), // root/libonnxruntime.1.24.2.dylib
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
      "libonnxruntime.1.24.2.dylib"
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

// Stop the Go binary and wait for it to actually exit.
// Returns once the process has terminated (or after a hard timeout).
const stopGoBinaryAsync = () => {
  return new Promise((resolve) => {
    if (!goProcess) {
      resolve();
      return;
    }

    const proc = goProcess;
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      goProcess = null;
      resolve();
    };

    proc.once("exit", finish);
    console.log("Stopping Go binary (async)...");
    proc.kill("SIGTERM");

    setTimeout(() => {
      if (!settled) {
        if (!proc.killed) {
          console.log("Force killing Go binary (async)...");
          proc.kill("SIGKILL");
        }
        // Give SIGKILL a brief moment, then resolve regardless.
        setTimeout(finish, 500);
      }
    }, 3000);
  });
};

// Restart the Go binary so it picks up updated env vars from the saved config.
const restartGoBinary = async () => {
  await stopGoBinaryAsync();
  launchGoBinary();
};

// Wait for the Go backend to be ready by polling the health endpoint
const waitForBackend = async (maxRetries = 30, retryInterval = 500) => {
  const { net } = require("electron");
  const healthUrl = "http://localhost:8080/api/health";

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
  const iconPath = path.join(__dirname, "..", "..", "assets", "kiji_proxy.svg");
  let imgSrc = "";
  try {
    const imgData = fs.readFileSync(iconPath, "utf-8");
    imgSrc = `data:image/svg+xml;base64,${Buffer.from(imgData).toString(
      "base64"
    )}`;
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

// Resolve the current tray icon image + tooltip. Returns null if no icon file
// exists. On macOS we prefer the `-Template.png` silhouette so the icon adapts
// to dark/light menu bars; other platforms use the full-color PNG.
const resolveTrayIconInfo = () => {
  const assetsDir = path.join(__dirname, "..", "..", "assets");
  const baseName = updateDownloaded ? "icon-16-update" : "icon-16";
  const macTemplatePath = path.join(assetsDir, `${baseName}-Template.png`);
  const colorPath = path.join(assetsDir, `${baseName}.png`);
  const useTemplate =
    process.platform === "darwin" && fs.existsSync(macTemplatePath);
  const iconPath = useTemplate ? macTemplatePath : colorPath;

  if (!fs.existsSync(iconPath)) {
    return null;
  }

  let image = nativeImage.createFromPath(iconPath);
  if (process.platform === "darwin") {
    image = image.resize({ width: 16, height: 16 });
    if (useTemplate) {
      image.setTemplateImage(true);
    }
  }
  return {
    image,
    tooltip: updateDownloaded
      ? "Kiji Privacy Proxy — Update available"
      : "Kiji Privacy Proxy",
  };
};

// Push the current icon + tooltip onto an existing tray (e.g. after an update
// is downloaded and we want to show the update badge).
const applyTrayIcon = () => {
  if (!tray) return;
  const info = resolveTrayIconInfo();
  if (!info) return;
  tray.setImage(info.image);
  tray.setToolTip(info.tooltip);
};

// Create system tray icon
function createTray() {
  const info = resolveTrayIconInfo();
  if (!info) {
    console.warn("Tray icon not found");
    return;
  }

  tray = new Tray(info.image);
  tray.setToolTip(info.tooltip);
  updateTrayMenu();

  // On macOS, left-click shows the context menu (default behavior).
  // On Windows/Linux, clicking the tray icon should open the main window.
  if (process.platform !== "darwin") {
    tray.on("click", () => {
      showMainWindow();
    });
  }
}

function updateTrayMenu() {
  if (!tray) return;

  const menuItems = [
    {
      label: mt("openApp", { name: app.getName() }),
      click: () => {
        showMainWindow();
      },
    },
    {
      label: mt("aboutApp", { name: app.getName() }),
      click: () => {
        showMainWindow();
        setTimeout(() => {
          if (mainWindow) {
            mainWindow.webContents.send("open-about");
          }
        }, 100);
      },
    },
    {
      label: mt("settings"),
      click: () => {
        showMainWindow();
        setTimeout(() => {
          if (mainWindow) {
            mainWindow.webContents.send("open-settings");
          }
        }, 100);
      },
    },

    { type: "separator" },
    {
      label: mt("documentation"),
      click: () =>
        shell.openExternal(
          "https://github.com/dataiku/kiji-proxy/blob/main/docs/README.md"
        ),
    },
    {
      label: mt("chromeExtension"),
      click: () =>
        shell.openExternal(
          "https://chromewebstore.google.com/detail/kiji-privacy-proxy-extens/knnjemahdeioghdgcpeikepmlajfihin"
        ),
    },
    {
      label: mt("bugReport"),
      click: () =>
        shell.openExternal(
          "https://github.com/dataiku/kiji-proxy/issues/new?template=10_bug_report.yml"
        ),
    },
    {
      label: mt("featureRequest"),
      click: () =>
        shell.openExternal(
          "https://github.com/dataiku/kiji-proxy/discussions/new/choose"
        ),
    },
    {
      label: mt("emailUs"),
      click: () =>
        shell.openExternal(
          "mailto:opensource@dataiku.com?subject=[Yaak Proxy User]"
        ),
    },
    { type: "separator" },
    ...(updateDownloaded
      ? [
          {
            label: mt("restartToUpdate"),
            click: () => autoUpdater.quitAndInstall(),
          },
        ]
      : []),
    {
      label: mt("quitApp", { name: app.getName() }),
      click: () => {
        app.quit();
      },
    },
  ];

  tray.setContextMenu(Menu.buildFromTemplate(menuItems));
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
  // In development, use the webpack dev server for full React errors and HMR.
  // In production, load the UI served by the Go backend.
  const startUrl = isDev ? "http://localhost:3000" : "http://localhost:8080";

  console.log("[DEBUG] Mode:", isDev ? "development" : "production");
  console.log("[DEBUG] Loading UI at:", startUrl);
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
    // Build the menu in the last-used language before showing the window. The
    // renderer re-asserts its detected language over set-language once it loads.
    setMenuLanguage(readConfig().language);
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
    shell.openExternal(url);
    return { action: "deny" };
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: mt("file"),
      submenu: [
        {
          label: mt("quit"),
          accelerator: process.platform === "darwin" ? "Cmd+Q" : "Ctrl+Q",
          click: () => {
            app.quit();
          },
        },
      ],
    },
    {
      label: mt("edit"),
      submenu: [
        { role: "undo", label: mt("undo") },
        { role: "redo", label: mt("redo") },
        { type: "separator" },
        { role: "cut", label: mt("cut") },
        { role: "copy", label: mt("copy") },
        { role: "paste", label: mt("paste") },
        { role: "selectAll", label: mt("selectAll") },
      ],
    },
    {
      label: mt("view"),
      submenu: [
        { role: "reload", label: mt("reload") },
        { role: "forceReload", label: mt("forceReload") },
        { role: "toggleDevTools", label: mt("toggleDevTools") },
        { type: "separator" },
        { role: "resetZoom", label: mt("actualSize") },
        { role: "zoomIn", label: mt("zoomIn") },
        { role: "zoomOut", label: mt("zoomOut") },
        { type: "separator" },
        { role: "togglefullscreen", label: mt("toggleFullscreen") },
      ],
    },
    {
      label: mt("window"),
      submenu: [
        { role: "minimize", label: mt("minimize") },
        { role: "close", label: mt("close") },
      ],
    },
    {
      label: mt("settings"),
      submenu: [
        {
          label: mt("preferences"),
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
      label: mt("help"),
      submenu: [
        {
          label: mt("aboutApp", { name: app.getName() }),
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-about");
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
          label: mt("aboutApp", { name: app.getName() }),
          click: () => {
            if (mainWindow) {
              mainWindow.webContents.send("open-about");
            }
          },
        },
        { type: "separator" },
        { role: "services", label: mt("services") },
        { type: "separator" },
        { role: "hide", label: mt("hideApp", { name: app.getName() }) },
        { role: "hideOthers", label: mt("hideOthers") },
        { role: "unhide", label: mt("showAll") },
        { type: "separator" },
        ...(updateDownloaded
          ? [
              {
                label: mt("restartToUpdate"),
                click: () => autoUpdater.quitAndInstall(),
              },
            ]
          : []),
        { role: "quit", label: mt("quitApp", { name: app.getName() }) },
      ],
    });

    // Window menu
    template[4].submenu = [
      { role: "close", label: mt("close") },
      { role: "minimize", label: mt("minimize") },
      { role: "zoom", label: mt("zoom") },
      { type: "separator" },
      { role: "front", label: mt("bringAllToFront") },
    ];
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// Switch the language of the native application and tray menus and rebuild them.
// Called at startup from the persisted config and at runtime when the renderer
// changes the language (over the set-language IPC channel).
function applyLanguage(language) {
  setMenuLanguage(language);
  if (mainWindow) {
    createMenu();
  }
  if (tray) {
    updateTrayMenu();
  }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(async () => {
  // Initialize telemetry first (opt-in) so early startup errors can be reported.
  initTelemetryMain();

  // Launch the Go binary backend first
  launchGoBinary();

  // Create the system tray icon
  createTray();

  // Show splash screen while backend starts up
  createSplashWindow();

  // Wait for backend to be ready before creating window
  await waitForBackend();
  createWindow();

  // Check for updates after launch
  autoUpdater.checkForUpdatesAndNotify();

  // Re-check for updates every hour for long-running sessions
  setInterval(() => autoUpdater.checkForUpdates(), 60 * 60 * 1000);

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
    custom: { apiKey: "", encrypted: false, model: "", baseUrl: "" },
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

// Register every ipcMain.handle channel. Definitions live in ipc-handlers.js;
// we inject the deps so that module stays decoupled from app/window lifecycle.
registerIpcHandlers({
  readConfig,
  saveConfig,
  encryptApiKey,
  decryptApiKey,
  restartGoBinary,
  waitForBackend,
  getMainWindow: () => mainWindow,
  // Rebuild the native menus whenever the renderer changes the UI language.
  onLanguageChange: applyLanguage,
});

// Security: Prevent new window creation
app.on("web-contents-created", (event, contents) => {
  contents.on("new-window", (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});
