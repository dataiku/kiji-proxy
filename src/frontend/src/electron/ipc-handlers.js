const { app, ipcMain, shell } = require("electron");
const path = require("path");
const fs = require("fs");

// Valid provider types accepted by the Go backend.
// Keep in sync with src/backend/main.go loadApplicationConfig().
const VALID_PROVIDERS = ["openai", "anthropic", "gemini", "mistral", "custom"];

// Make a JSON request to the local Go backend.
// Resolves with the parsed JSON body; rejects on network error or invalid JSON.
// Callers decide how to map errors into the response sent back to the renderer.
const backendRequest = (method, urlPath, body = null) => {
  const { net } = require("electron");
  return new Promise((resolve, reject) => {
    const request = net.request({
      method,
      url: `http://localhost:8080${urlPath}`,
    });
    if (body !== null) {
      request.setHeader("Content-Type", "application/json");
    }
    let responseData = "";
    request.on("response", (response) => {
      response.on("data", (chunk) => {
        responseData += chunk.toString();
      });
      response.on("end", () => {
        try {
          resolve(JSON.parse(responseData));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.on("error", reject);
    if (body !== null) {
      request.write(JSON.stringify(body));
    }
    request.end();
  });
};

// Notify the backend of a config change; treat failures as non-fatal because
// the local config write has already succeeded. The renderer just needs to
// know the save went through; the backend will pick up the value on its next
// read or restart.
const notifyBackendBestEffort = async (urlPath, body) => {
  try {
    return await backendRequest("POST", urlPath, body);
  } catch (error) {
    console.warn(`Backend notification to ${urlPath} failed:`, error.message);
    return { success: true };
  }
};

// Register every ipcMain.handle channel exposed to the renderer.
// Dependencies are injected so this module stays decoupled from the rest of
// the main process (window/tray/backend lifecycle live in electron-main.js).
const registerIpcHandlers = ({
  readConfig,
  saveConfig,
  encryptApiKey,
  decryptApiKey,
  restartGoBinary,
  waitForBackend,
  getMainWindow,
}) => {
  // Register matching get-/set- IPC handlers that read and write a single
  // top-level field of the persisted config. `coerce` transforms the inbound
  // value before saving — returning `undefined` deletes the field. `onChange`
  // runs after a successful save; its return value (e.g. backend notification
  // result) is sent back to the renderer in place of {success: true}.
  const defineConfigField = (
    key,
    getChannel,
    setChannel,
    { defaultValue, coerce = (v) => v, onChange } = {}
  ) => {
    ipcMain.handle(getChannel, async () => {
      try {
        const config = readConfig();
        return config[key] ?? defaultValue;
      } catch (error) {
        console.error(`Error reading ${key}:`, error);
        return defaultValue;
      }
    });

    ipcMain.handle(setChannel, async (_event, value) => {
      try {
        const config = readConfig();
        const coerced = coerce(value);
        if (coerced === undefined) {
          delete config[key];
        } else {
          config[key] = coerced;
        }
        saveConfig(config);
        if (onChange) {
          return await onChange(coerced);
        }
        return { success: true };
      } catch (error) {
        console.error(`Error saving ${key}:`, error);
        return { success: false, error: error.message };
      }
    });
  };

  // ---- Secure storage / API keys ----

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
  ipcMain.handle("set-api-key", async (_event, apiKey) => {
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

  ipcMain.handle("get-active-provider", async () => {
    try {
      const config = readConfig();
      return config.activeProvider || "openai";
    } catch (error) {
      console.error("Error reading active provider:", error);
      return "openai";
    }
  });

  ipcMain.handle("set-active-provider", async (_event, provider) => {
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

  ipcMain.handle("get-provider-api-key", async (_event, provider) => {
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

  ipcMain.handle("set-provider-api-key", async (_event, provider, apiKey) => {
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

  ipcMain.handle("get-provider-model", async (_event, provider) => {
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

  ipcMain.handle("set-provider-model", async (_event, provider, model) => {
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

  ipcMain.handle("get-provider-base-url", async (_event, provider) => {
    try {
      if (!VALID_PROVIDERS.includes(provider)) {
        console.error(`Invalid provider: ${provider}`);
        return "";
      }

      const config = readConfig();
      return config.providers?.[provider]?.baseUrl || "";
    } catch (error) {
      console.error(`Error reading base URL for ${provider}:`, error);
      return "";
    }
  });

  ipcMain.handle("set-provider-base-url", async (_event, provider, baseUrl) => {
    try {
      if (!VALID_PROVIDERS.includes(provider)) {
        return { success: false, error: `Invalid provider: ${provider}` };
      }

      const trimmed = (baseUrl || "").trim();
      if (trimmed && !/^https?:\/\//i.test(trimmed)) {
        return {
          success: false,
          error: "Base URL must start with http:// or https://",
        };
      }

      const config = readConfig();
      if (!config.providers) {
        config.providers = {};
      }
      if (!config.providers[provider]) {
        config.providers[provider] = { apiKey: "", encrypted: false };
      }

      config.providers[provider].baseUrl = trimmed;

      saveConfig(config);
      return { success: true };
    } catch (error) {
      console.error(`Error saving base URL for ${provider}:`, error);
      return { success: false, error: error.message };
    }
  });

  // Restart the Go backend so updated provider config (API keys, base URLs)
  // takes effect. Settings are injected as env vars at spawn time.
  ipcMain.handle("restart-backend", async () => {
    try {
      if (
        process.env.EXTERNAL_BACKEND === "true" ||
        process.env.SKIP_BACKEND_LAUNCH === "true"
      ) {
        return {
          success: false,
          error:
            "Backend is externally managed (EXTERNAL_BACKEND); restart it manually.",
        };
      }

      await restartGoBinary();
      const ready = await waitForBackend(30, 500);
      if (!ready) {
        return {
          success: false,
          error: "Backend failed to become ready after restart",
        };
      }
      return { success: true };
    } catch (error) {
      console.error("Error restarting backend:", error);
      return { success: false, error: error.message };
    }
  });

  // Open the CA cert location in the OS file manager. The Go backend writes the
  // CA to <userData>/certs/ca.crt — same path Electron knows via app.getPath().
  // If the file exists, highlight it; otherwise open the parent directory so the
  // user isn't left staring at an error dialog when the proxy hasn't generated
  // the cert yet.
  ipcMain.handle("reveal-ca-cert", async () => {
    try {
      const certPath = path.join(app.getPath("userData"), "certs", "ca.crt");
      if (fs.existsSync(certPath)) {
        shell.showItemInFolder(certPath);
        return { success: true, path: certPath, exists: true };
      }

      const certDir = path.dirname(certPath);
      fs.mkdirSync(certDir, { recursive: true });
      const errMsg = await shell.openPath(certDir);
      if (errMsg) {
        return { success: false, error: errMsg };
      }
      return { success: true, path: certDir, exists: false };
    } catch (error) {
      console.error("Error revealing CA cert:", error);
      return { success: false, error: error.message };
    }
  });

  // Get full providers config
  ipcMain.handle("get-providers-config", async () => {
    try {
      const config = readConfig();
      const activeProvider = config.activeProvider || "openai";

      // Build response with hasApiKey (boolean), model, and baseUrl for each provider
      const providers = {};
      for (const provider of VALID_PROVIDERS) {
        const providerConfig = config.providers?.[provider] || {};
        providers[provider] = {
          hasApiKey: !!(
            providerConfig.apiKey && providerConfig.apiKey.length > 0
          ),
          model: providerConfig.model || "",
          baseUrl: providerConfig.baseUrl || "",
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
          openai: { hasApiKey: false, model: "", baseUrl: "" },
          anthropic: { hasApiKey: false, model: "", baseUrl: "" },
          gemini: { hasApiKey: false, model: "", baseUrl: "" },
          mistral: { hasApiKey: false, model: "", baseUrl: "" },
          custom: { hasApiKey: false, model: "", baseUrl: "" },
        },
      };
    }
  });

  // ---- Onboarding / UI dismissal flags ----

  const booleanField = { defaultValue: false, coerce: (v) => !!v };

  defineConfigField(
    "caCertSetupDismissed",
    "get-ca-cert-setup-dismissed",
    "set-ca-cert-setup-dismissed",
    booleanField
  );

  defineConfigField(
    "welcomeDismissed",
    "get-welcome-dismissed",
    "set-welcome-dismissed",
    booleanField
  );

  defineConfigField(
    "tourCompleted",
    "get-tour-completed",
    "set-tour-completed",
    booleanField
  );

  // ---- Model directory management ----

  ipcMain.handle("select-model-directory", async () => {
    const { dialog } = require("electron");
    const result = await dialog.showOpenDialog(getMainWindow(), {
      properties: ["openDirectory"],
      title: "Select Model Directory",
      message: "Choose the directory containing your PII model files",
    });

    if (result.canceled || result.filePaths.length === 0) {
      return null;
    }

    return result.filePaths[0];
  });

  defineConfigField(
    "modelDirectory",
    "get-model-directory",
    "set-model-directory",
    {
      defaultValue: null,
      // Empty / whitespace-only strings clear the field instead of storing "".
      coerce: (v) => (v && v.trim() ? v.trim() : undefined),
    }
  );

  ipcMain.handle("reload-model", async (_event, directory) => {
    try {
      return await backendRequest("POST", "/api/model/reload", { directory });
    } catch (error) {
      console.error("Error reloading model:", error);
      return { success: false, error: error.message };
    }
  });

  ipcMain.handle("get-model-info", async () => {
    try {
      return await backendRequest("GET", "/api/model/info");
    } catch (error) {
      console.error("Error getting model info:", error);
      return { error: error.message };
    }
  });

  // ---- Transparent Proxy Settings ----
  defineConfigField(
    "transparentProxyEnabled",
    "get-transparent-proxy-enabled",
    "set-transparent-proxy-enabled",
    {
      defaultValue: false,
      coerce: (v) => !!v,
      onChange: (enabled) =>
        notifyBackendBestEffort("/api/proxy/transparent/toggle", { enabled }),
    }
  );

  // ---- PII Detection Confidence Threshold ----
  defineConfigField(
    "entityConfidence",
    "get-entity-confidence",
    "set-entity-confidence",
    {
      defaultValue: 0.25,
      onChange: (confidence) =>
        notifyBackendBestEffort("/api/pii/confidence", { confidence }),
    }
  );

  // ---- PII Entities to Mask ----
  // Persist the exclusion list — the entity types to leave UNMASKED — and push
  // it to the backend on change. Storing the exclusion (rather than the masked
  // set) is deliberate: null/empty means "nothing excluded, mask everything", so
  // the default and any accidental clearing fail closed toward masking.
  defineConfigField(
    "disabledEntities",
    "get-disabled-entities",
    "set-disabled-entities",
    {
      defaultValue: null,
      onChange: (disabled) =>
        notifyBackendBestEffort("/api/pii/entities", { disabled }),
    }
  );

  // The full set of selectable entity types comes from the loaded model, so it
  // is read live from the backend rather than persisted in the config.
  ipcMain.handle("get-available-entities", async () => {
    try {
      return await backendRequest("GET", "/api/pii/entities");
    } catch (error) {
      console.error("Error getting available entities:", error);
      return { error: error.message };
    }
  });

  // ---- PII Custom Regexes ----
  // The regex detector is the source of truth (seeded from the backend config),
  // so patterns are read live from the backend and written straight through.
  ipcMain.handle("get-custom-regexes", async () => {
    try {
      return await backendRequest("GET", "/api/pii/regexes");
    } catch (error) {
      console.error("Error getting custom regexes:", error);
      return { error: error.message };
    }
  });

  ipcMain.handle("set-custom-regexes", async (_event, regexes) => {
    try {
      return await backendRequest("POST", "/api/pii/regexes", { regexes });
    } catch (error) {
      console.error("Error setting custom regexes:", error);
      return { success: false, error: error.message };
    }
  });
};

module.exports = { registerIpcHandlers };
