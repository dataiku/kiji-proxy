// Preload script for Electron
// This runs in a context that has access to both DOM APIs and Node.js APIs
// but is isolated from the main renderer process for security

const { contextBridge, ipcRenderer } = require("electron");

// Expose protected methods that allow the renderer process to use
// the API endpoint configuration and secure storage
contextBridge.exposeInMainWorld("electronAPI", {
  // Legacy methods (delegate to active provider for backwards compatibility)
  // Get the stored API key for active provider
  getApiKey: async () => {
    return await ipcRenderer.invoke("get-api-key");
  },

  // Set the API key for active provider (securely stored)
  setApiKey: async (apiKey) => {
    return await ipcRenderer.invoke("set-api-key", apiKey);
  },

  // Multi-provider methods
  // Get the active provider
  getActiveProvider: async () => {
    return await ipcRenderer.invoke("get-active-provider");
  },

  // Set the active provider
  setActiveProvider: async (provider) => {
    return await ipcRenderer.invoke("set-active-provider", provider);
  },

  // Get API key for a specific provider
  getProviderApiKey: async (provider) => {
    return await ipcRenderer.invoke("get-provider-api-key", provider);
  },

  // Set API key for a specific provider
  setProviderApiKey: async (provider, apiKey) => {
    return await ipcRenderer.invoke("set-provider-api-key", provider, apiKey);
  },

  // Get custom model for a specific provider
  getProviderModel: async (provider) => {
    return await ipcRenderer.invoke("get-provider-model", provider);
  },

  // Set custom model for a specific provider
  setProviderModel: async (provider, model) => {
    return await ipcRenderer.invoke("set-provider-model", provider, model);
  },

  // Get full providers config (hasApiKey and model for each provider)
  getProvidersConfig: async () => {
    return await ipcRenderer.invoke("get-providers-config");
  },

  // Platform information
  platform: process.platform,

  // Version information
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron,
  },

  // Listen for settings menu command
  onSettingsOpen: (callback) => {
    ipcRenderer.on("open-settings", callback);
  },

  // Remove settings listener
  removeSettingsListener: () => {
    ipcRenderer.removeAllListeners("open-settings");
  },

  // Listen for about menu command
  onAboutOpen: (callback) => {
    ipcRenderer.on("open-about", callback);
  },

  // Remove about listener
  removeAboutListener: () => {
    ipcRenderer.removeAllListeners("open-about");
  },

  // Get CA cert setup dismissed flag
  getCACertSetupDismissed: async () => {
    return await ipcRenderer.invoke("get-ca-cert-setup-dismissed");
  },

  // Set CA cert setup dismissed flag
  setCACertSetupDismissed: async (dismissed) => {
    return await ipcRenderer.invoke("set-ca-cert-setup-dismissed", dismissed);
  },

  // Get terms accepted flag
  getTermsAccepted: async () => {
    return await ipcRenderer.invoke("get-terms-accepted");
  },

  // Set terms accepted flag
  setTermsAccepted: async (accepted) => {
    return await ipcRenderer.invoke("set-terms-accepted", accepted);
  },

  // Get welcome modal dismissed flag
  getWelcomeDismissed: async () => {
    return await ipcRenderer.invoke("get-welcome-dismissed");
  },

  // Set welcome modal dismissed flag
  setWelcomeDismissed: async (dismissed) => {
    return await ipcRenderer.invoke("set-welcome-dismissed", dismissed);
  },

  // Listen for terms menu command
  onTermsOpen: (callback) => {
    ipcRenderer.on("open-terms", callback);
  },

  // Remove terms listener
  removeTermsListener: () => {
    ipcRenderer.removeAllListeners("open-terms");
  },

  // Model directory management
  selectModelDirectory: async () => {
    return await ipcRenderer.invoke("select-model-directory");
  },

  getModelDirectory: async () => {
    return await ipcRenderer.invoke("get-model-directory");
  },

  setModelDirectory: async (directory) => {
    return await ipcRenderer.invoke("set-model-directory", directory);
  },

  reloadModel: async (directory) => {
    return await ipcRenderer.invoke("reload-model", directory);
  },

  getModelInfo: async () => {
    return await ipcRenderer.invoke("get-model-info");
  },

  // Transparent proxy settings
  getTransparentProxyEnabled: async () => {
    return await ipcRenderer.invoke("get-transparent-proxy-enabled");
  },

  setTransparentProxyEnabled: async (enabled) => {
    return await ipcRenderer.invoke("set-transparent-proxy-enabled", enabled);
  },
});
