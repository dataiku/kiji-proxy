// Preload script for Electron
// This runs in a context that has access to both DOM APIs and Node.js APIs
// but is isolated from the main renderer process for security

const { contextBridge, ipcRenderer } = require("electron");

// Expose protected methods that allow the renderer process to use
// the API endpoint configuration and secure storage
contextBridge.exposeInMainWorld("electronAPI", {
  // Get the forward endpoint
  getForwardEndpoint: async () => {
    return await ipcRenderer.invoke("get-forward-endpoint");
  },

  // Set the forward endpoint
  setForwardEndpoint: async (url) => {
    return await ipcRenderer.invoke("set-forward-endpoint", url);
  },

  // Get the stored OpenAI API key
  getApiKey: async () => {
    return await ipcRenderer.invoke("get-api-key");
  },

  // Set the OpenAI API key (securely stored)
  setApiKey: async (apiKey) => {
    return await ipcRenderer.invoke("set-api-key", apiKey);
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

  // Listen for terms menu command
  onTermsOpen: (callback) => {
    ipcRenderer.on("open-terms", callback);
  },

  // Remove terms listener
  removeTermsListener: () => {
    ipcRenderer.removeAllListeners("open-terms");
  },
});
