// TypeScript declarations for Electron API exposed via preload script

interface ElectronAPI {
  getForwardEndpoint: () => Promise<string>;
  setForwardEndpoint: (
    url: string
  ) => Promise<{ success: boolean; error?: string }>;
  getApiKey: () => Promise<string | null>;
  setApiKey: (apiKey: string) => Promise<{ success: boolean; error?: string }>;
  platform: string;
  versions: {
    node: string;
    chrome: string;
    electron: string;
  };
  onSettingsOpen: (callback: () => void) => void;
  removeSettingsListener: () => void;
  onAboutOpen: (callback: () => void) => void;
  removeAboutListener: () => void;
}

interface Window {
  electronAPI?: ElectronAPI;
}
