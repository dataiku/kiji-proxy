// TypeScript declarations for Electron API exposed via preload script

interface ElectronAPI {
  getForwardEndpoint: () => Promise<string>;
  setForwardEndpoint: (
    url: string
  ) => Promise<{ success: boolean; error?: string }>;
  getApiKey: () => Promise<string | null>;
  setApiKey: (apiKey: string) => Promise<{ success: boolean; error?: string }>;
  getCACertSetupDismissed: () => Promise<boolean>;
  setCACertSetupDismissed: (
    dismissed: boolean
  ) => Promise<{ success: boolean; error?: string }>;
  getTermsAccepted: () => Promise<boolean>;
  setTermsAccepted: (
    accepted: boolean
  ) => Promise<{ success: boolean; error?: string }>;
  getModelDirectory: () => Promise<string | null>;
  setModelDirectory: (
    path: string
  ) => Promise<{ success: boolean; error?: string }>;
  getModelInfo: () => Promise<{
    healthy: boolean;
    directory?: string;
    error?: string;
  }>;
  reloadModel: (path: string) => Promise<{ success: boolean; error?: string }>;
  selectModelDirectory: () => Promise<string | null>;
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
  onTermsOpen: (callback: () => void) => void;
  removeTermsListener: () => void;
}

interface Window {
  electronAPI?: ElectronAPI;
}
