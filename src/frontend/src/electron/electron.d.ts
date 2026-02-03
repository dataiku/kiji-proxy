// TypeScript declarations for Electron API exposed via preload script

// Provider types for multi-provider support
type ProviderType = "openai" | "anthropic" | "gemini" | "mistral";

interface ProviderSettings {
  hasApiKey: boolean;
  model: string; // Custom model or empty string for default
}

interface ProvidersConfig {
  activeProvider: ProviderType;
  providers: Record<ProviderType, ProviderSettings>;
}

interface ElectronAPI {
  // Legacy methods (delegate to active provider for backwards compatibility)
  getApiKey: () => Promise<string | null>;
  setApiKey: (apiKey: string) => Promise<{ success: boolean; error?: string }>;

  // Multi-provider methods
  getActiveProvider: () => Promise<ProviderType>;
  setActiveProvider: (
    provider: ProviderType
  ) => Promise<{ success: boolean; error?: string }>;
  getProviderApiKey: (provider: ProviderType) => Promise<string | null>;
  setProviderApiKey: (
    provider: ProviderType,
    apiKey: string
  ) => Promise<{ success: boolean; error?: string }>;
  getProviderModel: (provider: ProviderType) => Promise<string>;
  setProviderModel: (
    provider: ProviderType,
    model: string
  ) => Promise<{ success: boolean; error?: string }>;
  getProvidersConfig: () => Promise<ProvidersConfig>;

  // Other settings
  getCACertSetupDismissed: () => Promise<boolean>;
  setCACertSetupDismissed: (
    dismissed: boolean
  ) => Promise<{ success: boolean; error?: string }>;
  getTermsAccepted: () => Promise<boolean>;
  setTermsAccepted: (
    accepted: boolean
  ) => Promise<{ success: boolean; error?: string }>;
  getWelcomeDismissed: () => Promise<boolean>;
  setWelcomeDismissed: (
    dismissed: boolean
  ) => Promise<{ success: boolean; error?: string }>;

  // Platform and version info
  platform: string;
  versions: {
    node: string;
    chrome: string;
    electron: string;
  };

  // Event listeners
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
