import { useState, useEffect, useCallback, useRef } from "react";
import type { ProviderType, ProvidersConfig } from "../types/provider";
import { isElectron } from "../utils/providerHelpers";

interface ModalCallbacks {
  onSettingsOpen: () => void;
  onAboutOpen: () => void;
}

export function useElectronSettings(callbacks: ModalCallbacks) {
  const [activeProvider, setActiveProvider] = useState<ProviderType>("openai");
  const [providersConfig, setProvidersConfig] = useState<ProvidersConfig>({
    activeProvider: "openai",
    providers: {
      openai: { hasApiKey: false, model: "" },
      anthropic: { hasApiKey: false, model: "" },
      gemini: { hasApiKey: false, model: "" },
      mistral: { hasApiKey: false, model: "" },
      custom: { hasApiKey: false, model: "", baseUrl: "" },
    },
  });
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [isWelcomeOpen, setIsWelcomeOpen] = useState(false);

  // Keep callbacks in a ref so IPC listeners always call the latest version
  const callbacksRef = useRef(callbacks);
  useEffect(() => {
    callbacksRef.current = callbacks;
  });

  const loadSettings = useCallback(() => {
    if (!window.electronAPI) return;

    window.electronAPI
      .getProvidersConfig()
      .then((config) => {
        setProvidersConfig(config);
        setActiveProvider(config.activeProvider);
        return window.electronAPI!.getProviderApiKey(config.activeProvider);
      })
      .then((key) => {
        setApiKey(key);
      })
      .catch((error) => {
        console.error("Error loading settings:", error);
      });
  }, []);

  const switchProvider = useCallback(async (newProvider: ProviderType) => {
    setActiveProvider(newProvider);
    if (window.electronAPI) {
      await window.electronAPI.setActiveProvider(newProvider);
      const key = await window.electronAPI.getProviderApiKey(newProvider);
      setApiKey(key);
    }
  }, []);

  // Load settings on mount and listen for Electron menu commands
  useEffect(() => {
    if (isElectron && window.electronAPI) {
      loadSettings();

      window.electronAPI.getWelcomeDismissed().then((dismissed) => {
        if (!dismissed) {
          setTimeout(() => {
            setIsWelcomeOpen(true);
          }, 500);
        }
      });

      if (window.electronAPI.onSettingsOpen) {
        window.electronAPI.onSettingsOpen(() => {
          callbacksRef.current.onSettingsOpen();
        });
      }

      if (window.electronAPI.onAboutOpen) {
        window.electronAPI.onAboutOpen(() => {
          callbacksRef.current.onAboutOpen();
        });
      }

      return () => {
        if (window.electronAPI?.removeSettingsListener) {
          window.electronAPI.removeSettingsListener();
        }
        if (window.electronAPI?.removeAboutListener) {
          window.electronAPI.removeAboutListener();
        }
      };
    }

    return undefined;
  }, [loadSettings]);

  const closeWelcome = useCallback(() => {
    setIsWelcomeOpen(false);
  }, []);

  return {
    activeProvider,
    providersConfig,
    apiKey,
    isWelcomeOpen,
    loadSettings,
    switchProvider,
    closeWelcome,
  };
}
