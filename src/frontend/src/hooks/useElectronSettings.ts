import { useState, useEffect, useCallback, useRef } from "react";
import type { ProviderType, ProvidersConfig } from "../types/provider";

interface ModalCallbacks {
  onSettingsOpen: () => void;
  onAboutOpen: () => void;
  onTermsOpen: () => void;
  onTourStart: () => void;
}

export function useElectronSettings(
  isElectron: boolean,
  callbacks: ModalCallbacks
) {
  const [activeProvider, setActiveProvider] = useState<ProviderType>("openai");
  const [providersConfig, setProvidersConfig] = useState<ProvidersConfig>({
    activeProvider: "openai",
    providers: {
      openai: { hasApiKey: false, model: "" },
      anthropic: { hasApiKey: false, model: "" },
      gemini: { hasApiKey: false, model: "" },
      mistral: { hasApiKey: false, model: "" },
    },
  });
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [termsRequireAcceptance, setTermsRequireAcceptance] = useState(false);
  const [isTermsOpen, setIsTermsOpen] = useState(false);
  const [isWelcomeOpen, setIsWelcomeOpen] = useState(false);
  const [welcomeModalJustClosed, setWelcomeModalJustClosed] = useState(false);

  // Keep callbacks in a ref so IPC listeners always call the latest version
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  const loadSettings = useCallback(async () => {
    if (!window.electronAPI) return;

    try {
      const config = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(config);
      setActiveProvider(config.activeProvider);

      const key = await window.electronAPI.getProviderApiKey(
        config.activeProvider
      );
      setApiKey(key);
    } catch (error) {
      console.error("Error loading settings:", error);
    }
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

      window.electronAPI.getTermsAccepted().then((accepted) => {
        if (!accepted) {
          setTermsRequireAcceptance(true);
          setIsTermsOpen(true);
        }
      });

      window.electronAPI.getWelcomeDismissed().then((dismissed) => {
        if (!dismissed) {
          setTimeout(() => {
            setIsWelcomeOpen(true);
          }, 500);
        } else {
          setWelcomeModalJustClosed(true);
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

      if (window.electronAPI.onTermsOpen) {
        window.electronAPI.onTermsOpen(() => {
          setTermsRequireAcceptance(false);
          setIsTermsOpen(true);
        });
      }

      if (window.electronAPI.onTourOpen) {
        window.electronAPI.onTourOpen(() => {
          callbacksRef.current.onTourStart();
        });
      }

      return () => {
        if (window.electronAPI?.removeSettingsListener) {
          window.electronAPI.removeSettingsListener();
        }
        if (window.electronAPI?.removeAboutListener) {
          window.electronAPI.removeAboutListener();
        }
        if (window.electronAPI?.removeTermsListener) {
          window.electronAPI.removeTermsListener();
        }
        if (window.electronAPI?.removeTourListener) {
          window.electronAPI.removeTourListener();
        }
      };
    }

    // In web mode, no WelcomeModal persistence — just try to start tour
    if (!isElectron) {
      setWelcomeModalJustClosed(true);
    }

    return undefined;
  }, [isElectron, loadSettings]);

  const closeTerms = useCallback(() => {
    setIsTermsOpen(false);
    setTermsRequireAcceptance(false);
  }, []);

  const closeWelcome = useCallback(() => {
    setIsWelcomeOpen(false);
    setWelcomeModalJustClosed(true);
  }, []);

  return {
    activeProvider,
    providersConfig,
    apiKey,
    termsRequireAcceptance,
    isTermsOpen,
    isWelcomeOpen,
    welcomeModalJustClosed,
    loadSettings,
    switchProvider,
    closeTerms,
    closeWelcome,
  };
}
