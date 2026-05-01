import { useState, useEffect } from "react";
import { isElectron } from "../../utils/providerHelpers";
import type { ProvidersConfig, ProviderType } from "../../types/provider";
import {
  X,
  Save,
  Key,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Settings2,
  Lock,
  Unlock,
  Globe,
} from "lucide-react";

// Providers that support a user-configurable custom endpoint URL.
const PROVIDERS_WITH_CUSTOM_ENDPOINT: ReadonlySet<ProviderType> = new Set([
  "custom",
]);

// Provider display information
const PROVIDER_INFO: Record<
  ProviderType,
  {
    name: string;
    defaultModel: string;
    placeholder: string;
    helpLink?: string;
    baseUrlPlaceholder?: string;
    modelHelpText?: string;
    endpointHelpText?: string;
    apiKeyOptional?: boolean;
  }
> = {
  openai: {
    name: "OpenAI",
    defaultModel: "gpt-4o-mini",
    placeholder: "sk-...",
    helpLink:
      "https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key",
  },
  anthropic: {
    name: "Anthropic",
    defaultModel: "claude-haiku-4-5",
    placeholder: "sk-ant-...",
    helpLink: "https://platform.claude.com/docs/en/get-started",
  },
  gemini: {
    name: "Gemini",
    defaultModel: "gemini-flash-latest",
    placeholder: "AIza...",
    helpLink: "https://ai.google.dev/gemini-api/docs/api-key",
  },
  mistral: {
    name: "Mistral",
    defaultModel: "mistral-small-latest",
    placeholder: "...",
    helpLink: "https://console.mistral.ai/api-keys",
  },
  custom: {
    name: "Custom Provider",
    defaultModel: "your-model-id",
    placeholder: "...",
    baseUrlPlaceholder: "https://api.example.com/v1",
    apiKeyOptional: true,
    modelHelpText: "Use the exact model ID expected by your provider.",
    endpointHelpText:
      "Your custom provider must support an OpenAI-compliant chat completions API.",
  },
};

const PROVIDER_ORDER: ProviderType[] = [
  "openai",
  "anthropic",
  "gemini",
  "mistral",
  "custom",
];

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenAdvancedSettings: () => void;
}

export default function SettingsModal({
  isOpen,
  onClose,
  onOpenAdvancedSettings,
}: SettingsModalProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  // Provider state
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

  // Expanded accordion state
  const [expandedProvider, setExpandedProvider] = useState<ProviderType | null>(
    null
  );

  // Track which providers have their API key unlocked (visible/editable)
  const [unlockedProviders, setUnlockedProviders] = useState<
    Record<ProviderType, boolean>
  >({
    openai: false,
    anthropic: false,
    gemini: false,
    mistral: false,
    custom: false,
  });

  // Form state for each provider (API key inputs and model overrides)
  const [providerApiKeys, setProviderApiKeys] = useState<
    Record<ProviderType, string>
  >({
    openai: "",
    anthropic: "",
    gemini: "",
    mistral: "",
    custom: "",
  });

  const [providerModels, setProviderModels] = useState<
    Record<ProviderType, string>
  >({
    openai: "",
    anthropic: "",
    gemini: "",
    mistral: "",
    custom: "",
  });

  const [providerBaseUrls, setProviderBaseUrls] = useState<
    Record<ProviderType, string>
  >({
    openai: "",
    anthropic: "",
    gemini: "",
    mistral: "",
    custom: "",
  });

  const loadSettings = async () => {
    if (!window.electronAPI) return;

    setIsLoading(true);
    try {
      const config = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(config);

      // Load models and base URLs from config
      const models: Record<ProviderType, string> = {
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
        custom: "",
      };
      const baseUrls: Record<ProviderType, string> = {
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
        custom: "",
      };
      for (const provider of PROVIDER_ORDER) {
        models[provider] = config.providers[provider]?.model || "";
        baseUrls[provider] = config.providers[provider]?.baseUrl || "";
      }
      setProviderModels(models);
      setProviderBaseUrls(baseUrls);

      // Clear API key inputs
      setProviderApiKeys({
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
        custom: "",
      });
    } catch (error) {
      console.error("Error loading settings:", error);
      setMessage({ type: "error", text: "Failed to load settings" });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && isElectron) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      loadSettings();
    }
  }, [isOpen]);

  const handleSave = async () => {
    if (!window.electronAPI) return;

    setIsSaving(true);
    setMessage(null);

    try {
      // Save API keys and models for each provider
      for (const provider of PROVIDER_ORDER) {
        // Save API key if provided
        if (providerApiKeys[provider].trim()) {
          const keyResult = await window.electronAPI.setProviderApiKey(
            provider,
            providerApiKeys[provider].trim()
          );
          if (!keyResult.success) {
            setMessage({
              type: "error",
              text:
                keyResult.error ||
                `Failed to save ${PROVIDER_INFO[provider].name} API key`,
            });
            setIsSaving(false);
            return;
          }
        }

        // Save model override
        const modelResult = await window.electronAPI.setProviderModel(
          provider,
          providerModels[provider].trim()
        );
        if (!modelResult.success) {
          setMessage({
            type: "error",
            text:
              modelResult.error ||
              `Failed to save ${PROVIDER_INFO[provider].name} model`,
          });
          setIsSaving(false);
          return;
        }

        // Save custom base URL (only meaningful for providers that expose it,
        // but the backend stores it generically per provider)
        if (PROVIDERS_WITH_CUSTOM_ENDPOINT.has(provider)) {
          const baseUrlResult = await window.electronAPI.setProviderBaseUrl(
            provider,
            providerBaseUrls[provider].trim()
          );
          if (!baseUrlResult.success) {
            setMessage({
              type: "error",
              text:
                baseUrlResult.error ||
                `Failed to save ${PROVIDER_INFO[provider].name} endpoint URL`,
            });
            setIsSaving(false);
            return;
          }
        }
      }

      // Reload config to update hasApiKey status
      const updatedConfig = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(updatedConfig);

      // Clear API key inputs after successful save
      setProviderApiKeys({
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
        custom: "",
      });

      // Restart the backend so the new API keys / endpoint URLs take effect.
      // Without this, the Go process keeps using whatever env vars it spawned with.
      setMessage({ type: "success", text: "Saved. Restarting backend..." });
      const restartResult = await window.electronAPI.restartBackend();
      if (!restartResult.success) {
        setMessage({
          type: "error",
          text:
            restartResult.error ||
            "Settings saved, but backend restart failed. Restart the app to apply.",
        });
        setIsSaving(false);
        return;
      }

      setMessage({ type: "success", text: "Settings saved and applied!" });

      setTimeout(() => {
        onClose();
      }, 1000);
    } catch (error) {
      console.error("Error saving settings:", error);
      setMessage({ type: "error", text: "Failed to save settings" });
    } finally {
      setIsSaving(false);
    }
  };

  const handleClearApiKey = async (provider: ProviderType) => {
    if (!window.electronAPI) return;

    setIsSaving(true);
    setMessage(null);

    try {
      const result = await window.electronAPI.setProviderApiKey(provider, "");
      if (result.success) {
        // Update local state
        setProvidersConfig((prev) => ({
          ...prev,
          providers: {
            ...prev.providers,
            [provider]: { ...prev.providers[provider], hasApiKey: false },
          },
        }));
        setProviderApiKeys((prev) => ({ ...prev, [provider]: "" }));
        setMessage({
          type: "success",
          text: `${PROVIDER_INFO[provider].name} API key cleared`,
        });
      } else {
        setMessage({
          type: "error",
          text: result.error || "Failed to clear API key",
        });
      }
    } catch (error) {
      console.error("Error clearing API key:", error);
      setMessage({ type: "error", text: "Failed to clear API key" });
    } finally {
      setIsSaving(false);
    }
  };

  const toggleProviderLock = (provider: ProviderType) => {
    setUnlockedProviders((prev) => ({
      ...prev,
      [provider]: !prev[provider],
    }));
  };

  const toggleProviderExpansion = (provider: ProviderType) => {
    setExpandedProvider(expandedProvider === provider ? null : provider);
  };

  if (!isOpen) return null;

  if (!isElectron) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-slate-800">Settings</h2>
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-700 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <p className="text-slate-600">
            Settings are only available in Electron mode.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-lg w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-slate-800">Settings</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="space-y-6">
            {/* Provider Settings Accordion */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-3">
                Provider Settings
              </label>
              <div className="border-2 border-slate-200 rounded-lg overflow-hidden">
                {PROVIDER_ORDER.map((provider, index) => {
                  const info = PROVIDER_INFO[provider];
                  const config = providersConfig.providers[provider];
                  const isExpanded = expandedProvider === provider;
                  const isApiKeyOptional = info.apiKeyOptional === true;

                  return (
                    <div
                      key={provider}
                      className={index > 0 ? "border-t border-slate-200" : ""}
                    >
                      {/* Accordion Header */}
                      <button
                        onClick={() => toggleProviderExpansion(provider)}
                        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-50 transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          {isExpanded ? (
                            <ChevronDown className="w-4 h-4 text-slate-500" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-slate-500" />
                          )}
                          <span className="font-medium text-slate-700">
                            {info.name}
                          </span>
                        </div>
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            config?.hasApiKey
                              ? "bg-green-100 text-green-700"
                              : isApiKeyOptional
                              ? "bg-blue-100 text-blue-700"
                              : "bg-slate-100 text-slate-500"
                          }`}
                        >
                          {config?.hasApiKey
                            ? "Configured"
                            : isApiKeyOptional
                            ? "Key Optional"
                            : "Not Set"}
                        </span>
                      </button>

                      {/* Accordion Content */}
                      {isExpanded && (
                        <div className="px-4 pb-4 pt-2 bg-slate-50 space-y-4">
                          {/* API Key */}
                          <div>
                            {/* Header row with label, lock toggle, and clear button */}
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <label className="text-sm font-medium text-slate-600 flex items-center gap-2">
                                  <Key className="w-4 h-4" />
                                  {info.name} API Key
                                  {isApiKeyOptional ? " (optional)" : ""}
                                </label>
                                {config?.hasApiKey && (
                                  <button
                                    onClick={() => toggleProviderLock(provider)}
                                    className="p-1 rounded hover:bg-slate-200 transition-colors"
                                    title={
                                      unlockedProviders[provider]
                                        ? "Lock API key"
                                        : "Unlock to edit"
                                    }
                                  >
                                    {unlockedProviders[provider] ? (
                                      <Unlock className="w-4 h-4 text-orange-500" />
                                    ) : (
                                      <Lock className="w-4 h-4 text-slate-500" />
                                    )}
                                  </button>
                                )}
                              </div>
                              {config?.hasApiKey && (
                                <button
                                  onClick={() => handleClearApiKey(provider)}
                                  className="text-sm text-orange-500 hover:text-orange-600 transition-colors font-medium"
                                >
                                  Clear my key
                                </button>
                              )}
                            </div>

                            {/* Input field - only editable when unlocked or no key exists */}
                            <div className="relative">
                              <input
                                type={
                                  unlockedProviders[provider]
                                    ? "text"
                                    : "password"
                                }
                                value={providerApiKeys[provider]}
                                onChange={(e) =>
                                  setProviderApiKeys((prev) => ({
                                    ...prev,
                                    [provider]: e.target.value,
                                  }))
                                }
                                disabled={
                                  config?.hasApiKey &&
                                  !unlockedProviders[provider]
                                }
                                placeholder={
                                  config?.hasApiKey
                                    ? unlockedProviders[provider]
                                      ? "Enter new API key to update"
                                      : "API key is configured (unlock to edit)"
                                    : isApiKeyOptional
                                    ? `Optional ${info.name} API key (${info.placeholder})`
                                    : `Enter your ${info.name} API key (${info.placeholder})`
                                }
                                className={`w-full px-3 py-2 border rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm placeholder:text-gray-400 ${
                                  config?.hasApiKey &&
                                  !unlockedProviders[provider]
                                    ? "bg-slate-100 border-slate-200 text-slate-500 cursor-not-allowed"
                                    : "border-slate-300 bg-white"
                                }`}
                              />
                              {config?.hasApiKey &&
                                !unlockedProviders[provider] && (
                                  <CheckCircle2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-green-500" />
                                )}
                            </div>

                            {/* Help link */}
                            {info.helpLink && (
                              <a
                                href={info.helpLink}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 hover:underline transition-colors mt-1"
                              >
                                How to get your {info.name} API key?
                              </a>
                            )}
                          </div>

                          {/* Model Override */}
                          <div>
                            <label className="block text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                              <Cpu className="w-4 h-4" />
                              Model ID
                            </label>
                            <input
                              type="text"
                              value={providerModels[provider]}
                              onChange={(e) =>
                                setProviderModels((prev) => ({
                                  ...prev,
                                  [provider]: e.target.value,
                                }))
                              }
                              placeholder={info.defaultModel}
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm placeholder:text-gray-400"
                            />
                            <p className="text-xs text-slate-500 mt-1">
                              {info.modelHelpText || `Default: ${info.defaultModel}`}
                            </p>
                          </div>

                          {/* Custom Endpoint URL */}
                          {PROVIDERS_WITH_CUSTOM_ENDPOINT.has(provider) && (
                            <div>
                              <label className="block text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                                <Globe className="w-4 h-4" />
                                Custom Endpoint URL
                              </label>
                              <input
                                type="url"
                                value={providerBaseUrls[provider]}
                                onChange={(e) =>
                                  setProviderBaseUrls((prev) => ({
                                    ...prev,
                                    [provider]: e.target.value,
                                  }))
                                }
                                placeholder={
                                  info.baseUrlPlaceholder ||
                                  "https://api.example.com/v1"
                                }
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm placeholder:text-gray-400"
                              />
                              {info.endpointHelpText ? (
                                <p className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 mt-2">
                                  <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                                  <span>{info.endpointHelpText}</span>
                                </p>
                              ) : (
                                <p className="text-xs text-slate-500 mt-1">
                                  Override to use a custom endpoint.
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Your API keys are stored securely using system keychain
                encryption.
              </p>
            </div>

            {/* Advanced Settings Link */}
            <div
              onClick={() => {
                onOpenAdvancedSettings();
                onClose();
              }}
              className="border-2 border-slate-200 rounded-lg p-4 hover:border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Settings2 className="w-5 h-5 text-slate-600" />
                  <div>
                    <p className="font-medium text-slate-700">
                      Advanced Settings
                    </p>
                    <p className="text-xs text-slate-500">
                      Load custom PII model
                    </p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-slate-400" />
              </div>
            </div>

            {/* Message */}
            {message && (
              <div
                className={`flex items-center gap-2 p-3 rounded-lg ${
                  message.type === "success"
                    ? "bg-green-50 text-green-800 border border-green-200"
                    : "bg-red-50 text-red-800 border border-red-200"
                }`}
              >
                {message.type === "success" ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
                <span className="text-sm">{message.text}</span>
              </div>
            )}

            {/* Actions */}
            <div className="pt-4">
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium w-full"
              >
                {isSaving ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Settings
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
