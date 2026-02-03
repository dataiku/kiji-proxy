import { useState, useEffect } from "react";
import {
  X,
  Save,
  Key,
  Server,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Cpu,
  FolderOpen,
} from "lucide-react";

type ProviderType = "openai" | "anthropic" | "gemini" | "mistral";

interface ProviderSettings {
  hasApiKey: boolean;
  model: string;
}

interface ProvidersConfig {
  activeProvider: ProviderType;
  providers: Record<ProviderType, ProviderSettings>;
}

// Provider display information
const PROVIDER_INFO: Record<
  ProviderType,
  { name: string; defaultModel: string; placeholder: string }
> = {
  openai: {
    name: "OpenAI",
    defaultModel: "gpt-3.5-turbo",
    placeholder: "sk-...",
  },
  anthropic: {
    name: "Anthropic",
    defaultModel: "claude-3-haiku-20240307",
    placeholder: "sk-ant-...",
  },
  gemini: {
    name: "Gemini",
    defaultModel: "gemini-flash-latest",
    placeholder: "AIza...",
  },
  mistral: {
    name: "Mistral",
    defaultModel: "mistral-small-latest",
    placeholder: "...",
  },
};

const PROVIDER_ORDER: ProviderType[] = [
  "openai",
  "anthropic",
  "gemini",
  "mistral",
];

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
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
    },
  });

  // Expanded accordion state
  const [expandedProvider, setExpandedProvider] = useState<ProviderType | null>(
    null
  );

  // Form state for each provider (API key inputs and model overrides)
  const [providerApiKeys, setProviderApiKeys] = useState<
    Record<ProviderType, string>
  >({
    openai: "",
    anthropic: "",
    gemini: "",
    mistral: "",
  });

  const [providerModels, setProviderModels] = useState<
    Record<ProviderType, string>
  >({
    openai: "",
    anthropic: "",
    gemini: "",
    mistral: "",
  });

  // Model directory state
  const [modelDirectory, setModelDirectory] = useState("");
  const [_hasModelDirectory, setHasModelDirectory] = useState(false);
  const [modelInfo, setModelInfo] = useState<{
    healthy: boolean;
    directory?: string;
    error?: string;
  } | null>(null);
  const [isReloading, setIsReloading] = useState(false);
  const [reloadMessage, setReloadMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

  useEffect(() => {
    if (isOpen && isElectron) {
      loadSettings();
      loadModelInfo();
    }
  }, [isOpen, isElectron]);

  const loadSettings = async () => {
    if (!window.electronAPI) return;

    setIsLoading(true);
    try {
      const config = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(config);
      // TODO: CODE REVIEW DECISION: CAN WE COLLAPSE ALL PROVIDERS INITIALLY TO SAVE SPACE?
      // setExpandedProvider(config.activeProvider);

      // Load models from config
      const models: Record<ProviderType, string> = {
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
      };
      for (const provider of PROVIDER_ORDER) {
        models[provider] = config.providers[provider]?.model || "";
      }
      setProviderModels(models);

      // Clear API key inputs
      setProviderApiKeys({
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
      });
    } catch (error) {
      console.error("Error loading settings:", error);
      setMessage({ type: "error", text: "Failed to load settings" });
    } finally {
      setIsLoading(false);
    }
  };

  const loadModelInfo = async () => {
    if (!window.electronAPI) return;

    try {
      const [storedDir, info] = await Promise.all([
        window.electronAPI.getModelDirectory(),
        window.electronAPI.getModelInfo(),
      ]);

      setHasModelDirectory(!!storedDir);
      setModelDirectory(storedDir || "");
      setModelInfo(info);
    } catch (error) {
      console.error("Error loading model info:", error);
    }
  };

  const handleReloadModel = async () => {
    if (!window.electronAPI || !modelDirectory.trim()) return;

    setIsReloading(true);
    setReloadMessage(null);

    try {
      // First, save the directory to config
      const saveResult = await window.electronAPI.setModelDirectory(
        modelDirectory.trim()
      );

      if (!saveResult.success) {
        setReloadMessage({
          type: "error",
          text: saveResult.error || "Failed to save model directory",
        });
        setIsReloading(false);
        return;
      }

      setHasModelDirectory(true);

      // Then, reload the model
      const result = await window.electronAPI.reloadModel(
        modelDirectory.trim()
      );

      if (result.success) {
        setReloadMessage({
          type: "success",
          text: "Model saved and reloaded successfully!",
        });
        await loadModelInfo();
      } else {
        setReloadMessage({
          type: "error",
          text: result.error || "Failed to reload model",
        });
      }
    } catch (error) {
      console.error("Error reloading model:", error);
      setReloadMessage({
        type: "error",
        text: error instanceof Error ? error.message : "Unknown error",
      });
    } finally {
      setIsReloading(false);
    }
  };

  const handleBrowseModelDirectory = async () => {
    if (!window.electronAPI) return;

    try {
      const selectedPath = await window.electronAPI.selectModelDirectory();
      if (selectedPath) {
        setModelDirectory(selectedPath);
      }
    } catch (error) {
      console.error("Error selecting model directory:", error);
      setMessage({ type: "error", text: "Failed to open folder selector" });
    }
  };

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
      }

      setMessage({ type: "success", text: "Settings saved successfully!" });

      // Reload config to update hasApiKey status
      const updatedConfig = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(updatedConfig);

      // Clear API key inputs after successful save
      setProviderApiKeys({
        openai: "",
        anthropic: "",
        gemini: "",
        mistral: "",
      });

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
                              : "bg-slate-100 text-slate-500"
                          }`}
                        >
                          {config?.hasApiKey ? "Configured" : "Not Set"}
                        </span>
                      </button>

                      {/* Accordion Content */}
                      {isExpanded && (
                        <div className="px-4 pb-4 pt-2 bg-slate-50 space-y-4">
                          {/* API Key */}
                          <div>
                            <label className="block text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                              <Key className="w-4 h-4" />
                              API Key
                            </label>
                            {config?.hasApiKey &&
                              !providerApiKeys[provider] && (
                                <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 p-2 rounded mb-2">
                                  <CheckCircle2 className="w-4 h-4" />
                                  <span>API key is configured</span>
                                </div>
                              )}
                            <input
                              type="password"
                              value={providerApiKeys[provider]}
                              onChange={(e) =>
                                setProviderApiKeys((prev) => ({
                                  ...prev,
                                  [provider]: e.target.value,
                                }))
                              }
                              placeholder={
                                config?.hasApiKey
                                  ? "Enter new API key to update"
                                  : `Enter your ${info.name} API key (${info.placeholder})`
                              }
                              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm placeholder:text-gray-400"
                            />
                            {config?.hasApiKey && (
                              <button
                                onClick={() => handleClearApiKey(provider)}
                                className="text-sm text-red-600 hover:text-red-700 transition-colors mt-1"
                              >
                                Clear API key
                              </button>
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
                              Default: {info.defaultModel}
                            </p>
                          </div>
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

            {/* Load Custom Yaak PII Model */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                <Server className="w-4 h-4" />
                Load Custom Yaak PII Model
              </label>

              {/* Current Model Info */}
              {modelInfo && (
                <div
                  className={`mb-2 p-2 rounded ${
                    modelInfo.healthy
                      ? "bg-green-50 border border-green-200"
                      : "bg-red-50 border border-red-200"
                  }`}
                >
                  <div className="text-xs">
                    <span
                      className={
                        modelInfo.healthy ? "text-green-700" : "text-red-700"
                      }
                    >
                      Status: {modelInfo.healthy ? "Healthy" : "Unhealthy"}
                    </span>
                    {modelInfo.directory && (
                      <div className="text-slate-600 mt-1 break-all">
                        Current: {modelInfo.directory}
                      </div>
                    )}
                    {modelInfo.error && (
                      <div className="text-red-700 mt-1 break-all">
                        Error: {modelInfo.error}
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <input
                  type="text"
                  value={modelDirectory}
                  onChange={(e) => setModelDirectory(e.target.value)}
                  placeholder="/path/to/model/directory"
                  className="flex-1 px-4 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none font-mono text-sm placeholder:text-gray-400"
                />
                <button
                  onClick={handleBrowseModelDirectory}
                  className="px-4 py-2 bg-slate-100 border-2 border-slate-200 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors flex items-center gap-2"
                  title="Browse for folder"
                >
                  <FolderOpen className="w-4 h-4" />
                  Browse
                </button>
              </div>

              <p className="text-xs text-slate-500 mt-1">
                Directory must contain: model_quantized.onnx, tokenizer.json,
                label_mappings.json
              </p>

              {/* Action Button */}
              <div className="mt-2">
                <button
                  onClick={handleReloadModel}
                  disabled={isReloading || !modelDirectory.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm transition-colors w-full"
                >
                  {isReloading ? "Reloading..." : "Save & Reload Model"}
                </button>
              </div>

              {/* Reload Message */}
              {reloadMessage && (
                <div
                  className={`mt-2 p-2 rounded text-sm ${
                    reloadMessage.type === "success"
                      ? "bg-green-50 text-green-800 border border-green-200"
                      : "bg-red-50 text-red-800 border border-red-200"
                  }`}
                >
                  {reloadMessage.text}
                </div>
              )}
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
            <div className="flex gap-3 pt-4">
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium flex-1"
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
              <button
                onClick={onClose}
                className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
