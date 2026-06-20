import { useState, useEffect } from "react";
import { isElectron } from "../../utils/providerHelpers";
import type { ProvidersConfig, ProviderType } from "../../types/provider";
import {
  Save,
  Key,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Lock,
  Unlock,
  Globe,
  KeyRound,
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

interface ProvidersSectionProps {
  /** Called after a successful save + backend restart so the host can refresh
   *  any cached provider state (e.g. the Playground provider selector). */
  onSaved?: () => void;
}

export default function ProvidersSection({ onSaved }: ProvidersSectionProps) {
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
    if (isElectron) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      loadSettings();
    }
  }, []);

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
      setMessage({ type: "success", text: "Saved. Restarting backend…" });
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
      onSaved?.();
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

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <KeyRound className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            Providers
          </h2>
          <p className="text-[13px] text-stone-500">
            API keys and model IDs for each AI provider.
          </p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-10">
          <div className="w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="space-y-5">
          {/* Provider accordion */}
          <div className="rounded-xl ring-1 ring-stone-200 overflow-hidden divide-y divide-stone-100">
            {PROVIDER_ORDER.map((provider) => {
              const info = PROVIDER_INFO[provider];
              const config = providersConfig.providers[provider];
              const isExpanded = expandedProvider === provider;
              const isApiKeyOptional = info.apiKeyOptional === true;
              const configured = config?.hasApiKey;

              return (
                <div key={provider}>
                  {/* Accordion header */}
                  <button
                    onClick={() => toggleProviderExpansion(provider)}
                    className="w-full px-4 py-3 flex items-center justify-between hover:bg-stone-50/70 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      {isExpanded ? (
                        <ChevronDown className="w-4 h-4 text-stone-400" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-stone-400" />
                      )}
                      <span className="font-medium text-stone-700">
                        {info.name}
                      </span>
                    </div>
                    <span
                      className={`inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-full ring-1 ${
                        configured || isApiKeyOptional
                          ? "bg-brand-50 text-brand-700 ring-brand-200"
                          : "bg-stone-50 text-stone-500 ring-stone-200"
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          configured || isApiKeyOptional
                            ? "bg-brand-500"
                            : "bg-stone-300"
                        }`}
                      />
                      {configured
                        ? "Configured"
                        : isApiKeyOptional
                        ? "Key optional"
                        : "Not set"}
                    </span>
                  </button>

                  {/* Accordion content */}
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-1 bg-stone-50/60 space-y-4 animate-fade-in">
                      {/* API Key */}
                      <div>
                        {/* Header row with label, lock toggle, and clear button */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <label className="text-sm font-medium text-stone-600 flex items-center gap-2">
                              <Key className="w-4 h-4" />
                              {info.name} API Key
                              {isApiKeyOptional ? " (optional)" : ""}
                            </label>
                            {configured && (
                              <button
                                onClick={() => toggleProviderLock(provider)}
                                className="p-1 rounded hover:bg-stone-200 transition-colors"
                                title={
                                  unlockedProviders[provider]
                                    ? "Lock API key"
                                    : "Unlock to edit"
                                }
                              >
                                {unlockedProviders[provider] ? (
                                  <Unlock className="w-4 h-4 text-amber-500" />
                                ) : (
                                  <Lock className="w-4 h-4 text-stone-500" />
                                )}
                              </button>
                            )}
                          </div>
                          {configured && (
                            <button
                              onClick={() => handleClearApiKey(provider)}
                              className="text-sm text-amber-600 hover:text-amber-700 transition-colors font-medium"
                            >
                              Clear my key
                            </button>
                          )}
                        </div>

                        {/* Input field - only editable when unlocked or no key exists */}
                        <div className="relative">
                          <input
                            type={
                              unlockedProviders[provider] ? "text" : "password"
                            }
                            value={providerApiKeys[provider]}
                            onChange={(e) =>
                              setProviderApiKeys((prev) => ({
                                ...prev,
                                [provider]: e.target.value,
                              }))
                            }
                            disabled={configured && !unlockedProviders[provider]}
                            placeholder={
                              configured
                                ? unlockedProviders[provider]
                                  ? "Enter new API key to update"
                                  : "API key is configured (unlock to edit)"
                                : isApiKeyOptional
                                ? `Optional ${info.name} API key (${info.placeholder})`
                                : `Enter your ${info.name} API key (${info.placeholder})`
                            }
                            className={`w-full px-3 py-2 rounded-lg font-mono text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 placeholder:text-stone-400 ${
                              configured && !unlockedProviders[provider]
                                ? "bg-stone-100 border border-stone-200 text-stone-500 cursor-not-allowed"
                                : "border border-stone-200 bg-white"
                            }`}
                          />
                          {configured && !unlockedProviders[provider] && (
                            <CheckCircle2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-brand-500" />
                          )}
                        </div>

                        {/* Help link */}
                        {info.helpLink && (
                          <a
                            href={info.helpLink}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-xs text-brand-600 hover:text-brand-700 hover:underline underline-offset-2 transition-colors mt-1.5"
                          >
                            How to get your {info.name} API key?
                          </a>
                        )}
                      </div>

                      {/* Model Override */}
                      <div>
                        <label className="text-sm font-medium text-stone-600 mb-2 flex items-center gap-2">
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
                          className="w-full px-3 py-2 rounded-lg border border-stone-200 bg-white font-mono text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 placeholder:text-stone-400"
                        />
                        <p className="text-xs text-stone-500 mt-1.5">
                          {info.modelHelpText || `Default: ${info.defaultModel}`}
                        </p>
                      </div>

                      {/* Custom Endpoint URL */}
                      {PROVIDERS_WITH_CUSTOM_ENDPOINT.has(provider) && (
                        <div>
                          <label className="text-sm font-medium text-stone-600 mb-2 flex items-center gap-2">
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
                            className="w-full px-3 py-2 rounded-lg border border-stone-200 bg-white font-mono text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 placeholder:text-stone-400"
                          />
                          {info.endpointHelpText ? (
                            <p className="flex items-start gap-2 rounded-lg ring-1 ring-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 mt-2">
                              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                              <span>{info.endpointHelpText}</span>
                            </p>
                          ) : (
                            <p className="text-xs text-stone-500 mt-1.5">
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

          <p className="flex items-center gap-1.5 text-xs text-stone-500">
            <Lock className="w-3.5 h-3.5" />
            Your API keys are stored securely using system keychain encryption.
          </p>

          {/* Message */}
          {message && (
            <div
              className={`flex items-center gap-2 p-3 rounded-xl ring-1 ${
                message.type === "success"
                  ? "bg-brand-50 text-brand-800 ring-brand-200"
                  : "bg-red-50 text-red-800 ring-red-200"
              }`}
            >
              {message.type === "success" ? (
                <CheckCircle2 className="w-5 h-5 shrink-0" />
              ) : (
                <AlertCircle className="w-5 h-5 shrink-0" />
              )}
              <span className="text-sm">{message.text}</span>
            </div>
          )}

          {/* Save */}
          <div className="flex justify-end">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="btn-brand inline-flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed font-medium tracking-tight"
            >
              {isSaving ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Saving…
                </>
              ) : (
                <>
                  <Save className="w-5 h-5" />
                  Save providers
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
