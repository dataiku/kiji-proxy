import { useState, useEffect } from "react";
import { Settings2, Server, FolderOpen, Shield, AlertTriangle } from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface AdvancedSectionProps {
  /** Opens the shared CA certificate setup wizard (owned by SettingsView). */
  onOpenCACert: () => void;
}

export default function AdvancedSection({ onOpenCACert }: AdvancedSectionProps) {
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

  // Transparent proxy state
  const [transparentProxyEnabled, setTransparentProxyEnabled] = useState(false);
  const [isTogglingProxy, setIsTogglingProxy] = useState(false);

  const loadTransparentProxySetting = async () => {
    if (!window.electronAPI) return;

    try {
      const enabled = await window.electronAPI.getTransparentProxyEnabled();
      setTransparentProxyEnabled(enabled);
    } catch (error) {
      console.error("Error loading transparent proxy setting:", error);
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

  useEffect(() => {
    if (isElectron) {
      /* eslint-disable react-hooks/set-state-in-effect */
      loadModelInfo();
      loadTransparentProxySetting();
      /* eslint-enable react-hooks/set-state-in-effect */
    }
  }, []);

  const handleToggleTransparentProxy = async () => {
    if (!window.electronAPI) return;

    const newValue = !transparentProxyEnabled;

    // If enabling, show CA cert setup wizard first
    if (newValue) {
      onOpenCACert();
    }

    setIsTogglingProxy(true);
    try {
      const result = await window.electronAPI.setTransparentProxyEnabled(
        newValue
      );
      if (result.success) {
        setTransparentProxyEnabled(newValue);
      }
    } catch (error) {
      console.error("Error toggling transparent proxy:", error);
    } finally {
      setIsTogglingProxy(false);
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
      const result = await window.electronAPI.reloadModel(modelDirectory.trim());

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
      setReloadMessage({
        type: "error",
        text: "Failed to open folder selector",
      });
    }
  };

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <Settings2 className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            Advanced
          </h2>
          <p className="text-[13px] text-stone-500">
            System-wide interception and custom PII models.
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* Transparent Proxy Toggle */}
        <div className="rounded-xl ring-1 ring-stone-200 p-4">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-stone-700 flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Transparent Proxy
            </label>
            <button
              onClick={handleToggleTransparentProxy}
              disabled={isTogglingProxy}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 ${
                transparentProxyEnabled ? "bg-brand-600" : "bg-stone-300"
              } ${isTogglingProxy ? "opacity-50 cursor-not-allowed" : ""}`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  transparentProxyEnabled ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>
          <p className="text-xs text-stone-500 mt-2">
            Intercept HTTPS traffic system-wide for automatic PII protection.
          </p>
          <div className="mt-3 p-3 rounded-lg bg-amber-50 ring-1 ring-amber-200">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-amber-800">
                <strong>Experimental:</strong> This feature requires CA
                certificate installation and may affect system network settings.
              </p>
            </div>
          </div>
        </div>

        {/* Load Custom Kiji PII Model */}
        <div className="rounded-xl ring-1 ring-stone-200 p-4">
          <label className="text-sm font-semibold text-stone-700 mb-3 flex items-center gap-2">
            <Server className="w-4 h-4" />
            Load Custom Kiji PII Model
          </label>

          {/* Current Model Info */}
          {modelInfo && (
            <div
              className={`mb-3 p-3 rounded-lg ring-1 ${
                modelInfo.healthy
                  ? "bg-brand-50 ring-brand-200"
                  : "bg-red-50 ring-red-200"
              }`}
            >
              <div className="text-xs">
                <span
                  className={`font-medium ${
                    modelInfo.healthy ? "text-brand-700" : "text-red-700"
                  }`}
                >
                  Status: {modelInfo.healthy ? "Healthy" : "Unhealthy"}
                </span>
                {modelInfo.directory && (
                  <div className="text-stone-600 mt-1 break-all font-mono">
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
              className="flex-1 px-3 py-2 rounded-lg border border-stone-200 bg-white font-mono text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 placeholder:text-stone-400"
            />
            <button
              onClick={handleBrowseModelDirectory}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-stone-600 bg-white ring-1 ring-stone-200 hover:bg-stone-50 hover:text-stone-800 transition-colors"
              title="Browse for folder"
            >
              <FolderOpen className="w-4 h-4" />
              Browse
            </button>
          </div>

          <p className="text-xs text-stone-500 mt-2">
            Directory must contain: model.onnx, tokenizer.json,
            label_mappings.json
          </p>

          {/* Action Button */}
          <button
            onClick={handleReloadModel}
            disabled={isReloading || !modelDirectory.trim()}
            className="btn-brand mt-3 inline-flex items-center justify-center gap-2 px-5 py-2.5 text-white rounded-xl text-sm font-medium tracking-tight disabled:opacity-50 disabled:cursor-not-allowed w-full"
          >
            {isReloading ? "Reloading…" : "Reload Model"}
          </button>

          {/* Reload Message */}
          {reloadMessage && (
            <div
              className={`mt-3 p-3 rounded-lg text-sm ring-1 ${
                reloadMessage.type === "success"
                  ? "bg-brand-50 text-brand-800 ring-brand-200"
                  : "bg-red-50 text-red-800 ring-red-200"
              }`}
            >
              {reloadMessage.text}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
