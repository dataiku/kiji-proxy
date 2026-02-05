import { useState, useEffect } from "react";
import {
  X,
  Server,
  AlertCircle,
  CheckCircle2,
  FolderOpen,
} from "lucide-react";

interface AdvancedSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AdvancedSettingsModal({
  isOpen,
  onClose,
}: AdvancedSettingsModalProps) {
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
      loadModelInfo();
    }
  }, [isOpen, isElectron]);

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
      setReloadMessage({ type: "error", text: "Failed to open folder selector" });
    }
  };

  if (!isOpen) return null;

  if (!isElectron) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-slate-800">
              Advanced Settings
            </h2>
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-700 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <p className="text-slate-600">
            Advanced settings are only available in Electron mode.
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
          <h2 className="text-2xl font-bold text-slate-800">
            Advanced Settings
          </h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-6">
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

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <button
              onClick={onClose}
              className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium flex-1"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
