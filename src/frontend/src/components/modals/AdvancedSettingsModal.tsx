import { useState, useEffect } from "react";
import {
  X,
  Server,
  FolderOpen,
  Shield,
  AlertTriangle,
  ListChecks,
} from "lucide-react";
import CACertSetupModal from "./CACertSetupModal";
import { isElectron } from "../../utils/providerHelpers";

interface AdvancedSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Friendly display names for the model's raw entity labels. Unknown labels
// (e.g. from a custom model) fall back to a capitalized form.
const ENTITY_LABEL_NAMES: Record<string, string> = {
  SURNAME: "Surname",
  FIRSTNAME: "First Name",
  BUILDINGNUM: "Building Number",
  DATEOFBIRTH: "Date of Birth",
  EMAIL: "Email",
  PHONENUMBER: "Phone Number",
  CITY: "City",
  URL: "URL",
  COMPANYNAME: "Company Name",
  STATE: "State",
  ZIP: "ZIP Code",
  STREET: "Street",
  COUNTRY: "Country",
  SSN: "SSN",
  DRIVERLICENSENUM: "Driver License Number",
  PASSPORTID: "Passport ID",
  NATIONALID: "National ID",
  IDCARDNUM: "ID Card Number",
  TAXNUM: "Tax Number",
  LICENSEPLATENUM: "License Plate Number",
  PASSWORD: "Password",
  IBAN: "IBAN",
  AGE: "Age",
  SECURITYTOKEN: "Security Token",
  CREDITCARDNUMBER: "Credit Card Number",
  USERNAME: "Username",
};

const humanizeEntity = (label: string): string =>
  ENTITY_LABEL_NAMES[label] ??
  label.charAt(0).toUpperCase() + label.slice(1).toLowerCase();

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

  // Transparent proxy state
  const [transparentProxyEnabled, setTransparentProxyEnabled] = useState(false);
  const [isTogglingProxy, setIsTogglingProxy] = useState(false);
  const [isCACertSetupOpen, setIsCACertSetupOpen] = useState(false);

  // PII detection confidence state
  const [entityConfidence, setEntityConfidence] = useState(0.25);
  const [confidenceSaved, setConfidenceSaved] = useState(false);

  // Entities-to-mask state
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);
  const [enabledEntities, setEnabledEntities] = useState<Set<string>>(
    new Set()
  );
  const [entitiesSaved, setEntitiesSaved] = useState(false);

  const loadTransparentProxySetting = async () => {
    if (!window.electronAPI) return;

    try {
      const enabled = await window.electronAPI.getTransparentProxyEnabled();
      setTransparentProxyEnabled(enabled);
    } catch (error) {
      console.error("Error loading transparent proxy setting:", error);
    }
  };

  const loadEntityConfidence = async () => {
    if (!window.electronAPI) return;

    try {
      const confidence = await window.electronAPI.getEntityConfidence();
      setEntityConfidence(confidence);
    } catch (error) {
      console.error("Error loading entity confidence:", error);
    }
  };

  const loadEntities = async () => {
    if (!window.electronAPI) return;

    try {
      const [info, savedDisabled] = await Promise.all([
        window.electronAPI.getAvailableEntities(),
        window.electronAPI.getDisabledEntities(),
      ]);
      const available = info?.available ?? [];
      setAvailableEntities(available);
      // The stored value is the exclusion list (types left unmasked). A checkbox
      // is checked when its type is NOT excluded, so the default (nothing
      // excluded) shows everything checked => mask everything.
      const disabled = new Set(savedDisabled ?? []);
      setEnabledEntities(new Set(available.filter((e) => !disabled.has(e))));
    } catch (error) {
      console.error("Error loading entities:", error);
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
    if (isOpen && isElectron) {
      /* eslint-disable react-hooks/set-state-in-effect */
      loadModelInfo();
      loadTransparentProxySetting();
      loadEntityConfidence();
      loadEntities();
      /* eslint-enable react-hooks/set-state-in-effect */
    }
  }, [isOpen]);

  const handleSetEntityConfidence = async (confidence: number) => {
    if (!window.electronAPI) return;

    setEntityConfidence(confidence);
    try {
      await window.electronAPI.setEntityConfidence(confidence);
      setConfidenceSaved(true);
      setTimeout(() => setConfidenceSaved(false), 2000);
    } catch (error) {
      console.error("Error setting entity confidence:", error);
    }
  };

  const persistEnabledEntities = async (next: Set<string>) => {
    if (!window.electronAPI) return;

    setEnabledEntities(next);
    // Persist the inverse — the exclusion list of types to leave unmasked — so an
    // empty selection means "mask everything" and never leaks PII by accident.
    const disabled = availableEntities.filter((label) => !next.has(label));
    try {
      await window.electronAPI.setDisabledEntities(disabled);
      setEntitiesSaved(true);
      setTimeout(() => setEntitiesSaved(false), 2000);
    } catch (error) {
      console.error("Error saving entity selection:", error);
    }
  };

  const handleToggleEntity = (label: string) => {
    const next = new Set(enabledEntities);
    if (next.has(label)) {
      next.delete(label);
    } else {
      next.add(label);
    }
    persistEnabledEntities(next);
  };

  const handleSelectAllEntities = () =>
    persistEnabledEntities(new Set(availableEntities));

  const handleDeselectAllEntities = () => persistEnabledEntities(new Set());

  const handleToggleTransparentProxy = async () => {
    if (!window.electronAPI) return;

    const newValue = !transparentProxyEnabled;

    // If enabling, show CA cert setup modal first
    if (newValue) {
      setIsCACertSetupOpen(true);
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
      setReloadMessage({
        type: "error",
        text: "Failed to open folder selector",
      });
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
          {/* Transparent Proxy Toggle */}
          <div>
            <div className="flex items-center justify-between">
              <label className="block text-sm font-semibold text-slate-700 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                Transparent Proxy
              </label>
              <button
                onClick={handleToggleTransparentProxy}
                disabled={isTogglingProxy}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  transparentProxyEnabled ? "bg-blue-600" : "bg-slate-300"
                } ${isTogglingProxy ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    transparentProxyEnabled ? "translate-x-6" : "translate-x-1"
                  }`}
                />
              </button>
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Intercept HTTPS traffic system-wide for automatic PII protection.
            </p>
            <div className="mt-2 p-2 rounded bg-amber-50 border border-amber-200">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-700">
                  <strong>Experimental:</strong> This feature requires CA
                  certificate installation and may affect system network
                  settings.
                </p>
              </div>
            </div>
          </div>

          {/* PII Detection Sensitivity */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
              <Shield className="w-4 h-4" />
              PII Detection Sensitivity
            </label>
            <div className="flex rounded-lg border-2 border-slate-200 overflow-hidden">
              {(
                [
                  { label: "Low", value: 0.1 },
                  { label: "Medium", value: 0.25 },
                  { label: "High", value: 0.5 },
                ] as const
              ).map(({ label, value }) => (
                <button
                  key={value}
                  onClick={() => handleSetEntityConfidence(value)}
                  className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                    entityConfidence === value
                      ? "bg-blue-600 text-white"
                      : "bg-slate-50 text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-2">
              Controls how aggressively PII is detected. Low catches more
              potential PII but may have false positives. High is more precise
              but may miss some PII.
            </p>
            {confidenceSaved && (
              <p className="text-xs text-green-600 mt-1">Setting saved.</p>
            )}
          </div>

          {/* Entities to Mask */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-slate-700 flex items-center gap-2">
                <ListChecks className="w-4 h-4" />
                Entities to Mask
              </label>
              <div className="flex items-center gap-2 text-xs">
                <button
                  onClick={handleSelectAllEntities}
                  disabled={availableEntities.length === 0}
                  className="text-blue-600 hover:text-blue-800 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Select all
                </button>
                <span className="text-slate-300">|</span>
                <button
                  onClick={handleDeselectAllEntities}
                  disabled={availableEntities.length === 0}
                  className="text-blue-600 hover:text-blue-800 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Deselect all
                </button>
              </div>
            </div>

            {availableEntities.length === 0 ? (
              <p className="text-xs text-slate-500">
                No entity types available. Load a healthy PII model to configure
                masking.
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-1 max-h-56 overflow-y-auto border-2 border-slate-200 rounded-lg p-2">
                {availableEntities.map((label) => (
                  <label
                    key={label}
                    className="flex items-center gap-2 px-2 py-1 rounded hover:bg-slate-50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={enabledEntities.has(label)}
                      onChange={() => handleToggleEntity(label)}
                      className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-slate-700">
                      {humanizeEntity(label)}
                    </span>
                  </label>
                ))}
              </div>
            )}

            <p className="text-xs text-slate-500 mt-2">
              Unchecked types are left unmasked and sent to the AI provider
              as-is.
            </p>
            {entitiesSaved && (
              <p className="text-xs text-green-600 mt-1">Setting saved.</p>
            )}
          </div>

          {/* Load Custom Kiji PII Model */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
              <Server className="w-4 h-4" />
              Load Custom Kiji PII Model
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
              Directory must contain: model.onnx, tokenizer.json,
              label_mappings.json
            </p>

            {/* Action Button */}
            <div className="mt-2">
              <button
                onClick={handleReloadModel}
                disabled={isReloading || !modelDirectory.trim()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm transition-colors w-full"
              >
                {isReloading ? "Reloading..." : "Reload Model"}
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
        </div>
      </div>

      {/* CA Certificate Setup Modal */}
      <CACertSetupModal
        isOpen={isCACertSetupOpen}
        onClose={() => setIsCACertSetupOpen(false)}
      />
    </div>
  );
}
