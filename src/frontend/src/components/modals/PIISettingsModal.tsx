import { useState, useEffect } from "react";
import { X, Shield, ListChecks, Regex, Plus, Minus } from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface PIISettingsModalProps {
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

export default function PIISettingsModal({
  isOpen,
  onClose,
}: PIISettingsModalProps) {
  // PII detection confidence state
  const [entityConfidence, setEntityConfidence] = useState(0.25);
  const [confidenceSaved, setConfidenceSaved] = useState(false);

  // Entities-to-mask state
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);
  const [enabledEntities, setEnabledEntities] = useState<Set<string>>(
    new Set()
  );
  const [entitiesSaved, setEntitiesSaved] = useState(false);

  // Custom regex patterns state. Rows are edited locally (including incomplete
  // ones) and only complete rows are persisted to the backend.
  const [customRegexes, setCustomRegexes] = useState<
    Array<{ name: string; pattern: string }>
  >([]);
  const [selectedRegexIndex, setSelectedRegexIndex] = useState<number | null>(
    null
  );
  const [regexesSaved, setRegexesSaved] = useState(false);
  const [regexError, setRegexError] = useState<string | null>(null);

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

  const loadRegexes = async () => {
    if (!window.electronAPI) return;

    try {
      const info = await window.electronAPI.getCustomRegexes();
      setCustomRegexes(info?.regexes ?? []);
      setSelectedRegexIndex(null);
    } catch (error) {
      console.error("Error loading custom regexes:", error);
    }
  };

  useEffect(() => {
    if (isOpen && isElectron) {
      /* eslint-disable react-hooks/set-state-in-effect */
      loadEntityConfidence();
      loadEntities();
      loadRegexes();
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

  const persistRegexes = async (
    rows: Array<{ name: string; pattern: string }>
  ) => {
    if (!window.electronAPI) return;

    // Only send complete rows; a freshly added blank row stays in the editor
    // until the user fills both fields in. The name is used as the entity type,
    // so both name and pattern are required.
    const complete = rows
      .map((r) => ({ name: r.name.trim(), pattern: r.pattern.trim() }))
      .filter((r) => r.name && r.pattern);

    setRegexError(null);
    try {
      const result = await window.electronAPI.setCustomRegexes(complete);
      if (result?.success === false) {
        // The backend rejects invalid patterns with a 400; surface a friendly
        // hint rather than the low-level transport error.
        setRegexError(
          "Failed to save patterns. Check that each name and pattern is a valid RE2 expression."
        );
        return;
      }
      setRegexesSaved(true);
      setTimeout(() => setRegexesSaved(false), 2000);
    } catch (error) {
      console.error("Error saving custom regexes:", error);
      setRegexError("Failed to save patterns.");
    }
  };

  const handleRegexFieldChange = (
    index: number,
    field: "name" | "pattern",
    value: string
  ) => {
    setCustomRegexes((prev) =>
      prev.map((row, i) => (i === index ? { ...row, [field]: value } : row))
    );
  };

  const handleAddRegex = () => {
    const next = [...customRegexes, { name: "", pattern: "" }];
    setCustomRegexes(next);
    setSelectedRegexIndex(next.length - 1);
  };

  const handleRemoveRegex = () => {
    if (selectedRegexIndex === null) return;
    const next = customRegexes.filter((_, i) => i !== selectedRegexIndex);
    setSelectedRegexIndex(null);
    setCustomRegexes(next);
    persistRegexes(next);
  };

  if (!isOpen) return null;

  if (!isElectron) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold text-slate-800">PII Settings</h2>
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-700 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          <p className="text-slate-600">
            PII settings are only available in Electron mode.
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
          <h2 className="text-2xl font-bold text-slate-800">PII Settings</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-6">
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

          {/* Custom Regex Patterns */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
              <Regex className="w-4 h-4" />
              Custom Regex Patterns
            </label>

            <div className="border-2 border-slate-200 rounded-lg overflow-hidden">
              {/* Column headers */}
              <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-600">
                <span className="flex-1">Name</span>
                <span className="flex-[2]">Pattern</span>
              </div>

              {/* Rows */}
              <div className="max-h-48 overflow-y-auto">
                {customRegexes.length === 0 ? (
                  <p className="text-xs text-slate-500 px-3 py-3">
                    No custom patterns. Use the + button below to add one.
                  </p>
                ) : (
                  customRegexes.map((row, index) => (
                    <div
                      key={index}
                      onClick={() => setSelectedRegexIndex(index)}
                      className={`flex items-center gap-2 px-2 py-1 cursor-pointer border-l-2 ${
                        selectedRegexIndex === index
                          ? "bg-blue-50 border-blue-500"
                          : "border-transparent hover:bg-slate-50"
                      }`}
                    >
                      <input
                        type="text"
                        value={row.name}
                        onFocus={() => setSelectedRegexIndex(index)}
                        onChange={(e) =>
                          handleRegexFieldChange(index, "name", e.target.value)
                        }
                        onBlur={() => persistRegexes(customRegexes)}
                        placeholder="EMAIL"
                        className="flex-1 min-w-0 px-2 py-1 border border-slate-200 rounded font-mono text-xs focus:border-blue-500 focus:outline-none placeholder:text-gray-400"
                      />
                      <input
                        type="text"
                        value={row.pattern}
                        onFocus={() => setSelectedRegexIndex(index)}
                        onChange={(e) =>
                          handleRegexFieldChange(
                            index,
                            "pattern",
                            e.target.value
                          )
                        }
                        onBlur={() => persistRegexes(customRegexes)}
                        placeholder="\d{3}-\d{2}-\d{4}"
                        className="flex-[2] min-w-0 px-2 py-1 border border-slate-200 rounded font-mono text-xs focus:border-blue-500 focus:outline-none placeholder:text-gray-400"
                      />
                    </div>
                  ))
                )}
              </div>

              {/* Add / remove toolbar */}
              <div className="flex items-center gap-1 px-2 py-1.5 bg-slate-50 border-t border-slate-200">
                <button
                  onClick={handleAddRegex}
                  title="Add pattern"
                  className="w-7 h-7 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-100 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                </button>
                <button
                  onClick={handleRemoveRegex}
                  disabled={selectedRegexIndex === null}
                  title="Remove selected pattern"
                  className="w-7 h-7 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-700 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Minus className="w-4 h-4" />
                </button>
              </div>
            </div>

            <p className="text-xs text-slate-500 mt-2">
              Each pattern's name is used as the detected PII type. Patterns use
              RE2 (Go) syntax and are matched against request content.
            </p>
            {regexError && (
              <p className="text-xs text-red-600 mt-1">{regexError}</p>
            )}
            {regexesSaved && (
              <p className="text-xs text-green-600 mt-1">Setting saved.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
