import { useState, useEffect } from "react";
import {
  ShieldCheck,
  Shield,
  ListChecks,
  Regex,
  Plus,
  Minus,
} from "lucide-react";
import { apiUrl, isElectron } from "../../utils/providerHelpers";

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

function SavedHint({ show }: { show: boolean }) {
  if (!show) return null;
  return <span className="text-xs text-brand-600 font-medium">Saved</span>;
}

export default function PIISection() {
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

  // PII settings are read from / written to the backend over HTTP so this works
  // both in the desktop app and in the browser-served server build. apiUrl()
  // targets localhost:8080 under Electron and a relative path in web mode.
  const loadEntityConfidence = async () => {
    try {
      const res = await fetch(apiUrl("/api/pii/confidence", isElectron));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (typeof data.confidence === "number") {
        setEntityConfidence(data.confidence);
      }
    } catch (error) {
      console.error("Error loading entity confidence:", error);
    }
  };

  const loadEntities = async () => {
    try {
      const res = await fetch(apiUrl("/api/pii/entities", isElectron));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const available: string[] = data.available ?? [];
      setAvailableEntities(available);
      // The stored value is the exclusion list (types left unmasked). A checkbox
      // is checked when its type is NOT excluded, so the default (nothing
      // excluded) shows everything checked => mask everything.
      const disabled = new Set<string>(data.disabled ?? []);
      setEnabledEntities(new Set(available.filter((e) => !disabled.has(e))));
    } catch (error) {
      console.error("Error loading entities:", error);
    }
  };

  const loadRegexes = async () => {
    try {
      const res = await fetch(apiUrl("/api/pii/regexes", isElectron));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setCustomRegexes(data.regexes ?? []);
      setSelectedRegexIndex(null);
    } catch (error) {
      console.error("Error loading custom regexes:", error);
    }
  };

  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    loadEntityConfidence();
    loadEntities();
    loadRegexes();
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  const handleSetEntityConfidence = async (confidence: number) => {
    setEntityConfidence(confidence);
    try {
      const res = await fetch(apiUrl("/api/pii/confidence", isElectron), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confidence }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setConfidenceSaved(true);
      setTimeout(() => setConfidenceSaved(false), 2000);
    } catch (error) {
      console.error("Error setting entity confidence:", error);
    }
  };

  const persistEnabledEntities = async (next: Set<string>) => {
    setEnabledEntities(next);
    // Persist the inverse — the exclusion list of types to leave unmasked — so an
    // empty selection means "mask everything" and never leaks PII by accident.
    const disabled = availableEntities.filter((label) => !next.has(label));
    try {
      const res = await fetch(apiUrl("/api/pii/entities", isElectron), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disabled }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
    // Only send complete rows; a freshly added blank row stays in the editor
    // until the user fills both fields in. The name is used as the entity type,
    // so both name and pattern are required.
    const complete = rows
      .map((r) => ({ name: r.name.trim(), pattern: r.pattern.trim() }))
      .filter((r) => r.name && r.pattern);

    setRegexError(null);
    try {
      const res = await fetch(apiUrl("/api/pii/regexes", isElectron), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ regexes: complete }),
      });
      if (!res.ok) {
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

  // Values are the minimum confidence threshold to mask. A LOWER threshold
  // catches more (higher sensitivity), so "High" sensitivity maps to 0.1 and
  // "Low" to 0.5. Medium (0.25) stays the default.
  const SENSITIVITY = [
    { label: "Low", value: 0.5 },
    { label: "Medium", value: 0.25 },
    { label: "High", value: 0.1 },
  ] as const;

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            PII Detection
          </h2>
          <p className="text-[13px] text-stone-500">
            Tune how aggressively data is detected and what gets masked.
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {/* PII Detection Sensitivity */}
        <div>
          <label className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2">
            <Shield className="w-4 h-4" />
            Detection sensitivity
            <SavedHint show={confidenceSaved} />
          </label>
          <div className="inline-flex w-full p-1 rounded-xl bg-stone-100 ring-1 ring-stone-200/70">
            {SENSITIVITY.map(({ label, value }) => (
              <button
                key={value}
                onClick={() => handleSetEntityConfidence(value)}
                className={`flex-1 px-4 py-1.5 text-sm font-medium rounded-lg transition-all ${
                  entityConfidence === value
                    ? "bg-white text-brand-900 shadow-soft"
                    : "text-stone-500 hover:text-stone-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <p className="text-xs text-stone-500 mt-2">
            Controls how aggressively PII is detected. High catches more
            potential PII but may have false positives. Low is more precise but
            may miss some PII.
          </p>
        </div>

        {/* Entities to Mask */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-semibold text-stone-700 flex items-center gap-2">
              <ListChecks className="w-4 h-4" />
              Entities to mask
              <SavedHint show={entitiesSaved} />
            </label>
            <div className="flex items-center gap-2 text-xs">
              <button
                onClick={handleSelectAllEntities}
                disabled={availableEntities.length === 0}
                className="text-brand-600 hover:text-brand-700 disabled:opacity-40 disabled:cursor-not-allowed font-medium"
              >
                Select all
              </button>
              <span className="text-stone-300">|</span>
              <button
                onClick={handleDeselectAllEntities}
                disabled={availableEntities.length === 0}
                className="text-brand-600 hover:text-brand-700 disabled:opacity-40 disabled:cursor-not-allowed font-medium"
              >
                Deselect all
              </button>
            </div>
          </div>

          {availableEntities.length === 0 ? (
            <p className="text-xs text-stone-500">
              No entity types available. Load a healthy PII model to configure
              masking.
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-1 max-h-56 overflow-y-auto rounded-xl ring-1 ring-stone-200 p-2">
              {availableEntities.map((label) => (
                <label
                  key={label}
                  className="flex items-center gap-2 px-2 py-1 rounded-lg hover:bg-stone-50 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={enabledEntities.has(label)}
                    onChange={() => handleToggleEntity(label)}
                    className="rounded border-stone-300 text-brand-600 focus:ring-brand-500"
                  />
                  <span className="text-sm text-stone-700">
                    {humanizeEntity(label)}
                  </span>
                </label>
              ))}
            </div>
          )}

          <p className="text-xs text-stone-500 mt-2">
            Unchecked types are left unmasked and sent to the AI provider as-is.
          </p>
        </div>

        {/* Custom Regex Patterns */}
        <div>
          <label className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2">
            <Regex className="w-4 h-4" />
            Custom regex patterns
            <SavedHint show={regexesSaved} />
          </label>

          <div className="rounded-xl ring-1 ring-stone-200 overflow-hidden">
            {/* Column headers */}
            <div className="flex items-center gap-2 px-2 py-1.5 bg-stone-50 border-b border-stone-200 text-xs font-semibold text-stone-600">
              <span className="flex-1">Name</span>
              <span className="flex-[2]">Pattern</span>
            </div>

            {/* Rows */}
            <div className="max-h-48 overflow-y-auto">
              {customRegexes.length === 0 ? (
                <p className="text-xs text-stone-500 px-3 py-3">
                  No custom patterns. Use the + button below to add one.
                </p>
              ) : (
                customRegexes.map((row, index) => (
                  <div
                    key={index}
                    onClick={() => setSelectedRegexIndex(index)}
                    className={`flex items-center gap-2 px-2 py-1 cursor-pointer border-l-2 ${
                      selectedRegexIndex === index
                        ? "bg-brand-50 border-brand-500"
                        : "border-transparent hover:bg-stone-50"
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
                      className="flex-1 min-w-0 px-2 py-1 border border-stone-200 rounded font-mono text-xs focus:border-brand-500 focus:outline-none placeholder:text-stone-400"
                    />
                    <input
                      type="text"
                      value={row.pattern}
                      onFocus={() => setSelectedRegexIndex(index)}
                      onChange={(e) =>
                        handleRegexFieldChange(index, "pattern", e.target.value)
                      }
                      onBlur={() => persistRegexes(customRegexes)}
                      placeholder="\d{3}-\d{2}-\d{4}"
                      className="flex-[2] min-w-0 px-2 py-1 border border-stone-200 rounded font-mono text-xs focus:border-brand-500 focus:outline-none placeholder:text-stone-400"
                    />
                  </div>
                ))
              )}
            </div>

            {/* Add / remove toolbar */}
            <div className="flex items-center gap-1 px-2 py-1.5 bg-stone-50 border-t border-stone-200">
              <button
                onClick={handleAddRegex}
                title="Add pattern"
                className="w-7 h-7 flex items-center justify-center rounded-lg border border-stone-300 bg-white text-stone-700 hover:bg-stone-100 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
              <button
                onClick={handleRemoveRegex}
                disabled={selectedRegexIndex === null}
                title="Remove selected pattern"
                className="w-7 h-7 flex items-center justify-center rounded-lg border border-stone-300 bg-white text-stone-700 hover:bg-stone-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Minus className="w-4 h-4" />
              </button>
            </div>
          </div>

          <p className="text-xs text-stone-500 mt-2">
            Each pattern's name is used as the detected PII type. Patterns use
            RE2 (Go) syntax and are matched against request content.
          </p>
          {regexError && (
            <p className="text-xs text-red-600 mt-1">{regexError}</p>
          )}
        </div>
      </div>
    </section>
  );
}
