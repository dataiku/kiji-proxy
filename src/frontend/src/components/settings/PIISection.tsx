import { createProfileSaveQueue, type ProfilePatch } from "./profileSaveQueue";
import MappingExpirationControl from "./MappingExpirationControl";
import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  ShieldCheck,
  Shield,
  ListChecks,
  Regex,
  Plus,
  Minus,
  Star,
  Trash2,
  Layers,
} from "lucide-react";
import { apiUrl, isElectron } from "../../utils/providerHelpers";
import { usePIIProfiles } from "../../hooks/usePIIProfiles";
import type { PIIProfile, PIIRegexPattern } from "../../types/piiProfile";

// Friendly display names for the model's raw entity labels come from the
// `settings` locale (pii.entityLabels). Unknown labels (e.g. from a custom
// model) fall back to a capitalized form.
const humanizeEntity = (label: string, t: TFunction): string =>
  t(`pii.entityLabels.${label}`, {
    defaultValue: label.charAt(0).toUpperCase() + label.slice(1).toLowerCase(),
  });

const profileUrl = (id: string, suffix = "") =>
  apiUrl(`/api/pii/profiles/${encodeURIComponent(id)}${suffix}`, isElectron);

const jsonHeaders = { "Content-Type": "application/json" };

function SavedHint({ show }: { show: boolean }) {
  const { t } = useTranslation("settings");
  if (!show) return null;
  return (
    <span className="text-xs text-brand-600 font-medium">{t("pii.saved")}</span>
  );
}

type SavedKey = "name" | "confidence" | "entities" | "regexes";

interface PIISectionProps {
  /** Called after the profile list changes (create, rename, delete, default)
   *  so the host can refresh other profile pickers (e.g. the Playground). */
  onProfilesChanged?: () => void;
}

/**
 * Manages PII detection profiles: named sets of sensitivity, entities to mask
 * and custom regexes. One profile is the default; requests choose another via
 * the X-Kiji-Profile header. All reads/writes go to the backend over HTTP so
 * this works both in the desktop app and in the browser-served server build.
 */
export default function PIISection({ onProfilesChanged }: PIISectionProps) {
  const { t } = useTranslation("settings");
  const { profiles, defaultId, reload } = usePIIProfiles();

  // The profile being edited. Falls back to the default until the user picks
  // one (or when the picked one disappears).
  const [pickedId, setPickedId] = useState("");
  const selectedId = profiles.some((p) => p.id === pickedId)
    ? pickedId
    : defaultId;

  const [profile, setProfile] = useState<PIIProfile | null>(null);
  // Mirrors the latest profile (including optimistic edits) so overlapping
  // saves each build on the previous one instead of on a stale render, which
  // would let one PUT overwrite fields another just changed.
  const profileRef = useRef<PIIProfile | null>(null);
  const applyProfile = (next: PIIProfile | null) => {
    profileRef.current = next;
    setProfile(next);
  };
  const [nameDraft, setNameDraft] = useState("");
  const [nameError, setNameError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Entity types selectable for this profile (model labels + its regex names).
  const [availableEntities, setAvailableEntities] = useState<string[]>([]);

  // Custom regex rows are edited locally (including incomplete ones) and only
  // complete rows are persisted.
  const [customRegexes, setCustomRegexes] = useState<PIIRegexPattern[]>([]);
  const [selectedRegexIndex, setSelectedRegexIndex] = useState<number | null>(
    null
  );
  const [regexError, setRegexError] = useState<string | null>(null);

  const [saved, setSaved] = useState<Record<SavedKey, boolean>>({
    name: false,
    confidence: false,
    entities: false,
    regexes: false,
  });
  const flashSaved = (key: SavedKey) => {
    setSaved((s) => ({ ...s, [key]: true }));
    setTimeout(() => setSaved((s) => ({ ...s, [key]: false })), 2000);
  };

  const loadEntities = useCallback(async (id: string) => {
    try {
      const res = await fetch(
        apiUrl(
          `/api/pii/entities?profile=${encodeURIComponent(id)}`,
          isElectron
        )
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAvailableEntities(data.available ?? []);
    } catch (error) {
      console.error("Error loading entities:", error);
    }
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(profileUrl(selectedId));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: PIIProfile = await res.json();
        if (cancelled) return;
        applyProfile(data);
        setNameDraft(data.name);
        setCustomRegexes(data.custom_regexes ?? []);
        setSelectedRegexIndex(null);
        setNameError(null);
        setRegexError(null);
        setActionError(null);
        setConfirmDelete(false);
      } catch (error) {
        console.error("Error loading PII profile:", error);
        if (!cancelled) setActionError(t("pii.profiles.loadFailed"));
      }
    })();
    /* eslint-disable-next-line react-hooks/set-state-in-effect */
    loadEntities(selectedId);
    return () => {
      cancelled = true;
    };
  }, [selectedId, loadEntities, t]);

  const listChanged = async () => {
    await reload();
    onProfilesChanged?.();
  };

  const saveQueueRef = useRef<ReturnType<typeof createProfileSaveQueue> | null>(null);
  const saveProfile = (patch: ProfilePatch) => {
    if (!saveQueueRef.current) {
      saveQueueRef.current = createProfileSaveQueue(
        () => profileRef.current,
        applyProfile,
        (next) => fetch(profileUrl(next.id), {
          method: "PUT",
          headers: jsonHeaders,
          body: JSON.stringify({
            name: next.name,
            confidence: next.confidence,
            disabled_entities: next.disabled_entities,
            custom_regexes: next.custom_regexes,
          }),
        })
      );
    }
    return saveQueueRef.current(patch);
  };

  // --- profile list actions ---

  const handleCreate = async () => {
    setActionError(null);
    const base = t("pii.profiles.newName");
    const taken = new Set(profiles.map((p) => p.name.toLowerCase()));
    let name = base;
    for (let n = 2; taken.has(name.toLowerCase()); n++) name = `${base} ${n}`;
    try {
      const res = await fetch(apiUrl("/api/pii/profiles", isElectron), {
        method: "POST",
        headers: jsonHeaders,
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const created: PIIProfile = await res.json();
      setPickedId(created.id);
      await listChanged();
    } catch (error) {
      console.error("Error creating PII profile:", error);
      setActionError(t("pii.profiles.actionFailed"));
    }
  };

  const handleSetDefault = async () => {
    if (!profile) return;
    setActionError(null);
    try {
      const res = await fetch(profileUrl(profile.id, "/default"), {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (profileRef.current?.id === profile.id) {
        applyProfile({ ...profileRef.current, is_default: true });
      }
      await listChanged();
    } catch (error) {
      console.error("Error setting default PII profile:", error);
      setActionError(t("pii.profiles.actionFailed"));
    }
  };

  const handleDelete = async () => {
    if (!profile) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setActionError(null);
    try {
      const res = await fetch(profileUrl(profile.id), { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPickedId("");
      await listChanged();
    } catch (error) {
      console.error("Error deleting PII profile:", error);
      setActionError(t("pii.profiles.actionFailed"));
    }
  };

  const handleRename = async () => {
    if (!profile) return;
    const name = nameDraft.trim();
    if (name === profile.name) return;
    if (!name || name.length > 64) {
      setNameError(t("pii.profiles.nameInvalid"));
      return;
    }
    setNameError(null);
    const result = await saveProfile({ name });
    if (result.ok) {
      flashSaved("name");
      await listChanged();
    } else {
      setNameError(
        result.status === 409
          ? t("pii.profiles.nameTaken")
          : t("pii.profiles.actionFailed")
      );
    }
  };

  // --- profile settings ---

  const handleSetConfidence = async (confidence: number) => {
    if ((await saveProfile({ confidence })).ok) flashSaved("confidence");
  };

  // The profile stores the exclusion list (types left unmasked), so an empty
  // selection means "mask everything" and never leaks PII by accident.
  const persistDisabled = async (disabled: string[]) => {
    if ((await saveProfile({ disabled_entities: disabled })).ok) {
      flashSaved("entities");
    }
  };

  const disabledSet = new Set(profile?.disabled_entities ?? []);

  const handleToggleEntity = (label: string) => {
    const next = new Set(profileRef.current?.disabled_entities ?? []);
    if (next.has(label)) {
      next.delete(label);
    } else {
      next.add(label);
    }
    persistDisabled([...next]);
  };

  const persistRegexes = async (rows: PIIRegexPattern[]) => {
    const current = profileRef.current;
    if (!current) return;
    // Only send complete rows; a freshly added blank row stays in the editor
    // until the user fills both fields in. The name is used as the entity type,
    // so both name and pattern are required.
    const complete = rows
      .map((r) => ({ name: r.name.trim(), pattern: r.pattern.trim() }))
      .filter((r) => r.name && r.pattern);

    // A removed or renamed regex can no longer be detected, so drop it from
    // the disabled list too (the backend rejects labels it can't produce).
    const oldNames = new Set(current.custom_regexes.map((r) => r.name));
    const newNames = new Set(complete.map((r) => r.name));
    const disabled = current.disabled_entities.filter(
      (label) => !oldNames.has(label) || newNames.has(label)
    );

    setRegexError(null);
    const result = await saveProfile({
      custom_regexes: complete,
      disabled_entities: disabled,
    });
    if (result.ok) {
      flashSaved("regexes");
      loadEntities(current.id);
    } else {
      // The backend rejects invalid patterns with a 400; surface a friendly
      // hint rather than the low-level transport error.
      setRegexError(
        result.status === 400
          ? t("pii.regex.saveFailedInvalid")
          : t("pii.regex.saveFailed")
      );
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
    { key: "low", value: 0.5 },
    { key: "medium", value: 0.25 },
    { key: "high", value: 0.1 },
  ] as const;

  const iconButton =
    "inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-stone-200 bg-white text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors";

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            {t("pii.title")}
          </h2>
          <p className="text-[13px] text-stone-500">{t("pii.subtitle")}</p>
        </div>
      </div>

      <div className="space-y-6">
        <MappingExpirationControl />
        {/* Profile picker + actions */}
        <div>
          <label
            htmlFor="kiji-pii-profile-edit"
            className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2"
          >
            <Layers className="w-4 h-4" />
            {t("pii.profiles.label")}
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <select
              id="kiji-pii-profile-edit"
              value={selectedId}
              onChange={(e) => setPickedId(e.target.value)}
              className="w-full sm:w-64 px-3 py-2 rounded-lg border border-stone-200 bg-white text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            >
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.is_default
                    ? t("pii.profiles.defaultSuffix", { name: p.name })
                    : p.name}
                </option>
              ))}
            </select>
            <button onClick={handleCreate} className={iconButton}>
              <Plus className="w-4 h-4" />
              {t("pii.profiles.new")}
            </button>
            <button
              onClick={handleSetDefault}
              disabled={!profile || profile.is_default}
              className={iconButton}
            >
              <Star className="w-4 h-4" />
              {profile?.is_default
                ? t("pii.profiles.isDefault")
                : t("pii.profiles.setDefault")}
            </button>
            <button
              onClick={handleDelete}
              onBlur={() => setConfirmDelete(false)}
              disabled={!profile || profile.is_default}
              title={
                profile?.is_default
                  ? t("pii.profiles.cannotDeleteDefault")
                  : undefined
              }
              className={`${iconButton} ${
                confirmDelete ? "border-red-300 text-red-700 bg-red-50" : ""
              }`}
            >
              <Trash2 className="w-4 h-4" />
              {confirmDelete
                ? t("pii.profiles.confirmDelete")
                : t("pii.profiles.delete")}
            </button>
          </div>
          <p className="text-xs text-stone-500 mt-2">
            {t("pii.profiles.help")}
          </p>
          {actionError && (
            <p className="text-xs text-red-600 mt-1">{actionError}</p>
          )}
        </div>

        {profile && (
          <>
            {/* Profile name */}
            <div>
              <label
                htmlFor="kiji-pii-profile-name"
                className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2"
              >
                {t("pii.profiles.nameLabel")}
                <SavedHint show={saved.name} />
              </label>
              <input
                id="kiji-pii-profile-name"
                type="text"
                value={nameDraft}
                maxLength={64}
                onChange={(e) => setNameDraft(e.target.value)}
                onBlur={handleRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") e.currentTarget.blur();
                }}
                className="w-full sm:w-64 px-3 py-2 rounded-lg border border-stone-200 bg-white text-sm transition-shadow focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              />
              <p className="text-xs text-stone-500 mt-2">
                {t("pii.profiles.idHint")}{" "}
                <code className="font-mono text-stone-700">
                  {`X-Kiji-Profile: ${profile.id}`}
                </code>
              </p>
              {nameError && (
                <p className="text-xs text-red-600 mt-1">{nameError}</p>
              )}
            </div>

            {/* PII Detection Sensitivity */}
            <div>
              <label className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                {t("pii.sensitivity.label")}
                <SavedHint show={saved.confidence} />
              </label>
              <div className="inline-flex w-full p-1 rounded-xl bg-stone-100 ring-1 ring-stone-200/70">
                {SENSITIVITY.map(({ key, value }) => (
                  <button
                    key={value}
                    onClick={() => handleSetConfidence(value)}
                    className={`flex-1 px-4 py-1.5 text-sm font-medium rounded-lg transition-all ${
                      profile.confidence === value
                        ? "bg-white text-brand-900 shadow-soft"
                        : "text-stone-500 hover:text-stone-700"
                    }`}
                  >
                    {t(`pii.sensitivity.${key}`)}
                  </button>
                ))}
              </div>
              <p className="text-xs text-stone-500 mt-2">
                {t("pii.sensitivity.help")}
              </p>
            </div>

            {/* Entities to Mask */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-semibold text-stone-700 flex items-center gap-2">
                  <ListChecks className="w-4 h-4" />
                  {t("pii.entities.label")}
                  <SavedHint show={saved.entities} />
                </label>
                <div className="flex items-center gap-2 text-xs">
                  <button
                    onClick={() => persistDisabled([])}
                    disabled={availableEntities.length === 0}
                    className="text-brand-600 hover:text-brand-700 disabled:opacity-40 disabled:cursor-not-allowed font-medium"
                  >
                    {t("pii.entities.selectAll")}
                  </button>
                  <span className="text-stone-300">|</span>
                  <button
                    onClick={() => persistDisabled([...availableEntities])}
                    disabled={availableEntities.length === 0}
                    className="text-brand-600 hover:text-brand-700 disabled:opacity-40 disabled:cursor-not-allowed font-medium"
                  >
                    {t("pii.entities.deselectAll")}
                  </button>
                </div>
              </div>

              {availableEntities.length === 0 ? (
                <p className="text-xs text-stone-500">
                  {t("pii.entities.none")}
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
                        checked={!disabledSet.has(label)}
                        onChange={() => handleToggleEntity(label)}
                        className="rounded border-stone-300 text-brand-600 focus:ring-brand-500"
                      />
                      <span className="text-sm text-stone-700">
                        {humanizeEntity(label, t)}
                      </span>
                    </label>
                  ))}
                </div>
              )}

              <p className="text-xs text-stone-500 mt-2">
                {t("pii.entities.help")}
              </p>
            </div>

            {/* Custom Regex Patterns */}
            <div>
              <label className="text-sm font-semibold text-stone-700 mb-2 flex items-center gap-2">
                <Regex className="w-4 h-4" />
                {t("pii.regex.label")}
                <SavedHint show={saved.regexes} />
              </label>

              <div className="rounded-xl ring-1 ring-stone-200 overflow-hidden">
                {/* Column headers */}
                <div className="flex items-center gap-2 px-2 py-1.5 bg-stone-50 border-b border-stone-200 text-xs font-semibold text-stone-600">
                  <span className="flex-1">{t("pii.regex.nameColumn")}</span>
                  <span className="flex-[2]">
                    {t("pii.regex.patternColumn")}
                  </span>
                </div>

                {/* Rows */}
                <div className="max-h-48 overflow-y-auto">
                  {customRegexes.length === 0 ? (
                    <p className="text-xs text-stone-500 px-3 py-3">
                      {t("pii.regex.none")}
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
                            handleRegexFieldChange(
                              index,
                              "name",
                              e.target.value
                            )
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
                            handleRegexFieldChange(
                              index,
                              "pattern",
                              e.target.value
                            )
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
                    title={t("pii.regex.addTitle")}
                    className="w-7 h-7 flex items-center justify-center rounded-lg border border-stone-300 bg-white text-stone-700 hover:bg-stone-100 transition-colors"
                  >
                    <Plus className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleRemoveRegex}
                    disabled={selectedRegexIndex === null}
                    title={t("pii.regex.removeTitle")}
                    className="w-7 h-7 flex items-center justify-center rounded-lg border border-stone-300 bg-white text-stone-700 hover:bg-stone-100 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <Minus className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <p className="text-xs text-stone-500 mt-2">
                {t("pii.regex.help")}
              </p>
              {regexError && (
                <p className="text-xs text-red-600 mt-1">{regexError}</p>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
