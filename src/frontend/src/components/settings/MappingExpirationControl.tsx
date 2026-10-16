import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { apiUrl, isElectron } from "../../utils/providerHelpers";

export default function MappingExpirationControl() {
  const { t } = useTranslation("settings");
  const [enabled, setEnabled] = useState(true);
  const [seconds, setSeconds] = useState("300");
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<"saved" | "error" | "invalid" | null>(null);

  useEffect(() => {
    let active = true;
    fetch(apiUrl("/api/pii/mapping-expiration", isElectron))
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((policy) => {
        if (!active) return;
        setEnabled(policy.enabled);
        setSeconds(String(policy.ttl_seconds));
        setLoaded(true);
      })
      .catch(() => { if (active) setStatus("error"); });
    return () => { active = false; };
  }, []);

  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    const ttl = Number(seconds);
    if (!Number.isInteger(ttl) || ttl < 1 || ttl > 2147483647) {
      setStatus("invalid");
      return;
    }
    setSaving(true);
    setStatus(null);
    try {
      const res = await fetch(apiUrl("/api/pii/mapping-expiration", isElectron), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, ttl_seconds: ttl }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus("saved");
    } catch {
      setStatus("error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={save} className="space-y-2">
      <h3 className="text-sm font-semibold text-stone-700">{t("pii.expiration.title")}</h3>
      <fieldset disabled={!loaded || saving} className="space-y-2 disabled:opacity-50">
        <label className="flex items-center gap-2 text-sm text-stone-700">
          <input type="checkbox" checked={enabled} onChange={(e) => { setEnabled(e.target.checked); setStatus(null); }} />
          {t("pii.expiration.enabled")}
        </label>
        <label className="flex items-center gap-3 text-sm text-stone-700">
          {t("pii.expiration.seconds")}
          <input type="number" min="1" max="2147483647" step="1" required disabled={!enabled}
            value={seconds} onChange={(e) => { setSeconds(e.target.value); setStatus(null); }}
            className="w-32 rounded border border-stone-200 px-2 py-1" />
        </label>
        <button type="submit" className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white">
          {t("pii.expiration.save")}
        </button>
      </fieldset>
      <p className="text-xs text-stone-500">{t("pii.expiration.help")}</p>
      {status && <p role="status" className={status === "saved" ? "text-xs text-brand-600" : "text-xs text-red-600"}>
        {t(status === "saved" ? "pii.saved" : `pii.expiration.${status}`)}
      </p>}
    </form>
  );
}
