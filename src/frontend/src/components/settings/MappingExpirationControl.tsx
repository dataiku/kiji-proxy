import { useEffect, useState, useSyncExternalStore } from "react";
import { useTranslation } from "react-i18next";
import { apiUrl, isElectron } from "../../utils/providerHelpers";
import { createMappingExpirationStore, hasExpirationEdits } from "./mappingExpirationStore";

export default function MappingExpirationControl() {
  const { t } = useTranslation("settings");
  const [store] = useState(() => createMappingExpirationStore(apiUrl("/api/pii/mapping-expiration", isElectron)));
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot);
  const { enabled, seconds, baseline, latest, saving, status } = state;

  useEffect(() => store.watch(window, document), [store]);

  return (
    <form onSubmit={(event) => { event.preventDefault(); void store.save(); }} className="space-y-2">
      <h3 className="text-sm font-semibold text-stone-700">{t("pii.expiration.title")}</h3>
      <fieldset disabled={!baseline || saving} className="space-y-2 disabled:opacity-50">
        <label className="flex items-center gap-2 text-sm text-stone-700">
          <input type="checkbox" checked={enabled} onChange={(e) => store.edit({ enabled: e.target.checked })} />
          {t("pii.expiration.enabled")}
        </label>
        <label className="flex items-center gap-3 text-sm text-stone-700">
          {t("pii.expiration.seconds")}
          <input type="number" min="1" max="2147483647" step="1" required disabled={!enabled}
            value={seconds} onChange={(e) => store.edit({ seconds: e.target.value })}
            className="w-32 rounded border border-stone-200 px-2 py-1" />
        </label>
        <button type="submit" disabled={status === "conflict"}
          className="rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
          {t("pii.expiration.save")}
        </button>
      </fieldset>
      <button type="button" disabled={saving} onClick={() => { void store.refresh(); }} className="text-sm text-brand-700 underline">
        {t("pii.expiration.refresh")}
      </button>
      <p className="text-xs text-stone-500">{t("pii.expiration.help")}</p>
      {hasExpirationEdits(state) && <p className="text-xs text-stone-500">{t("pii.expiration.unsaved")}</p>}
      {status === "conflict" && latest ? <div role="status" className="space-y-2 text-sm text-amber-800">
        <p>{t("pii.expiration.conflict", { seconds: latest.ttl_seconds,
          state: t(latest.enabled ? "pii.expiration.on" : "pii.expiration.off") })}</p>
        <div className="flex gap-3">
          <button type="button" onClick={store.useServerSettings} className="underline">{t("pii.expiration.useServer")}</button>
          <button type="button" onClick={store.keepEdits} className="underline">{t("pii.expiration.keepEdits")}</button>
        </div>
      </div> : status && <p role="status" className={status === "saved" ? "text-xs text-brand-600" : "text-xs text-red-600"}>
        {t(status === "saved" ? "pii.saved" : `pii.expiration.${status}`)}
      </p>}
    </form>
  );
}
