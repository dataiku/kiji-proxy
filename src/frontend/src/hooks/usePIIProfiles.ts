import { useState, useEffect, useCallback } from "react";
import type { PIIProfileSummary } from "../types/piiProfile";
import { apiUrl, isElectron } from "../utils/providerHelpers";

/**
 * Loads the PII profile list (ids, names and the default id) from the backend.
 * Re-fetches whenever `reloadSignal` changes, so hosts can refresh it after
 * profiles are edited in Settings.
 */
export function usePIIProfiles(reloadSignal?: number) {
  const [profiles, setProfiles] = useState<PIIProfileSummary[]>([]);
  const [defaultId, setDefaultId] = useState("");

  const reload = useCallback(async () => {
    try {
      const res = await fetch(apiUrl("/api/pii/profiles", isElectron));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setProfiles(data.profiles ?? []);
      setDefaultId(data.default ?? "");
    } catch (error) {
      console.error("Error loading PII profiles:", error);
    }
  }, []);

  useEffect(() => {
    /* eslint-disable-next-line react-hooks/set-state-in-effect */
    reload();
  }, [reload, reloadSignal]);

  return { profiles, defaultId, reload };
}
