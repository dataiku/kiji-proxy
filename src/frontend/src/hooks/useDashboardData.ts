import { useCallback, useEffect, useState } from "react";
import { apiUrl } from "../utils/providerHelpers";
import type { DashboardResponse, DashboardRange } from "../types/dashboard";

interface UseDashboardData {
  data: DashboardResponse | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/**
 * Fetches `GET /v1/dashboard` and refreshes on an interval.
 *
 * Mirrors the conventions of the other server hooks (useServerHealth, useLogs):
 * it relies on `apiUrl()` so it works in both Electron (direct to the Go server)
 * and web (proxied) modes. Polling acts as the fallback for the SSE
 * `/v1/dashboard/stream` endpoint described in the API spec.
 */
export function useDashboardData(
  range: DashboardRange,
  isElectron: boolean,
  pollMs = 10000
): UseDashboardData {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const res = await fetch(
          apiUrl(`/v1/dashboard?range=${range}`, isElectron),
          { signal }
        );
        if (!res.ok) {
          throw new Error(`Dashboard request failed (${res.status})`);
        }
        const json = (await res.json()) as DashboardResponse;
        setData(json);
        setError(null);
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return;
        setError((err as Error)?.message || "Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    },
    [range, isElectron]
  );

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    load(controller.signal);
    const id = pollMs ? window.setInterval(() => load(), pollMs) : undefined;
    return () => {
      controller.abort();
      if (id) window.clearInterval(id);
    };
  }, [load, pollMs]);

  return { data, loading, error, reload: () => load() };
}
