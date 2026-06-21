import { useEffect, useState } from "react";
import { apiUrl } from "../utils/providerHelpers";
import type { DashboardResponse, DashboardRange } from "../types/dashboard";

interface UseDashboardData {
  data: DashboardResponse | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/**
 * Fetches `GET /api/dashboard` and refreshes on an interval.
 *
 * Mirrors the conventions of the other server hooks (useServerHealth, useLogs):
 * it relies on `apiUrl()` so it works in both Electron (direct to the Go server)
 * and web (proxied) modes. The aggregate response embeds the timeseries and
 * recent-activity data, so a single polled endpoint is all the UI needs.
 */
export function useDashboardData(
  range: DashboardRange,
  isElectron: boolean,
  pollMs = 10000
): UseDashboardData {
  const [data, setData] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadN, setReloadN] = useState(0);
  const [prevRange, setPrevRange] = useState(range);

  // Re-show the loading state when the range changes. Done during render (React's
  // "adjust state on prop change" pattern) so background polls — which don't
  // change `range` — refresh silently without flashing a skeleton.
  if (range !== prevRange) {
    setPrevRange(range);
    setLoading(true);
  }

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const res = await fetch(
          apiUrl(`/api/dashboard?range=${range}`, isElectron),
          { signal: controller.signal }
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
    };
    load();
    const id = pollMs ? window.setInterval(load, pollMs) : undefined;
    return () => {
      controller.abort();
      if (id) window.clearInterval(id);
    };
  }, [range, isElectron, pollMs, reloadN]);

  return { data, loading, error, reload: () => setReloadN((n) => n + 1) };
}
