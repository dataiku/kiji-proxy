export interface ExpirationPolicy {
  enabled: boolean;
  ttl_seconds: number;
}

interface ExpirationState {
  baseline: ExpirationPolicy | null;
  latest: ExpirationPolicy | null;
  enabled: boolean;
  seconds: string;
  saving: boolean;
  status: "saved" | "error" | "invalid" | "conflict" | null;
}

function samePolicy(a: ExpirationPolicy | null, b: ExpirationPolicy | null) {
  return a?.enabled === b?.enabled && a?.ttl_seconds === b?.ttl_seconds;
}

export function hasExpirationEdits(state: ExpirationState) {
  return state.baseline !== null && (
    state.enabled !== state.baseline.enabled ||
    state.seconds.trim() === "" || Number(state.seconds) !== state.baseline.ttl_seconds
  );
}

function readPolicy(value: ExpirationPolicy): ExpirationPolicy {
  if (!value || typeof value.enabled !== "boolean" || !Number.isInteger(value.ttl_seconds) ||
      value.ttl_seconds < 1 || value.ttl_seconds > 2147483647) {
    throw new Error("Invalid expiration policy");
  }
  return { enabled: value.enabled, ttl_seconds: value.ttl_seconds };
}

// Keep request ordering and draft reconciliation separate from rendering so
// focus events and slow responses cannot silently replace a user's edits.
export function createMappingExpirationStore(url: string, request: typeof fetch = fetch) {
  let state: ExpirationState = {
    baseline: null, latest: null, enabled: true, seconds: "300", saving: false, status: null,
  };
  let sequence = 0;
  let pendingReads = 0;
  const listeners: Set<() => void> = new Set();
  const publish = (patch: Partial<ExpirationState>) => {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  };
  const accept = (policy: ExpirationPolicy, status: ExpirationState["status"] = null) => {
    publish({ baseline: policy, latest: policy, enabled: policy.enabled,
      seconds: String(policy.ttl_seconds), status });
  };

  const refresh = async () => {
    if (state.saving) return;
    const current = ++sequence;
    ++pendingReads;
    try {
      const response = await request(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const policy = readPolicy(await response.json());
      if (current !== sequence) return;
      // Inspect the draft when the response arrives, not when the GET began.
      if (!hasExpirationEdits(state)) {
        accept(policy);
      } else {
        publish({ latest: policy, status: samePolicy(state.baseline, policy) ? null : "conflict" });
      }
    } catch {
      if (current === sequence) publish({ status: state.status === "conflict" ? "conflict" : "error" });
    } finally {
      --pendingReads;
    }
  };

  return {
    getSnapshot: () => state,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    refresh,
    watch: (focusTarget: EventTarget, visibilityTarget: EventTarget & { visibilityState: string },
      schedule: (callback: () => void) => () => void = (callback) => {
        const timer = setInterval(callback, 5000);
        return () => clearInterval(timer);
      }) => {
      const onFocus = () => { void refresh(); };
      const onVisible = () => { if (visibilityTarget.visibilityState === "visible") void refresh(); };
      focusTarget.addEventListener("focus", onFocus);
      visibilityTarget.addEventListener("visibilitychange", onVisible);
      // Two clients can remain visible side by side without a focus event.
      const stopPolling = schedule(() => {
        if (visibilityTarget.visibilityState === "visible" && pendingReads === 0) void refresh();
      });
      void refresh();
      return () => {
        ++sequence;
        focusTarget.removeEventListener("focus", onFocus);
        visibilityTarget.removeEventListener("visibilitychange", onVisible);
        stopPolling();
      };
    },
    edit: (patch: Partial<Pick<ExpirationState, "enabled" | "seconds">>) => {
      if (state.saving) return;
      publish({ ...patch, status: samePolicy(state.baseline, state.latest) ? null : "conflict" });
    },
    useServerSettings: () => { if (state.latest && !state.saving) accept(state.latest); },
    keepEdits: () => {
      if (state.latest && !state.saving) publish({ baseline: state.latest, status: null });
    },
    save: async () => {
      if (!state.baseline || state.saving || state.status === "conflict") return;
      const ttl = Number(state.seconds);
      if (!Number.isInteger(ttl) || ttl < 1 || ttl > 2147483647) {
        publish({ status: "invalid" });
        return;
      }
      const body = { enabled: state.enabled, ttl_seconds: ttl, expected: state.baseline };
      const current = ++sequence; // invalidate any GET started before this save
      publish({ saving: true, status: null });
      try {
        const response = await request(url, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        });
        if (response.status === 409) {
          const latest = readPolicy(await response.json());
          if (current === sequence) publish({ latest, status: "conflict" });
        } else {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const policy = readPolicy(await response.json());
          if (current === sequence) accept(policy, "saved");
        }
      } catch {
        if (current === sequence) publish({ status: "error" });
      } finally {
        if (current === sequence) publish({ saving: false });
      }
    },
  };
}
