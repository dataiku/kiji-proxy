export interface ExpirationPolicy {
  ttl_seconds: number;
}

interface ExpirationState {
  baseline: ExpirationPolicy | null;
  latest: ExpirationPolicy | null;
  seconds: string;
  saving: boolean;
  status: "saved" | "error" | "uncertain" | "invalid" | "conflict" | null;
}

function samePolicy(a: ExpirationPolicy | null, b: ExpirationPolicy | null) {
  return a?.ttl_seconds === b?.ttl_seconds;
}

export function hasExpirationEdits(state: ExpirationState) {
  return state.baseline !== null && (
    state.seconds.trim() === "" || Number(state.seconds) !== state.baseline.ttl_seconds
  );
}

function readPolicy(value: ExpirationPolicy): ExpirationPolicy {
  if (!value || !Number.isInteger(value.ttl_seconds) ||
      value.ttl_seconds < 0 || value.ttl_seconds > 2147483647) {
    throw new Error("Invalid expiration policy");
  }
  return { ttl_seconds: value.ttl_seconds };
}

class SettingsHTTPError extends Error {}

// Keep request ordering and draft reconciliation separate from rendering so
// focus events and slow responses cannot silently replace a user's edits.
export function createMappingExpirationStore(url: string, request: typeof fetch = fetch,
  scheduleTimeout: (callback: () => void) => () => void = (callback) => {
    const timer = setTimeout(callback, 10000);
    return () => clearTimeout(timer);
  }) {
  let state: ExpirationState = {
    baseline: null, latest: null, seconds: "300", saving: false, status: null,
  };
  let sequence = 0;
  let activeRead: AbortController | null = null;
  let activeSave: AbortController | null = null;
  let saveFailed = false;
  let definiteSaveFailure = false;
  let pendingSave: ExpirationPolicy | null = null;
  const listeners: Set<() => void> = new Set();
  const publish = (patch: Partial<ExpirationState>) => {
    state = { ...state, ...patch };
    listeners.forEach((listener) => listener());
  };
  const accept = (policy: ExpirationPolicy, status: ExpirationState["status"] = null) => {
    saveFailed = false;
    definiteSaveFailure = false;
    pendingSave = null;
    publish({ baseline: policy, latest: policy,
      seconds: String(policy.ttl_seconds), status });
  };

  const fetchPolicy = async (controller: AbortController, init: RequestInit = {}) => {
    let onAbort = () => {};
    const aborted: Promise<never> = new Promise((_, reject) => {
      onAbort = () => reject(new Error("Settings request cancelled or timed out"));
      controller.signal.addEventListener("abort", onAbort, { once: true });
    });
    const stopTimeout = scheduleTimeout(() => controller.abort());
    try {
      // Bound both the headers and body. The race also settles callers whose
      // transport does not reject promptly after cancellation.
      return await Promise.race([
        (async () => {
          const response = await request(url, { ...init, signal: controller.signal });
          if (!response.ok && !(init.method === "POST" && response.status === 409)) {
            throw new SettingsHTTPError(`HTTP ${response.status}`);
          }
          return { response, policy: readPolicy(await response.json()) };
        })(),
        aborted,
      ]);
    } finally {
      stopTimeout();
      controller.signal.removeEventListener("abort", onAbort);
    }
  };

  const hasConflict = () => state.status === "conflict";

  const reconcile = (policy: ExpirationPolicy) => {
    if (pendingSave && samePolicy(policy, pendingSave)) {
      // The requested state is now on the server. A newer local draft must
      // survive, but its baseline can advance past our uncertain save.
      const draftMatches = state.seconds.trim() !== "" && Number(state.seconds) === policy.ttl_seconds;
      pendingSave = null;
      if (draftMatches) accept(policy, "saved");
      else {
        saveFailed = definiteSaveFailure;
        publish({ baseline: policy, latest: policy, status: definiteSaveFailure ? "error" : null });
      }
    } else if (pendingSave) {
      publish({ latest: policy, status: samePolicy(state.baseline, policy) ? definiteSaveFailure ? "error" : "uncertain" : "conflict" });
    } else if (!hasExpirationEdits(state)) {
      accept(policy, samePolicy(state.baseline, policy) && state.status === "saved" ? "saved" : null);
    } else {
      const status = saveFailed ? "error" : state.status === "invalid" ? "invalid" : null;
      publish({ latest: policy, status: samePolicy(state.baseline, policy) ? status : "conflict" });
    }
  };

  const refresh = async () => {
    if (state.saving) return false;
    const current = ++sequence;
    activeRead?.abort();
    const controller = new AbortController();
    activeRead = controller;
    try {
      const { policy } = await fetchPolicy(controller, { cache: "no-store" });
      if (current !== sequence) return false;
      reconcile(policy);
      return true;
    } catch {
      if (current === sequence) publish({ status: state.status === "conflict" ? "conflict" : pendingSave && !definiteSaveFailure ? "uncertain" : "error" });
      return false;
    } finally {
      if (activeRead === controller) activeRead = null;
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
        if (visibilityTarget.visibilityState === "visible" && activeRead === null) void refresh();
      });
      void refresh();
      return () => {
        ++sequence;
        activeRead?.abort();
        activeSave?.abort();
        if (state.saving) publish({ saving: false });
        focusTarget.removeEventListener("focus", onFocus);
        visibilityTarget.removeEventListener("visibilitychange", onVisible);
        stopPolling();
      };
    },
    edit: (patch: Partial<Pick<ExpirationState, "seconds">>) => {
      if (state.saving) return;
      saveFailed = false;
      definiteSaveFailure = false;
      publish({ ...patch, status: samePolicy(state.baseline, state.latest) ? pendingSave ? "uncertain" : null : "conflict" });
    },
    useServerSettings: () => { if (state.latest && !state.saving) accept(state.latest); },
    keepEdits: () => {
      if (state.latest && !state.saving) {
        saveFailed = false;
        definiteSaveFailure = false;
        pendingSave = null;
        publish({ baseline: state.latest, status: null });
      }
    },
    save: async () => {
      if (!state.baseline || state.saving || state.status === "conflict") return;
      if (pendingSave) {
        // Verify before retrying an ambiguous POST. Another pending save can
        // still finish later; conditional writes remain the final guard.
        if (!await refresh() || state.saving || hasConflict()) return;
        if (!pendingSave && !hasExpirationEdits(state)) return;
      }
      const ttl = Number(state.seconds);
      if (state.seconds.trim() === "" || !Number.isInteger(ttl) || ttl < 0 || ttl > 2147483647) {
        publish({ status: "invalid" });
        return;
      }
      const body = { ttl_seconds: ttl, expected: state.baseline };
      const current = ++sequence; // invalidate any GET started before this save
      activeRead?.abort();
      const controller = new AbortController();
      activeSave = controller;
      saveFailed = false;
      definiteSaveFailure = false;
      publish({ saving: true, status: null });
      try {
        const { response, policy } = await fetchPolicy(controller, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        });
        if (response.status === 409) {
          if (current === sequence) {
            if (samePolicy(policy, { ttl_seconds: body.ttl_seconds })) accept(policy, "saved");
            else reconcile(policy);
          }
        } else {
          if (current === sequence) accept(policy, "saved");
        }
      } catch (error) {
        if (current === sequence) {
          saveFailed = true;
          definiteSaveFailure = error instanceof SettingsHTTPError;
          if (!definiteSaveFailure) {
            pendingSave = { ttl_seconds: body.ttl_seconds };
          }
          publish({ status: definiteSaveFailure ? "error" : "uncertain" });
        }
      } finally {
        if (activeSave === controller) activeSave = null;
        if (current === sequence) publish({ saving: false });
      }
    },
  };
}
