import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { PauseCircle } from "lucide-react";
import {
  DEFAULT_PAUSE_MINUTES,
  PAUSE_DURATIONS_MINUTES,
  type PauseMinutes,
  formatCountdown,
  formatResumeTime,
  secondsUntil,
} from "./proxyPause";

interface ProxyPauseControlProps {
  /** Pausing only makes sense while the transparent proxy is turned on. */
  proxyEnabled: boolean;
}

export default function ProxyPauseControl({
  proxyEnabled,
}: ProxyPauseControlProps) {
  const { t, i18n } = useTranslation("settings");
  const [minutes, setMinutes] = useState<PauseMinutes>(DEFAULT_PAUSE_MINUTES);
  const [pausedUntil, setPausedUntil] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);

  const apply = useCallback((result: ProxyPauseResult) => {
    setFailed(!result.success);
    if (result.success) {
      setPausedUntil(result.paused ? (result.paused_until ?? null) : null);
    }
    setNow(Date.now());
  }, []);

  // State is only set from the resolved promise, never synchronously, so this
  // is safe to call from effects.
  const refresh = useCallback(() => {
    window.electronAPI
      ?.getProxyPause()
      .then(apply)
      .catch((error: unknown) => {
        console.error("Error loading proxy pause:", error);
        setFailed(true);
      });
  }, [apply]);

  // Load the current pause, and re-check whenever the window regains focus
  // (timers do not run while the machine sleeps).
  useEffect(() => {
    refresh();
    const onFocus = () => refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  const remaining = secondsUntil(pausedUntil, now);
  const paused = pausedUntil !== null;

  // Tick the countdown once a second while paused.
  const expired = paused && remaining === 0;
  useEffect(() => {
    if (!paused || expired) return;
    const id = window.setTimeout(() => setNow(Date.now()), 1000);
    return () => window.clearTimeout(id);
  }, [paused, expired, now]);

  // At zero, poll the backend until it reports the pause is over, so the
  // switch only flips back once interception has actually resumed.
  useEffect(() => {
    if (!expired) return;
    refresh();
    const id = window.setInterval(refresh, 1000);
    return () => window.clearInterval(id);
  }, [expired, refresh]);

  const run = async (call: () => Promise<ProxyPauseResult>) => {
    setBusy(true);
    try {
      apply(await call());
    } catch (error) {
      console.error("Error updating proxy pause:", error);
      setFailed(true);
    } finally {
      setBusy(false);
    }
  };

  const pause = (value: PauseMinutes) => {
    setMinutes(value);
    if (!window.electronAPI) return;
    const api = window.electronAPI;
    void run(() => api.pauseProxy(value));
  };

  const handleToggle = () => {
    if (!window.electronAPI) return;
    const api = window.electronAPI;
    void run(() => (paused ? api.resumeProxy() : api.pauseProxy(minutes)));
  };

  const disabled = busy || (!proxyEnabled && !paused);

  return (
    <div className="mt-4 pt-4 border-t border-stone-200">
      <div className="flex items-center justify-between">
        <label
          htmlFor="proxy-pause-toggle"
          className="text-sm font-semibold text-stone-700 flex items-center gap-2"
        >
          <PauseCircle className="w-4 h-4" />
          {t("advanced.transparentProxy.pause.label")}
        </label>
        <button
          id="proxy-pause-toggle"
          type="button"
          role="switch"
          aria-checked={paused}
          aria-label={t("advanced.transparentProxy.pause.label")}
          onClick={handleToggle}
          disabled={disabled}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 ${
            paused ? "bg-amber-500" : "bg-stone-300"
          } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              paused ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      <div
        role="radiogroup"
        aria-label={t("advanced.transparentProxy.pause.durationLabel")}
        className="mt-3 flex flex-wrap gap-2"
      >
        {PAUSE_DURATIONS_MINUTES.map((value) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={minutes === value}
            disabled={disabled}
            onClick={() => (paused ? pause(value) : setMinutes(value))}
            className={`px-3 py-1 rounded-lg text-xs font-medium ring-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              minutes === value
                ? "bg-brand-50 text-brand-700 ring-brand-300"
                : "bg-white text-stone-600 ring-stone-200 hover:bg-stone-50"
            }`}
          >
            {t("advanced.transparentProxy.pause.minutes", { minutes: value })}
          </button>
        ))}
      </div>

      <p role="status" className="text-xs mt-2">
        {paused && pausedUntil ? (
          <span className="text-amber-800">
            {t("advanced.transparentProxy.pause.pausedUntil", {
              time: formatResumeTime(pausedUntil, i18n.language),
              remaining: formatCountdown(remaining),
            })}
          </span>
        ) : (
          <span className="text-stone-500">
            {proxyEnabled
              ? t("advanced.transparentProxy.pause.help")
              : t("advanced.transparentProxy.pause.requiresEnabled")}
          </span>
        )}
      </p>
      {failed && (
        <p className="text-xs text-red-600 mt-1">
          {t("advanced.transparentProxy.pause.error")}
        </p>
      )}
    </div>
  );
}
