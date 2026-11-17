// Pure helpers for the transparent-proxy pause control. Kept free of React so
// tests/proxy-pause.test.cjs can exercise them directly.

// Must match allowedPauseMinutes in src/backend/server/proxy_pause.go.
export const PAUSE_DURATIONS_MINUTES = [1, 5, 15, 30, 60] as const;
export type PauseMinutes = (typeof PAUSE_DURATIONS_MINUTES)[number];
export const DEFAULT_PAUSE_MINUTES: PauseMinutes = 15;

// Whole seconds left until pausedUntil (rounded up, never negative).
export function secondsUntil(
  pausedUntil: string | null | undefined,
  nowMs: number
): number {
  if (!pausedUntil) return 0;
  const deadline = Date.parse(pausedUntil);
  if (Number.isNaN(deadline)) return 0;
  return Math.max(0, Math.ceil((deadline - nowMs) / 1000));
}

// Countdown label such as "14:05" or "0:09".
export function formatCountdown(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

// Local wall-clock time the pause ends, e.g. "14:30" or "2:30 PM".
export function formatResumeTime(pausedUntil: string, locale?: string): string {
  return new Date(pausedUntil).toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
  });
}
