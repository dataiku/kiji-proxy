/**
 * Types for the Kiji proxy dashboard API (`GET /api/dashboard`).
 */

export type DashboardRange = "24h" | "7d" | "30d" | "90d" | "all";

export interface DashboardServer {
  status: "online" | "degraded" | "offline";
  uptime_seconds: number;
  version: string;
  port: number;
  model: {
    signature: string;
    hash: string;
    healthy: boolean;
  };
}

export interface KpiPiiProtected {
  total: number;
  delta: number;
  delta_window: string;
  spark: number[];
}

export interface DashboardKpis {
  pii_protected: KpiPiiProtected;
  requests_proxied: { total: number; today: number };
  pii_leaked: { total: number; masked_rate: number };
  latency_ms: { avg_added: number; p95_added: number };
  detection_confidence: { avg: number };
}

export interface TimeseriesPoint {
  t: string;
  v: number;
}

export interface DashboardTimeseries {
  metric: string;
  bucket: "hour" | "day" | "week";
  points: TimeseriesPoint[];
}

export interface CompositionEntry {
  type: string;
  label: string;
  count: number;
  share: number; // 0..1
}

export interface DashboardComposition {
  total: number;
  by_type: CompositionEntry[];
}

export interface ProviderEntry {
  provider: string;
  label: string;
  requests: number;
  share: number; // 0..1, relative to the top provider
}

export interface RecentIntercept {
  id: string;
  ts: string;
  source: string;
  provider: string;
  pii_count: number;
  types: string[];
  /** Already masked, generic descriptor — never raw PII. May be null. */
  preview: string | null;
}

export interface DashboardResponse {
  generated_at: string;
  range: DashboardRange;
  server: DashboardServer;
  kpis: DashboardKpis;
  timeseries: DashboardTimeseries;
  composition: DashboardComposition;
  by_provider: ProviderEntry[];
  recent: RecentIntercept[];
  highlights?: {
    peak_rpm_today?: number;
    busiest_source?: string;
  };
}
