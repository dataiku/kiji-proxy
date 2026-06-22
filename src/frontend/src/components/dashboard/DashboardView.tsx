import { useState } from "react";
import { Shield, TrendingUp, Send, ArrowRight } from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";
import { useDashboardData } from "../../hooks/useDashboardData";
import type {
  DashboardRange,
  TimeseriesPoint,
  CompositionEntry,
  RecentIntercept,
} from "../../types/dashboard";

const RANGES: { id: DashboardRange; label: string }[] = [
  { id: "24h", label: "24 hours" },
  { id: "7d", label: "7 days" },
  { id: "30d", label: "30 days" },
  { id: "90d", label: "90 days" },
];

const SEGMENT_COLORS = ["#1f8568", "#5dc1a6", "#ecaa4f", "#195545", "#d6d3d1"];

// The dashboard feed is a glance, not a log — show only the latest few and send
// users to the Activity tab for the full, paginated history.
const RECENT_LIMIT = 5;

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

function relativeTime(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  if (Number.isNaN(diff)) return "";
  const s = Math.max(0, Math.floor(diff / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/* ---- tiny inline charts (no chart dependency) ---- */

function Sparkline({ values }: { values: number[] }) {
  if (!values || values.length < 2) return null;
  const w = 120;
  const h = 34;
  const mx = Math.max(...values) * 1.1 || 1;
  const pts = values
    .map((v, i) => `${(i * w) / (values.length - 1)},${h - (v / mx) * h}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={120} height={34} aria-hidden="true">
      <polyline
        points={pts}
        fill="none"
        stroke="#1f8568"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function AreaChart({ points }: { points: TimeseriesPoint[] }) {
  if (!points || points.length < 2) {
    return (
      <div className="h-[200px] flex items-center justify-center text-sm text-stone-400">
        Not enough activity yet to chart.
      </div>
    );
  }
  const W = 620;
  const H = 200;
  const pad = 14;
  const vals = points.map((p) => p.v);
  const mx = Math.max(...vals) * 1.08 || 1;
  const X = (i: number) => pad + (i * (W - 2 * pad)) / (points.length - 1);
  const Y = (v: number) => H - pad - (v / mx) * (H - 2 * pad);
  const line = points
    .map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`)
    .join(" ");
  const area =
    `M${X(0).toFixed(1)},${H - pad} ` +
    points.map((p, i) => `L${X(i).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ") +
    ` L${X(points.length - 1).toFixed(1)},${H - pad} Z`;
  const lx = X(points.length - 1);
  const ly = Y(points[points.length - 1].v);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={200}
      preserveAspectRatio="none"
      style={{ display: "block" }}
    >
      <defs>
        <linearGradient id="kijiArea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5dc1a6" stopOpacity="0.38" />
          <stop offset="1" stopColor="#5dc1a6" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[1, 2, 3, 4].map((g) => (
        <line
          key={g}
          x1={pad}
          x2={W - pad}
          y1={H - pad - (g / 4) * (H - 2 * pad)}
          y2={H - pad - (g / 4) * (H - 2 * pad)}
          stroke="rgba(6,49,46,.06)"
          strokeWidth={1}
        />
      ))}
      <path d={area} fill="url(#kijiArea)" />
      <path
        d={line}
        fill="none"
        stroke="#1f8568"
        strokeWidth={2.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={lx} cy={ly} r={5} fill="#1f8568" stroke="#fff" strokeWidth={2.5} />
    </svg>
  );
}

function Donut({
  total,
  byType,
}: {
  total: number;
  byType: CompositionEntry[];
}) {
  const segments: string[] = [];
  let acc = 0;
  byType.forEach((e, i) => {
    const start = acc * 100;
    acc += e.share;
    const end = acc * 100;
    segments.push(`${SEGMENT_COLORS[i % SEGMENT_COLORS.length]} ${start}% ${end}%`);
  });
  const gradient =
    segments.length > 0
      ? `conic-gradient(${segments.join(", ")})`
      : "conic-gradient(#e7e5e4 0% 100%)";
  return (
    <div className="flex items-center gap-5 mt-3.5">
      <div
        className="kiji-donut w-[138px] h-[138px]"
        style={{ background: gradient }}
      >
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center">
          <b className="font-mono text-xl text-brand-900">{fmt(total)}</b>
          <small className="text-[10px] tracking-[0.1em] uppercase text-stone-400">
            entities
          </small>
        </div>
      </div>
      <div className="flex-1 flex flex-col gap-2.5 text-[13px]">
        {byType.map((e, i) => (
          <div key={e.type} className="flex items-center gap-2.5">
            <span
              className="w-2.5 h-2.5 rounded-[3px]"
              style={{ background: SEGMENT_COLORS[i % SEGMENT_COLORS.length] }}
            />
            {e.label}
            <span className="ml-auto font-mono text-xs text-stone-500">
              {Math.round(e.share * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeedRow({ item }: { item: RecentIntercept }) {
  return (
    <div className="grid grid-cols-[auto_1fr_auto] gap-3 items-center py-3 border-b border-brand-900/[0.06] last:border-0">
      <div className="w-[34px] h-[34px] rounded-[9px] bg-brand-50 border border-brand-100 grid place-items-center text-brand-600">
        <Shield className="w-4 h-4" />
      </div>
      <div className="min-w-0">
        <div className="text-[13px] font-semibold truncate">
          {item.source} <span className="text-stone-500 font-medium">&rarr; {item.provider}</span>
        </div>
        <div className="text-xs text-stone-400 font-mono mt-0.5 truncate">
          {item.preview ?? "request"} &middot; masked{" "}
          <span className="bg-brand-900 text-brand-100 rounded-[3px] px-1.5 tracking-widest">
            ██████
          </span>
        </div>
      </div>
      <div className="text-right text-xs">
        <div className="font-bold text-brand-700">{item.pii_count} PII</div>
        <div className="text-stone-400 text-[11px] mt-0.5">
          {relativeTime(item.ts)}
        </div>
      </div>
    </div>
  );
}

/* ---- the view ---- */

export default function DashboardView({
  onShowActivity,
}: {
  onShowActivity?: () => void;
}) {
  const [range, setRange] = useState<DashboardRange>("30d");
  const { data, loading, error } = useDashboardData(range, isElectron);

  const intercepting = data?.server.status === "online";
  const isEmpty = data != null && data.kpis.pii_protected.total === 0;

  return (
    <div>
      {/* top bar */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
            Overview
          </h1>
          <p className="text-stone-500 text-[13px] mt-0.5">
            Everything routed through Kiji, masked before it leaves your machine.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <select
            value={range}
            onChange={(e) => setRange(e.target.value as DashboardRange)}
            className="text-[13px] font-medium text-stone-700 bg-white border border-brand-900/10 rounded-lg px-3 py-2 shadow-soft focus:outline-none focus:border-brand-500"
          >
            {RANGES.map((r) => (
              <option key={r.id} value={r.id}>
                Last {r.label}
              </option>
            ))}
          </select>
          <span
            className={`inline-flex items-center gap-2 text-[13px] font-medium bg-white border border-brand-900/10 rounded-lg px-3 py-2 shadow-soft ${
              intercepting ? "text-brand-700" : "text-stone-500"
            }`}
          >
            <span
              className={`w-[7px] h-[7px] rounded-full ${
                intercepting ? "bg-brand-500" : "bg-stone-300"
              }`}
            />
            {intercepting ? "Intercepting" : "Idle"}
          </span>
        </div>
      </div>

      {error && !data && (
        <div className="card p-5 text-sm text-red-700 bg-red-50 ring-1 ring-red-200 border-0">
          Couldn&rsquo;t reach the Kiji server: {error}. Make sure the proxy is
          running on localhost:8080.
        </div>
      )}

      {loading && !data && (
        <div className="grid grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="card p-5 animate-pulse">
              <div className="h-3 w-24 bg-stone-100 rounded mb-4" />
              <div className="h-8 w-28 bg-stone-100 rounded" />
            </div>
          ))}
        </div>
      )}

      {data && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-[1.4fr_1fr_1fr] gap-4 mb-4">
            <div className="card p-5">
              <div className="flex justify-between items-end">
                <div>
                  <div className="text-[11px] font-semibold tracking-[0.13em] uppercase text-stone-400">
                    PII items protected
                  </div>
                  <div className="font-mono text-[34px] font-semibold text-brand-700 mt-2 leading-none">
                    {fmt(data.kpis.pii_protected.total)}
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-stone-500 mt-2.5">
                    <span className="inline-flex items-center gap-1 text-brand-600 font-semibold">
                      <TrendingUp className="w-3.5 h-3.5" /> +
                      {fmt(data.kpis.pii_protected.delta)}
                    </span>
                    in the last {data.kpis.pii_protected.delta_window}
                  </div>
                </div>
                <Sparkline values={data.kpis.pii_protected.spark} />
              </div>
            </div>

            <div className="card p-5">
              <div className="text-[11px] font-semibold tracking-[0.13em] uppercase text-stone-400">
                Requests proxied
              </div>
              <div className="font-mono text-[34px] font-semibold text-brand-900 mt-2 leading-none">
                {fmt(data.kpis.requests_proxied.total)}
              </div>
              <div className="text-xs text-stone-500 mt-2.5">
                {fmt(data.kpis.requests_proxied.today)} today
              </div>
            </div>

            <div className="card p-5">
              <div className="text-[11px] font-semibold tracking-[0.13em] uppercase text-stone-400">
                Avg added latency
              </div>
              <div className="font-mono text-[34px] font-semibold text-brand-900 mt-2 leading-none">
                {data.kpis.latency_ms.avg_added}
                <span className="text-lg text-stone-500">ms</span>
              </div>
              <div className="text-xs text-stone-500 mt-2.5">
                privacy, no slowdown
              </div>
            </div>
          </div>

          {/* chart + donut */}
          <div className="grid grid-cols-[1.55fr_1fr] gap-4 mb-4 items-stretch">
            <div className="card p-5">
              <h3 className="text-sm font-bold flex items-center justify-between">
                PII masked over time
                <span className="text-xs text-stone-400 font-medium">
                  last {RANGES.find((r) => r.id === range)?.label}
                </span>
              </h3>
              <div className="mt-3.5">
                {isEmpty ? (
                  <div className="h-[200px] flex flex-col items-center justify-center text-center gap-1">
                    <p className="text-sm font-medium text-stone-600">
                      Kiji is running and ready.
                    </p>
                    <p className="text-xs text-stone-400">
                      Route your first request through the proxy to start
                      protecting data.
                    </p>
                  </div>
                ) : (
                  <AreaChart points={data.timeseries.points} />
                )}
              </div>
            </div>

            <div className="card p-5 flex flex-col">
              <h3 className="text-sm font-bold flex items-center justify-between">
                What we masked
                <span className="text-xs text-stone-400 font-medium">
                  this period
                </span>
              </h3>
              <div className="flex-1 flex flex-col justify-center">
                {data.composition.by_type.length > 0 ? (
                  <Donut
                    total={data.composition.total}
                    byType={data.composition.by_type}
                  />
                ) : (
                  <div className="h-[138px] flex items-center justify-center text-sm text-stone-400">
                    Nothing masked yet.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* providers + recent */}
          <div className="grid grid-cols-2 gap-4 items-start">
            <div className="card p-5">
              <h3 className="text-sm font-bold flex items-center justify-between">
                Traffic by provider
                <span className="text-xs text-stone-400 font-medium">requests</span>
              </h3>
              <div className="flex flex-col gap-3.5 mt-4">
                {data.by_provider.map((p) => (
                  <div key={p.provider}>
                    <div className="flex justify-between text-[13px] mb-1.5">
                      <b className="font-semibold">{p.label}</b>
                      <span className="text-stone-500 font-mono text-xs">
                        {fmt(p.requests)}
                      </span>
                    </div>
                    <div className="h-[9px] bg-stone-100 rounded-md overflow-hidden">
                      <div
                        className="h-full rounded-md bg-gradient-to-r from-brand-400 to-brand-600"
                        style={{ width: `${Math.max(3, p.share * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
                {data.by_provider.length === 0 && (
                  <p className="text-sm text-stone-400 py-6 text-center">
                    No traffic yet.
                  </p>
                )}
              </div>
              {data.highlights && (
                <div className="flex gap-2.5 mt-4 pt-4 border-t border-brand-900/[0.06]">
                  <div className="flex-1 bg-stone-50 border border-brand-900/[0.06] rounded-[10px] px-3 py-2.5">
                    <div className="text-[10px] tracking-[0.12em] uppercase text-stone-400 font-semibold">
                      Peak today
                    </div>
                    <div className="text-[15px] font-semibold text-brand-900 mt-1">
                      {data.highlights.peak_rpm_today ?? "—"}{" "}
                      <span className="font-mono font-normal text-stone-500 text-xs">
                        req/min
                      </span>
                    </div>
                  </div>
                  <div className="flex-1 bg-stone-50 border border-brand-900/[0.06] rounded-[10px] px-3 py-2.5">
                    <div className="text-[10px] tracking-[0.12em] uppercase text-stone-400 font-semibold">
                      Busiest source
                    </div>
                    <div className="text-[15px] font-semibold text-brand-900 mt-1 truncate">
                      {data.highlights.busiest_source ?? "—"}
                    </div>
                  </div>
                  <div className="flex-1 bg-stone-50 border border-brand-900/[0.06] rounded-[10px] px-3 py-2.5">
                    <div className="text-[10px] tracking-[0.12em] uppercase text-stone-400 font-semibold">
                      Detection conf.
                    </div>
                    <div className="text-[15px] font-semibold text-brand-900 mt-1">
                      {(data.kpis.detection_confidence.avg * 100).toFixed(1)}
                      <span className="font-mono font-normal text-stone-500 text-xs">
                        % avg
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="card p-5">
              <h3 className="text-sm font-bold flex items-center justify-between">
                Recent intercepts
                <span className="text-xs text-stone-400 font-medium">live</span>
              </h3>
              <div className="flex flex-col mt-2">
                {data.recent.slice(0, RECENT_LIMIT).map((item) => (
                  <FeedRow key={item.id} item={item} />
                ))}
                {data.recent.length === 0 && (
                  <div className="flex flex-col items-center justify-center text-center gap-2 py-10">
                    <Send className="w-5 h-5 text-stone-300" />
                    <p className="text-sm text-stone-400">
                      No requests intercepted yet.
                    </p>
                  </div>
                )}
              </div>
              {data.recent.length > 0 && onShowActivity && (
                <button
                  type="button"
                  onClick={onShowActivity}
                  className="mt-3 w-full flex items-center justify-center gap-1.5 text-xs font-semibold text-brand-700 hover:text-brand-800"
                >
                  View all activity
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
