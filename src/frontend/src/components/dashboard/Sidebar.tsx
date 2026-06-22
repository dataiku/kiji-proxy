import {
  LayoutDashboard,
  Wand2,
  List,
  Database,
  Settings as SettingsIcon,
  Info,
} from "lucide-react";
import logoImage from "../../../assets/kiji-dark.svg";

export type ViewId =
  | "dashboard"
  | "playground"
  | "activity"
  | "mappings"
  | "settings"
  | "about";

interface SidebarProps {
  active: ViewId | null;
  onNavigate: (view: ViewId) => void;
  server: {
    status: "online" | "degraded" | "offline";
    version: string | null;
    model: string | null;
    port?: number;
    uptimeSeconds?: number;
  };
  counts?: { activity?: number; mappings?: string };
}

function formatUptime(seconds?: number): string {
  if (!seconds || seconds < 0) return "—";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return d > 0 ? `${d}d ${pad(h)}h ${pad(m)}m` : `${pad(h)}h ${pad(m)}m`;
}

const STATUS = {
  online: { label: "Server online", dot: "#7fe8c8", glow: true },
  degraded: { label: "Model degraded", dot: "#ecaa4f", glow: true },
  offline: { label: "Server offline", dot: "#ff7c6c", glow: false },
} as const;

export default function Sidebar({
  active,
  onNavigate,
  server,
  counts,
}: SidebarProps) {
  const status = STATUS[server.status];

  return (
    <aside className="kiji-sidebar">
      <div className="flex items-center gap-3 px-2 pb-6 pt-1">
        <img src={logoImage} alt="" className="w-9 h-9" />
        <div className="text-[16px] font-bold text-white">
          Kiji <span className="font-medium text-brand-400">Proxy</span>
        </div>
      </div>

      <nav className="flex flex-col gap-0.5">
        <span className="px-2.5 pt-3 pb-2 text-[10px] font-semibold tracking-[0.18em] uppercase text-brand-400/70">
          Workspace
        </span>
        <button
          className={`kiji-navitem${active === "dashboard" ? " is-active" : ""}`}
          onClick={() => onNavigate("dashboard")}
        >
          <LayoutDashboard /> Dashboard
        </button>
        <button
          className={`kiji-navitem${active === "playground" ? " is-active" : ""}`}
          onClick={() => onNavigate("playground")}
        >
          <Wand2 /> Playground <span className="cnt">mask</span>
        </button>
        <button
          className={`kiji-navitem${active === "activity" ? " is-active" : ""}`}
          onClick={() => onNavigate("activity")}
        >
          <List /> Activity
          {counts?.activity != null && (
            <span className="cnt">{counts.activity}</span>
          )}
        </button>
        <button
          className={`kiji-navitem${active === "mappings" ? " is-active" : ""}`}
          onClick={() => onNavigate("mappings")}
        >
          <Database /> Mappings
          {counts?.mappings && <span className="cnt">{counts.mappings}</span>}
        </button>

        <span className="px-2.5 pt-4 pb-2 text-[10px] font-semibold tracking-[0.18em] uppercase text-brand-400/70">
          Configure
        </span>
        <button
          className={`kiji-navitem${active === "settings" ? " is-active" : ""}`}
          onClick={() => onNavigate("settings")}
        >
          <SettingsIcon /> Settings
        </button>
        <button
          className={`kiji-navitem${active === "about" ? " is-active" : ""}`}
          onClick={() => onNavigate("about")}
        >
          <Info /> About Kiji
        </button>
      </nav>

      <div className="mt-auto border-t border-white/10 pt-3.5 text-xs">
        <div className="flex items-center gap-2.5 font-semibold text-[#eafaf3]">
          <span
            className="w-2 h-2 rounded-full"
            style={{
              background: status.dot,
              boxShadow: status.glow ? `0 0 9px 1px ${status.dot}b3` : "none",
              animation: status.glow ? "pulse 2.4s ease-in-out infinite" : "none",
            }}
          />
          {status.label}
        </div>
        <div className="mt-2 font-mono text-[11px] leading-relaxed text-brand-400/80">
          uptime&nbsp;&nbsp;{formatUptime(server.uptimeSeconds)}
          <br />
          port&nbsp;&nbsp;&nbsp;&nbsp;
          <span className="text-[#a9cdc1]">localhost:{server.port ?? 8080}</span>
          <br />
          model&nbsp;&nbsp;
          <span className="text-[#a9cdc1]">{server.model ?? "—"}</span>
          {server.version && (
            <>
              {" "}
              <span className="text-brand-400/60">v{server.version}</span>
            </>
          )}
        </div>
      </div>
    </aside>
  );
}
