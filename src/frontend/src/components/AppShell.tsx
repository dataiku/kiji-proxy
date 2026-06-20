import { useState } from "react";
import { isElectron } from "../utils/providerHelpers";
import { useServerHealth } from "../hooks/useServerHealth";
import Sidebar, { ViewId, ModalId } from "./dashboard/Sidebar";
import DashboardView from "./dashboard/DashboardView";
import SettingsView from "./settings/SettingsView";
import PrivacyProxyUI from "./privacy-proxy-ui";

/**
 * Top-level shell for the proxy app.
 *
 * The deep-forest sidebar is the always-on server (persistent nav + live
 * status); the light area is the workspace. The Dashboard is home; the original
 * masking tool lives under the "Playground" view; Settings is its own
 * (SettingsView) page. Activity / Mappings reuse the existing modals (rendered
 * by PrivacyProxyUI) and are triggered via a signal so we don't duplicate their
 * state here.
 */
export default function AppShell() {
  const [view, setView] = useState<ViewId>("dashboard");
  const [modalSignal, setModalSignal] = useState<
    { type: "mappings" | "logging"; n: number } | undefined
  >(undefined);
  // Bumped after a successful provider save so the Playground re-reads its
  // cached provider config (the selector ✓ marks, active provider, etc.).
  const [settingsReloadN, setSettingsReloadN] = useState(0);

  const { serverStatus, serverHealth, modelSignature, version } =
    useServerHealth(isElectron);

  const server = {
    status:
      serverStatus === "online"
        ? serverHealth.modelHealthy
          ? "online"
          : "degraded"
        : "offline",
    version,
    model: modelSignature,
    port: 8080,
  } as const;

  const openModal = (modal: ModalId) => {
    setView("playground");
    setModalSignal({ type: modal, n: Date.now() });
  };

  return (
    <div className="kiji-shell">
      <Sidebar
        active={view}
        onNavigate={setView}
        onOpenModal={openModal}
        server={server}
      />
      <main className="kiji-main">
        {view === "dashboard" && <DashboardView />}
        {view === "settings" && (
          <SettingsView
            onProvidersSaved={() => setSettingsReloadN((n) => n + 1)}
          />
        )}
        {/* Kept mounted so Playground state and modals persist across nav */}
        <div hidden={view !== "playground"}>
          <PrivacyProxyUI
            embedded
            openModalSignal={modalSignal}
            reloadSettingsSignal={settingsReloadN}
          />
        </div>
      </main>
    </div>
  );
}
