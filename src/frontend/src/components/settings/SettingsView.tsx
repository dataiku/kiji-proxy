import { useState } from "react";
import { isElectron } from "../../utils/providerHelpers";
import ProvidersSection from "./ProvidersSection";
import PIISection from "./PIISection";
import AdvancedSection from "./AdvancedSection";
import CertificatesSection from "./CertificatesSection";
import CACertSetupModal from "../modals/CACertSetupModal";

interface SettingsViewProps {
  /** Called after a successful provider save so the host can refresh cached
   *  provider state (e.g. the Playground provider selector). */
  onProvidersSaved?: () => void;
}

export default function SettingsView({ onProvidersSaved }: SettingsViewProps) {
  // A single CA cert wizard, shared by the Advanced and Certificates sections.
  const [isCACertOpen, setIsCACertOpen] = useState(false);
  const openCACert = () => setIsCACertOpen(true);

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
          Settings
        </h1>
        <p className="text-stone-500 text-[13px] mt-0.5">
          Providers, PII detection, and advanced proxy configuration.
        </p>
      </div>

      <div className="space-y-4 animate-rise-in">
        {/* PII detection is configurable everywhere (talks to the backend over
            HTTP). The remaining sections rely on the desktop app's native
            integration — provider keys come from env vars on a server, the model
            directory uses a native picker, and certificate install is OS-level —
            so they are only shown in the desktop app. */}
        {isElectron && <ProvidersSection onSaved={onProvidersSaved} />}
        <PIISection />
        {isElectron ? (
          <>
            <AdvancedSection onOpenCACert={openCACert} />
            <CertificatesSection onOpenCACert={openCACert} />
          </>
        ) : (
          <div className="card p-6 text-sm text-stone-600">
            Provider keys, model directory, and certificate installation are
            configured via environment variables and the desktop app on server
            deployments.
          </div>
        )}
      </div>

      {/* Shared CA certificate setup wizard */}
      <CACertSetupModal
        isOpen={isCACertOpen}
        onClose={() => setIsCACertOpen(false)}
      />
    </div>
  );
}
