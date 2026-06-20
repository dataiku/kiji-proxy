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
    <div className="w-full">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
          Settings
        </h1>
        <p className="text-stone-500 text-[13px] mt-0.5">
          Providers, PII detection, and advanced proxy configuration.
        </p>
      </div>

      {!isElectron ? (
        <div className="card p-6 text-sm text-stone-600">
          Settings are only available in the desktop app.
        </div>
      ) : (
        <div className="space-y-4 animate-rise-in">
          <ProvidersSection onSaved={onProvidersSaved} />
          <PIISection />
          <AdvancedSection onOpenCACert={openCACert} />
          <CertificatesSection onOpenCACert={openCACert} />
        </div>
      )}

      {/* Shared CA certificate setup wizard */}
      <CACertSetupModal
        isOpen={isCACertOpen}
        onClose={() => setIsCACertOpen(false)}
      />
    </div>
  );
}
