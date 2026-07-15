import { useState } from "react";
import { useTranslation } from "react-i18next";
import { isElectron } from "../../utils/providerHelpers";
import { useIsAdmin } from "../../hooks/useIsAdmin";
import LanguageSection from "./LanguageSection";
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
  const { t } = useTranslation("settings");
  // A single CA cert wizard, shared by the Advanced and Certificates sections.
  const [isCACertOpen, setIsCACertOpen] = useState(false);
  const openCACert = () => setIsCACertOpen(true);

  // Access policy: the desktop app grants every user all features. On a web
  // deployment, backend-configuration sections are admin-only. `isAdmin` is null
  // until it resolves, so web users see nothing gated until then; on the desktop
  // `isElectron` short-circuits so there's no wait or flash.
  const isAdmin = useIsAdmin();
  const canConfigureServer = isElectron || isAdmin === true;

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
          {t("title")}
        </h1>
        <p className="text-stone-500 text-[13px] mt-0.5">{t("subtitle")}</p>
      </div>

      <div className="space-y-4 animate-rise-in">
        {/* Language is a universal UI preference (renderer-only, persisted via
            i18next's localStorage cache), so it is shown to every user —
            including non-admins and web-mode users. */}
        <LanguageSection />
        {/* PII detection talks to the backend over HTTP, but the rules are
            server-wide, so on a web deployment only admins may change them; the
            desktop app shows it to everyone. The remaining sections rely on the
            desktop app's native integration — provider keys come from env vars
            on a server, the model directory uses a native picker, and
            certificate install is OS-level — so they are desktop-only. */}
        {isElectron && <ProvidersSection onSaved={onProvidersSaved} />}
        {canConfigureServer && <PIISection />}
        {isElectron ? (
          <>
            <AdvancedSection onOpenCACert={openCACert} />
            <CertificatesSection onOpenCACert={openCACert} />
          </>
        ) : (
          <div className="card p-6 text-sm text-stone-600">
            {t("serverNote")}
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
