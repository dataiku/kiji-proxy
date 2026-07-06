import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ShieldCheck, FolderOpen, ChevronRight } from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface CertificatesSectionProps {
  /** Opens the shared CA certificate setup wizard (owned by SettingsView). */
  onOpenCACert: () => void;
}

export default function CertificatesSection({
  onOpenCACert,
}: CertificatesSectionProps) {
  const { t } = useTranslation("settings");
  const [error, setError] = useState<string | null>(null);

  const handleRevealCert = async () => {
    setError(null);
    if (!isElectron || !window.electronAPI) {
      setError(t("certificates.messages.revealDesktopOnly"));
      return;
    }
    const result = await window.electronAPI.revealCACert();
    if (!result.success) {
      setError(result.error || t("certificates.messages.openFolderFailed"));
    }
  };

  const revealLabel =
    isElectron && window.electronAPI?.platform === "darwin"
      ? t("certificates.revealFinder")
      : t("certificates.revealExplorer");

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            {t("certificates.title")}
          </h2>
          <p className="text-[13px] text-stone-500">
            {t("certificates.subtitle")}
          </p>
        </div>
      </div>

      <div className="space-y-3">
        {/* Set up CA certificate (opens the wizard) */}
        <button
          onClick={onOpenCACert}
          className="group w-full flex items-center justify-between gap-3 rounded-xl ring-1 ring-stone-200 p-4 text-left hover:ring-brand-200 hover:bg-brand-50/40 transition-all"
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-stone-100 group-hover:bg-brand-50 flex items-center justify-center text-stone-600 group-hover:text-brand-600 transition-colors shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <p className="font-medium text-stone-700">
                {t("certificates.setup.title")}
              </p>
              <p className="text-xs text-stone-500">
                {t("certificates.setup.help")}
              </p>
            </div>
          </div>
          <ChevronRight className="w-5 h-5 text-stone-400 group-hover:text-brand-500 transition-colors shrink-0" />
        </button>

        {/* Reveal CA cert in Finder / Explorer */}
        {isElectron && window.electronAPI && (
          <button
            onClick={handleRevealCert}
            className="group w-full flex items-center justify-between gap-3 rounded-xl ring-1 ring-stone-200 p-4 text-left hover:ring-brand-200 hover:bg-brand-50/40 transition-all"
          >
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-stone-100 group-hover:bg-brand-50 flex items-center justify-center text-stone-600 group-hover:text-brand-600 transition-colors shrink-0">
                <FolderOpen className="w-5 h-5" />
              </div>
              <div>
                <p className="font-medium text-stone-700">{revealLabel}</p>
                <p className="text-xs text-stone-500">
                  {t("certificates.revealHelp")}
                </p>
              </div>
            </div>
            <ChevronRight className="w-5 h-5 text-stone-400 group-hover:text-brand-500 transition-colors shrink-0" />
          </button>
        )}

        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>
    </section>
  );
}
