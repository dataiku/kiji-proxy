import { useState } from "react";
import { ShieldCheck, FolderOpen, ChevronRight } from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface CertificatesSectionProps {
  /** Opens the shared CA certificate setup wizard (owned by SettingsView). */
  onOpenCACert: () => void;
}

export default function CertificatesSection({
  onOpenCACert,
}: CertificatesSectionProps) {
  const [error, setError] = useState<string | null>(null);

  const handleRevealCert = async () => {
    setError(null);
    if (!isElectron || !window.electronAPI) {
      setError("Reveal in Finder is only available in the desktop app.");
      return;
    }
    const result = await window.electronAPI.revealCACert();
    if (!result.success) {
      setError(result.error || "Failed to open the certificate folder.");
    }
  };

  const revealLabel =
    isElectron && window.electronAPI?.platform === "darwin"
      ? "Reveal CA cert in Finder"
      : "Show CA cert in Explorer";

  return (
    <section className="card p-6 md:p-7">
      {/* Section header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
          <ShieldCheck className="w-5 h-5" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-brand-900 tracking-tight">
            Certificates
          </h2>
          <p className="text-[13px] text-stone-500">
            Trust Kiji's root certificate to intercept HTTPS traffic.
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
              <p className="font-medium text-stone-700">Set up CA certificate</p>
              <p className="text-xs text-stone-500">
                Step-by-step instructions to trust the certificate system-wide
                or per browser.
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
                  Open the folder containing the proxy's root certificate.
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
