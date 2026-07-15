import { useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import {
  X,
  ShieldCheck,
  Terminal,
  Globe,
  ExternalLink,
  FolderOpen,
  Copy,
  Check,
} from "lucide-react";
import { isElectron } from "../../utils/providerHelpers";

interface CACertSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Inline-markup tags shared by the instruction strings rendered via <Trans>.
// Provided to every step so each translation can use whichever it needs.
const STEP_COMPONENTS = {
  code: <code className="bg-stone-100 px-1.5 py-0.5 rounded text-xs" />,
  b: <strong />,
  accent: <strong className="text-brand-600" />,
  amber: <strong className="text-amber-600" />,
};

const SYSTEM_STEP_KEYS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"];
const FIREFOX_STEP_KEYS = ["s1", "s2", "s3", "s4", "s5"];
const CHROME_STEP_KEYS = ["s1", "s2", "s3"];

export default function CACertSetupModal({
  isOpen,
  onClose,
}: CACertSetupModalProps) {
  const { t } = useTranslation("modals");
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [currentTab, setCurrentTab] = useState<"system" | "browsers">("system");
  const [revealError, setRevealError] = useState<string | null>(null);
  const [commandCopied, setCommandCopied] = useState(false);

  const certInstallCommand = `sudo security add-trusted-cert \\
  -d \\
  -r trustRoot \\
  -k /Library/Keychains/System.keychain \\
  ~/"Library/Application Support/Kiji Privacy Proxy/certs/ca.crt"`;

  const handleCopyCommand = async () => {
    try {
      await navigator.clipboard.writeText(certInstallCommand);
      setCommandCopied(true);
      setTimeout(() => setCommandCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy cert install command:", error);
    }
  };

  if (!isOpen) return null;

  const handleConfirm = async () => {
    if (dontShowAgain && isElectron && window.electronAPI) {
      // Store preference using electron API
      try {
        await window.electronAPI.setCACertSetupDismissed(true);
      } catch (error) {
        console.error("Failed to save CA cert setup preference:", error);
      }
    }
    onClose();
  };

  const handleRevealCert = async () => {
    setRevealError(null);
    if (!isElectron || !window.electronAPI) {
      setRevealError(t("caCert.revealDesktopOnly"));
      return;
    }
    const result = await window.electronAPI.revealCACert();
    if (!result.success) {
      setRevealError(result.error || t("caCert.revealFailed"));
    }
  };

  return (
    <div className="fixed inset-0 bg-brand-950/40 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="bg-white rounded-2xl shadow-lift ring-1 ring-brand-900/10 max-w-2xl w-full max-h-[90vh] overflow-y-auto animate-rise-in">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-stone-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-brand-50 ring-1 ring-brand-100 flex items-center justify-center text-brand-600 shrink-0">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h2 className="text-lg font-semibold text-brand-900 tracking-tight">
              {t("caCert.title")}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-stone-400 hover:text-stone-600 hover:bg-stone-100 rounded-lg transition-colors"
            aria-label={t("common:actions.close")}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Introduction */}
          <div className="bg-brand-50 ring-1 ring-brand-200 rounded-xl p-4">
            <p className="text-sm text-brand-900">{t("caCert.intro")}</p>
          </div>

          {/* Reveal in Finder shortcut */}
          {isElectron && (
            <div>
              <button
                onClick={handleRevealCert}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-stone-600 bg-white ring-1 ring-stone-200 hover:bg-stone-50 hover:text-stone-800 transition-colors"
              >
                <FolderOpen className="w-4 h-4" />
                {window.electronAPI?.platform === "darwin"
                  ? t("caCert.revealFinder")
                  : t("caCert.revealExplorer")}
              </button>
              {revealError && (
                <p className="text-xs text-red-600 mt-2">{revealError}</p>
              )}
            </div>
          )}

          {/* Tabs */}
          <div className="inline-flex p-1 rounded-xl bg-stone-100 ring-1 ring-stone-200/70">
            <button
              onClick={() => setCurrentTab("system")}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                currentTab === "system"
                  ? "bg-white text-brand-900 shadow-soft"
                  : "text-stone-500 hover:text-stone-700"
              }`}
            >
              <Terminal className="w-4 h-4" />
              {t("caCert.tabs.system")}
            </button>
            <button
              onClick={() => setCurrentTab("browsers")}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                currentTab === "browsers"
                  ? "bg-white text-brand-900 shadow-soft"
                  : "text-stone-500 hover:text-stone-700"
              }`}
            >
              <Globe className="w-4 h-4" />
              {t("caCert.tabs.browsers")}
            </button>
          </div>

          {/* System-Wide Instructions */}
          {currentTab === "system" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-stone-700 mb-2">
                  {t("caCert.system.option1Title")}
                </h3>
                <div className="relative bg-brand-950 rounded-xl p-4 text-sm font-mono text-stone-100 overflow-x-auto">
                  <button
                    onClick={handleCopyCommand}
                    className="absolute top-2 right-2 inline-flex items-center gap-1 px-2 py-1 text-xs text-stone-300 hover:text-white hover:bg-stone-700 rounded transition-colors"
                    aria-label={
                      commandCopied
                        ? t("caCert.system.copiedAria")
                        : t("caCert.system.copyAria")
                    }
                    title={
                      commandCopied
                        ? t("caCert.system.copiedTitle")
                        : t("caCert.system.copyTitle")
                    }
                  >
                    {commandCopied ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-brand-400" />
                        <span>{t("caCert.system.copiedLabel")}</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>{t("caCert.system.copyLabel")}</span>
                      </>
                    )}
                  </button>
                  <pre className="whitespace-pre pr-20">
                    <code>{certInstallCommand}</code>
                  </pre>
                </div>
                <p className="text-xs text-stone-600 mt-2">
                  {t("caCert.system.commandHelp")}
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-stone-700 mb-2">
                  {t("caCert.system.option2Title")}
                </h3>
                <ol className="space-y-2 text-sm text-stone-700">
                  {SYSTEM_STEP_KEYS.map((key, i) => (
                    <li key={key} className="flex gap-2">
                      <span className="font-semibold text-brand-600 min-w-[20px]">
                        {i + 1}.
                      </span>
                      <span>
                        <Trans
                          t={t}
                          i18nKey={`caCert.system.steps.${key}`}
                          components={STEP_COMPONENTS}
                        />
                      </span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="bg-amber-50 ring-1 ring-amber-200 rounded-xl p-4">
                <p className="text-xs text-amber-900">
                  <Trans
                    t={t}
                    i18nKey="caCert.system.important"
                    components={STEP_COMPONENTS}
                  />
                </p>
              </div>

              <div className="bg-red-50 ring-1 ring-red-200 rounded-xl p-4">
                <p className="text-xs text-red-900">
                  <Trans
                    t={t}
                    i18nKey="caCert.system.commonIssue"
                    components={STEP_COMPONENTS}
                  />
                </p>
              </div>
            </div>
          )}

          {/* Browser-Specific Instructions */}
          {currentTab === "browsers" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-stone-700 mb-3">
                  {t("caCert.browsers.firefox")}
                </h3>
                <ol className="space-y-2 text-sm text-stone-700">
                  {FIREFOX_STEP_KEYS.map((key, i) => (
                    <li key={key} className="flex gap-2">
                      <span className="font-semibold text-brand-600 min-w-[20px]">
                        {i + 1}.
                      </span>
                      <span>
                        <Trans
                          t={t}
                          i18nKey={`caCert.browsers.firefoxSteps.${key}`}
                          components={STEP_COMPONENTS}
                        />
                      </span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="border-t border-stone-200 pt-4">
                <h3 className="text-sm font-semibold text-stone-700 mb-3">
                  {t("caCert.browsers.chrome")}
                </h3>
                <ol className="space-y-2 text-sm text-stone-700">
                  {CHROME_STEP_KEYS.map((key, i) => (
                    <li key={key} className="flex gap-2">
                      <span className="font-semibold text-brand-600 min-w-[20px]">
                        {i + 1}.
                      </span>
                      <span>
                        <Trans
                          t={t}
                          i18nKey={`caCert.browsers.chromeSteps.${key}`}
                          components={STEP_COMPONENTS}
                        />
                      </span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="bg-brand-50 ring-1 ring-brand-200 rounded-xl p-4">
                <p className="text-xs text-brand-900">
                  <Trans
                    t={t}
                    i18nKey="caCert.browsers.chromeTip"
                    components={STEP_COMPONENTS}
                  />
                </p>
              </div>
            </div>
          )}

          {/* Documentation Link */}
          <div className="pt-2">
            <a
              href="https://github.com/dataiku/kiji-private/blob/main/docs/01-getting-started.md#installing-ca-certificate-required-for-https"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-brand-600 hover:text-brand-700 hover:underline"
            >
              <ExternalLink className="w-4 h-4" />
              {t("caCert.docLink")}
            </a>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 pt-0 space-y-4">
          {/* Don't show again checkbox */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
              className="w-4 h-4 rounded border-stone-300 text-brand-600 focus:ring-brand-500 cursor-pointer"
            />
            <span className="text-sm text-stone-600">
              {t("caCert.dontShowAgain")}
            </span>
          </label>

          {/* Action button */}
          <button
            onClick={handleConfirm}
            className="btn-brand w-full px-4 py-3 text-white rounded-xl font-medium tracking-tight"
          >
            {t("caCert.confirm")}
          </button>
        </div>
      </div>
    </div>
  );
}
