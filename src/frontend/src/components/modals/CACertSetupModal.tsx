import { useState } from "react";
import { X, ShieldCheck, Terminal, Chrome, ExternalLink } from "lucide-react";

interface CACertSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CACertSetupModal({
  isOpen,
  onClose,
}: CACertSetupModalProps) {
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [currentTab, setCurrentTab] = useState<"system" | "browsers">("system");

  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

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

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-6 h-6 text-green-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              CA Certificate Setup Required
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Introduction */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-900">
              To intercept and analyze HTTPS traffic, Kiji Privacy Proxy uses a
              self-signed Certificate Authority (CA). You must trust this
              certificate on your system and/or browsers.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 border-b border-slate-200">
            <button
              onClick={() => setCurrentTab("system")}
              className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
                currentTab === "system"
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-slate-600 hover:text-slate-800"
              }`}
            >
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4" />
                System-Wide Trust
              </div>
            </button>
            <button
              onClick={() => setCurrentTab("browsers")}
              className={`px-4 py-2 font-medium text-sm transition-colors border-b-2 ${
                currentTab === "browsers"
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-slate-600 hover:text-slate-800"
              }`}
            >
              <div className="flex items-center gap-2">
                <Chrome className="w-4 h-4" />
                Browser-Specific
              </div>
            </button>
          </div>

          {/* System-Wide Instructions */}
          {currentTab === "system" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">
                  Option 1: Command Line (Recommended)
                </h3>
                <div className="bg-slate-900 rounded-lg p-4 text-sm font-mono text-slate-100 overflow-x-auto">
                  <code>
                    sudo security add-trusted-cert \
                    <br />
                    {"  "}-d \<br />
                    {"  "}-r trustRoot \<br />
                    {"  "}-k /Library/Keychains/System.keychain \<br />
                    {"  "}~/Library/Application Support/Kiji Privacy
                    Proxy/certs/ca.crt
                  </code>
                </div>
                <p className="text-xs text-slate-600 mt-2">
                  This command requires administrator privileges and will
                  install the certificate system-wide.
                </p>
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">
                  Option 2: Keychain Access GUI
                </h3>
                <ol className="space-y-2 text-sm text-slate-700">
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      1.
                    </span>
                    <span>
                      Double-click the certificate file:{" "}
                      <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">
                        ~/Library/Application Support/Kiji Privacy
                        Proxy/certs/ca.crt
                      </code>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      2.
                    </span>
                    <span>
                      This opens <strong>Keychain Access</strong> - click{" "}
                      <strong>Add</strong> to install
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      3.
                    </span>
                    <span>
                      In Keychain Access, select <strong>System</strong>{" "}
                      keychain in the left sidebar
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      4.
                    </span>
                    <span>
                      Search for <strong>"Kiji Privacy Proxy CA"</strong> and
                      double-click it
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      5.
                    </span>
                    <span>
                      Click the <strong>▶ triangle next to "Trust"</strong> to
                      expand the section
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      6.
                    </span>
                    <span>
                      Set <strong>"When using this certificate"</strong> to{" "}
                      <strong className="text-green-600">"Always Trust"</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      7.
                    </span>
                    <span>
                      Set <strong>"Secure Sockets Layer (SSL)"</strong> to{" "}
                      <strong className="text-green-600">"Always Trust"</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      8.
                    </span>
                    <span>
                      <strong>Close the window</strong> and enter your password
                      when prompted
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      9.
                    </span>
                    <span>
                      <strong className="text-amber-600">
                        Restart your browser completely
                      </strong>{" "}
                      (Cmd+Q, then reopen)
                    </span>
                  </li>
                </ol>
              </div>

              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <p className="text-xs text-amber-900">
                  <strong>Important:</strong> You must restart your browser
                  after trusting the certificate. System-wide trust works for
                  Safari and Chrome. Firefox requires separate configuration
                  (see Browser-Specific tab).
                </p>
              </div>

              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-xs text-red-900">
                  <strong>⚠️ Common Issue:</strong> If the certificate shows
                  "Number of trust settings: 0" in terminal, it means you
                  haven't set it to "Always Trust" in steps 6-7 above. The
                  certificate must be marked as trusted for SSL, not just
                  installed.
                </p>
              </div>
            </div>
          )}

          {/* Browser-Specific Instructions */}
          {currentTab === "browsers" && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-3">
                  Firefox
                </h3>
                <ol className="space-y-2 text-sm text-slate-700">
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      1.
                    </span>
                    <span>
                      Settings → <strong>Privacy & Security</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      2.
                    </span>
                    <span>
                      Certificates → <strong>View Certificates</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      3.
                    </span>
                    <span>
                      Authorities → <strong>Import</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      4.
                    </span>
                    <span>
                      Select{" "}
                      <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">
                        ~/Library/Application Support/Kiji Privacy
                        Proxy/certs/ca.crt
                      </code>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      5.
                    </span>
                    <span>
                      Check <strong>"Trust for websites"</strong>
                    </span>
                  </li>
                </ol>
              </div>

              <div className="border-t border-slate-200 pt-4">
                <h3 className="text-sm font-semibold text-slate-700 mb-3">
                  Chrome/Chromium
                </h3>
                <ol className="space-y-2 text-sm text-slate-700">
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      1.
                    </span>
                    <span>
                      Settings → <strong>Privacy and Security</strong> →{" "}
                      <strong>Security</strong>
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      2.
                    </span>
                    <span>
                      <strong>Manage certificates</strong> → Authorities
                    </span>
                  </li>
                  <li className="flex gap-2">
                    <span className="font-semibold text-blue-600 min-w-[20px]">
                      3.
                    </span>
                    <span>
                      <strong>Import</strong> CA certificate (
                      <code className="bg-slate-100 px-1.5 py-0.5 rounded text-xs">
                        ~/Library/Application Support/Kiji Privacy
                        Proxy/certs/ca.crt
                      </code>
                      )
                    </span>
                  </li>
                </ol>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-xs text-blue-900">
                  <strong>Tip:</strong> Chrome on macOS typically uses the
                  system keychain, so system-wide trust (see other tab) should
                  be sufficient.
                </p>
              </div>
            </div>
          )}

          {/* Documentation Link */}
          <div className="pt-2">
            <a
              href="https://github.com/hanneshapke/kiji-private/blob/main/docs/01-getting-started.md#installing-ca-certificate-required-for-https"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              <ExternalLink className="w-4 h-4" />
              View full documentation
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
              className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
            <span className="text-sm text-slate-600">
              Don't show this message again
            </span>
          </label>

          {/* Action button */}
          <button
            onClick={handleConfirm}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            I Understand
          </button>
        </div>
      </div>
    </div>
  );
}
