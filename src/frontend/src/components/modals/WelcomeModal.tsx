import { useState } from "react";
import { X, Shield, Cpu, Lock } from "lucide-react";

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function WelcomeModal({ isOpen, onClose }: WelcomeModalProps) {
  const [dontShowAgain, setDontShowAgain] = useState(false);

  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

  if (!isOpen) return null;

  const handleClose = async () => {
    if (dontShowAgain && isElectron && window.electronAPI) {
      try {
        await window.electronAPI.setWelcomeDismissed(true);
      } catch (error) {
        console.error("Failed to save welcome dismissed preference:", error);
      }
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <Shield className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              Welcome to Yaak Privacy Proxy
            </h2>
          </div>
          <button
            onClick={handleClose}
            className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Theme 1: Why this app? */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-5 border border-blue-100">
            <div className="flex items-start gap-4">
              <div className="bg-blue-100 rounded-full p-3 flex-shrink-0">
                <Shield className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-800 mb-2">
                  Why Yaak Privacy Proxy?
                </h3>
                <p className="text-slate-600 leading-relaxed">
                  When using AI services like OpenAI or Anthropic, sensitive
                  data in your prompts gets sent to external servers. Yaak
                  Privacy Proxy helps you avoid your personal information going
                  to big companies by{" "}
                  <strong>
                    automatically detecting and masking personally identifiable
                    information (PII)
                  </strong>{" "}
                  before it leaves your device.
                </p>
              </div>
            </div>
          </div>

          {/* Theme 2: How does it work? */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-5 border border-green-100">
            <div className="flex items-start gap-4">
              <div className="bg-green-100 rounded-full p-3 flex-shrink-0">
                <Cpu className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-800 mb-2">
                  How Does It Work?
                </h3>
                <p className="text-slate-600 leading-relaxed mb-3">
                  Yaak acts as a transparent proxy between you and AI services:
                </p>
                <ol className="text-slate-600 space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="font-semibold text-green-600 min-w-[20px]">
                      1.
                    </span>
                    <span>Your app sends a request to Yaak proxy</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-semibold text-green-600 min-w-[20px]">
                      2.
                    </span>
                    <span>
                      Yaak detects PII using a local ML model (no 3rd party)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-semibold text-green-600 min-w-[20px]">
                      3.
                    </span>
                    <span>
                      PII is replaced with realistic dummy data before the API
                      call
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-semibold text-green-600 min-w-[20px]">
                      4.
                    </span>
                    <span>
                      The response is received and your original information is
                      restored automatically
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="font-semibold text-green-600 min-w-[20px]">
                      5.
                    </span>
                    <span>
                      Your app receives the response with original data intact
                    </span>
                  </li>
                </ol>
              </div>
            </div>
          </div>

          {/* Theme 3: What happens to your data? */}
          <div className="bg-gradient-to-r from-purple-50 to-violet-50 rounded-lg p-5 border border-purple-100">
            <div className="flex items-start gap-4">
              <div className="bg-purple-100 rounded-full p-3 flex-shrink-0">
                <Lock className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-slate-800 mb-2">
                  What Happens to Your Data?
                </h3>
                <p className="text-slate-600 leading-relaxed mb-3">
                  Yaak Privacy Proxy uses machine learning models to identify
                  PII and removes it from traffic. Your privacy is our priority:
                </p>
                <ul className="text-slate-600 space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold">•</span>
                    <span>
                      <strong>All processing happens locally</strong> - no data
                      is sent to external services for PII detection
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold">•</span>
                    <span>
                      <strong>No data is shared with 3rd parties</strong> beyond
                      your chosen model provider (OpenAI, Anthropic, etc.)
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold">•</span>
                    <span>
                      <strong>16+ PII types detected</strong> including emails,
                      phone numbers, SSNs, credit cards, and IP addresses
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-600 font-bold">•</span>
                    <span>
                      <strong>Open source</strong> - you can review the code,
                      verify our privacy claims, and customize the AI models for
                      your use cases
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 pt-0 border-t border-slate-100 flex-shrink-0 space-y-4">
          {/* Don't show again checkbox */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={dontShowAgain}
              onChange={(e) => setDontShowAgain(e.target.checked)}
              className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
            <span className="text-sm text-slate-600">
              Don't show this again
            </span>
          </label>

          {/* Action button */}
          <button
            onClick={handleClose}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}
