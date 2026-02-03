import { useEffect, useState } from "react";
import { X, Info } from "lucide-react";
import logoImage from "../../../assets/logo.png";
import TermsModal from "./TermsModal";

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AboutModal({ isOpen, onClose }: AboutModalProps) {
  const [version, setVersion] = useState<string>("Loading...");
  const [isTermsModalOpen, setIsTermsModalOpen] = useState(false);

  useEffect(() => {
    if (isOpen) {
      const loadVersion = async () => {
        try {
          const response = await fetch("http://localhost:8080/version");
          if (response.ok) {
            const data = await response.json();
            setVersion(data.version || "Unknown");
          } else {
            setVersion("Unknown");
          }
        } catch (error) {
          console.error("Failed to fetch version:", error);
          setVersion("Unknown");
        }
      };
      loadVersion();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 relative">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200">
          <div className="flex items-center gap-3">
            <Info className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              About Yaak Proxy
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
          {/* Logo and Name */}
          <div className="flex flex-col items-center text-center">
            <img src={logoImage} alt="Yaak Logo" className="w-20 h-20 mb-4" />
            <h3 className="text-2xl font-bold text-slate-800">
              Yaak Privacy Proxy
            </h3>
            <p className="text-slate-600 mt-2">
              PII Detection and Masking Proxy
            </p>
          </div>

          {/* Version Information */}
          <div className="bg-slate-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-600">
                Version:
              </span>
              <span className="text-sm font-mono font-semibold text-slate-800">
                {version}
              </span>
            </div>
          </div>

          {/* Description */}
          <div className="text-sm text-slate-600 space-y-2">
            <p>
              Yaak Privacy Proxy automatically detects and masks personally
              identifiable information (PII) in your API requests, ensuring data
              privacy when interacting with external services.
            </p>
          </div>

          {/* Links */}
          <div className="space-y-2">
            <a
              href="https://github.com/hannes/yaak-proxy/"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              View on GitHub →
            </a>
            <a
              href="https://github.com/hanneshapke/yaak-proxy/blob/main/docs/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              Documentation →
            </a>
            <a
              href="https://github.com/hanneshapke/yaak-proxy/issues/new?template=10_bug_report.yml"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              File a Bug Report →
            </a>
            <a
              href="https://github.com/hanneshapke/yaak-proxy/discussions/new/choose"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              Request a Feature →
            </a>
            <a
              href="mailto:opensource@dataiku.com?subject=[Yaak Proxy User]"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              Email us →
            </a>
            <a
              href="https://github.com/hannes/yaak-private/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline"
            >
              Apache 2.0 License →
            </a>
            <button
              onClick={() => setIsTermsModalOpen(true)}
              className="block text-sm text-blue-600 hover:text-blue-700 hover:underline text-left"
            >
              Terms &amp; Conditions →
            </button>
          </div>

          {/* Copyright */}
          <div className="pt-4 border-t border-slate-200 text-center">
            <p className="text-xs text-slate-500">
              © {new Date().getFullYear()} Dataiku Open Source Lab. All rights
              reserved.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-slate-50 rounded-b-xl">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>

      <TermsModal
        isOpen={isTermsModalOpen}
        onClose={() => setIsTermsModalOpen(false)}
        requireAcceptance={false}
      />
    </div>
  );
}
