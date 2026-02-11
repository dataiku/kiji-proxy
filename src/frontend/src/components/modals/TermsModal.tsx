import { useState, useMemo } from "react";
import { X, FileText } from "lucide-react";
import DOMPurify from "dompurify";
import termsMarkdown from "./Terms.md";
import { isElectron } from "../../utils/providerHelpers";

interface TermsModalProps {
  isOpen: boolean;
  onClose: () => void;
  requireAcceptance?: boolean; // If true, user must accept before closing
}

export default function TermsModal({
  isOpen,
  onClose,
  requireAcceptance = false,
}: TermsModalProps) {
  const [hasAccepted, setHasAccepted] = useState(false);

  const termsHtml = useMemo(() => {
    // Convert markdown to HTML
    // Simple markdown parser (supports headers and basic formatting)
    let html = termsMarkdown
      // Headers
      .replace(/^## (.+)$/gm, "<h2>$1</h2>")
      .replace(/^# (.+)$/gm, "<h1>$1</h1>")
      // Bold
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      // Italic
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      // Line breaks
      .replace(/\n\n/g, "</p><p>")
      // Single line breaks
      .replace(/\n/g, "<br />");

    // Wrap in paragraph tags if not already wrapped
    if (!html.startsWith("<h")) {
      html = "<p>" + html + "</p>";
    }

    return DOMPurify.sanitize(html);
  }, []);

  if (!isOpen) return null;

  const handleAccept = async () => {
    if (requireAcceptance && isElectron && window.electronAPI) {
      try {
        await window.electronAPI.setTermsAccepted(true);
      } catch (error) {
        console.error("Failed to save terms acceptance:", error);
      }
    }
    onClose();
  };

  const handleClose = () => {
    if (!requireAcceptance) {
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200 flex-shrink-0">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              Terms & Conditions
            </h2>
          </div>
          {!requireAcceptance && (
            <button
              onClick={handleClose}
              className="p-1 text-slate-400 hover:text-slate-600 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Content - Scrollable */}
        <div className="flex-1 overflow-y-auto p-6">
          <div
            className="prose prose-slate max-w-none"
            style={{
              fontSize: "0.95rem",
              lineHeight: "1.6",
            }}
          >
            <style>{`
              .prose h1 {
                font-size: 1.75rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 1rem;
                margin-top: 0;
              }
              .prose h2 {
                font-size: 1.25rem;
                font-weight: 600;
                color: #334155;
                margin-top: 1.5rem;
                margin-bottom: 0.75rem;
              }
              .prose h2:first-child {
                margin-top: 0;
              }
              .prose p {
                margin-bottom: 1rem;
                color: #475569;
              }
              .prose strong {
                font-weight: 600;
                color: #1e293b;
              }
            `}</style>
            <div dangerouslySetInnerHTML={{ __html: termsHtml }} />
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 pt-0 border-t border-slate-100 flex-shrink-0">
          {requireAcceptance ? (
            <div className="space-y-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hasAccepted}
                  onChange={(e) => setHasAccepted(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                />
                <span className="text-sm text-slate-700">
                  I have read and accept the Terms & Conditions
                </span>
              </label>
              <button
                onClick={handleAccept}
                disabled={!hasAccepted}
                className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Accept & Continue
              </button>
            </div>
          ) : (
            <button
              onClick={handleClose}
              className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
