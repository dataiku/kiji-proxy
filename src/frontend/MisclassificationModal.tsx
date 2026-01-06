import React, { useState } from "react";
import { X, Flag, AlertCircle } from "lucide-react";

interface Entity {
  type: string;
  original: string;
  token: string;
  confidence: number;
}

interface MisclassificationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (comment: string) => void;
  entities?: Entity[];
  originalInput?: string;
  maskedInput?: string;
  source?: string; // "main" or "log"
}

export default function MisclassificationModal({
  isOpen,
  onClose,
  onSubmit,
  entities = [],
  originalInput = "",
  maskedInput: _maskedInput = "",
  source: _source = "main",
}: MisclassificationModalProps) {
  const [comment, setComment] = useState("");

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(comment);
    setComment("");
    onClose();
  };

  const handleCancel = () => {
    setComment("");
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <Flag className="w-5 h-5 text-amber-600" />
            <h2 className="text-xl font-semibold text-slate-800">
              Report Misclassification
            </h2>
          </div>
          <button
            onClick={handleCancel}
            className="text-slate-400 hover:text-slate-600 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="p-6 space-y-4">
            {/* Entity Details Section */}
            {entities.length > 0 && (
              <div className="bg-slate-50 rounded-lg p-4 space-y-3">
                <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4" />
                  Detected Entities ({entities.length})
                </h3>
                <div className="space-y-2">
                  {entities.map((entity, index) => (
                    <div
                      key={index}
                      className="bg-white rounded-md p-3 border border-slate-200"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                              {entity.type}
                            </span>
                            <span className="text-xs text-slate-500">
                              {(entity.confidence * 100).toFixed(1)}% confidence
                            </span>
                          </div>
                          <p className="text-sm text-slate-900 font-mono break-all">
                            "{entity.original}"
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Original Input Preview (optional, for context) */}
            {originalInput && (
              <div className="bg-slate-50 rounded-lg p-3">
                <h4 className="text-xs font-semibold text-slate-600 mb-2">
                  Original Input
                </h4>
                <p className="text-sm text-slate-700 font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto">
                  {originalInput.substring(0, 500)}
                  {originalInput.length > 500 ? "..." : ""}
                </p>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Your Feedback
              </label>
              <p className="text-sm text-slate-600 mb-3">
                Please describe what was incorrectly classified. Your feedback
                helps us improve the PII detection model.
              </p>
              <textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                className="w-full h-32 p-3 border-2 border-slate-200 rounded-lg focus:border-amber-500 focus:outline-none resize-none text-sm placeholder:text-gray-400"
                placeholder="Example: 'John Smith' was detected as a person name but it's actually a company name..."
                autoFocus
              />
            </div>
          </div>

          <div className="flex gap-3 p-6 pt-0 border-t border-slate-100">
            <button
              type="button"
              onClick={handleCancel}
              className="flex-1 px-4 py-2 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors font-medium"
            >
              Submit Report
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
