import { useState } from "react";
import {
  X,
  FileText,
  ArrowDownCircle,
  ArrowUpCircle,
  Code,
  MessageSquare,
  Flag,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import type { LogEntry } from "../../types/provider";
import { useLogs } from "../../hooks/useLogs";
import {
  formatTimestamp,
  formatMessage,
  isJson,
  getDirectionLabel,
  getRowBackground,
} from "../../utils/logFormatters";

interface LoggingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onReportMisclassification?: (logEntry: LogEntry) => void;
}

const MAX_PAGE_SIZE = 500;

function getDirectionIcon(direction: string) {
  if (
    direction === "request_original" ||
    direction === "request" ||
    direction === "In"
  ) {
    return <ArrowDownCircle className="w-4 h-4 text-blue-600" />;
  }
  if (direction === "request_masked") {
    return <ArrowDownCircle className="w-4 h-4 text-purple-600" />;
  }
  if (direction === "response_masked") {
    return <ArrowUpCircle className="w-4 h-4 text-orange-600" />;
  }
  return <ArrowUpCircle className="w-4 h-4 text-green-600" />;
}

export default function LoggingModal({
  isOpen,
  onClose,
  onReportMisclassification,
}: LoggingModalProps) {
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showFullJson, setShowFullJson] = useState(false);

  const {
    logs,
    isLoading,
    isClearing,
    error,
    hasMore,
    total,
    handleLoadMore,
    handleClearLogs,
    retry,
  } = useLogs(isOpen);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      {/* Clear Confirmation Dialog */}
      {showClearConfirm && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-60">
          <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-red-100 rounded-full">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <h3 className="text-xl font-bold text-slate-800">
                Clear All Logs?
              </h3>
            </div>
            <p className="text-slate-600 mb-6">
              This will permanently delete all {total} log entries. This action
              cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowClearConfirm(false)}
                disabled={isClearing}
                className="px-4 py-2 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  await handleClearLogs();
                  setShowClearConfirm(false);
                }}
                disabled={isClearing}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium flex items-center gap-2 disabled:opacity-50"
              >
                {isClearing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Clearing...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    Clear All Logs
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-6xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <FileText className="w-6 h-6 text-slate-700" />
            <h2 className="text-2xl font-bold text-slate-800">Logging</h2>
            {total > 0 && (
              <span className="ml-2 px-2 py-1 bg-slate-100 text-slate-600 text-sm rounded-full">
                {total} entries
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {total > 0 && (
              <button
                onClick={() => setShowClearConfirm(true)}
                className="flex items-center gap-2 px-3 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                title="Clear all logs"
              >
                <Trash2 className="w-5 h-5" />
                <span className="text-sm font-medium">Clear Logs</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-700 transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Toggle Slider */}
        <div className="mb-4 flex items-center gap-4">
          <div className="flex items-center gap-3">
            <MessageSquare className="w-4 h-4 text-slate-600" />
            <span className="text-sm font-medium text-slate-700">
              Messages Only
            </span>
          </div>
          <button
            onClick={() => setShowFullJson(!showFullJson)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              showFullJson ? "bg-blue-600" : "bg-slate-300"
            }`}
            role="switch"
            aria-checked={showFullJson}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showFullJson ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-slate-700">
              Full JSON
            </span>
            <Code className="w-4 h-4 text-slate-600" />
          </div>
          <span className="text-sm text-slate-500 ml-2">
            {showFullJson
              ? "Showing complete request/response"
              : "Showing message content only"}
          </span>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <div className="text-red-600 text-sm flex-1">
              <strong>Error loading logs:</strong> {error}
            </div>
            <button
              onClick={retry}
              className="text-red-600 hover:text-red-800 text-sm font-medium"
            >
              Retry
            </button>
          </div>
        )}

        {/* Table */}
        <div className="flex-1 overflow-auto scrollbar-always-visible">
          {isLoading && logs.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
              <FileText className="w-12 h-12 mb-4 opacity-50" />
              <p className="text-lg">No log entries found</p>
            </div>
          ) : (
            <div>
              <table className="border-collapse" style={{ minWidth: "1200px" }}>
                <thead className="bg-slate-100 sticky top-0">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Direction
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Model
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Message
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Detected PII
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Blocked
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      Time Stamp
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
                      <Flag className="w-4 h-4" />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => (
                    <tr
                      key={log.id}
                      className={`hover:opacity-90 transition-colors border-b border-slate-100 ${getRowBackground(
                        log.direction
                      )}`}
                    >
                      <td className="px-4 py-3 text-sm">
                        <div className="flex items-center gap-2">
                          {getDirectionIcon(log.direction)}
                          <span className="font-medium">
                            {getDirectionLabel(log.direction, log.model)}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-600">
                        {log.model ? (
                          <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-medium">
                            {log.model}
                          </span>
                        ) : (
                          <span className="text-slate-400">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-700">
                        <div className="relative">
                          {log.messages && log.messages.length > 0 && (
                            <div className="absolute -top-1 -right-1 px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full font-medium">
                              {log.messages.length} message
                              {log.messages.length !== 1 ? "s" : ""}
                            </div>
                          )}
                          <pre
                            className={`font-mono text-xs whitespace-pre-wrap break-words ${
                              showFullJson && isJson(log.message)
                                ? "bg-slate-50 p-2 rounded border border-slate-200"
                                : (log.messages && log.messages.length > 0) ||
                                  isJson(log.message)
                                ? "p-2 rounded bg-gradient-to-br from-blue-50 to-slate-50 border border-blue-100"
                                : ""
                            }`}
                          >
                            {formatMessage(log, showFullJson)}
                          </pre>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-700">
                        {log.detectedPII}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium ${
                            log.blocked
                              ? "bg-red-100 text-red-700"
                              : "bg-green-100 text-green-700"
                          }`}
                        >
                          {log.blocked ? "Yes" : "No"}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-600">
                        {formatTimestamp(log.timestamp)}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {onReportMisclassification &&
                          (() => {
                            const hasPII =
                              log.detectedPII && log.detectedPII !== "None";
                            return (
                              <button
                                onClick={() => onReportMisclassification(log)}
                                disabled={!hasPII}
                                className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${
                                  !hasPII
                                    ? "text-slate-300 cursor-not-allowed"
                                    : "text-amber-600 hover:text-amber-700 hover:bg-amber-50"
                                }`}
                                title={
                                  !hasPII
                                    ? "No PII detected in this log"
                                    : "Report misclassification"
                                }
                              >
                                <Flag className="w-4 h-4" />
                                <span className="text-xs">Report</span>
                              </button>
                            );
                          })()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Load More Button */}
              {hasMore && !isLoading && (
                <div className="flex justify-center py-4 border-t border-slate-200">
                  <button
                    onClick={handleLoadMore}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
                  >
                    Load More Logs
                  </button>
                </div>
              )}

              {/* Loading More Indicator */}
              {isLoading && logs.length > 0 && (
                <div className="flex justify-center py-4 border-t border-slate-200">
                  <div className="w-6 h-6 border-3 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  <span className="ml-3 text-sm text-slate-600">
                    Loading more...
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-4 border-t border-slate-200 flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <p className="text-sm text-slate-500">
              Showing {logs.length} of {total} log{" "}
              {total === 1 ? "entry" : "entries"}
              {hasMore && (
                <span className="ml-2 text-slate-400">(more available)</span>
              )}
            </p>
            {total > MAX_PAGE_SIZE && (
              <p className="text-xs text-amber-600 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Large log count detected. Consider clearing old logs to improve
                performance.
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="px-6 py-2 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
