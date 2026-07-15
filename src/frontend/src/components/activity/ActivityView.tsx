import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  FileText,
  ArrowDownCircle,
  ArrowUpCircle,
  Code,
  MessageSquare,
  Flag,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { useLogs } from "../../hooks/useLogs";
import { useMisclassificationReport } from "../../hooks/useMisclassificationReport";
import MisclassificationModal from "../modals/MisclassificationModal";
import {
  formatTimestamp,
  formatMessage,
  isJson,
  getDirectionLabel,
  getRowBackground,
} from "../../utils/logFormatters";

interface ActivityViewProps {
  /** Verified model signature, used as a fallback when a log has no model of
   *  its own while reporting a misclassification. */
  modelSignature?: string | null;
}

const MAX_PAGE_SIZE = 500;

function getDirectionIcon(direction: string) {
  if (
    direction === "request_original" ||
    direction === "request" ||
    direction === "In"
  ) {
    return <ArrowDownCircle className="w-4 h-4 text-brand-600" />;
  }
  if (direction === "request_masked") {
    return <ArrowDownCircle className="w-4 h-4 text-stone-600" />;
  }
  if (direction === "response_masked") {
    return <ArrowUpCircle className="w-4 h-4 text-orange-600" />;
  }
  return <ArrowUpCircle className="w-4 h-4 text-brand-600" />;
}

/**
 * Activity (logging) workspace view.
 *
 * The view counterpart of the former LoggingModal: it lives in the shell's main
 * area and is reached from the sidebar. Log data + actions come from useLogs;
 * misclassification reporting is self-contained (its own modal) so the view does
 * not depend on the Playground.
 */
export default function ActivityView({ modelSignature }: ActivityViewProps) {
  const { t } = useTranslation("activity");
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
  } = useLogs(true);

  const misclassification = useMisclassificationReport();

  return (
    <div className="w-full">
      {/* Clear confirmation dialog */}
      {showClearConfirm && (
        <div className="fixed inset-0 bg-brand-950/40 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-red-100 rounded-full">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <h3 className="text-xl font-bold text-stone-800">
                {t("confirm.title")}
              </h3>
            </div>
            <p className="text-stone-600 mb-6">
              {t("confirm.body", { count: total })}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowClearConfirm(false)}
                disabled={isClearing}
                className="px-4 py-2 border-2 border-stone-300 text-stone-700 rounded-lg hover:bg-stone-50 transition-colors font-medium disabled:opacity-50"
              >
                {t("confirm.cancel")}
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
                    {t("confirm.clearing")}
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    {t("confirm.clearAll")}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Page header */}
      <div className="flex items-start justify-between mb-6 gap-4">
        <div>
          <h1 className="text-[23px] font-semibold tracking-tight text-stone-900">
            {t("title")}
          </h1>
          <p className="text-stone-500 text-[13px] mt-0.5">{t("subtitle")}</p>
        </div>
        <div className="flex items-center gap-2.5 shrink-0">
          {total > 0 && (
            <span className="inline-flex items-center text-[13px] font-medium text-stone-600 bg-white border border-brand-900/10 rounded-lg px-3 py-2 shadow-soft">
              {t("entryCount", { count: total })}
            </span>
          )}
          {total > 0 && (
            <button
              onClick={() => setShowClearConfirm(true)}
              className="inline-flex items-center gap-2 px-3 py-2 text-[13px] font-medium text-red-600 bg-white border border-red-200 rounded-lg shadow-soft hover:bg-red-50 transition-colors"
              title={t("clearLogsTitle")}
            >
              <Trash2 className="w-4 h-4" />
              {t("clearLogs")}
            </button>
          )}
        </div>
      </div>

      <div className="animate-rise-in">
          {/* Messages-only / Full JSON toggle */}
          <div className="mb-4 flex items-center gap-4">
            <div className="flex items-center gap-3">
              <MessageSquare className="w-4 h-4 text-stone-600" />
              <span className="text-sm font-medium text-stone-700">
                {t("toggle.messagesOnly")}
              </span>
            </div>
            <button
              onClick={() => setShowFullJson(!showFullJson)}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 ${
                showFullJson ? "bg-brand-600" : "bg-stone-300"
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
              <span className="text-sm font-medium text-stone-700">
                {t("toggle.fullJson")}
              </span>
              <Code className="w-4 h-4 text-stone-600" />
            </div>
            <span className="text-sm text-stone-500 ml-2">
              {showFullJson
                ? t("toggle.showingFull")
                : t("toggle.showingMessages")}
            </span>
          </div>

          {/* Error display */}
          {error && (
            <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
              <div className="text-red-600 text-sm flex-1">
                <strong>{t("error.loading")}</strong> {error}
              </div>
              <button
                onClick={retry}
                className="text-red-600 hover:text-red-800 text-sm font-medium"
              >
                {t("error.retry")}
              </button>
            </div>
          )}

          {/* Table */}
          <div className="card p-0 overflow-hidden">
            <div className="overflow-x-auto scrollbar-always-visible">
              {isLoading && logs.length === 0 ? (
                <div className="flex items-center justify-center py-16">
                  <div className="w-8 h-8 border-4 border-brand-600 border-t-transparent rounded-full animate-spin" />
                </div>
              ) : logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-stone-500">
                  <FileText className="w-12 h-12 mb-4 opacity-50" />
                  <p className="text-lg">{t("empty")}</p>
                </div>
              ) : (
                <table className="border-collapse w-full">
                  <thead className="bg-stone-100 sticky top-0">
                    <tr>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        {t("table.direction")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        {t("table.model")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200 w-full">
                        {t("table.message")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        {t("table.detectedPII")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        {t("table.blocked")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        {t("table.timestamp")}
                      </th>
                      <th className="px-4 py-3 text-left text-sm font-semibold text-stone-700 border-b border-stone-200">
                        <Flag className="w-4 h-4" />
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((log) => (
                      <tr
                        key={log.id}
                        className={`hover:opacity-90 transition-colors border-b border-stone-100 ${getRowBackground(
                          log.direction
                        )}`}
                      >
                        <td className="px-4 py-3 text-sm">
                          <div className="flex items-center gap-2">
                            {getDirectionIcon(log.direction)}
                            <span className="font-medium">
                              {getDirectionLabel(log.direction, log.model, t)}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-stone-600">
                          {log.model ? (
                            <span className="px-2 py-1 bg-stone-100 text-stone-700 rounded text-xs font-medium">
                              {log.model}
                            </span>
                          ) : (
                            <span className="text-stone-400">-</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-stone-700">
                          <div className="relative">
                            {log.messages && log.messages.length > 0 && (
                              <div className="absolute -top-1 -right-1 px-2 py-0.5 bg-brand-100 text-brand-700 text-xs rounded-full font-medium">
                                {t("messageCount", {
                                  count: log.messages.length,
                                })}
                              </div>
                            )}
                            <pre
                              className={`font-mono text-xs whitespace-pre-wrap break-all ${
                                showFullJson && isJson(log.message)
                                  ? "bg-stone-50 p-2 rounded border border-stone-200"
                                  : (log.messages &&
                                      log.messages.length > 0) ||
                                    isJson(log.message)
                                  ? "p-2 rounded bg-gradient-to-br from-brand-50 to-stone-50 border border-brand-100"
                                  : ""
                              }`}
                            >
                              {formatMessage(log, showFullJson, t)}
                            </pre>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm text-stone-700">
                          <div className="max-w-[240px] break-words">
                            {log.detectedPII}
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <span
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              log.blocked
                                ? "bg-red-100 text-red-700"
                                : "bg-brand-100 text-brand-700"
                            }`}
                          >
                            {log.blocked ? t("blocked.yes") : t("blocked.no")}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-stone-600">
                          {formatTimestamp(log.timestamp)}
                        </td>
                        <td className="px-4 py-3 text-sm">
                          {(() => {
                            const hasPII =
                              log.detectedPII && log.detectedPII !== "None";
                            return (
                              <button
                                onClick={() =>
                                  misclassification.handleReportFromLog(
                                    log,
                                    modelSignature ?? null
                                  )
                                }
                                disabled={!hasPII}
                                className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${
                                  !hasPII
                                    ? "text-stone-300 cursor-not-allowed"
                                    : "text-amber-600 hover:text-amber-700 hover:bg-amber-50"
                                }`}
                                title={
                                  !hasPII
                                    ? t("report.noPII")
                                    : t("report.report")
                                }
                              >
                                <Flag className="w-4 h-4" />
                                <span className="text-xs">
                                  {t("report.label")}
                                </span>
                              </button>
                            );
                          })()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              {/* Load more */}
              {hasMore && !isLoading && logs.length > 0 && (
                <div className="flex justify-center py-4 border-t border-stone-200">
                  <button
                    onClick={handleLoadMore}
                    className="px-6 py-2 bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition-colors font-medium"
                  >
                    {t("loadMore")}
                  </button>
                </div>
              )}

              {/* Loading more */}
              {isLoading && logs.length > 0 && (
                <div className="flex justify-center py-4 border-t border-stone-200">
                  <div className="w-6 h-6 border-3 border-brand-600 border-t-transparent rounded-full animate-spin" />
                  <span className="ml-3 text-sm text-stone-600">
                    {t("loadingMore")}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Footer summary */}
          {logs.length > 0 && (
            <div className="mt-4 flex flex-col gap-1">
              <p className="text-sm text-stone-500">
                {t("footer.showing", { shown: logs.length, count: total })}
                {hasMore && (
                  <span className="ml-2 text-stone-400">
                    {t("footer.moreAvailable")}
                  </span>
                )}
              </p>
              {total > MAX_PAGE_SIZE && (
                <p className="text-xs text-amber-600 flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3" />
                  {t("footer.largeCount")}
                </p>
              )}
            </div>
          )}
        </div>

      {/* Misclassification report modal */}
      <MisclassificationModal
        isOpen={misclassification.isMisclassificationModalOpen}
        onClose={misclassification.closeModal}
        onSubmit={misclassification.handleSubmitMisclassification}
        entities={misclassification.reportingData?.entities || []}
        originalInput={misclassification.reportingData?.originalInput || ""}
        maskedInput={misclassification.reportingData?.maskedInput || ""}
        source={misclassification.reportingData?.source || "log"}
      />
    </div>
  );
}
