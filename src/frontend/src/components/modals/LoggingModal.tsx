import { useState, useEffect, useCallback } from "react";
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

interface OpenAIMessage {
  role: string;
  content: string;
}

interface PIIEntity {
  pii_type: string;
  original_pii: string;
  confidence?: number;
}

interface LogEntry {
  id: string;
  direction:
  | "request_original"
  | "request_masked"
  | "response_masked"
  | "response_original"
  | "request"
  | "response"
  | "In"
  | "Out";
  message?: string;
  messages?: OpenAIMessage[];
  formatted_messages?: string;
  model?: string;
  detectedPII: string; // Human-readable formatted string for display
  detectedPIIRaw?: PIIEntity[]; // Raw JSON array from backend for reporting
  blocked: boolean;
  timestamp: Date;
  transactionId?: string;
}

interface LoggingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onReportMisclassification?: (logEntry: LogEntry) => void;
}

export default function LoggingModal({
  isOpen,
  onClose,
  onReportMisclassification,
}: LoggingModalProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [showFullJson, setShowFullJson] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);
  const pageSize = 50;
  const maxPageSize = 500; // Server-enforced maximum

  // Load logs when modal opens
  useEffect(() => {
    if (isOpen) {
      setPage(0);
      setLogs([]);
      setError(null);
      loadLogs(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const sortLogs = (logsToSort: LogEntry[]): LogEntry[] => {
    // 1. Group logs by transactionId
    const groups: { [key: string]: LogEntry[] } = {};
    const singles: LogEntry[] = [];

    logsToSort.forEach((log) => {
      if (log.transactionId) {
        if (!groups[log.transactionId]) {
          groups[log.transactionId] = [];
        }
        groups[log.transactionId].push(log);
      } else {
        singles.push(log);
      }
    });

    // 2. Sort within groups
    const sortedGroups: LogEntry[][] = Object.values(groups).map((group) => {
      return group.sort((a, b) => {
        const order = [
          "request_original",
          "request_masked",
          "response_masked",
          "response_original",
        ];

        // Define indices for sorting, mapping other types to -1 or a low priority
        const aIndex = order.indexOf(a.direction);
        const bIndex = order.indexOf(b.direction);

        // If both are in the known order list, sort by that order
        if (aIndex !== -1 && bIndex !== -1) {
          return aIndex - bIndex;
        }

        // If one is known and the other isn't, known comes first (or however appropriate)
        // Actually, if we want strict ordering, we should stick to the list.
        // If unknown types, maybe fall back to timestamp.
        return a.timestamp.getTime() - b.timestamp.getTime();
      });
    });

    // 3. Create a list of "items" to sort (groups vs singles)
    // We treat each group as a single item for sorting purposes, using its latest timestamp
    // wrapper objects to help sorting
    type SortableItem =
      | { type: 'group'; logs: LogEntry[]; latestTimestamp: number }
      | { type: 'single'; log: LogEntry; latestTimestamp: number };

    const sortableItems: SortableItem[] = [
      ...sortedGroups.map(g => ({
        type: 'group' as const,
        logs: g,
        latestTimestamp: Math.max(...g.map(l => l.timestamp.getTime()))
      })),
      ...singles.map(l => ({
        type: 'single' as const,
        log: l,
        latestTimestamp: l.timestamp.getTime()
      }))
    ];

    // 4. Sort all items by latest timestamp descending
    sortableItems.sort((a, b) => b.latestTimestamp - a.latestTimestamp);

    // 5. Flatten
    const flattened: LogEntry[] = [];
    sortableItems.forEach(item => {
      if (item.type === 'group') {
        flattened.push(...item.logs);
      } else {
        flattened.push(item.log);
      }
    });

    return flattened;
  };

  const loadLogs = useCallback(
    async (pageNum: number) => {
      // If we have already loaded all logs, don't load more
      // BUT, if we are purely relying on hasMore, that's fine.
      // We need to be careful not to fetch page 1 if page 0 returned everything.
      if (!hasMore && pageNum > 0) return;

      setIsLoading(true);
      setError(null);
      try {
        // Determine the API URL
        const isElectron =
          typeof window !== "undefined" && window.electronAPI !== undefined;
        const offset = pageNum * pageSize;
        const baseUrl = isElectron
          ? "http://localhost:8080/logs" // Direct call to main server in Electron
          : "/logs"; // Proxied call in web mode
        const apiUrl = `${baseUrl}?limit=${pageSize}&offset=${offset}`;

        const response = await fetch(apiUrl);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        setTotal(data.total || 0);
        // data.logs.length can be less than pageSize if it's the last page
        setHasMore(data.logs && data.logs.length === pageSize);

        // Transform the API response to match LogEntry format
        const transformedLogs: LogEntry[] = (data.logs || []).map(
          (log: Record<string, unknown>) => {
            // Parse timestamp - handle both string and Date formats
            let timestamp: Date;
            if (typeof log.timestamp === "string") {
              timestamp = new Date(log.timestamp);
            } else if (log.timestamp instanceof Date) {
              timestamp = log.timestamp;
            } else {
              timestamp = new Date();
            }

            // Extract transaction_id if present in the message
            let transactionId: string | undefined;
            if (typeof log.message === 'string') {
              try {
                const parsed = JSON.parse(log.message);
                if (parsed && typeof parsed === 'object' && '_transaction_id' in parsed) {
                  transactionId = parsed._transaction_id;
                }
              } catch {
                // Ignore parsing errors
              }
            }

            // Format detected_pii for display
            let formattedPII = "None";
            const rawPII = log.detected_pii;
            let typedRawPII: PIIEntity[] | undefined;

            if (rawPII && Array.isArray(rawPII) && rawPII.length > 0) {
              // Type guard to ensure it's PIIEntity[]
              typedRawPII = rawPII as PIIEntity[];
              formattedPII = typedRawPII
                .map(
                  (entity: PIIEntity) =>
                    `${entity.pii_type}: ${entity.original_pii}`
                )
                .join(", ");
            } else if (typeof rawPII === "string" && rawPII !== "None") {
              // Backend is returning a string instead of JSON array
              // This shouldn't happen with new backend, but handle it as fallback
              formattedPII = rawPII;
              typedRawPII = undefined;
            }

            const entry: LogEntry = {
              id: String(log.id),
              direction: (log.direction as LogEntry["direction"]) || "Unknown",
              message: log.message as string | undefined,
              messages: log.messages as OpenAIMessage[] | undefined,
              formatted_messages: log.formatted_messages as string | undefined,
              model: log.model as string | undefined,
              detectedPII: formattedPII,
              detectedPIIRaw: typedRawPII, // Keep raw JSON for reporting
              blocked: (log.blocked as boolean) || false,
              timestamp: timestamp,
              transactionId: transactionId,
            };

            return entry;
          }
        );

        setLogs((prev) => {
          const combined = pageNum === 0 ? transformedLogs : [...prev, ...transformedLogs];
          // Remove duplicates just in case, though backend shouldn't return overlap
          const unique = Array.from(new Map(combined.map(item => [item.id, item])).values());
          return sortLogs(unique);
        });

        setPage(pageNum);
      } catch (err) {
        console.error("Error loading logs:", err);

        const errorMessage =
          err instanceof Error ? err.message : "Failed to load logs";
        setError(errorMessage);

        // Only clear logs on first page error
        if (pageNum === 0) {
          setLogs([]);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [hasMore, pageSize]
  );

  const handleLoadMore = () => {
    if (!isLoading && hasMore) {
      loadLogs(page + 1);
    }
  };

  const handleClearLogs = async () => {
    setIsClearing(true);
    setError(null);

    try {
      const isElectron =
        typeof window !== "undefined" && window.electronAPI !== undefined;
      const baseUrl = isElectron ? "http://localhost:8080/logs" : "/logs";

      const response = await fetch(baseUrl, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Reset state after successful clear
      setLogs([]);
      setTotal(0);
      setPage(0);
      setHasMore(false);
      setShowClearConfirm(false);
    } catch (err) {
      console.error("Error clearing logs:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to clear logs";
      setError(errorMessage);
    } finally {
      setIsClearing(false);
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleString("en-US", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const extractMessageFromJson = (message: string): string => {
    try {
      const parsed = JSON.parse(message);

      // Check if it's an OpenAI chat completion request (In message)
      if (parsed.messages && Array.isArray(parsed.messages)) {
        const messages = parsed.messages
          .map((msg: Record<string, unknown>) => {
            const role = msg.role ? `[${msg.role}]` : "";
            const content = msg.content || "";
            return role ? `${role} ${content}` : content;
          })
          .filter((content: string) => content.trim())
          .join("\n\n");
        return messages || message;
      }

      // Check if it's an OpenAI response (Out message)
      if (parsed.choices && Array.isArray(parsed.choices)) {
        const messages = parsed.choices
          .map((choice: Record<string, unknown>) => {
            // Handle chat completion format
            if (choice.message && typeof choice.message === "object") {
              const messageObj = choice.message as Record<string, unknown>;
              if (messageObj.content) {
                const role = messageObj.role
                  ? `[${messageObj.role}]`
                  : "[assistant]";
                return `${role} ${messageObj.content}`;
              }
            }
            // Handle legacy completion format
            if (choice.text) {
              return `[completion] ${choice.text}`;
            }
            return "";
          })
          .filter((content: string) => content.trim())
          .join("\n\n");
        return messages || message;
      }

      // Return the original if we can't parse it
      return message;
    } catch (_error) {
      // Not JSON or parsing failed, return as-is
      return message;
    }
  };

  const formatStructuredMessages = (messages: OpenAIMessage[]): string => {
    return messages.map((msg) => `[${msg.role}] ${msg.content}`).join("\n\n");
  };

  const formatMessage = (log: LogEntry, useFullJson: boolean): string => {
    if (!useFullJson) {
      // Messages Only mode - prefer structured messages
      if (log.messages && log.messages.length > 0) {
        const formatted = formatStructuredMessages(log.messages);
        if (formatted.length > 5000) {
          return (
            formatted.substring(0, 5000) +
            "\n\n... [Message truncated for display]"
          );
        }
        return formatted;
      }

      // Fall back to extracting from JSON message
      if (log.message) {
        const extracted = extractMessageFromJson(log.message);
        if (extracted.length > 5000) {
          return (
            extracted.substring(0, 5000) +
            "\n\n... [Message truncated for display]"
          );
        }
        return extracted;
      }

      return "No message content";
    }

    // Full JSON mode - pretty-print the entire JSON
    if (log.message) {
      try {
        const parsed = JSON.parse(log.message);
        const formatted = JSON.stringify(parsed, null, 2);
        if (formatted.length > 10000) {
          return (
            formatted.substring(0, 10000) +
            "\n\n... [JSON truncated for display]"
          );
        }
        return formatted;
      } catch {
        // Not valid JSON, show as-is with truncation
        if (log.message && log.message.length > 5000) {
          return (
            log.message.substring(0, 5000) +
            "\n\n... [Message truncated for display]"
          );
        }
        return log.message || "No message content";
      }
    }

    // If no raw message but we have structured messages, show them
    if (log.messages && log.messages.length > 0) {
      const formatted = formatStructuredMessages(log.messages);
      if (formatted.length > 5000) {
        return (
          formatted.substring(0, 5000) +
          "\n\n... [Message truncated for display]"
        );
      }
      return formatted;
    }

    return "No message content";
  };

  const isJson = (message?: string): boolean => {
    if (!message) return false;
    try {
      JSON.parse(message);
      return true;
    } catch {
      return false;
    }
  };

  // Infer provider name from model string
  const getProviderFromModel = (model?: string): string => {
    if (!model) return "Provider";
    const modelLower = model.toLowerCase();
    if (modelLower.includes("gpt") || modelLower.includes("openai")) return "OpenAI";
    if (modelLower.includes("claude") || modelLower.includes("anthropic")) return "Anthropic";
    if (modelLower.includes("gemini") || modelLower.includes("google")) return "Gemini";
    if (modelLower.includes("mistral")) return "Mistral";
    return "Provider";
  };

  const getDirectionLabel = (direction: string, model?: string): string => {
    const providerName = getProviderFromModel(model);
    if (direction === "request_original") return "Request (Original)";
    if (direction === "request_masked") return `Request (To ${providerName})`;
    if (direction === "response_masked") return `Response (From ${providerName})`;
    if (direction === "response_original") return "Response (Restored)";
    if (direction === "request" || direction === "In") return "Request";
    if (direction === "response" || direction === "Out") return "Response";
    return direction;
  };

  const getDirectionIcon = (direction: string) => {
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
  };

  const getRowBackground = (direction: string): string => {
    if (direction === "request_original") return "bg-blue-50";
    if (direction === "request_masked") return "bg-purple-50";
    if (direction === "response_masked") return "bg-orange-50";
    if (direction === "response_original") return "bg-green-50";
    return "";
  };

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
                onClick={handleClearLogs}
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
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${showFullJson ? "bg-blue-600" : "bg-slate-300"
              }`}
            role="switch"
            aria-checked={showFullJson}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${showFullJson ? "translate-x-6" : "translate-x-1"
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
              onClick={() => {
                setError(null);
                loadLogs(0);
              }}
              className="text-red-600 hover:text-red-800 text-sm font-medium"
            >
              Retry
            </button>
          </div>
        )}

        {/* Table */}
        <div className="flex-1 overflow-auto">
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
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
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
                            className={`font-mono text-xs whitespace-pre-wrap break-words ${showFullJson && isJson(log.message)
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
                          className={`px-2 py-1 rounded text-xs font-medium ${log.blocked
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
                                className={`flex items-center gap-1 px-2 py-1 rounded transition-colors ${!hasPII
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
            {total > maxPageSize && (
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
