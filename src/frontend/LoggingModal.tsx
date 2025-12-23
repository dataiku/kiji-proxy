import React, { useState, useEffect } from "react";
import {
  X,
  FileText,
  ArrowDownCircle,
  ArrowUpCircle,
  Code,
  MessageSquare,
} from "lucide-react";

interface LogEntry {
  id: string;
  direction: "In" | "Out";
  message: string;
  detectedPII: string;
  blocked: boolean;
  timestamp: Date;
}

interface LoggingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function LoggingModal({ isOpen, onClose }: LoggingModalProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showFullJson, setShowFullJson] = useState(false);

  // TODO: Replace with actual API call to fetch logs
  useEffect(() => {
    if (isOpen) {
      loadLogs();
    }
  }, [isOpen]);

  const loadLogs = async () => {
    setIsLoading(true);
    try {
      // Determine the API URL
      const isElectron =
        typeof window !== "undefined" && window.electronAPI !== undefined;
      const apiUrl = isElectron
        ? "http://localhost:8080/logs" // Direct call to Go server in Electron
        : "/logs"; // Proxied call in web mode

      const response = await fetch(apiUrl);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Transform the API response to match LogEntry format
      const transformedLogs: LogEntry[] = (data.logs || []).map((log: any) => {
        // Parse timestamp - handle both string and Date formats
        let timestamp: Date;
        if (typeof log.timestamp === "string") {
          timestamp = new Date(log.timestamp);
        } else if (log.timestamp instanceof Date) {
          timestamp = log.timestamp;
        } else {
          timestamp = new Date();
        }

        return {
          id: String(log.id),
          direction: log.direction || "Unknown",
          message: log.message || "",
          detectedPII: log.detected_pii || "None",
          blocked: log.blocked || false,
          timestamp: timestamp,
        };
      });

      setLogs(transformedLogs);
    } catch (error) {
      console.error("Error loading logs:", error);
      // Set empty logs on error
      setLogs([]);
    } finally {
      setIsLoading(false);
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
          .map((msg: any) => {
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
          .map((choice: any) => {
            // Handle chat completion format
            if (choice.message?.content) {
              const role = choice.message.role
                ? `[${choice.message.role}]`
                : "[assistant]";
              return `${role} ${choice.message.content}`;
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
    } catch (error) {
      // Not JSON or parsing failed, return as-is
      return message;
    }
  };

  const formatMessage = (message: string): string => {
    if (!showFullJson) {
      // Messages Only mode - extract and show just the message content
      const extracted = extractMessageFromJson(message);
      // Truncate very long messages for display (5000 chars)
      if (extracted.length > 5000) {
        return (
          extracted.substring(0, 5000) +
          "\n\n... [Message truncated for display]"
        );
      }
      return extracted;
    }

    // Full JSON mode - pretty-print the entire JSON
    try {
      const parsed = JSON.parse(message);
      const formatted = JSON.stringify(parsed, null, 2);
      // Truncate very long JSON for display (10000 chars)
      if (formatted.length > 10000) {
        return (
          formatted.substring(0, 10000) + "\n\n... [JSON truncated for display]"
        );
      }
      return formatted;
    } catch {
      // Not valid JSON, show as-is with truncation
      if (message.length > 5000) {
        return (
          message.substring(0, 5000) + "\n\n... [Message truncated for display]"
        );
      }
      return message;
    }
  };

  const isJson = (message: string): boolean => {
    try {
      JSON.parse(message);
      return true;
    } catch {
      return false;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-6xl w-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <FileText className="w-6 h-6 text-slate-700" />
            <h2 className="text-2xl font-bold text-slate-800">Logging</h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
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

        {/* Table */}
        <div className="flex-1 overflow-auto">
          {isLoading ? (
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
                      In/Out
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
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log) => (
                    <tr
                      key={log.id}
                      className="hover:bg-slate-50 transition-colors border-b border-slate-100"
                    >
                      <td className="px-4 py-3 text-sm">
                        <div className="flex items-center gap-2">
                          {log.direction === "In" ? (
                            <ArrowDownCircle className="w-4 h-4 text-blue-600" />
                          ) : (
                            <ArrowUpCircle className="w-4 h-4 text-green-600" />
                          )}
                          <span className="font-medium">{log.direction}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-700">
                        <div className="relative">
                          {!showFullJson && isJson(log.message) && (
                            <div className="absolute -top-1 -right-1 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
                              {log.direction === "In" ? "Request" : "Response"}
                            </div>
                          )}
                          <pre
                            className={`font-mono text-xs whitespace-pre-wrap break-words ${
                              showFullJson && isJson(log.message)
                                ? "bg-slate-50 p-2 rounded border border-slate-200"
                                : !showFullJson && isJson(log.message)
                                  ? "p-2 rounded bg-gradient-to-br from-blue-50 to-slate-50 border border-blue-100"
                                  : ""
                            }`}
                          >
                            {formatMessage(log.message)}
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
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-4 border-t border-slate-200 flex items-center justify-between">
          <p className="text-sm text-slate-500">
            {logs.length} log {logs.length === 1 ? "entry" : "entries"}
          </p>
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
