import { useState, useEffect, useCallback } from "react";
import type { LogEntry, PIIEntity } from "../types/provider";
import { apiUrl, isElectron } from "../utils/providerHelpers";
import { sortLogs } from "../utils/logFormatters";

const PAGE_SIZE = 50;

export function useLogs(isOpen: boolean) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);

  const loadLogs = useCallback(
    async (pageNum: number) => {
      if (!hasMore && pageNum > 0) return;

      setIsLoading(true);
      setError(null);
      try {
        const offset = pageNum * PAGE_SIZE;
        const logsUrl = `${apiUrl("/logs", isElectron)}?limit=${PAGE_SIZE}&offset=${offset}`;

        const response = await fetch(logsUrl);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        setTotal(data.total || 0);
        setHasMore(data.logs && data.logs.length === PAGE_SIZE);

        const transformedLogs: LogEntry[] = (data.logs || []).map(
          (log: Record<string, unknown>) => {
            let timestamp: Date;
            if (typeof log.timestamp === "string") {
              timestamp = new Date(log.timestamp);
            } else if (log.timestamp instanceof Date) {
              timestamp = log.timestamp;
            } else {
              timestamp = new Date();
            }

            let transactionId: string | undefined;
            if (typeof log.message === "string") {
              try {
                const parsed = JSON.parse(log.message);
                if (
                  parsed &&
                  typeof parsed === "object" &&
                  "_transaction_id" in parsed
                ) {
                  transactionId = parsed._transaction_id;
                }
              } catch {
                // Ignore parsing errors
              }
            }

            let formattedPII = "None";
            const rawPII = log.detected_pii;
            let typedRawPII: PIIEntity[] | undefined;

            if (rawPII && Array.isArray(rawPII) && rawPII.length > 0) {
              typedRawPII = rawPII as PIIEntity[];
              formattedPII = typedRawPII
                .map(
                  (entity: PIIEntity) =>
                    `${entity.pii_type}: ${entity.original_pii}`
                )
                .join(", ");
            } else if (typeof rawPII === "string" && rawPII !== "None") {
              formattedPII = rawPII;
              typedRawPII = undefined;
            }

            const entry: LogEntry = {
              id: String(log.id),
              direction:
                (log.direction as LogEntry["direction"]) || "Unknown",
              message: log.message as string | undefined,
              messages: log.messages as
                | Array<{ role: string; content: string }>
                | undefined,
              formatted_messages: log.formatted_messages as string | undefined,
              model: log.model as string | undefined,
              detectedPII: formattedPII,
              detectedPIIRaw: typedRawPII,
              blocked: (log.blocked as boolean) || false,
              timestamp: timestamp,
              transactionId: transactionId,
            };

            return entry;
          }
        );

        setLogs((prev) => {
          const combined =
            pageNum === 0 ? transformedLogs : [...prev, ...transformedLogs];
          const unique = Array.from(
            new Map(combined.map((item) => [item.id, item])).values()
          );
          return sortLogs(unique);
        });

        setPage(pageNum);
      } catch (err) {
        console.error("Error loading logs:", err);

        const errorMessage =
          err instanceof Error ? err.message : "Failed to load logs";
        setError(errorMessage);

        if (pageNum === 0) {
          setLogs([]);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [hasMore]
  );

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

  const handleLoadMore = useCallback(() => {
    if (!isLoading && hasMore) {
      loadLogs(page + 1);
    }
  }, [isLoading, hasMore, loadLogs, page]);

  const handleClearLogs = useCallback(async () => {
    setIsClearing(true);
    setError(null);

    try {
      const response = await fetch(apiUrl("/logs", isElectron), {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setLogs([]);
      setTotal(0);
      setPage(0);
      setHasMore(false);
    } catch (err) {
      console.error("Error clearing logs:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to clear logs";
      setError(errorMessage);
    } finally {
      setIsClearing(false);
    }
  }, []);

  const retry = useCallback(() => {
    setError(null);
    loadLogs(0);
  }, [loadLogs]);

  return {
    logs,
    isLoading,
    isClearing,
    error,
    hasMore,
    total,
    handleLoadMore,
    handleClearLogs,
    retry,
  };
}
