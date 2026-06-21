import { useState, useEffect, useCallback } from "react";
import type {
  Mapping,
  MappingSortColumn,
  SortOrder,
} from "../types/provider";
import { apiUrl, isElectron } from "../utils/providerHelpers";

const PAGE_SIZE = 50;

export function useMappings(isOpen: boolean) {
  const [mappings, setMappings] = useState<Mapping[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);
  const [sortColumn, setSortColumn] = useState<MappingSortColumn>("created_at");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  // column/order are passed explicitly so handleSort can reload with new values
  // immediately, rather than reading stale state right after setState.
  const loadMappings = useCallback(
    async (pageNum: number, column: MappingSortColumn, order: SortOrder) => {
      setIsLoading(true);
      setError(null);
      if (pageNum === 0) {
        setMappings([]);
        setPage(0);
      }
      try {
        const offset = pageNum * PAGE_SIZE;
        const url = `${apiUrl(
          "/api/mappings",
          isElectron
        )}?limit=${PAGE_SIZE}&offset=${offset}&sort=${column}&order=${order}`;

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        setTotal(data.total || 0);
        setHasMore(data.mappings && data.mappings.length === PAGE_SIZE);

        const transformed: Mapping[] = (data.mappings || []).map(
          (m: Record<string, unknown>) => ({
            id: String(m.id),
            piiType: (m.pii_type as string) || "",
            original: (m.original_pii as string) || "",
            masked: (m.dummy_pii as string) || "",
            createdAt:
              typeof m.created_at === "string"
                ? new Date(m.created_at)
                : new Date(),
          })
        );

        // The server returns globally-sorted rows, so appending keeps order.
        // De-dupe by id defensively.
        setMappings((prev) => {
          const combined =
            pageNum === 0 ? transformed : [...prev, ...transformed];
          return Array.from(
            new Map(combined.map((item) => [item.id, item])).values()
          );
        });

        setPage(pageNum);
      } catch (err) {
        console.error("Error loading mappings:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load mappings";
        setError(errorMessage);
        if (pageNum === 0) {
          setMappings([]);
        }
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  // Load mappings when the modal opens
  useEffect(() => {
    if (isOpen) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      loadMappings(0, sortColumn, sortOrder);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const handleLoadMore = useCallback(() => {
    if (!isLoading && hasMore) {
      loadMappings(page + 1, sortColumn, sortOrder);
    }
  }, [isLoading, hasMore, loadMappings, page, sortColumn, sortOrder]);

  // Changing sort resets pagination to page 0 and reloads the whole dataset.
  const handleSort = useCallback(
    (column: MappingSortColumn, order: SortOrder) => {
      setSortColumn(column);
      setSortOrder(order);
      loadMappings(0, column, order);
    },
    [loadMappings]
  );

  const retry = useCallback(() => {
    setError(null);
    loadMappings(0, sortColumn, sortOrder);
  }, [loadMappings, sortColumn, sortOrder]);

  // Delete every mapping (DELETE /api/mappings with no id).
  const handleClearAll = useCallback(async () => {
    setIsClearing(true);
    setError(null);
    try {
      const response = await fetch(apiUrl("/api/mappings", isElectron), {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setMappings([]);
      setTotal(0);
      setPage(0);
      setHasMore(false);
    } catch (err) {
      console.error("Error clearing mappings:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to clear mappings";
      setError(errorMessage);
    } finally {
      setIsClearing(false);
    }
  }, []);

  // Delete a single mapping by id; optimistically drop the row on success.
  const handleDeleteOne = useCallback(async (id: string) => {
    setDeletingId(id);
    setError(null);
    try {
      const response = await fetch(
        `${apiUrl("/api/mappings", isElectron)}?id=${encodeURIComponent(id)}`,
        { method: "DELETE" }
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      setMappings((prev) => prev.filter((m) => m.id !== id));
      setTotal((prev) => (prev > 0 ? prev - 1 : 0));
    } catch (err) {
      console.error("Error deleting mapping:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to delete mapping";
      setError(errorMessage);
    } finally {
      setDeletingId(null);
    }
  }, []);

  return {
    mappings,
    isLoading,
    isClearing,
    deletingId,
    error,
    hasMore,
    total,
    sortColumn,
    sortOrder,
    handleLoadMore,
    handleSort,
    handleClearAll,
    handleDeleteOne,
    retry,
  };
}
