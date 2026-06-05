import { useState } from "react";
import {
  X,
  Database,
  ChevronUp,
  ChevronDown,
  Trash2,
  AlertTriangle,
  Check,
} from "lucide-react";
import type { MappingSortColumn, SortOrder } from "../../types/provider";
import { useMappings } from "../../hooks/useMappings";
import { formatTimestamp } from "../../utils/logFormatters";

interface MappingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface SortableHeaderProps {
  label: string;
  column: MappingSortColumn;
  activeColumn: MappingSortColumn;
  activeOrder: SortOrder;
  onSort: (column: MappingSortColumn, order: SortOrder) => void;
}

function SortableHeader({
  label,
  column,
  activeColumn,
  activeOrder,
  onSort,
}: SortableHeaderProps) {
  const isActive = activeColumn === column;
  return (
    <th className="px-4 py-3 text-left text-sm font-semibold text-slate-700 border-b border-slate-200">
      <div className="flex items-center gap-2">
        <span>{label}</span>
        <span className="flex flex-col leading-none">
          <button
            type="button"
            aria-label={`Sort ${label} ascending`}
            onClick={() => onSort(column, "asc")}
            className={`-mb-0.5 transition-colors ${
              isActive && activeOrder === "asc"
                ? "text-blue-600"
                : "text-slate-300 hover:text-slate-500"
            }`}
          >
            <ChevronUp className="w-3 h-3" />
          </button>
          <button
            type="button"
            aria-label={`Sort ${label} descending`}
            onClick={() => onSort(column, "desc")}
            className={`transition-colors ${
              isActive && activeOrder === "desc"
                ? "text-blue-600"
                : "text-slate-300 hover:text-slate-500"
            }`}
          >
            <ChevronDown className="w-3 h-3" />
          </button>
        </span>
      </div>
    </th>
  );
}

export default function MappingsModal({ isOpen, onClose }: MappingsModalProps) {
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(
    null
  );

  const {
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
  } = useMappings(isOpen);

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
                Clear All Mappings?
              </h3>
            </div>
            <p className="text-slate-600 mb-6">
              This will permanently delete all {total} PII mapping
              {total === 1 ? "" : "s"}. This action cannot be undone.
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
                  await handleClearAll();
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
                    Clear All Mappings
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
            <Database className="w-6 h-6 text-slate-700" />
            <h2 className="text-2xl font-bold text-slate-800">PII Mappings</h2>
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
                title="Clear all mappings"
              >
                <Trash2 className="w-5 h-5" />
                <span className="text-sm font-medium">Clear All</span>
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

        {/* Error Display */}
        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
            <div className="text-red-600 text-sm flex-1">
              <strong>Error:</strong> {error}
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
          {isLoading && mappings.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : mappings.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-500">
              <Database className="w-12 h-12 mb-4 opacity-50" />
              <p className="text-lg">No PII mappings found</p>
            </div>
          ) : (
            <div>
              <table className="border-collapse w-full">
                <thead className="bg-slate-100 sticky top-0">
                  <tr>
                    <SortableHeader
                      label="Entity Type"
                      column="pii_type"
                      activeColumn={sortColumn}
                      activeOrder={sortOrder}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      label="Original"
                      column="original_pii"
                      activeColumn={sortColumn}
                      activeOrder={sortOrder}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      label="Masked"
                      column="dummy_pii"
                      activeColumn={sortColumn}
                      activeOrder={sortOrder}
                      onSort={handleSort}
                    />
                    <SortableHeader
                      label="Date of first entity"
                      column="created_at"
                      activeColumn={sortColumn}
                      activeOrder={sortOrder}
                      onSort={handleSort}
                    />
                    <th className="px-4 py-3 text-right text-sm font-semibold text-slate-700 border-b border-slate-200 w-px">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {mappings.map((m) => (
                    <tr
                      key={m.id}
                      className="group hover:bg-slate-50 transition-colors border-b border-slate-100"
                    >
                      <td className="px-4 py-3 text-sm">
                        <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs font-medium">
                          {m.piiType}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-slate-700 break-all">
                        {m.original}
                      </td>
                      <td className="px-4 py-3 text-sm font-mono text-slate-700 break-all">
                        {m.masked}
                      </td>
                      <td className="px-4 py-3 text-sm text-slate-600 whitespace-nowrap">
                        {formatTimestamp(m.createdAt)}
                      </td>
                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        {deletingId === m.id ? (
                          <div className="inline-flex w-5 h-5 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
                        ) : confirmingDeleteId === m.id ? (
                          <div className="inline-flex items-center gap-1">
                            <span className="text-xs text-slate-500 mr-1">
                              Delete?
                            </span>
                            <button
                              type="button"
                              aria-label="Confirm delete"
                              onClick={async () => {
                                await handleDeleteOne(m.id);
                                setConfirmingDeleteId(null);
                              }}
                              className="p-1 text-red-600 hover:bg-red-100 rounded transition-colors"
                            >
                              <Check className="w-4 h-4" />
                            </button>
                            <button
                              type="button"
                              aria-label="Cancel delete"
                              onClick={() => setConfirmingDeleteId(null)}
                              className="p-1 text-slate-500 hover:bg-slate-200 rounded transition-colors"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            aria-label="Delete mapping"
                            title="Delete mapping"
                            onClick={() => setConfirmingDeleteId(m.id)}
                            className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        )}
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
                    Load More Mappings
                  </button>
                </div>
              )}

              {/* Loading More Indicator */}
              {isLoading && mappings.length > 0 && (
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
          <p className="text-sm text-slate-500">
            Showing {mappings.length} of {total} mapping
            {total === 1 ? "" : "s"}
            {hasMore && (
              <span className="ml-2 text-slate-400">(more available)</span>
            )}
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
