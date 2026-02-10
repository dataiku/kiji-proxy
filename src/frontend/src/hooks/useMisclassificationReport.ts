import { useState, useCallback } from "react";
import type {
  DetectedEntity,
  LogEntry,
  PIIEntity,
  ReportSource,
} from "../types/provider";
import { reportMisclassification } from "../utils/misclassificationReporter";

interface ReportingData {
  entities: DetectedEntity[];
  originalInput: string;
  maskedInput: string;
  source: ReportSource;
  modelVersion?: string;
}

export function useMisclassificationReport() {
  const [isMisclassificationModalOpen, setIsMisclassificationModalOpen] =
    useState(false);
  const [reportingData, setReportingData] = useState<ReportingData | null>(
    null
  );

  const handleReportMisclassification = useCallback(
    (
      inputData: string,
      maskedInput: string,
      detectedEntities: DetectedEntity[],
      modelSignature: string | null
    ) => {
      if (!inputData || detectedEntities.length === 0) {
        alert(
          "Please process some data first before reporting misclassification."
        );
        return;
      }

      setReportingData({
        entities: detectedEntities,
        originalInput: inputData,
        maskedInput: maskedInput,
        source: "main",
        modelVersion: modelSignature || undefined,
      });
      setIsMisclassificationModalOpen(true);
    },
    []
  );

  const handleReportFromLog = useCallback(
    (logEntry: LogEntry, modelSignature: string | null) => {
      const message = logEntry.message || logEntry.formatted_messages || "";
      const detectedPIIRaw = logEntry.detectedPIIRaw || [];

      let entities: DetectedEntity[] = [];

      if (Array.isArray(detectedPIIRaw) && detectedPIIRaw.length > 0) {
        entities = detectedPIIRaw.map((entity: PIIEntity) => ({
          type: entity.pii_type || "unknown",
          original: entity.original_pii || "",
          token: "[Filtered]",
          confidence: entity.confidence || 0,
        }));
      } else {
        entities = [
          {
            type: "log_entry",
            original: logEntry.detectedPII || "None",
            token: "[Filtered]",
            confidence: 0,
          },
        ];
      }

      setReportingData({
        entities: entities,
        originalInput: message,
        maskedInput: `Log Entry ID: ${logEntry.id}, Direction: ${logEntry.direction}`,
        source: "log",
        modelVersion: logEntry.model || modelSignature || undefined,
      });
      setIsMisclassificationModalOpen(true);
    },
    []
  );

  const handleSubmitMisclassification = useCallback(
    async (comment: string) => {
      if (!reportingData) {
        console.error("No reporting data available");
        return;
      }

      try {
        await reportMisclassification({
          originalInput: reportingData.originalInput,
          maskedInput: reportingData.maskedInput,
          detectedEntities: reportingData.entities,
          userComment: comment || undefined,
          modelVersion: reportingData.modelVersion,
          timestamp: new Date().toISOString(),
        });

        alert(
          "Thank you for your feedback! The misclassification has been reported."
        );

        setReportingData(null);
        setIsMisclassificationModalOpen(false);
      } catch (error) {
        console.error("Error submitting misclassification:", error);
        alert("Failed to submit report. Please try again.");
      }
    },
    [reportingData]
  );

  const closeModal = useCallback(() => {
    setIsMisclassificationModalOpen(false);
    setReportingData(null);
  }, []);

  return {
    isMisclassificationModalOpen,
    reportingData,
    handleReportMisclassification,
    handleReportFromLog,
    handleSubmitMisclassification,
    closeModal,
  };
}
