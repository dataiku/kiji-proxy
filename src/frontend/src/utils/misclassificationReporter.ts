import * as Sentry from "@sentry/electron/renderer";

export interface MisclassificationReport {
  originalInput: string;
  maskedInput: string;
  detectedEntities: Array<{
    type: string;
    original: string;
    token: string;
    confidence: number;
  }>;
  userComment?: string;
  modelVersion?: string;
  timestamp: string;
}

/**
 * Report a misclassification to Sentry for tracking and analysis
 */
export async function reportMisclassification(
  report: MisclassificationReport
): Promise<void> {
  try {
    // Format entity details for better readability
    const entitySummary = report.detectedEntities
      .map(
        (e) =>
          `${e.type}: "${e.original}" (confidence: ${(
            e.confidence * 100
          ).toFixed(1)}%)`
      )
      .join(", ");

    // Create a descriptive message that includes key details
    const message = `PII Misclassification: ${
      report.detectedEntities.length
    } entities detected - ${entitySummary.substring(0, 100)}${
      entitySummary.length > 100 ? "..." : ""
    }`;

    // Capture as a custom message/event with enhanced data
    const eventId = Sentry.captureMessage(message, {
      level: "info",
      tags: {
        type: "misclassification",
        entity_count: report.detectedEntities.length.toString(),
        model_version: report.modelVersion || "unknown",
        has_user_comment: report.userComment ? "yes" : "no",
      },
      extra: {
        // Extra fields are shown prominently in Sentry UI
        user_comment: report.userComment || "(no comment provided)",
        original_input: report.originalInput,
        masked_input: report.maskedInput,
        entity_details: report.detectedEntities.map((e) => ({
          type: e.type,
          original_text: e.original,
          replacement_token: e.token,
          confidence: `${(e.confidence * 100).toFixed(1)}%`,
        })),
        model_version: report.modelVersion || "unknown",
        timestamp: report.timestamp,
      },
      contexts: {
        misclassification: {
          original_input: report.originalInput,
          masked_input: report.maskedInput,
          detected_entities: report.detectedEntities,
          user_comment: report.userComment || "",
          model_version: report.modelVersion || "unknown",
          timestamp: report.timestamp,
        },
      },
      fingerprint: ["misclassification", report.modelVersion || "unknown"],
    });

    // eventId available if needed for support reference
    void eventId;
  } catch (error) {
    console.error("Failed to send misclassification report:", error);
  }
}

/**
 * Report a general error to Sentry
 */
export function reportError(
  error: Error,
  context?: Record<string, unknown>
): void {
  try {
    Sentry.captureException(error, {
      contexts: context ? { additional: context } : undefined,
    });
  } catch (err) {
    console.error("Failed to report error to Sentry:", err);
  }
}
