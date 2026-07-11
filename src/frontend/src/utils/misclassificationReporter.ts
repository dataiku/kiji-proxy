import * as Sentry from "@sentry/electron/renderer";

export interface MisclassificationReport {
  // NOTE: the raw, pre-masking input is deliberately NOT part of this payload.
  // Kiji is a privacy proxy; sending the unmasked text (which by definition
  // contains the PII we exist to strip) to a third party would defeat the point.
  // Only the masked text and non-identifying entity metadata are reported.
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
 * Report a misclassification to Sentry for tracking and analysis.
 *
 * Privacy: only the MASKED input and non-identifying entity metadata (type,
 * replacement token, confidence) are sent. The original text and the original
 * matched substrings are never transmitted.
 *
 * Returns true if the report was actually sent, false if telemetry is disabled
 * (Sentry not initialized) so the caller can tell the user nothing was sent.
 */
export async function reportMisclassification(
  report: MisclassificationReport
): Promise<boolean> {
  // When telemetry is opt-out (default), Sentry is never initialized, so there
  // is no client and captureMessage would silently no-op. Detect that up front.
  if (!Sentry.getClient()) {
    return false;
  }

  try {
    // Non-identifying entity summary: type + confidence only, never the matched
    // text. Safe to include in the human-readable event title.
    const entitySummary = report.detectedEntities
      .map(
        (e) => `${e.type} (confidence: ${(e.confidence * 100).toFixed(1)}%)`
      )
      .join(", ");

    // Create a descriptive message that includes key (non-PII) details
    const message = `PII Misclassification: ${
      report.detectedEntities.length
    } entities detected - ${entitySummary.substring(0, 100)}${
      entitySummary.length > 100 ? "..." : ""
    }`;

    // Strip the raw matched text from entity details — keep only the metadata
    // that is safe to leave the machine.
    const safeEntityDetails = report.detectedEntities.map((e) => ({
      type: e.type,
      replacement_token: e.token,
      confidence: `${(e.confidence * 100).toFixed(1)}%`,
    }));

    // Capture as a custom message/event with enhanced (non-PII) data
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
        masked_input: report.maskedInput,
        entity_details: safeEntityDetails,
        model_version: report.modelVersion || "unknown",
        timestamp: report.timestamp,
      },
      contexts: {
        misclassification: {
          masked_input: report.maskedInput,
          detected_entities: safeEntityDetails,
          user_comment: report.userComment || "",
          model_version: report.modelVersion || "unknown",
          timestamp: report.timestamp,
        },
      },
      fingerprint: ["misclassification", report.modelVersion || "unknown"],
    });

    // eventId available if needed for support reference
    void eventId;
    return true;
  } catch (error) {
    console.error("Failed to send misclassification report:", error);
    return false;
  }
}

/**
 * Report a general error to Sentry. No-op when telemetry is disabled (Sentry
 * not initialized), so this is safe to call unconditionally.
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
