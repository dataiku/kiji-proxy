import type { LogEntry } from "../types/provider";

export function sortLogs(logsToSort: LogEntry[]): LogEntry[] {
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

  const sortedGroups: LogEntry[][] = Object.values(groups).map((group) => {
    return group.sort((a, b) => {
      const order = [
        "request_original",
        "request_masked",
        "response_masked",
        "response_original",
      ];

      const aIndex = order.indexOf(a.direction);
      const bIndex = order.indexOf(b.direction);

      if (aIndex !== -1 && bIndex !== -1) {
        return aIndex - bIndex;
      }

      return a.timestamp.getTime() - b.timestamp.getTime();
    });
  });

  type SortableItem =
    | { type: "group"; logs: LogEntry[]; latestTimestamp: number }
    | { type: "single"; log: LogEntry; latestTimestamp: number };

  const sortableItems: SortableItem[] = [
    ...sortedGroups.map((g) => ({
      type: "group" as const,
      logs: g,
      latestTimestamp: Math.max(...g.map((l) => l.timestamp.getTime())),
    })),
    ...singles.map((l) => ({
      type: "single" as const,
      log: l,
      latestTimestamp: l.timestamp.getTime(),
    })),
  ];

  sortableItems.sort((a, b) => b.latestTimestamp - a.latestTimestamp);

  const flattened: LogEntry[] = [];
  sortableItems.forEach((item) => {
    if (item.type === "group") {
      flattened.push(...item.logs);
    } else {
      flattened.push(item.log);
    }
  });

  return flattened;
}

export function formatTimestamp(date: Date): string {
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function extractTextContent(content: unknown): string {
  if (typeof content === "string") return content;

  if (Array.isArray(content)) {
    return content
      .map((item: unknown) => extractTextContent(item))
      .filter((text: string) => text.trim().length > 0)
      .join("");
  }

  if (content && typeof content === "object") {
    const contentObj = content as Record<string, unknown>;
    if (typeof contentObj.text === "string") return contentObj.text;
    if (Array.isArray(contentObj.parts)) {
      return extractTextContent(contentObj.parts);
    }
    if (typeof contentObj.content === "string") return contentObj.content;
  }

  return "";
}

function extractMessageFromJson(message: string): string {
  try {
    const parsed = JSON.parse(message);

    if (parsed.messages && Array.isArray(parsed.messages)) {
      const messages = parsed.messages
        .map((msg: Record<string, unknown>) => {
          const role = msg.role ? `[${msg.role}]` : "";
          const content = extractTextContent(msg.content);
          return role ? `${role} ${content}` : content;
        })
        .filter((content: string) => content.trim())
        .join("\n\n");
      return messages || message;
    }

    if (parsed.choices && Array.isArray(parsed.choices)) {
      const messages = parsed.choices
        .map((choice: Record<string, unknown>) => {
          if (choice.message && typeof choice.message === "object") {
            const messageObj = choice.message as Record<string, unknown>;
            const content = extractTextContent(messageObj.content);
            if (content) {
              const role = messageObj.role
                ? `[${messageObj.role}]`
                : "[assistant]";
              return `${role} ${content}`;
            }
          }
          if (choice.text) {
            return `[completion] ${choice.text}`;
          }
          return "";
        })
        .filter((content: string) => content.trim())
        .join("\n\n");
      return messages || message;
    }

    if (parsed.content && Array.isArray(parsed.content)) {
      const textBlocks = parsed.content
        .map((block: Record<string, unknown>) => {
          if (block.type === "text" && typeof block.text === "string") {
            return block.text;
          }
          return "";
        })
        .filter((text: string) => text.trim());
      if (textBlocks.length > 0) {
        const role = parsed.role ? `[${parsed.role}]` : "[assistant]";
        return `${role} ${textBlocks.join("")}`;
      }
    }

    if (parsed.candidates && Array.isArray(parsed.candidates)) {
      const messages = parsed.candidates
        .map((candidate: Record<string, unknown>) => {
          if (!candidate.content || typeof candidate.content !== "object") {
            return "";
          }
          const contentObj = candidate.content as Record<string, unknown>;
          const text = extractTextContent(contentObj.parts);
          return text ? `[assistant] ${text}` : "";
        })
        .filter((text: string) => text.trim())
        .join("\n\n");
      if (messages) {
        return messages;
      }
    }

    return message;
  } catch (_error) {
    return message;
  }
}

function formatStructuredMessages(
  messages: Array<{ role: string; content: string }>
): string {
  return messages.map((msg) => `[${msg.role}] ${msg.content}`).join("\n\n");
}

export function formatMessage(log: LogEntry, useFullJson: boolean): string {
  if (!useFullJson) {
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
      if (log.message && log.message.length > 5000) {
        return (
          log.message.substring(0, 5000) +
          "\n\n... [Message truncated for display]"
        );
      }
      return log.message || "No message content";
    }
  }

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
}

export function isJson(message?: string): boolean {
  if (!message) return false;
  try {
    JSON.parse(message);
    return true;
  } catch {
    return false;
  }
}

export function getProviderFromModel(model?: string): string {
  if (!model) return "Provider";
  const modelLower = model.toLowerCase();
  if (modelLower.includes("gpt") || modelLower.includes("openai"))
    return "OpenAI";
  if (modelLower.includes("claude") || modelLower.includes("anthropic"))
    return "Anthropic";
  if (modelLower.includes("gemini") || modelLower.includes("google"))
    return "Gemini";
  if (modelLower.includes("mistral")) return "Mistral";
  return "Provider";
}

export function getDirectionLabel(direction: string, model?: string): string {
  const providerName = getProviderFromModel(model);
  if (direction === "request_original") return "Request (Original)";
  if (direction === "request_masked") return `Request (To ${providerName})`;
  if (direction === "response_masked")
    return `Response (From ${providerName})`;
  if (direction === "response_original") return "Response (Restored)";
  if (direction === "request" || direction === "In") return "Request";
  if (direction === "response" || direction === "Out") return "Response";
  return direction;
}

export function getRowBackground(direction: string): string {
  if (direction === "request_original") return "bg-blue-50";
  if (direction === "request_masked") return "bg-purple-50";
  if (direction === "response_masked") return "bg-orange-50";
  if (direction === "response_original") return "bg-green-50";
  return "";
}
