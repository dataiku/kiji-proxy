import type {
  ProviderType,
  ContentBlock,
  Part,
  ProviderResponse,
} from "../types/provider";
import { DEFAULT_MODELS } from "../types/provider";

export const GO_SERVER_ADDRESS = "http://localhost:8080";
export const GO_SERVER_PORT = 8080;

export function getGoServerAddress(isElectron: boolean): string {
  if (isElectron) {
    return GO_SERVER_ADDRESS;
  }
  // In web mode, use relative path (proxied)
  return "";
}

/**
 * Build an API URL that works in both Electron (direct) and web (proxied) modes.
 */
export function apiUrl(path: string, isElectron: boolean): string {
  return isElectron ? `${GO_SERVER_ADDRESS}${path}` : path;
}

export function getModel(provider: ProviderType, customModel: string): string {
  return customModel || DEFAULT_MODELS[provider] || "gpt-3.5-turbo";
}

export function buildRequestBody(
  provider: ProviderType,
  model: string,
  content: string
) {
  const baseFields = { provider };

  switch (provider) {
    case "openai":
    case "mistral":
      return {
        ...baseFields,
        model,
        messages: [{ role: "user", content }],
        max_tokens: 1000,
      };

    case "anthropic":
      return {
        ...baseFields,
        model,
        messages: [{ role: "user", content }],
        max_tokens: 1024,
      };

    case "gemini":
      return {
        ...baseFields,
        model,
        contents: [{ parts: [{ text: content }] }],
        generationConfig: { maxOutputTokens: 1000 },
      };

    default:
      return {
        ...baseFields,
        model,
        messages: [{ role: "user", content }],
        max_tokens: 1000,
      };
  }
}

export function buildHeaders(
  provider: ProviderType,
  providerApiKey: string
): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  switch (provider) {
    case "anthropic":
      headers["x-api-key"] = providerApiKey;
      headers["anthropic-version"] = "2023-06-01";
      break;
    case "gemini":
      headers["x-goog-api-key"] = providerApiKey;
      break;
    default: // openai, mistral
      headers["Authorization"] = `Bearer ${providerApiKey}`;
  }

  return headers;
}

export function getProviderEndpoint(
  provider: ProviderType,
  model: string
): string {
  switch (provider) {
    case "openai":
    case "mistral":
      return "/v1/chat/completions";
    case "anthropic":
      return "/v1/messages";
    case "gemini":
      return `/v1beta/models/${model}:generateContent`;
    default:
      return "/v1/chat/completions";
  }
}

export function extractAssistantMessage(
  provider: ProviderType,
  data: ProviderResponse
): string {
  try {
    switch (provider) {
      case "openai":
      case "mistral":
        return data.choices?.[0]?.message?.content || "";

      case "anthropic":
        if (Array.isArray(data.content)) {
          return data.content
            .filter((block: ContentBlock) => block.type === "text")
            .map((block: ContentBlock) => block.text)
            .join("");
        }
        return "";

      case "gemini": {
        const parts = data.candidates?.[0]?.content?.parts;
        if (Array.isArray(parts)) {
          return parts
            .filter((part: Part) => part.text !== undefined)
            .map((part: Part) => part.text)
            .join("");
        }
        return "";
      }

      default:
        return data.choices?.[0]?.message?.content || "";
    }
  } catch (error) {
    console.error(`[ERROR] Failed to extract message for ${provider}:`, error);
    return "";
  }
}

export function getConfidenceColor(confidence: number): string {
  const clampedConfidence = Math.max(0, Math.min(1, confidence));
  const red = Math.round(220 - (220 - 34) * clampedConfidence);
  const green = Math.round(38 + (197 - 38) * clampedConfidence);
  const blue = Math.round(38 + (94 - 38) * clampedConfidence);
  return `rgb(${red}, ${green}, ${blue})`;
}
