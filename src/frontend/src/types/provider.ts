// Provider types shared across the application

export type ProviderType = "openai" | "anthropic" | "gemini" | "mistral";

export interface ProviderSettings {
  hasApiKey: boolean;
  model: string;
}

export interface ProvidersConfig {
  activeProvider: ProviderType;
  providers: Record<ProviderType, ProviderSettings>;
}

// Default models per provider
export const DEFAULT_MODELS: Record<ProviderType, string> = {
  openai: "gpt-3.5-turbo",
  anthropic: "claude-3-haiku-20240307",
  gemini: "gemini-flash-latest",
  mistral: "mistral-small-latest",
};

// Provider display names
export const PROVIDER_NAMES: Record<ProviderType, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Gemini",
  mistral: "Mistral",
};

export interface PerformanceWithMemory extends Performance {
  memory?: {
    jsHeapSizeLimit: number;
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  };
}

export interface ContentBlock {
  type: string;
  text: string;
}

export interface Part {
  text?: string;
}

export interface PiiEntityForProcessing {
  label: string;
  text: string;
  masked_text: string;
  confidence: number;
}

export interface ProviderResponse {
  choices?: { message: { content: string } }[];
  content?: { type: string; text: string }[];
  candidates?: { content: { parts: { text?: string }[] } }[];
}

export interface PIIEntity {
  pii_type: string;
  original_pii: string;
  confidence?: number;
}

export interface DetectedEntity {
  type: string;
  original: string;
  token: string;
  confidence: number;
}

export interface LogEntry {
  id: string;
  direction: string;
  message?: string;
  messages?: Array<{ role: string; content: string }>;
  formatted_messages?: string;
  model?: string;
  detectedPII: string;
  detectedPIIRaw?: PIIEntity[];
  blocked: boolean;
  timestamp: Date;
}
