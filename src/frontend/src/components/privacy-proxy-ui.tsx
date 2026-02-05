import { useState, useEffect, useRef, useMemo } from "react";
import {
  Send,
  AlertCircle,
  Settings,
  FileText,
  Info,
  Menu,
  Flag,
} from "lucide-react";
import logoImage from "../../assets/logo.png";
import yaakMascot from "../../assets/yaak.png";
import SettingsModal from "./modals/SettingsModal";
import LoggingModal from "./modals/LoggingModal";
import AboutModal from "./modals/AboutModal";
import MisclassificationModal from "./modals/MisclassificationModal";
import TermsModal from "./modals/TermsModal";

import WelcomeModal from "./modals/WelcomeModal";
import {
  highlightTextByCharacter,
  highlightEntitiesByToken,
  highlightEntitiesByOriginal,
} from "../utils/textHighlight";
import { reportMisclassification } from "../utils/misclassificationReporter";

interface PerformanceWithMemory extends Performance {
  memory?: {
    jsHeapSizeLimit: number;
    totalJSHeapSize: number;
    usedJSHeapSize: number;
  };
}

interface ContentBlock {
  type: string;
  text: string;
}

interface Part {
  text?: string;
}

interface PiiEntityForProcessing {
  label: string;
  text: string;
  masked_text: string;
  confidence: number;
}

interface ProviderResponse {
  choices?: { message: { content: string } }[];
  content?: { type: string; text: string }[];
  candidates?: { content: { parts: { text?: string }[] } }[];
}

// Provider types
type ProviderType = "openai" | "anthropic" | "gemini" | "mistral";

interface ProviderSettings {
  hasApiKey: boolean;
  model: string;
}

interface ProvidersConfig {
  activeProvider: ProviderType;
  providers: Record<ProviderType, ProviderSettings>;
}

// Default models per provider
const DEFAULT_MODELS: Record<ProviderType, string> = {
  openai: "gpt-3.5-turbo",
  anthropic: "claude-3-haiku-20240307",
  gemini: "gemini-flash-latest",
  mistral: "mistral-small-latest",
};

// Provider display names
const PROVIDER_NAMES: Record<ProviderType, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Gemini",
  mistral: "Mistral",
};

interface PIIEntity {
  pii_type: string;
  original_pii: string;
  confidence?: number;
}

interface LogEntry {
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

export default function PrivacyProxyUI() {
  const [inputData, setInputData] = useState("");
  const [maskedInput, setMaskedInput] = useState("");
  const [maskedOutput, setMaskedOutput] = useState("");
  const [finalOutput, setFinalOutput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedEntities, setDetectedEntities] = useState<
    Array<{ type: string; original: string; token: string; confidence: number }>
  >([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isLoggingOpen, setIsLoggingOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [isMisclassificationModalOpen, setIsMisclassificationModalOpen] =
    useState(false);
  const [isTermsOpen, setIsTermsOpen] = useState(false);

  const [isWelcomeOpen, setIsWelcomeOpen] = useState(false);
  const [termsRequireAcceptance, setTermsRequireAcceptance] = useState(false);
  const [reportingData, setReportingData] = useState<{
    entities: Array<{
      type: string;
      original: string;
      token: string;
      confidence: number;
    }>;
    originalInput: string;
    maskedInput: string;
    source: string;
    modelVersion?: string;
  } | null>(null);

  // Multi-provider state
  const [activeProvider, setActiveProvider] = useState<ProviderType>("openai");
  const [providersConfig, setProvidersConfig] = useState<ProvidersConfig>({
    activeProvider: "openai",
    providers: {
      openai: { hasApiKey: false, model: "" },
      anthropic: { hasApiKey: false, model: "" },
      gemini: { hasApiKey: false, model: "" },
      mistral: { hasApiKey: false, model: "" },
    },
  });
  const [apiKey, setApiKey] = useState<string | null>(null);

  const [serverStatus, setServerStatus] = useState<"online" | "offline">(
    "offline"
  );
  const [serverHealth, setServerHealth] = useState<{
    status: "online" | "offline";
    modelHealthy: boolean;
    modelError?: string;
  }>({
    status: "offline",
    modelHealthy: false,
  });
  const [modelSignature, setModelSignature] = useState<string | null>(null);
  const [showModelTooltip, setShowModelTooltip] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [version, setVersion] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Fixed Go server address - always call the Go server at this address
  const GO_SERVER_ADDRESS = "http://localhost:8080";

  // Detect if running in Electron
  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

  // Memoize highlighted text to prevent re-computation and memory explosion
  // Safety limit for text highlighting to prevent memory issues
  const MAX_HIGHLIGHT_SIZE = 50000; // 50KB max for highlighting

  const truncateForHighlighting = (text: string): string => {
    if (text.length > MAX_HIGHLIGHT_SIZE) {
      console.warn(
        `[SAFETY] Text truncated from ${text.length} to ${MAX_HIGHLIGHT_SIZE} chars for highlighting`
      );
      return (
        text.substring(0, MAX_HIGHLIGHT_SIZE) +
        "\n\n... [Text truncated for display - too large to highlight safely]"
      );
    }
    return text;
  };

  // Generate HTML strings for highlighting (no React components)
  const highlightedInputOriginalHTML = useMemo(
    () =>
      highlightTextByCharacter(
        truncateForHighlighting(inputData),
        detectedEntities,
        "bg-red-200 text-red-900"
      ),
    [inputData, detectedEntities]
  );

  const highlightedInputMaskedHTML = useMemo(
    () =>
      highlightEntitiesByToken(
        truncateForHighlighting(maskedInput),
        detectedEntities,
        "bg-green-200 text-green-900 font-bold"
      ),
    [maskedInput, detectedEntities]
  );

  const highlightedOutputMaskedHTML = useMemo(
    () =>
      highlightEntitiesByToken(
        truncateForHighlighting(maskedOutput),
        detectedEntities,
        "bg-purple-200 text-purple-900 font-bold"
      ),
    [maskedOutput, detectedEntities]
  );

  const highlightedOutputFinalHTML = useMemo(
    () =>
      highlightEntitiesByOriginal(
        truncateForHighlighting(finalOutput),
        detectedEntities,
        "bg-blue-200 text-blue-900 font-bold"
      ),
    [finalOutput, detectedEntities]
  );

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    };

    if (isMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isMenuOpen]);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Load settings on mount and listen for menu command
  useEffect(() => {
    if (isElectron && window.electronAPI) {
      loadSettings();

      // Check if user has accepted terms on app load
      window.electronAPI.getTermsAccepted().then((accepted) => {
        if (!accepted) {
          setTermsRequireAcceptance(true);
          setIsTermsOpen(true);
        }
      });

      // Check if welcome modal has been dismissed
      window.electronAPI.getWelcomeDismissed().then((dismissed) => {
        if (!dismissed) {
          // Show welcome modal after a short delay (after other modals)
          setTimeout(() => {
            setIsWelcomeOpen(true);
          }, 500);
        }
      });

      // Listen for settings menu command
      if (window.electronAPI.onSettingsOpen) {
        window.electronAPI.onSettingsOpen(() => {
          setIsSettingsOpen(true);
        });
      }

      // Listen for about menu command
      if (window.electronAPI.onAboutOpen) {
        window.electronAPI.onAboutOpen(() => {
          setIsAboutOpen(true);
        });
      }

      // Listen for terms menu command
      if (window.electronAPI.onTermsOpen) {
        window.electronAPI.onTermsOpen(() => {
          setTermsRequireAcceptance(false);
          setIsTermsOpen(true);
        });
      }

      // Cleanup
      return () => {
        if (window.electronAPI?.removeSettingsListener) {
          window.electronAPI.removeSettingsListener();
        }
        if (window.electronAPI?.removeAboutListener) {
          window.electronAPI.removeAboutListener();
        }
        if (window.electronAPI?.removeTermsListener) {
          window.electronAPI.removeTermsListener();
        }
      };
    }
    return undefined;
  }, [isElectron]);

  // Check server status periodically
  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const url =
          isElectron && window.electronAPI
            ? GO_SERVER_ADDRESS
            : "http://localhost:8080";
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout

        const response = await fetch(`${url}/health`, {
          method: "GET",
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (response.ok) {
          const data = await response.json();
          setServerStatus("online");
          setServerHealth({
            status: "online",
            modelHealthy: data.model_healthy !== false,
            modelError: data.model_error,
          });
        } else {
          setServerStatus("offline");
          setServerHealth({
            status: "offline",
            modelHealthy: false,
          });
        }
      } catch (_error) {
        setServerStatus("offline");
        setServerHealth({
          status: "offline",
          modelHealthy: false,
        });
      }
    };

    const loadModelSignature = async () => {
      try {
        const apiUrl = isElectron
          ? "http://localhost:8080/api/model/security" // Direct call to Go server in Electron
          : "/api/model/security"; // Proxied call in web mode

        const response = await fetch(apiUrl);
        if (response.ok) {
          const data = await response.json();
          const hash = data.hash;
          if (hash) {
            setModelSignature(hash.substring(0, 7));
          }
        }
      } catch (_error) {
        // Silently fail - model signature is optional UI enhancement
      }
    };

    const loadVersion = async () => {
      try {
        const apiUrl = isElectron
          ? "http://localhost:8080/version" // Direct call to Go server in Electron
          : "/version"; // Proxied call in web mode

        const response = await fetch(apiUrl);
        if (response.ok) {
          const data = await response.json();
          if (data.version) {
            setVersion(data.version);
          }
        }
      } catch (_error) {
        // Silently fail - version is optional UI enhancement
      }
    };

    // Check immediately
    checkServerStatus();
    loadModelSignature();
    loadVersion();

    // Check every 5 seconds
    const interval = setInterval(checkServerStatus, 5000);

    return () => clearInterval(interval);
  }, [isElectron]);

  const loadSettings = async () => {
    if (!window.electronAPI) return;

    try {
      // Load providers config
      const config = await window.electronAPI.getProvidersConfig();
      setProvidersConfig(config);
      setActiveProvider(config.activeProvider);

      // Load API key for active provider
      const key = await window.electronAPI.getProviderApiKey(
        config.activeProvider
      );
      setApiKey(key);

      // Debug logging
      if (key) {
        console.log(
          `API key loaded for ${config.activeProvider} (length: ${key.length})`
        );
      } else {
        console.log(`No API key found for ${config.activeProvider}`);
      }
    } catch (error) {
      console.error("Error loading settings:", error);
    }
  };

  // Get Go server address - always use fixed address in Electron, relative path in web mode
  const getGoServerAddress = () => {
    if (isElectron && window.electronAPI) {
      return GO_SERVER_ADDRESS;
    }
    // In web mode, use relative path (proxied)
    return "";
  };

  // Get model for provider (custom or default)
  const getModel = (provider: ProviderType, customModel: string): string => {
    return customModel || DEFAULT_MODELS[provider] || "gpt-3.5-turbo";
  };

  // Build provider-specific request body
  const buildRequestBody = (
    provider: ProviderType,
    model: string,
    content: string
  ) => {
    // Always include provider field for backend routing
    const baseFields = { provider };

    switch (provider) {
      case "openai":
      case "mistral":
        // OpenAI/Mistral format
        return {
          ...baseFields,
          model,
          messages: [{ role: "user", content }],
          max_tokens: 1000,
        };

      case "anthropic":
        // Anthropic format - max_tokens is REQUIRED
        return {
          ...baseFields,
          model,
          messages: [{ role: "user", content }],
          max_tokens: 1024,
        };

      case "gemini":
        // Gemini format - completely different structure
        return {
          ...baseFields,
          model, // Backend uses this to build the URL
          contents: [{ parts: [{ text: content }] }],
          generationConfig: { maxOutputTokens: 1000 },
        };

      default:
        // Fallback to OpenAI format
        return {
          ...baseFields,
          model,
          messages: [{ role: "user", content }],
          max_tokens: 1000,
        };
    }
  };

  // Build provider-specific headers
  const buildHeaders = (
    provider: ProviderType,
    providerApiKey: string
  ): Record<string, string> => {
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
  };

  // Get provider-specific API endpoint path
  const getProviderEndpoint = (
    provider: ProviderType,
    model: string
  ): string => {
    switch (provider) {
      case "openai":
      case "mistral":
        return "/v1/chat/completions";
      case "anthropic":
        return "/v1/messages";
      case "gemini":
        // Gemini requires the model name in the URL path
        return `/v1beta/models/${model}:generateContent`;
      default:
        return "/v1/chat/completions";
    }
  };

  // Extract assistant message from provider-specific response format
  const extractAssistantMessage = (
    provider: ProviderType,
    data: ProviderResponse
  ): string => {
    try {
      switch (provider) {
        case "openai":
        case "mistral":
          // OpenAI/Mistral format: { choices: [{ message: { content: "..." } }] }
          return data.choices?.[0]?.message?.content || "";

        case "anthropic":
          // Anthropic format: { content: [{ type: "text", text: "..." }] }
          // Content is an array of content blocks, we concatenate all text blocks
          if (Array.isArray(data.content)) {
            return data.content
              .filter((block: ContentBlock) => block.type === "text")
              .map((block: ContentBlock) => block.text)
              .join("");
          }
          return "";

        case "gemini": {
          // Gemini format: { candidates: [{ content: { parts: [{ text: "..." }] } }] }
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
          // Fallback to OpenAI format
          return data.choices?.[0]?.message?.content || "";
      }
    } catch (error) {
      console.error(
        `[ERROR] Failed to extract message for ${provider}:`,
        error
      );
      return "";
    }
  };

  // Call the real /details endpoint
  const handleSubmit = async () => {
    if (!inputData.trim()) return;

    // Safety check: Prevent submitting huge inputs that could cause memory issues
    const MAX_INPUT_SIZE = 500000; // 500KB max input
    if (inputData.length > MAX_INPUT_SIZE) {
      alert(
        `Input is too large (${(inputData.length / 1024).toFixed(
          1
        )}KB). Maximum allowed is ${(MAX_INPUT_SIZE / 1024).toFixed(
          0
        )}KB. Please reduce the input size.`
      );
      return;
    }

    setIsProcessing(true);

    // Performance timing and memory logging
    const startTime = performance.now();
    console.log("[DEBUG] handleSubmit started");
    console.log(`[DEBUG] Using provider: ${activeProvider}`);

    if (
      typeof window !== "undefined" &&
      (window.performance as PerformanceWithMemory)?.memory
    ) {
      const mem = (window.performance as PerformanceWithMemory).memory;
      if (mem) {
        console.log("[DEBUG] Memory before request:", {
          usedJSHeapSize: `${(mem.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
          totalJSHeapSize: `${(mem.totalJSHeapSize / 1024 / 1024).toFixed(
            2
          )} MB`,
          jsHeapSizeLimit: `${(mem.jsHeapSizeLimit / 1024 / 1024).toFixed(
            2
          )} MB`,
        });
      }
    }

    try {
      // Get model for current provider
      const customModel =
        providersConfig.providers[activeProvider]?.model || "";
      const model = getModel(activeProvider, customModel);

      // Build provider-specific request body
      const requestBody = buildRequestBody(activeProvider, model, inputData);

      // Get provider-specific endpoint path
      const endpointPath = getProviderEndpoint(activeProvider, model);

      // Call provider-specific proxy endpoint with details parameter for PII metadata
      const goServerUrl = getGoServerAddress();
      const apiUrl = isElectron
        ? `${goServerUrl}${endpointPath}?details=true`
        : `${endpointPath}?details=true`;

      // Build provider-specific headers
      let headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      if (isElectron && apiKey) {
        headers = buildHeaders(activeProvider, apiKey);
        console.log(
          `Sending request to ${activeProvider} with API key (length: ${apiKey.length})`
        );
      } else if (isElectron && !apiKey) {
        console.warn(
          `No API key available for ${activeProvider} - request will likely fail`
        );
      }

      console.log("[DEBUG] Starting fetch request");
      const fetchStart = performance.now();

      // Call the standard OpenAI-compatible endpoint
      const response = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
      });

      console.log(
        `[DEBUG] Fetch completed in ${(performance.now() - fetchStart).toFixed(
          2
        )}ms`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log("[DEBUG] Parsing JSON response");
      const jsonStart = performance.now();
      const data = await response.json();
      console.log(
        `[DEBUG] JSON parsed in ${(performance.now() - jsonStart).toFixed(2)}ms`
      );
      console.log(
        `[DEBUG] Response size: ${JSON.stringify(data).length} bytes`
      );

      // Extract assistant message using provider-specific format
      console.log(`[DEBUG] Extracting assistant message for ${activeProvider}`);
      let assistantMessage = extractAssistantMessage(activeProvider, data);

      // Safety check: Truncate very large responses to prevent memory issues
      const MAX_RESPONSE_SIZE = 500000; // 500KB max per field
      if (assistantMessage.length > MAX_RESPONSE_SIZE) {
        console.warn(
          `[SAFETY] Assistant message truncated from ${assistantMessage.length} to ${MAX_RESPONSE_SIZE} chars`
        );
        assistantMessage =
          assistantMessage.substring(0, MAX_RESPONSE_SIZE) +
          "\n\n... [Response truncated - too large to display safely]";
      }

      // Extract PII details BEFORE setting state to prevent holding large object
      let maskedInputText = "";
      let maskedOutputText = "";
      let transformedEntities: Array<{
        type: string;
        original: string;
        token: string;
        confidence: number;
      }> = [];

      if (data.x_pii_details) {
        console.log("[DEBUG] Processing PII details");
        const piiStart = performance.now();

        // Extract values into local variables with safety limits
        maskedInputText = data.x_pii_details.masked_message || "";
        maskedOutputText = data.x_pii_details.masked_response || "";

        // Safety check: Truncate large masked text
        if (maskedInputText.length > MAX_RESPONSE_SIZE) {
          console.warn(
            `[SAFETY] Masked input truncated from ${maskedInputText.length} to ${MAX_RESPONSE_SIZE} chars`
          );
          maskedInputText =
            maskedInputText.substring(0, MAX_RESPONSE_SIZE) +
            "\n\n... [Masked input truncated - too large]";
        }
        if (maskedOutputText.length > MAX_RESPONSE_SIZE) {
          console.warn(
            `[SAFETY] Masked output truncated from ${maskedOutputText.length} to ${MAX_RESPONSE_SIZE} chars`
          );
          maskedOutputText =
            maskedOutputText.substring(0, MAX_RESPONSE_SIZE) +
            "\n\n... [Masked output truncated - too large]";
        }

        // Transform PII entities to match UI format
        const entityCount = data.x_pii_details.pii_entities?.length || 0;
        console.log(`[DEBUG] Transforming ${entityCount} PII entities`);

        if (data.x_pii_details.pii_entities) {
          // Safety check: Limit number of entities to prevent memory issues
          const MAX_ENTITIES = 500;
          const entitiesToProcess =
            entityCount > MAX_ENTITIES
              ? data.x_pii_details.pii_entities.slice(0, MAX_ENTITIES)
              : data.x_pii_details.pii_entities;

          if (entityCount > MAX_ENTITIES) {
            console.warn(
              `[SAFETY] Entity count ${entityCount} exceeds limit ${MAX_ENTITIES}, limiting entities`
            );
          }

          transformedEntities = entitiesToProcess.map(
            (entity: PiiEntityForProcessing) => ({
              type: entity.label.toLowerCase(),
              original: entity.text,
              token: entity.masked_text,
              confidence: entity.confidence,
            })
          );
        }

        console.log(
          `[DEBUG] PII details processed in ${(
            performance.now() - piiStart
          ).toFixed(2)}ms`
        );
      } else {
        console.log("[DEBUG] No PII details in response");
      }

      // Clear the large data object before setting state
      console.log("[DEBUG] Clearing response object from memory");

      // Set state with extracted values only (not the full response object)
      setFinalOutput(assistantMessage);
      setMaskedInput(maskedInputText);
      setMaskedOutput(maskedOutputText);
      setDetectedEntities(transformedEntities);

      console.log("[DEBUG] State updated, response object can be GC'd");

      console.log(
        `[DEBUG] handleSubmit completed successfully in ${(
          performance.now() - startTime
        ).toFixed(2)}ms`
      );

      if (
        typeof window !== "undefined" &&
        (window.performance as PerformanceWithMemory)?.memory
      ) {
        const mem = (window.performance as PerformanceWithMemory).memory;
        if (mem) {
          console.log("[DEBUG] Memory after processing:", {
            usedJSHeapSize: `${(mem.usedJSHeapSize / 1024 / 1024).toFixed(
              2
            )} MB`,
            totalJSHeapSize: `${(mem.totalJSHeapSize / 1024 / 1024).toFixed(
              2
            )} MB`,
            jsHeapSizeLimit: `${(mem.jsHeapSizeLimit / 1024 / 1024).toFixed(
              2
            )} MB`,
          });
        }
      }
    } catch (error) {
      console.error("[DEBUG] Error in handleSubmit:", error);
      console.error(
        "Error calling OpenAI proxy endpoint:",
        error instanceof Error ? error.message : String(error)
      );
      alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      console.log("[DEBUG] Setting isProcessing to false");
      setIsProcessing(false);
      console.log(
        `[DEBUG] Total handleSubmit time: ${(
          performance.now() - startTime
        ).toFixed(2)}ms`
      );
    }
  };

  const handleReset = () => {
    setInputData("");
    setMaskedInput("");
    setMaskedOutput("");
    setFinalOutput("");
    setDetectedEntities([]);
  };

  const handleReportMisclassification = () => {
    if (!inputData || detectedEntities.length === 0) {
      alert(
        "Please process some data first before reporting misclassification."
      );
      return;
    }

    // Set the reporting data from current state
    setReportingData({
      entities: detectedEntities,
      originalInput: inputData,
      maskedInput: maskedInput,
      source: "main",
      modelVersion: modelSignature || undefined,
    });
    setIsMisclassificationModalOpen(true);
  };

  const handleReportFromLog = (logEntry: LogEntry) => {
    // Extract information from log entry
    const message = logEntry.message || logEntry.formatted_messages || "";
    const detectedPIIRaw = logEntry.detectedPIIRaw || [];

    // Parse detectedPIIRaw array to extract individual entities with confidence
    let entities: Array<{
      type: string;
      original: string;
      token: string;
      confidence: number;
    }> = [];

    if (Array.isArray(detectedPIIRaw) && detectedPIIRaw.length > 0) {
      entities = detectedPIIRaw.map((entity: PIIEntity) => ({
        type: entity.pii_type || "unknown",
        original: entity.original_pii || "",
        token: "[Filtered]",
        confidence: entity.confidence || 0,
      }));
    } else {
      // Fallback if no raw data available
      entities = [
        {
          type: "log_entry",
          original: logEntry.detectedPII || "None",
          token: "[Filtered]",
          confidence: 0,
        },
      ];
    }

    // Set the reporting data and open modal
    setReportingData({
      entities: entities,
      originalInput: message,
      maskedInput: `Log Entry ID: ${logEntry.id}, Direction: ${logEntry.direction}`,
      source: "log",
      modelVersion: logEntry.model || modelSignature || undefined,
    });
    setIsMisclassificationModalOpen(true);
  };

  const handleSubmitMisclassification = async (comment: string) => {
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

      // Clear reporting data and close modal
      setReportingData(null);
      setIsMisclassificationModalOpen(false);
    } catch (error) {
      console.error("Error submitting misclassification:", error);
      alert("Failed to submit report. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8 pb-16">
      {/* Yaak Mascot Loading Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in">
          <div className="flex flex-col items-center gap-4">
            <img
              src={yaakMascot}
              alt="Yaak mascot"
              className="w-32 h-32 animate-bounce-slow drop-shadow-2xl"
            />
            <div className="flex items-center gap-3 bg-white/90 px-6 py-3 rounded-full shadow-lg">
              <div className="w-5 h-5 border-3 border-blue-600 border-t-transparent rounded-full animate-spin" />
              <span className="text-lg font-medium text-slate-700">
                Processing your data...
              </span>
            </div>
          </div>
        </div>
      )}
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div
          className={`sticky top-0 z-40 transition-all duration-300 -mx-4 md:-mx-8 px-4 md:px-8 py-4 mb-8 ${isScrolled
            ? "bg-white/80 backdrop-blur-md shadow-md py-2 border-b border-slate-200"
            : "bg-transparent"
            }`}
        >
          <div className="relative">
            {isElectron && (
              <div className="absolute left-0 top-1/2 -translate-y-1/2 z-50" ref={menuRef}>
                <button
                  onClick={() => setIsMenuOpen(!isMenuOpen)}
                  className="p-2 text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
                  title="Menu"
                >
                  <Menu className="w-6 h-6" />
                </button>
                {isMenuOpen && (
                  <div className="absolute left-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-slate-200 z-50">
                    <button
                      onClick={() => {
                        setIsSettingsOpen(true);
                        setIsMenuOpen(false);
                      }}
                      className="w-full text-left px-4 py-3 text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2 first:rounded-t-lg"
                    >
                      <Settings className="w-4 h-4" />
                      Settings
                    </button>
                    <button
                      onClick={() => {
                        setIsLoggingOpen(true);
                        setIsMenuOpen(false);
                      }}
                      className="w-full text-left px-4 py-3 text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2"
                    >
                      <FileText className="w-4 h-4" />
                      Logging
                    </button>
                    <button
                      onClick={() => {
                        setIsAboutOpen(true);
                        setIsMenuOpen(false);
                      }}
                      className="w-full text-left px-4 py-3 text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2 last:rounded-b-lg"
                    >
                      <Info className="w-4 h-4" />
                      About Yaak Proxy
                    </button>
                  </div>
                )}
              </div>
            )}
            <div className={`flex flex-col items-center justify-center transition-all duration-300 ${isScrolled ? "scale-90" : "scale-100"}`}>
              <div className="flex items-center justify-center gap-3">
                <img src={logoImage} alt="Yaak Logo" className={`transition-all duration-300 ${isScrolled ? "w-8 h-8" : "w-12 h-12"}`} />
                <h1 className={`font-bold text-slate-800 transition-all duration-300 ${isScrolled ? "text-2xl" : "text-4xl"}`}>
                  Yaak Privacy Proxy
                </h1>
              </div>
              {!isScrolled && (
                <p className="text-slate-600 text-lg mt-2 animate-fade-in">
                  PII Detection and Masking Proxy
                </p>
              )}
            </div>
          </div>

          {/* Model Health Banner */}
          {serverHealth.status === "online" && !serverHealth.modelHealthy && (
            <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-lg inline-block max-w-2xl">
              <p className="text-sm text-red-900 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span className="font-semibold">Model is unhealthy</span>
              </p>
              {serverHealth.modelError && (
                <p className="text-xs text-red-700 mt-2 break-all">
                  {serverHealth.modelError}
                </p>
              )}
              <p className="text-xs text-red-700 mt-2">
                Please check model configuration in{" "}
                {isElectron ? (
                  <button
                    onClick={() => setIsSettingsOpen(true)}
                    className="underline font-semibold"
                  >
                    Settings
                  </button>
                ) : (
                  "Settings"
                )}
              </p>
            </div>
          )}

          {isElectron && !apiKey && (
            <div className="mt-4 p-2 bg-amber-50 border border-amber-200 rounded-lg inline-block">
              <p className="text-xs text-amber-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                <span>
                  {PROVIDER_NAMES[activeProvider]} API key not configured.{" "}
                </span>
                <button
                  onClick={() => setIsSettingsOpen(true)}
                  className="underline font-semibold"
                >
                  Configure in Settings
                </button>
              </p>
            </div>
          )}
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            {isElectron && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-600">
                  Type your request to:
                </label>
                <select
                  value={activeProvider}
                  onChange={async (e) => {
                    const newProvider = e.target.value as ProviderType;
                    setActiveProvider(newProvider);
                    if (window.electronAPI) {
                      await window.electronAPI.setActiveProvider(newProvider);
                      // Load API key for new provider
                      const key = await window.electronAPI.getProviderApiKey(
                        newProvider
                      );
                      setApiKey(key);
                    }
                  }}
                  className="px-3 py-2 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none text-sm bg-white"
                >
                  {(
                    [
                      "openai",
                      "anthropic",
                      "gemini",
                      "mistral",
                    ] as ProviderType[]
                  ).map((provider) => (
                    <option key={provider} value={provider}>
                      {PROVIDER_NAMES[provider]}
                      {providersConfig.providers[provider]?.hasApiKey
                        ? " ✓"
                        : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
          <textarea
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter your message with sensitive information...&#10;&#10;Example: Hi, my name is John Smith and my email is john.smith@email.com. My phone is 555-123-4567.&#10;&#10;This will be processed through the real PII detection and masking pipeline."
            className={`w-full h-32 p-4 border-2 rounded-lg focus:outline-none resize-none font-mono text-sm placeholder:text-gray-400 ${
              serverStatus === "offline"
                ? "border-red-200 bg-red-50 cursor-not-allowed opacity-60"
                : "border-slate-200 focus:border-blue-500"
            }`}
            disabled={serverStatus === "offline"}
          />
          <div className="flex gap-3 mt-4 items-center">
            <button
              onClick={handleSubmit}
              disabled={
                !inputData.trim() || isProcessing || serverStatus === "offline"
              }
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
              title={
                serverStatus === "offline" ? "Backend server is offline" : ""
              }
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  Process Data
                </>
              )}
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors font-medium"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Diff View */}
        {maskedInput && (
          <div className="space-y-6">
            {/* Input Diff */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600" />
                Input Transformation (A → A')
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Original (A)</span>
                    <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded">
                      PII Exposed
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{
                      __html: highlightedInputOriginalHTML,
                    }}
                  />
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Masked (A')</span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">
                      PII Protected
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{
                      __html: highlightedInputMaskedHTML,
                    }}
                  />
                </div>
              </div>
              <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <p className="text-sm text-amber-900">
                  <span className="font-semibold">Changes:</span>{" "}
                  {detectedEntities.length} PII entities detected and replaced
                  with tokens
                </p>
              </div>
            </div>

            {/* Output Diff */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-blue-600" />
                Output Transformation (B' → B)
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Masked Output (B')</span>
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">
                      From {PROVIDER_NAMES[activeProvider]}
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{
                      __html: highlightedOutputMaskedHTML,
                    }}
                  />
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Final Output (B)</span>
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                      Restored
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap"
                    dangerouslySetInnerHTML={{
                      __html: highlightedOutputFinalHTML,
                    }}
                  />
                </div>
              </div>
              <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-900">
                  <span className="font-semibold">Changes:</span>{" "}
                  {detectedEntities.length} tokens replaced with original PII
                  values
                </p>
              </div>
            </div>

            {/* Transformation Summary */}
            <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-4">
                Transformation Summary
              </h3>
              <div className="grid md:grid-cols-3 gap-4">
                <div className="bg-white rounded-lg p-4 border-l-4 border-amber-500">
                  <div className="text-2xl font-bold text-slate-800">
                    {detectedEntities.length}
                  </div>
                  <div className="text-sm text-slate-600">
                    Entities Detected
                  </div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-green-500">
                  <div className="text-2xl font-bold text-slate-800">100%</div>
                  <div className="text-sm text-slate-600">PII Protected</div>
                </div>
                <div className="bg-white rounded-lg p-4 border-l-4 border-blue-500">
                  <div className="text-2xl font-bold text-slate-800">
                    {detectedEntities.length > 0
                      ? (
                          (detectedEntities.reduce(
                            (sum, e) => sum + (e.confidence || 0),
                            0
                          ) /
                            detectedEntities.length) *
                          100
                        ).toFixed(1)
                      : 0}
                    %
                  </div>
                  <div className="text-sm text-slate-600">Avg. Confidence</div>
                </div>
              </div>

              {/* Report Misclassification Button */}
              <div className="mt-6 flex justify-center">
                <button
                  onClick={handleReportMisclassification}
                  className="flex items-center gap-2 px-6 py-3 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors shadow-md hover:shadow-lg"
                  title="Report incorrect PII classification"
                >
                  <Flag className="w-5 h-5" />
                  Report Misclassification
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Info Footer */}
        <div className="mt-8 text-center text-sm text-slate-500">
          <p>
            Yaak Privacy Proxy - Made by Dataiku's Open Source Lab
            {version && (
              <span className="ml-2 text-xs text-slate-400">v{version}</span>
            )}
          </p>
        </div>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 bg-slate-800 text-slate-200 px-4 py-2 flex items-center justify-between border-t border-slate-700">
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full ${
              serverStatus === "online" ? "bg-green-500" : "bg-red-500"
            } ${serverStatus === "online" ? "animate-pulse" : ""}`}
            title={
              serverStatus === "online" ? "Server online" : "Server offline"
            }
          />
          <span className="text-sm">
            {serverStatus === "online" ? (
              "Server online"
            ) : (
              <span className="flex items-center gap-2">
                Server offline - Please ensure the Go backend server is running
                at localhost:8080
              </span>
            )}
          </span>
        </div>
        {modelSignature && (
          <div className="relative">
            <div
              className="flex items-center gap-2 cursor-help"
              role="status"
              aria-label="Model signature"
              onMouseEnter={() => setShowModelTooltip(true)}
              onMouseLeave={() => setShowModelTooltip(false)}
            >
              <span className="text-xs text-slate-400">Model:</span>
              <code
                className="text-xs font-mono text-slate-300 bg-slate-700/50 px-1 rounded"
                aria-label={`Model signature ${modelSignature}`}
              >
                {modelSignature}
              </code>
            </div>
            {showModelTooltip && (
              <div className="absolute bottom-full right-0 mb-2 px-2 py-1 text-xs text-white bg-gray-900 border border-gray-700 rounded shadow-lg whitespace-nowrap z-50">
                Verified model signature
                <div className="absolute top-full right-2 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => {
          setIsSettingsOpen(false);
          loadSettings(); // Reload settings after closing
        }}
      />

      {/* Logging Modal */}
      <LoggingModal
        isOpen={isLoggingOpen}
        onClose={() => setIsLoggingOpen(false)}
        onReportMisclassification={handleReportFromLog}
      />

      {/* About Modal */}
      <AboutModal isOpen={isAboutOpen} onClose={() => setIsAboutOpen(false)} />

      {/* Misclassification Modal */}
      <MisclassificationModal
        isOpen={isMisclassificationModalOpen}
        onClose={() => {
          setIsMisclassificationModalOpen(false);
          setReportingData(null);
        }}
        onSubmit={handleSubmitMisclassification}
        entities={reportingData?.entities || []}
        originalInput={reportingData?.originalInput || ""}
        maskedInput={reportingData?.maskedInput || ""}
        source={reportingData?.source || "main"}
      />

      {/* Terms Modal */}
      <TermsModal
        isOpen={isTermsOpen}
        onClose={() => {
          setIsTermsOpen(false);
          setTermsRequireAcceptance(false);
        }}
        requireAcceptance={termsRequireAcceptance}
      />

      {/* Welcome Modal */}
      <WelcomeModal
        isOpen={isWelcomeOpen}
        onClose={() => setIsWelcomeOpen(false)}
      />
    </div>
  );
}
