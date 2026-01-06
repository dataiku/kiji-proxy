import { useState, useEffect, useRef } from "react";
import {
  Eye,
  Send,
  AlertCircle,
  Settings,
  Menu,
  FileText,
  Info,
  WifiOff,
  CheckCircle,
  Flag,
} from "lucide-react";
import SettingsModal from "./modals/SettingsModal";
import LoggingModal from "./modals/LoggingModal";
import AboutModal from "./modals/AboutModal";
import MisclassificationModal from "./modals/MisclassificationModal";
import logoImage from "../../assets/logo.png";
import {
  highlightTextByCharacter,
  highlightEntitiesByToken,
  highlightEntitiesByOriginal,
} from "../utils/textHighlight";
import { reportMisclassification } from "../utils/misclassificationReporter";

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
  const [_, setForwardEndpoint] = useState("https://api.openai.com/v1");
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<"online" | "offline">(
    "offline"
  );
  const [modelSignature, setModelSignature] = useState<string | null>(null);
  const [showModelTooltip, setShowModelTooltip] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [version, setVersion] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Fixed Go server address - always call the Go server at this address
  const GO_SERVER_ADDRESS = "http://localhost:8080";

  // Detect if running in Electron
  const isElectron =
    typeof window !== "undefined" && window.electronAPI !== undefined;

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

  // Load settings on mount and listen for menu command
  useEffect(() => {
    if (isElectron && window.electronAPI) {
      loadSettings();

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

      // Cleanup
      return () => {
        if (window.electronAPI?.removeSettingsListener) {
          window.electronAPI.removeSettingsListener();
        }
        if (window.electronAPI?.removeAboutListener) {
          window.electronAPI.removeAboutListener();
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
        setServerStatus(response.ok ? "online" : "offline");
      } catch (_error) {
        setServerStatus("offline");
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
      const [url, key] = await Promise.all([
        window.electronAPI.getForwardEndpoint(),
        window.electronAPI.getApiKey(),
      ]);
      setForwardEndpoint(url);
      setApiKey(key);
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

  // Call the real /details endpoint
  const handleSubmit = async () => {
    if (!inputData.trim()) return;

    setIsProcessing(true);

    try {
      // Create OpenAI chat completion request format (standard format)
      const requestBody = {
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "user",
            content: inputData,
          },
        ],
        max_tokens: 1000,
      };

      // Call standard OpenAI proxy endpoint with details parameter for PII metadata
      const goServerUrl = getGoServerAddress();
      const apiUrl = isElectron
        ? `${goServerUrl}/v1/chat/completions?details=true` // Standard proxy with PII details
        : "/v1/chat/completions?details=true"; // Proxied call in web mode

      // Prepare headers with standard Authorization format
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      // Add API key as Bearer token (standard OpenAI format)
      if (isElectron && apiKey) {
        headers["Authorization"] = `Bearer ${apiKey}`;
      }

      // Call the standard OpenAI-compatible endpoint
      const response = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Extract standard OpenAI response
      const assistantMessage = data.choices?.[0]?.message?.content || "";
      setFinalOutput(assistantMessage);

      // Extract PII details from custom field (if available)
      if (data.x_pii_details) {
        // Use masked_message for display (just the content, not full JSON)
        setMaskedInput(data.x_pii_details.masked_message || "");
        setMaskedOutput(data.x_pii_details.masked_response || "");

        // Transform PII entities to match UI format
        const transformedEntities = (data.x_pii_details.pii_entities || []).map(
          (entity: {
            label: string;
            text: string;
            masked_text: string;
            confidence: number;
          }) => ({
            type: entity.label.toLowerCase(),
            original: entity.text,
            token: entity.masked_text,
            confidence: entity.confidence,
          })
        );
        setDetectedEntities(transformedEntities);
      } else {
        // No PII details available (shouldn't happen with ?details=true)
        setMaskedInput("");
        setMaskedOutput("");
        setDetectedEntities([]);
      }
    } catch (error) {
      console.error(
        "Error calling OpenAI proxy endpoint:",
        error instanceof Error ? error.message : String(error)
      );
      alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsProcessing(false);
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
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4 relative">
            {isElectron && (
              <div className="absolute left-0" ref={menuRef}>
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
            <img src={logoImage} alt="Yaak Logo" className="w-12 h-12" />
            <h1 className="text-4xl font-bold text-slate-800">
              Yaak - Privacy Proxy
            </h1>
          </div>
          <p className="text-slate-600 text-lg">Privacy Proxy</p>

          {/* Server Status Banner */}
          {serverStatus === "offline" ? (
            <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-lg inline-block">
              <p className="text-sm text-red-900 flex items-center gap-2">
                <WifiOff className="w-5 h-5" />
                <span className="font-semibold">
                  Backend server is offline.
                </span>
              </p>
              <p className="text-xs text-red-700 mt-2">
                Please ensure the Go backend server is running at localhost:8080
              </p>
            </div>
          ) : (
            <div className="mt-4 p-2 bg-green-50 border border-green-200 rounded-lg inline-block">
              <p className="text-xs text-green-800 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                <span>Backend server is online</span>
              </p>
            </div>
          )}

          {isElectron && !apiKey && (
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg inline-block">
              <p className="text-sm text-amber-900 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                <span>OpenAI API key not configured. </span>
                <button
                  onClick={() => setIsSettingsOpen(true)}
                  className="underline font-semibold"
                >
                  Configure it in Settings
                </button>
              </p>
            </div>
          )}
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-slate-800 flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Input Data (A)
            </h2>
            <span className="text-sm text-slate-500">
              Original data with PII
            </span>
          </div>
          <textarea
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter your message with sensitive information...&#10;&#10;Example: Hi, my name is John Smith and my email is john.smith@email.com. My phone is 555-123-4567.&#10;&#10;This will be processed through the real PII detection and masking pipeline."
            className="w-full h-32 p-4 border-2 border-slate-200 rounded-lg focus:border-blue-500 focus:outline-none resize-none font-mono text-sm placeholder:text-gray-400"
          />
          <div className="flex gap-3 mt-4">
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
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {highlightTextByCharacter(
                      inputData,
                      detectedEntities,
                      "bg-red-200 text-red-900"
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Masked (A')</span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">
                      PII Protected
                    </span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {highlightEntitiesByToken(
                      maskedInput,
                      detectedEntities,
                      "bg-green-200 text-green-900 font-bold"
                    )}
                  </div>
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
                      From OpenAI
                    </span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {highlightEntitiesByToken(
                      maskedOutput,
                      detectedEntities,
                      "bg-purple-200 text-purple-900 font-bold"
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Final Output (B)</span>
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                      Restored
                    </span>
                  </div>
                  <div className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap">
                    {highlightEntitiesByOriginal(
                      finalOutput,
                      detectedEntities,
                      "bg-blue-200 text-blue-900 font-bold"
                    )}
                  </div>
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
            Yaak - Privacy Proxy - Diff View
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
            {serverStatus === "online" ? "Server online" : "Server offline"}
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
    </div>
  );
}
