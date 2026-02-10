import { useState, useEffect, useRef } from "react";
import {
  Send,
  AlertCircle,
  Settings,
  FileText,
  Info,
  Menu,
  Flag,
  HelpCircle,
} from "lucide-react";
import logoImage from "../../assets/logo.png";
import kijiMascot from "../../assets/yaak.png";
import SettingsModal from "./modals/SettingsModal";
import AdvancedSettingsModal from "./modals/AdvancedSettingsModal";
import LoggingModal from "./modals/LoggingModal";
import AboutModal from "./modals/AboutModal";
import MisclassificationModal from "./modals/MisclassificationModal";
import TermsModal from "./modals/TermsModal";
import WelcomeModal from "./modals/WelcomeModal";
import { useTour } from "../tour/useTour";
import { useServerHealth } from "../hooks/useServerHealth";
import { useElectronSettings } from "../hooks/useElectronSettings";
import { useMisclassificationReport } from "../hooks/useMisclassificationReport";
import { useProxySubmit } from "../hooks/useProxySubmit";
import { getConfidenceColor } from "../utils/providerHelpers";
import type { ProviderType, LogEntry } from "../types/provider";
import { PROVIDER_NAMES } from "../types/provider";

// Detect if running in Electron
const isElectron =
  typeof window !== "undefined" && window.electronAPI !== undefined;

export default function PrivacyProxyUI() {
  // UI toggle state (simple enough to stay in the component)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isAdvancedSettingsOpen, setIsAdvancedSettingsOpen] = useState(false);
  const [isLoggingOpen, setIsLoggingOpen] = useState(false);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [showModelTooltip, setShowModelTooltip] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Settings & provider state
  const settings = useElectronSettings(isElectron, {
    onSettingsOpen: () => setIsSettingsOpen(true),
    onAboutOpen: () => setIsAboutOpen(true),
    onTermsOpen: () => {},
    onTourStart: () => startTour(),
  });

  // Product tour
  const { startTour, isTourActive, cancelTour } = useTour(
    settings.welcomeModalJustClosed,
    !settings.termsRequireAcceptance
  );

  // Server health polling
  const { serverStatus, serverHealth, modelSignature, version } =
    useServerHealth(isElectron);

  // Proxy submit & input/output state
  const {
    inputData,
    setInputData,
    maskedInput,
    isProcessing,
    detectedEntities,
    averageConfidence,
    highlightedInputOriginalHTML,
    highlightedInputMaskedHTML,
    highlightedOutputMaskedHTML,
    highlightedOutputFinalHTML,
    handleSubmit,
    handleReset,
  } = useProxySubmit({
    activeProvider: settings.activeProvider,
    providersConfig: settings.providersConfig,
    apiKey: settings.apiKey,
    isElectron,
    isTourActive,
    cancelTour,
  });

  // Misclassification reporting
  const misclassification = useMisclassificationReport();

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

  // Track scroll position for sticky header
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8 pb-16">
      {/* Yaak Mascot Loading Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in">
          <div className="flex flex-col items-center gap-4">
            <img
              src={kijiMascot}
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
          className={`sticky top-0 z-40 transition-all duration-300 -mx-4 md:-mx-8 px-4 md:px-8 py-4 mb-8 ${
            isScrolled
              ? "bg-white/80 backdrop-blur-md shadow-md py-2 border-b border-slate-200"
              : "bg-transparent"
          }`}
        >
          <div className="relative">
            {isElectron && (
              <div
                className="absolute left-0 top-1/2 -translate-y-1/2 z-50"
                ref={menuRef}
              >
                <button
                  id="tour-menu-button"
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
                      className="w-full text-left px-4 py-3 text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2"
                    >
                      <Info className="w-4 h-4" />
                      About Kiji Privacy Proxy
                    </button>
                    <button
                      onClick={() => {
                        startTour();
                        setIsMenuOpen(false);
                      }}
                      className="w-full text-left px-4 py-3 text-slate-700 hover:bg-slate-50 transition-colors flex items-center gap-2 last:rounded-b-lg"
                    >
                      <HelpCircle className="w-4 h-4" />
                      Start Tour
                    </button>
                  </div>
                )}
              </div>
            )}
            <div
              id="tour-header"
              className={`flex flex-col items-center justify-center transition-all duration-300 ${
                isScrolled ? "scale-90" : "scale-100"
              }`}
            >
              <div className="flex items-center justify-center gap-3">
                <img
                  src={logoImage}
                  alt="Yaak Logo"
                  className={`transition-all duration-300 ${
                    isScrolled ? "w-8 h-8" : "w-12 h-12"
                  }`}
                />
                <h1
                  className={`font-bold text-slate-800 transition-all duration-300 ${
                    isScrolled ? "text-2xl" : "text-4xl"
                  }`}
                >
                  Kiji Privacy Proxy
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

          {isElectron && !settings.apiKey && (
            <div className="mt-4 p-2 bg-amber-50 border border-amber-200 rounded-lg inline-block">
              <p className="text-xs text-amber-800 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                <span>
                  {PROVIDER_NAMES[settings.activeProvider]} API key not
                  configured.{" "}
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
        <div
          id="tour-input-section"
          className="bg-white rounded-xl shadow-lg p-6 mb-6"
        >
          <div className="flex items-center justify-between mb-4">
            {isElectron && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium text-slate-600">
                  Type your request to:
                </label>
                <select
                  id="tour-provider-selector"
                  value={settings.activeProvider}
                  onChange={(e) =>
                    settings.switchProvider(e.target.value as ProviderType)
                  }
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
                      {settings.providersConfig.providers[provider]?.hasApiKey
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
              id="tour-process-button"
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
            {/* Combined Input and Output Diff */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              {/* Input Diff */}
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-600" />
                {isElectron
                  ? `What was sent to ${
                      PROVIDER_NAMES[settings.activeProvider]
                    }`
                  : "What was sent to the LLM: "}
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="flex flex-col">
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Request submitted</span>
                    <span className="px-2 py-0.5 bg-red-100 text-red-700 text-xs rounded">
                      PII Exposed
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap flex-1"
                    dangerouslySetInnerHTML={{
                      __html: highlightedInputOriginalHTML,
                    }}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <p className="text-sm font-semibold text-slate-700">
                      {detectedEntities.length} PII detected
                    </p>
                    <p
                      className="text-sm font-semibold"
                      style={{
                        color: getConfidenceColor(averageConfidence),
                      }}
                    >
                      {(averageConfidence * 100).toFixed(1)}% avg confidence
                    </p>
                  </div>
                </div>
                <div className="flex flex-col">
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>
                      Request submitted with personal information removed
                    </span>
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded">
                      PII Protected
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap flex-1"
                    dangerouslySetInnerHTML={{
                      __html: highlightedInputMaskedHTML,
                    }}
                  />
                  <div className="mt-2">
                    <p className="text-sm font-semibold text-green-600">
                      {detectedEntities.length} PII replaced
                    </p>
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className="my-6 border-t-2 border-slate-200"></div>

              {/* Output Diff */}
              <h2 className="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-blue-600" />
                {isElectron
                  ? `What ${PROVIDER_NAMES[settings.activeProvider]} returned`
                  : "What the LLM returned"}
              </h2>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="flex flex-col">
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>Request received</span>
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">
                      From {PROVIDER_NAMES[settings.activeProvider]}
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap flex-1"
                    dangerouslySetInnerHTML={{
                      __html: highlightedOutputMaskedHTML,
                    }}
                  />
                  <div className="mt-2">
                    <p className="text-sm font-semibold text-slate-700">
                      {detectedEntities.length} fake PIIs received
                    </p>
                  </div>
                </div>
                <div className="flex flex-col">
                  <div className="text-sm font-medium text-slate-600 mb-2 flex items-center gap-2">
                    <span>
                      Request received with personal information restored
                    </span>
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                      Restored
                    </span>
                  </div>
                  <div
                    className="bg-slate-50 rounded-lg p-4 font-mono text-sm border-2 border-slate-200 whitespace-pre-wrap flex-1"
                    dangerouslySetInnerHTML={{
                      __html: highlightedOutputFinalHTML,
                    }}
                  />
                  <div className="mt-2">
                    <p className="text-sm font-semibold text-green-600">
                      {detectedEntities.length} PII restored
                    </p>
                  </div>
                </div>
              </div>

              {/* Report Misclassification Button */}
              <div className="mt-6 flex justify-end">
                <button
                  onClick={() =>
                    misclassification.handleReportMisclassification(
                      inputData,
                      maskedInput,
                      detectedEntities,
                      modelSignature
                    )
                  }
                  className="flex items-center gap-2 px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors font-medium"
                  title="Report incorrect PII classification"
                >
                  <Flag className="w-4 h-4" />
                  Report Misclassification
                </button>
              </div>
            </div>

            {/* Transformation Summary */}
            {false && (
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
                    <div className="text-2xl font-bold text-slate-800">
                      100%
                    </div>
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
                    <div className="text-sm text-slate-600">
                      Avg. Confidence
                    </div>
                  </div>
                </div>

                {/* Report Misclassification Button */}
                <div className="mt-6 flex justify-center">
                  <button
                    onClick={() =>
                      misclassification.handleReportMisclassification(
                        inputData,
                        maskedInput,
                        detectedEntities,
                        modelSignature
                      )
                    }
                    className="flex items-center gap-2 px-6 py-3 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors shadow-md hover:shadow-lg"
                    title="Report incorrect PII classification"
                  >
                    <Flag className="w-5 h-5" />
                    Report Misclassification
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Info Footer */}
        <div className="mt-8 text-center text-sm text-slate-500">
          <p>
            Kiji Privacy Proxy - Made by 575 Lab - Dataiku's Open Source Office
            {version && (
              <span className="ml-2 text-xs text-slate-400">v{version}</span>
            )}
          </p>
        </div>
      </div>

      {/* Status Bar */}
      <div
        id="tour-status-bar"
        className="fixed bottom-0 left-0 right-0 bg-slate-800 text-slate-200 px-4 py-2 flex items-center justify-between border-t border-slate-700"
      >
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
          settings.loadSettings();
        }}
        onOpenAdvancedSettings={() => setIsAdvancedSettingsOpen(true)}
      />

      {/* Advanced Settings Modal */}
      <AdvancedSettingsModal
        isOpen={isAdvancedSettingsOpen}
        onClose={() => setIsAdvancedSettingsOpen(false)}
      />

      {/* Logging Modal */}
      <LoggingModal
        isOpen={isLoggingOpen}
        onClose={() => setIsLoggingOpen(false)}
        onReportMisclassification={(logEntry: LogEntry) =>
          misclassification.handleReportFromLog(logEntry, modelSignature)
        }
      />

      {/* About Modal */}
      <AboutModal isOpen={isAboutOpen} onClose={() => setIsAboutOpen(false)} />

      {/* Misclassification Modal */}
      <MisclassificationModal
        isOpen={misclassification.isMisclassificationModalOpen}
        onClose={misclassification.closeModal}
        onSubmit={misclassification.handleSubmitMisclassification}
        entities={misclassification.reportingData?.entities || []}
        originalInput={misclassification.reportingData?.originalInput || ""}
        maskedInput={misclassification.reportingData?.maskedInput || ""}
        source={misclassification.reportingData?.source || "main"}
      />

      {/* Terms Modal */}
      <TermsModal
        isOpen={settings.isTermsOpen}
        onClose={settings.closeTerms}
        requireAcceptance={settings.termsRequireAcceptance}
      />

      {/* Welcome Modal */}
      <WelcomeModal
        isOpen={settings.isWelcomeOpen}
        onClose={settings.closeWelcome}
      />
    </div>
  );
}
