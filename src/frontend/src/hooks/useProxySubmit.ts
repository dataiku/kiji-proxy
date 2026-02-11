import { useState, useEffect, useMemo, useCallback } from "react";
import type {
  ProviderType,
  DetectedEntity,
  PiiEntityForProcessing,
} from "../types/provider";
import {
  getGoServerAddress,
  getModel,
  buildRequestBody,
  buildHeaders,
  getProviderEndpoint,
  extractAssistantMessage,
} from "../utils/providerHelpers";
import {
  highlightTextByCharacter,
  highlightEntitiesByToken,
  highlightEntitiesByOriginal,
} from "../utils/textHighlight";

const MAX_HIGHLIGHT_SIZE = 50000; // 50KB max for highlighting

function truncateForHighlighting(text: string): string {
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
}

interface UseProxySubmitOptions {
  activeProvider: ProviderType;
  providersConfig: {
    providers: Record<ProviderType, { hasApiKey: boolean; model: string }>;
  };
  apiKey: string | null;
  isElectron: boolean;
  isTourActive: () => boolean;
  cancelTour: () => void;
}

export function useProxySubmit({
  activeProvider,
  providersConfig,
  apiKey,
  isElectron,
  isTourActive,
  cancelTour,
}: UseProxySubmitOptions) {
  const [inputData, setInputData] = useState("");
  const [maskedInput, setMaskedInput] = useState("");
  const [maskedOutput, setMaskedOutput] = useState("");
  const [finalOutput, setFinalOutput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedEntities, setDetectedEntities] = useState<DetectedEntity[]>(
    []
  );

  // Cancel tour if processing starts
  useEffect(() => {
    if (isProcessing && isTourActive()) {
      cancelTour();
    }
  }, [isProcessing, isTourActive, cancelTour]);

  const averageConfidence = useMemo(() => {
    if (detectedEntities.length === 0) return 0;
    const sum = detectedEntities.reduce(
      (acc, entity) => acc + (entity.confidence || 0),
      0
    );
    return sum / detectedEntities.length;
  }, [detectedEntities]);

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

  const handleSubmit = useCallback(async () => {
    if (!inputData.trim()) return;

    const MAX_INPUT_SIZE = 500000;
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

    try {
      const customModel =
        providersConfig.providers[activeProvider]?.model || "";
      const model = getModel(activeProvider, customModel);

      const requestBody = buildRequestBody(activeProvider, model, inputData);
      const endpointPath = getProviderEndpoint(activeProvider, model);

      const goServerUrl = getGoServerAddress(isElectron);
      const apiUrl = isElectron
        ? `${goServerUrl}${endpointPath}?details=true`
        : `${endpointPath}?details=true`;

      let headers: Record<string, string> = {
        "Content-Type": "application/json",
      };

      if (isElectron && apiKey) {
        headers = buildHeaders(activeProvider, apiKey);
      }

      const response = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      let assistantMessage = extractAssistantMessage(activeProvider, data);

      const MAX_RESPONSE_SIZE = 500000;
      if (assistantMessage.length > MAX_RESPONSE_SIZE) {
        console.warn(
          `[SAFETY] Assistant message truncated from ${assistantMessage.length} to ${MAX_RESPONSE_SIZE} chars`
        );
        assistantMessage =
          assistantMessage.substring(0, MAX_RESPONSE_SIZE) +
          "\n\n... [Response truncated - too large to display safely]";
      }

      let maskedInputText = "";
      let maskedOutputText = "";
      let transformedEntities: DetectedEntity[] = [];

      if (data.x_pii_details) {
        maskedInputText = data.x_pii_details.masked_message || "";
        maskedOutputText = data.x_pii_details.masked_response || "";

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

        if (data.x_pii_details.pii_entities) {
          const entityCount = data.x_pii_details.pii_entities.length;
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
      }

      setFinalOutput(assistantMessage);
      setMaskedInput(maskedInputText);
      setMaskedOutput(maskedOutputText);
      setDetectedEntities(transformedEntities);
    } catch (error) {
      console.error(
        "Error calling proxy endpoint:",
        error instanceof Error ? error.message : String(error)
      );
      alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsProcessing(false);
    }
  }, [inputData, activeProvider, providersConfig, apiKey, isElectron]);

  const handleReset = useCallback(() => {
    setInputData("");
    setMaskedInput("");
    setMaskedOutput("");
    setFinalOutput("");
    setDetectedEntities([]);
  }, []);

  return {
    inputData,
    setInputData,
    maskedInput,
    maskedOutput,
    finalOutput,
    isProcessing,
    detectedEntities,
    averageConfidence,
    highlightedInputOriginalHTML,
    highlightedInputMaskedHTML,
    highlightedOutputMaskedHTML,
    highlightedOutputFinalHTML,
    handleSubmit,
    handleReset,
  };
}
