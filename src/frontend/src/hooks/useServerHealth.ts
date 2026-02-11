import { useState, useEffect } from "react";
import { apiUrl } from "../utils/providerHelpers";

interface ServerHealth {
  status: "online" | "offline";
  modelHealthy: boolean;
  modelError?: string;
}

export function useServerHealth(isElectron: boolean) {
  const [serverStatus, setServerStatus] = useState<"online" | "offline">(
    "offline"
  );
  const [serverHealth, setServerHealth] = useState<ServerHealth>({
    status: "offline",
    modelHealthy: false,
  });
  const [modelSignature, setModelSignature] = useState<string | null>(null);
  const [version, setVersion] = useState<string | null>(null);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);

        const response = await fetch(apiUrl("/health", isElectron), {
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
        const response = await fetch(apiUrl("/api/model/security", isElectron));
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
        const response = await fetch(apiUrl("/version", isElectron));
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

    checkServerStatus();
    loadModelSignature();
    loadVersion();

    const interval = setInterval(checkServerStatus, 5000);

    return () => clearInterval(interval);
  }, [isElectron]);

  return { serverStatus, serverHealth, modelSignature, version };
}
