import { useEffect, useState } from "react";
import { apiUrl, isElectron } from "../utils/providerHelpers";

/**
 * Resolves whether the current user is an admin, using the same signal the app
 * shell uses for launch routing:
 *
 *   - Desktop: the role chosen during onboarding, read over IPC.
 *   - Web: a configured HTTP Basic Auth credential — whoever set the credentials
 *     and can load the gated UI is the admin. /api/auth/status exposes only this
 *     boolean, never the secrets.
 *
 * Returns `null` while the async check is in flight so callers can keep
 * admin-only UI hidden until the role is known (avoids flashing it to
 * non-admins on a web deployment).
 */
export function useIsAdmin(): boolean | null {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const resolve = async () => {
      let result = false;
      if (isElectron && window.electronAPI) {
        try {
          result = await window.electronAPI.getAdmin();
        } catch (error) {
          console.error("Failed to read admin preference:", error);
        }
      } else {
        try {
          const res = await fetch(apiUrl("/api/auth/status", isElectron));
          if (res.ok) {
            const data = await res.json();
            result = data.basicAuthActive === true;
          }
        } catch (error) {
          console.error("Failed to read auth status:", error);
        }
      }
      if (!cancelled) setIsAdmin(result);
    };
    resolve();
    return () => {
      cancelled = true;
    };
  }, []);

  return isAdmin;
}
