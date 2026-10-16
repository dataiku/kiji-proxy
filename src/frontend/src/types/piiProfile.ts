// PII detection profiles, as served by /api/pii/profiles. A request picks a
// profile by `id` (X-Kiji-Profile header); `name` is display-only.

export interface PIIProfileSummary {
  id: string;
  name: string;
  is_default: boolean;
}

export interface PIIRegexPattern {
  name: string;
  pattern: string;
}

export interface PIIProfile extends PIIProfileSummary {
  confidence: number;
  /** Entity labels left unmasked; empty means "mask everything". */
  disabled_entities: string[];
  custom_regexes: PIIRegexPattern[];
}

/** localStorage key remembering the Playground's selected profile id. */
export const PLAYGROUND_PROFILE_STORAGE_KEY = "kiji.piiProfile";
