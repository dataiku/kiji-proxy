import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";

// Namespaced English base + French draft translations.
// French files start as machine-drafted strings pending native review.
import enCommon from "./locales/en/common.json";
import enSettings from "./locales/en/settings.json";
import enDashboard from "./locales/en/dashboard.json";
import enOnboarding from "./locales/en/onboarding.json";
import enActivity from "./locales/en/activity.json";
import enMappings from "./locales/en/mappings.json";
import enAbout from "./locales/en/about.json";

import frCommon from "./locales/fr/common.json";
import frSettings from "./locales/fr/settings.json";
import frDashboard from "./locales/fr/dashboard.json";
import frOnboarding from "./locales/fr/onboarding.json";
import frActivity from "./locales/fr/activity.json";
import frMappings from "./locales/fr/mappings.json";
import frAbout from "./locales/fr/about.json";

export const SUPPORTED_LANGUAGES = ["en", "fr"] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const NAMESPACES = [
  "common",
  "settings",
  "dashboard",
  "onboarding",
  "activity",
  "mappings",
  "about",
] as const;

export const DEFAULT_NAMESPACE = "common";

const resources = {
  en: {
    common: enCommon,
    settings: enSettings,
    dashboard: enDashboard,
    onboarding: enOnboarding,
    activity: enActivity,
    mappings: enMappings,
    about: enAbout,
  },
  fr: {
    common: frCommon,
    settings: frSettings,
    dashboard: frDashboard,
    onboarding: frOnboarding,
    activity: frActivity,
    mappings: frMappings,
    about: frAbout,
  },
} as const;

i18next
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    // Detect from navigator.language (and persisted choice, once #577 wires
    // localStorage), falling back to English. Region subtags like "fr-FR"
    // resolve to the "fr" base via load: "languageOnly".
    fallbackLng: "en",
    supportedLngs: SUPPORTED_LANGUAGES as unknown as string[],
    load: "languageOnly",
    ns: NAMESPACES as unknown as string[],
    defaultNS: DEFAULT_NAMESPACE,
    interpolation: {
      // React already escapes values, so i18next must not double-escape.
      escapeValue: false,
    },
    detection: {
      order: ["localStorage", "navigator"],
      caches: ["localStorage"],
    },
    returnNull: false,
    // Resources are bundled and initialized synchronously, so no async load
    // gate is needed. Disabling Suspense also keeps the class-based
    // ErrorBoundary (which renders via withTranslation) safe.
    react: {
      useSuspense: false,
    },
  });

export default i18next;
