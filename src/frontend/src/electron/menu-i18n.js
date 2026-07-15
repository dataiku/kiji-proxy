// Main-process menu translations.
//
// The renderer localizes with react-i18next, but the native application and
// tray menus are built in the main process, which has no access to that
// runtime. This module is a tiny, self-contained string table + selector so
// the menus can be rebuilt in the user's language. Keep the keys here in sync
// with what createMenu()/updateTrayMenu() reference in electron-main.js.

const SUPPORTED_LANGUAGES = ["en", "fr"];
const DEFAULT_LANGUAGE = "en";

const STRINGS = {
  en: {
    // File
    file: "File",
    quit: "Quit",
    // Edit
    edit: "Edit",
    undo: "Undo",
    redo: "Redo",
    cut: "Cut",
    copy: "Copy",
    paste: "Paste",
    selectAll: "Select All",
    // View
    view: "View",
    reload: "Reload",
    forceReload: "Force Reload",
    toggleDevTools: "Toggle Developer Tools",
    actualSize: "Actual Size",
    zoomIn: "Zoom In",
    zoomOut: "Zoom Out",
    toggleFullscreen: "Toggle Fullscreen",
    // Window
    window: "Window",
    minimize: "Minimize",
    close: "Close",
    zoom: "Zoom",
    bringAllToFront: "Bring All to Front",
    // Settings / Help
    settings: "Settings",
    preferences: "Preferences...",
    help: "Help",
    // macOS app menu
    services: "Services",
    hideOthers: "Hide Others",
    showAll: "Show All",
    restartToUpdate: "Restart to Update",
    aboutApp: "About {{name}}",
    hideApp: "Hide {{name}}",
    quitApp: "Quit {{name}}",
    openApp: "Open {{name}}",
    // Tray / links
    documentation: "Documentation",
    chromeExtension: "Kiji Chrome Extension",
    bugReport: "File a Bug Report",
    featureRequest: "Request a Feature",
    emailUs: "Email us",
  },
  fr: {
    // File
    file: "Fichier",
    quit: "Quitter",
    // Edit
    edit: "Édition",
    undo: "Annuler",
    redo: "Rétablir",
    cut: "Couper",
    copy: "Copier",
    paste: "Coller",
    selectAll: "Tout sélectionner",
    // View
    view: "Affichage",
    reload: "Recharger",
    forceReload: "Forcer le rechargement",
    toggleDevTools: "Afficher/masquer les outils de développement",
    actualSize: "Taille réelle",
    zoomIn: "Zoom avant",
    zoomOut: "Zoom arrière",
    toggleFullscreen: "Basculer en plein écran",
    // Window
    window: "Fenêtre",
    minimize: "Réduire",
    close: "Fermer",
    zoom: "Zoom",
    bringAllToFront: "Tout ramener au premier plan",
    // Settings / Help
    settings: "Paramètres",
    preferences: "Préférences...",
    help: "Aide",
    // macOS app menu
    services: "Services",
    hideOthers: "Masquer les autres",
    showAll: "Tout afficher",
    restartToUpdate: "Redémarrer pour mettre à jour",
    aboutApp: "À propos de {{name}}",
    hideApp: "Masquer {{name}}",
    quitApp: "Quitter {{name}}",
    openApp: "Ouvrir {{name}}",
    // Tray / links
    documentation: "Documentation",
    chromeExtension: "Extension Chrome Kiji",
    bugReport: "Signaler un bogue",
    featureRequest: "Proposer une fonctionnalité",
    emailUs: "Nous écrire",
  },
};

/**
 * Normalize an arbitrary language input to a supported base language.
 * Region subtags are stripped ("fr-FR" -> "fr"); unknown values fall back to
 * English.
 */
function normalizeLanguage(lng) {
  if (typeof lng !== "string") return DEFAULT_LANGUAGE;
  const base = lng.split("-")[0].toLowerCase();
  return SUPPORTED_LANGUAGES.includes(base) ? base : DEFAULT_LANGUAGE;
}

let currentLanguage = DEFAULT_LANGUAGE;

function setMenuLanguage(lng) {
  currentLanguage = normalizeLanguage(lng);
  return currentLanguage;
}

function getMenuLanguage() {
  return currentLanguage;
}

/**
 * Look up a menu string for the active language, interpolating {{vars}}.
 * Falls back to English, then to the raw key, so a missing translation can
 * never blank out a menu item.
 */
function mt(key, vars) {
  const table = STRINGS[currentLanguage] || STRINGS[DEFAULT_LANGUAGE];
  let value = table[key] ?? STRINGS[DEFAULT_LANGUAGE][key] ?? key;
  if (vars) {
    value = value.replace(/\{\{(\w+)\}\}/g, (_, name) =>
      name in vars ? vars[name] : `{{${name}}}`
    );
  }
  return value;
}

module.exports = {
  SUPPORTED_LANGUAGES,
  DEFAULT_LANGUAGE,
  normalizeLanguage,
  setMenuLanguage,
  getMenuLanguage,
  mt,
};
