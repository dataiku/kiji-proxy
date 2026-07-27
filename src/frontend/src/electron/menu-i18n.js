// Main-process menu translations.
//
// The renderer localizes with react-i18next, but the native application and
// tray menus are built in the main process, which has no access to that
// runtime. This module is a tiny, self-contained string table + selector so
// the menus can be rebuilt in the user's language. Keep the keys here in sync
// with what createMenu()/updateTrayMenu() reference in electron-main.js.

const SUPPORTED_LANGUAGES = ["en", "fr", "ja", "ko"];
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
  ja: {
    // File
    file: "ファイル",
    quit: "終了",
    // Edit
    edit: "編集",
    undo: "取り消す",
    redo: "やり直す",
    cut: "カット",
    copy: "コピー",
    paste: "ペースト",
    selectAll: "すべてを選択",
    // View
    view: "表示",
    reload: "再読み込み",
    forceReload: "強制的に再読み込み",
    toggleDevTools: "デベロッパーツールを表示/非表示",
    actualSize: "実際のサイズ",
    zoomIn: "拡大",
    zoomOut: "縮小",
    toggleFullscreen: "フルスクリーンを切り替え",
    // Window
    window: "ウインドウ",
    minimize: "しまう",
    close: "閉じる",
    zoom: "拡大/縮小",
    bringAllToFront: "すべてを手前に移動",
    // Settings / Help
    settings: "設定",
    preferences: "環境設定...",
    help: "ヘルプ",
    // macOS app menu
    services: "サービス",
    hideOthers: "ほかを隠す",
    showAll: "すべてを表示",
    restartToUpdate: "再起動して更新",
    aboutApp: "{{name}} について",
    hideApp: "{{name}} を隠す",
    quitApp: "{{name}} を終了",
    openApp: "{{name}} を開く",
    // Tray / links
    documentation: "ドキュメント",
    chromeExtension: "Kiji Chrome 拡張機能",
    bugReport: "不具合を報告",
    featureRequest: "機能をリクエスト",
    emailUs: "メールで問い合わせる",
  },
  ko: {
    // File
    file: "파일",
    quit: "종료",
    // Edit
    edit: "편집",
    undo: "실행 취소",
    redo: "다시 실행",
    cut: "오려두기",
    copy: "복사하기",
    paste: "붙여넣기",
    selectAll: "전체 선택",
    // View
    view: "보기",
    reload: "다시 불러오기",
    forceReload: "강제로 다시 불러오기",
    toggleDevTools: "개발자 도구 표시/숨기기",
    actualSize: "실제 크기",
    zoomIn: "확대",
    zoomOut: "축소",
    toggleFullscreen: "전체 화면 전환",
    // Window
    window: "윈도우",
    minimize: "최소화",
    close: "닫기",
    zoom: "확대/축소",
    bringAllToFront: "모두 앞으로 가져오기",
    // Settings / Help
    settings: "설정",
    preferences: "환경설정...",
    help: "도움말",
    // macOS app menu
    services: "서비스",
    hideOthers: "나머지 가리기",
    showAll: "모두 보기",
    restartToUpdate: "다시 시작하여 업데이트",
    aboutApp: "{{name}} 정보",
    hideApp: "{{name}} 가리기",
    quitApp: "{{name}} 종료",
    openApp: "{{name}} 열기",
    // Tray / links
    documentation: "문서",
    chromeExtension: "Kiji Chrome 확장 프로그램",
    bugReport: "버그 신고",
    featureRequest: "기능 요청",
    emailUs: "이메일 보내기",
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
