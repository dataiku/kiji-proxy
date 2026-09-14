# Chapter 10: Internationalization (i18n)

The desktop/renderer UI is localized with [react-i18next](https://react.i18next.com/).
English (`en`) is the base language; French (`fr`), Japanese (`ja`), and Korean
(`ko`) are the translations.
This chapter covers how the translation system is wired and how to add or change
strings.

## Overview

- **Framework:** `i18next` + `react-i18next`, initialized in
  `src/frontend/src/i18n/index.ts` and imported once for its side effects from
  `src/frontend/index.js`.
- **Detection & persistence:** `i18next-browser-languagedetector` reads the
  language from `localStorage` (falling back to `navigator.language`) and caches
  the choice back to `localStorage`. Region subtags collapse to the base
  language (`fr-FR` → `fr`) via `load: "languageOnly"`.
- **Supported languages:** declared in `SUPPORTED_LANGUAGES`
  (`["en", "fr", "ja", "ko"]`) in `src/i18n/index.ts`. The Electron main process
  keeps its own copy in `src/electron/menu-i18n.js`, derived from the keys of its
  `STRINGS` table (see below).
- **Bundled resources:** all locale JSON is imported statically and initialized
  synchronously, so `react: { useSuspense: false }` is safe (the class-based
  `ErrorBoundary` renders translations without an async gate).

## Where the strings live

Translations are namespaced JSON under
`src/frontend/src/i18n/locales/<lang>/<namespace>.json`:

| Namespace    | Covers                                                        |
| ------------ | ------------------------------------------------------------- |
| `common`     | Shared UI, the Playground, language labels, error boundary    |
| `settings`   | Settings view and all its sections                            |
| `dashboard`  | Dashboard view and the sidebar                                |
| `activity`   | Activity (logs) view                                          |
| `mappings`   | Mappings view                                                 |
| `about`      | About view                                                    |
| `onboarding` | Welcome / first-run modal                                     |
| `modals`     | CA-certificate setup and misclassification-report dialogs     |

Every namespace exists in every locale (`en/`, `fr/`, `ja/`, `ko/`). To add a
namespace, create the JSON in all locales and register it in `NAMESPACES` +
`resources` in `src/i18n/index.ts`.

## Using translations in components

Function components use the hook; class components use the HOC (needed for
`ErrorBoundary`):

```tsx
const { t } = useTranslation("dashboard");
return <h1>{t("title")}</h1>;
```

- **Interpolation:** `t("kpi.requestsToday", { value: fmt(n) })` with
  `"{{value}} today"`. Pre-format numbers/dates and pass them as plain vars so
  formatting is preserved; avoid the reserved `count` var unless you want
  pluralization.
- **Pluralization:** use `key_one` / `key_other` (English) with `{{count}}`:
  `t("entryCount", { count: total })`. See the plural note below for the other
  locales.
- **Embedded markup** (links, bold): use `<Trans>` with a `components` map, e.g.
  the CA-cert instructions in `modals.json` render `<b>`, `<code>`, `<accent>`,
  and `<amber>` tags. Keep the tag pairs balanced in every locale.
- **Cross-namespace keys:** reference another namespace with a prefix, e.g.
  `t("common:actions.close")`.

Brand names (`Kiji`, `OpenAI`, …), shell commands, certificate paths, and other
technical literals are intentionally left untranslated.

### Plural categories per locale

i18next resolves plural suffixes with `Intl.PluralRules`, so each locale needs
exactly the cardinal categories its CLDR rules define, and **i18next does not
fall back from a missing category to `other`** — a missing suffix renders the
raw key. The parity check (below) enforces exactly the categories each locale's
CLDR rules require:

| Locale | Required suffixes         | Note                                              |
| ------ | ------------------------ | ------------------------------------------------- |
| `en`   | `_one`, `_other`         |                                                   |
| `fr`   | `_one`, `_many`, `_other`| `many` triggers on multiples of 1,000,000         |
| `ja`   | `_other` only            | No singular/plural distinction; a `_one` key fails |
| `ko`   | `_other` only            | Same as `ja`                                      |

Do not copy `_one` keys from `en` into an `other`-only locale: the parity check
rejects categories the locale's rules never produce.

## Language selector and the Electron menu

- The in-app selector lives in `src/components/settings/LanguageSection.tsx` and
  is shown in Settings for both the web and desktop builds. It calls
  `i18n.changeLanguage(...)`; persistence is handled by the detector's
  `localStorage` cache.
- The native application and tray menus are built in the Electron **main**
  process, which has no access to react-i18next. Their strings live in
  `src/electron/menu-i18n.js`. The renderer pushes its resolved language to the
  main process over the `set-language` IPC channel (on init and on every
  `languageChanged`); the main process persists it and rebuilds the menus, and
  seeds the menu language from the persisted config at startup.
- `menu-i18n.js` derives its `SUPPORTED_LANGUAGES` from the keys of its
  `STRINGS` table. A locale that exists in the renderer but not in `STRINGS`
  falls back to English menus with no error, so **every new locale must add a
  `STRINGS` entry in `menu-i18n.js`** in the same PR.

## Adding or changing strings

1. Add the key to the **English** namespace JSON (the base/source of truth).
2. Add the same key to the **French**, **Japanese**, and **Korean** JSON with
   translations, using only the plural suffixes each locale requires (table
   above). Drafts are machine-translated pending native review — flag anything
   uncertain.
3. Reference it from the component with `t(...)` (or `<Trans>` for markup).
4. Run the checks:

   ```bash
   cd src/frontend
   npm run i18n:check   # plural-aware parity across all locales (hard gate, CI)
   npm run lint         # includes an advisory no-literal-string warning
   npm run type-check
   ```

### Tooling

- **`scripts/check-i18n-parity.js`** (`npm run i18n:check`) — the hard gate, run
  in CI. Per namespace it verifies: base-key parity between locales (plural
  suffixes normalized away), plural completeness against each locale's CLDR
  categories, matching `{{placeholder}}` sets, and matching/balanced `<Trans>`
  tags.
- **`eslint-plugin-i18next`** — `no-literal-string` runs in `jsx-text-only` mode
  as a **warning** over `src/**/*.{jsx,tsx}`, surfacing un-extracted visible text
  without gating CI (attributes/expressions and technical literals are out of
  scope).
