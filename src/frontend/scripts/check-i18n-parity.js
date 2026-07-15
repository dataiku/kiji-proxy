#!/usr/bin/env node
/*
 * i18n parity check (plural-aware).
 *
 * Validates that the English base and every translation stay structurally in
 * sync. Run from src/frontend as `node scripts/check-i18n-parity.js` (wired to
 * `npm run i18n:check` and CI).
 *
 * What it checks, per namespace, for each non-English locale against English:
 *   1. Base-key parity — the set of keys, with CLDR plural suffixes stripped,
 *      must match exactly. Missing or stray keys fail.
 *   2. Plural completeness — a key that is pluralized must provide exactly the
 *      plural categories the locale's own CLDR rules require. English needs
 *      {one, other}; French needs {one, many, other}. This is why a naive
 *      key-for-key diff is wrong: the correct French file legitimately has
 *      `_many` variants English does not.
 *   3. Interpolation parity — a key present in both locales must use the same
 *      set of {{placeholders}}.
 *   4. Markup parity — <tag> names used by <Trans> strings must match across
 *      locales, and open/close tags must balance within every string.
 */

const fs = require("fs");
const path = require("path");

const LOCALES_DIR = path.join(__dirname, "..", "src", "i18n", "locales");
const BASE_LOCALE = "en";
const PLURAL_SUFFIXES = ["zero", "one", "two", "few", "many", "other"];
const PLURAL_RE = new RegExp(`_(${PLURAL_SUFFIXES.join("|")})$`);

// The plural categories each locale must supply, taken from the locale's own
// CLDR cardinal rules (e.g. en -> [one, other], fr -> [one, many, other]).
function requiredCategories(locale) {
  return new Set(
    new Intl.PluralRules(locale).resolvedOptions().pluralCategories
  );
}

function baseKey(key) {
  return key.replace(PLURAL_RE, "");
}

function pluralCategory(key) {
  const m = key.match(PLURAL_RE);
  return m ? m[1] : null;
}

// Flatten a nested JSON object into dot-joined leaf keys.
function flatten(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      flatten(v, key, out);
    } else {
      out[key] = v;
    }
  }
  return out;
}

function placeholders(value) {
  if (typeof value !== "string") return new Set();
  return new Set([...value.matchAll(/\{\{(\w+)\}\}/g)].map((m) => m[1]));
}

function tagNames(value) {
  if (typeof value !== "string") return [];
  return [...value.matchAll(/<\/?(\w+)\s*\/?>/g)].map((m) => m[1]);
}

function tagsBalanced(value) {
  if (typeof value !== "string") return true;
  const opens = [...value.matchAll(/<(\w+)>/g)].map((m) => m[1]).sort();
  const closes = [...value.matchAll(/<\/(\w+)>/g)].map((m) => m[1]).sort();
  return JSON.stringify(opens) === JSON.stringify(closes);
}

function eq(a, b) {
  return a.size === b.size && [...a].every((x) => b.has(x));
}

function listNamespaces() {
  return fs
    .readdirSync(path.join(LOCALES_DIR, BASE_LOCALE))
    .filter((f) => f.endsWith(".json"))
    .map((f) => f.replace(/\.json$/, ""));
}

function listLocales() {
  return fs
    .readdirSync(LOCALES_DIR)
    .filter((f) => fs.statSync(path.join(LOCALES_DIR, f)).isDirectory())
    .filter((l) => l !== BASE_LOCALE);
}

function load(locale, ns) {
  return flatten(
    JSON.parse(
      fs.readFileSync(path.join(LOCALES_DIR, locale, `${ns}.json`), "utf8")
    )
  );
}

function baseKeySet(flat) {
  return new Set(Object.keys(flat).map(baseKey));
}

// Map base key -> set of plural categories present for it in a flat locale.
function pluralCategoriesByBase(flat) {
  const map = new Map();
  for (const key of Object.keys(flat)) {
    const cat = pluralCategory(key);
    if (!cat) continue;
    const b = baseKey(key);
    if (!map.has(b)) map.set(b, new Set());
    map.get(b).add(cat);
  }
  return map;
}

function checkNamespace(locale, ns, errors) {
  const en = load(BASE_LOCALE, ns);
  let loc;
  try {
    loc = load(locale, ns);
  } catch {
    errors.push(`[${locale}/${ns}] missing or unreadable translation file`);
    return;
  }

  const enBases = baseKeySet(en);
  const locBases = baseKeySet(loc);

  for (const b of enBases) {
    if (!locBases.has(b)) errors.push(`[${locale}/${ns}] missing key: ${b}`);
  }
  for (const b of locBases) {
    if (!enBases.has(b)) errors.push(`[${locale}/${ns}] stray key: ${b}`);
  }

  // Plural completeness for keys pluralized in either locale.
  const enPlurals = pluralCategoriesByBase(en);
  const locPlurals = pluralCategoriesByBase(loc);
  const pluralBases = new Set([...enPlurals.keys(), ...locPlurals.keys()]);
  const enReq = requiredCategories(BASE_LOCALE);
  const locReq = requiredCategories(locale);

  for (const b of pluralBases) {
    if (enBases.has(b)) {
      const have = enPlurals.get(b) || new Set();
      for (const cat of enReq) {
        if (!have.has(cat)) {
          errors.push(`[${BASE_LOCALE}/${ns}] ${b}: missing plural _${cat}`);
        }
      }
    }
    if (locBases.has(b)) {
      const have = locPlurals.get(b) || new Set();
      for (const cat of locReq) {
        if (!have.has(cat)) {
          errors.push(`[${locale}/${ns}] ${b}: missing plural _${cat}`);
        }
      }
    }
  }

  // Interpolation + markup parity for keys present in both locales.
  for (const key of Object.keys(en)) {
    if (!(key in loc)) continue;
    if (!eq(placeholders(en[key]), placeholders(loc[key]))) {
      errors.push(`[${locale}/${ns}] ${key}: placeholder mismatch`);
    }
    const enTags = new Set(tagNames(en[key]));
    const locTags = new Set(tagNames(loc[key]));
    if (!eq(enTags, locTags)) {
      errors.push(`[${locale}/${ns}] ${key}: markup <tag> mismatch`);
    }
    if (!tagsBalanced(loc[key])) {
      errors.push(`[${locale}/${ns}] ${key}: unbalanced markup tags`);
    }
  }
  // English strings should have balanced markup too.
  for (const key of Object.keys(en)) {
    if (!tagsBalanced(en[key])) {
      errors.push(`[${BASE_LOCALE}/${ns}] ${key}: unbalanced markup tags`);
    }
  }
}

function main() {
  const namespaces = listNamespaces();
  const locales = listLocales();
  const errors = [];

  for (const locale of locales) {
    for (const ns of namespaces) {
      checkNamespace(locale, ns, errors);
    }
  }

  if (errors.length > 0) {
    console.error(
      `i18n parity check failed with ${errors.length} issue(s):\n` +
        errors.map((e) => `  - ${e}`).join("\n")
    );
    process.exit(1);
  }

  console.log(
    `i18n parity OK: ${namespaces.length} namespace(s) across locales [${[
      BASE_LOCALE,
      ...locales,
    ].join(", ")}].`
  );
}

main();
