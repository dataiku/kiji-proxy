const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

// check-i18n-parity.js covers the renderer JSON only; this covers the native menu table.
const menuPath = path.join(__dirname, '../src/electron/menu-i18n.js');
const menu = require(menuPath);
const STRINGS = new Function('module', 'require', `${fs.readFileSync(menuPath, 'utf8')}\nreturn STRINGS;`)({ exports: {} }, require);

const rendererSource = fs.readFileSync(path.join(__dirname, '../src/i18n/index.ts'), 'utf8');
const rendererLanguages = JSON.parse(rendererSource.match(/SUPPORTED_LANGUAGES = (\[[^\]]*\])/)[1]);
const placeholders = value => (value.match(/\{\{\w+\}\}/g) || []).sort();

test('every renderer UI language has a native menu translation', () => {
  assert.deepEqual([...menu.SUPPORTED_LANGUAGES].sort(), [...rendererLanguages].sort());
});

test('every menu key has a non-empty value with the same placeholders in every language', () => {
  const keys = Object.keys(STRINGS.en);
  for (const lng of menu.SUPPORTED_LANGUAGES) {
    assert.deepEqual(Object.keys(STRINGS[lng]).sort(), [...keys].sort(), `${lng} menu keys differ from en`);
    for (const key of keys) {
      const value = STRINGS[lng][key];
      assert.ok(typeof value === 'string' && value.trim(), `${lng}.${key} is empty`);
      assert.deepEqual(placeholders(value), placeholders(STRINGS.en[key]), `${lng}.${key} placeholders differ from en`);
    }
  }
});

test('Indonesian menus are selected for id and id-ID', () => {
  assert.equal(menu.normalizeLanguage('id-ID'), 'id');
  menu.setMenuLanguage('id');
  try {
    assert.equal(menu.mt('quitApp', { name: 'Kiji' }), 'Keluar dari Kiji');
  } finally {
    menu.setMenuLanguage('en');
  }
});
