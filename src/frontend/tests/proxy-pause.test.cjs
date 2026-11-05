const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const electronPause = require('../src/electron/proxy-pause');

// Compile the renderer helpers in memory, like mapping-expiration.test.cjs.
const source = fs.readFileSync(path.join(__dirname, '../src/components/settings/proxyPause.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } });
const ui = {};
new Function('exports', compiled.outputText)(ui);

const NOW = Date.parse('2026-06-20T12:00:00Z');

test('renderer and Electron offer the same durations as the backend', () => {
  const goSource = fs.readFileSync(path.join(__dirname, '../../backend/server/proxy_pause.go'), 'utf8');
  const goMinutes = goSource.match(/allowedPauseMinutes = map\[int\]bool\{([^}]*)\}/)[1]
    .match(/\d+(?=:)/g).map(Number);
  assert.deepEqual([...ui.PAUSE_DURATIONS_MINUTES], goMinutes);
  assert.deepEqual(electronPause.PAUSE_DURATIONS_MINUTES, goMinutes);
  assert.ok(ui.PAUSE_DURATIONS_MINUTES.includes(ui.DEFAULT_PAUSE_MINUTES));
});

test('secondsUntil rounds up and never goes negative', () => {
  assert.equal(ui.secondsUntil('2026-06-20T12:15:00Z', NOW), 900);
  assert.equal(ui.secondsUntil('2026-06-20T12:15:00Z', NOW + 899_001), 1);
  assert.equal(ui.secondsUntil('2026-06-20T12:15:00Z', NOW + 900_000), 0);
  assert.equal(ui.secondsUntil('2026-06-20T11:00:00Z', NOW), 0);
  assert.equal(ui.secondsUntil(null, NOW), 0);
  assert.equal(ui.secondsUntil('not a date', NOW), 0);
});

test('formatCountdown pads seconds', () => {
  assert.equal(ui.formatCountdown(3600), '60:00');
  assert.equal(ui.formatCountdown(845), '14:05');
  assert.equal(ui.formatCountdown(9), '0:09');
  assert.equal(ui.formatCountdown(-3), '0:00');
});

test('activePausedUntil only returns a deadline that is still ahead', () => {
  const key = electronPause.PAUSED_UNTIL_CONFIG_KEY;
  assert.equal(electronPause.activePausedUntil({ [key]: '2026-06-20T12:05:00Z' }, NOW), '2026-06-20T12:05:00Z');
  assert.equal(electronPause.activePausedUntil({ [key]: '2026-06-20T12:00:00Z' }, NOW), null);
  assert.equal(electronPause.activePausedUntil({ [key]: 'garbage' }, NOW), null);
  assert.equal(electronPause.activePausedUntil({ [key]: 12345 }, NOW), null);
  assert.equal(electronPause.activePausedUntil({}, NOW), null);
  assert.equal(electronPause.activePausedUntil(undefined, NOW), null);
});

test('applyPausedUntilEnv forwards a running pause and scrubs stale values', () => {
  const key = electronPause.PAUSED_UNTIL_CONFIG_KEY;
  const envName = electronPause.PAUSED_UNTIL_ENV;
  assert.equal(envName, 'TRANSPARENT_PROXY_PAUSED_UNTIL');

  const running = electronPause.applyPausedUntilEnv({}, { [key]: '2026-06-20T12:30:00Z' }, NOW);
  assert.equal(running[envName], '2026-06-20T12:30:00Z');

  const inherited = { [envName]: '2099-01-01T00:00:00Z', PATH: '/bin' };
  electronPause.applyPausedUntilEnv(inherited, { [key]: '2026-06-20T11:59:00Z' }, NOW);
  assert.equal(envName in inherited, false, 'an expired pause must not reach the backend');
  assert.equal(inherited.PATH, '/bin');
});

test('persistPauseStatus mirrors backend status into config', () => {
  const key = electronPause.PAUSED_UNTIL_CONFIG_KEY;
  const config = { language: 'fr' };
  electronPause.persistPauseStatus(config, { paused: true, paused_until: '2026-06-20T12:05:00Z', remaining_seconds: 300 });
  assert.equal(config[key], '2026-06-20T12:05:00Z');
  electronPause.persistPauseStatus(config, { paused: false, paused_until: null, remaining_seconds: 0 });
  assert.equal(key in config, false);
  assert.equal(config.language, 'fr');
});

test('isValidPauseMinutes matches the offered durations exactly', () => {
  for (const m of [1, 5, 15, 30, 60]) assert.equal(electronPause.isValidPauseMinutes(m), true);
  for (const m of [0, 2, 61, -5, 1.5, '5', null, undefined]) assert.equal(electronPause.isValidPauseMinutes(m), false);
});
