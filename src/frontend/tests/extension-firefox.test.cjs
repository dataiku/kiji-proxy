const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../../..');
const source = path.join(root, 'chrome-extension');
const build = (...args) => spawnSync(process.execPath,
  [path.join(root, 'src/scripts/build-firefox-extension.js'), ...args], { encoding: 'utf8' });

test('Firefox package shares Chrome assets and permissions with a separate background manifest', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-firefox-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const output = path.join(directory, 'extension');
  const result = build('--output', output, '--version', '2.3.4');
  assert.equal(result.status, 0, result.stderr);
  const firefox = JSON.parse(fs.readFileSync(path.join(output, 'manifest.json')));
  const chrome = JSON.parse(fs.readFileSync(path.join(source, 'manifest.json')));
  assert.equal(firefox.version, '2.3.4');
  assert.deepEqual(firefox.background, { scripts: ['config.js', 'background.js'] });
  assert.ok(firefox.browser_specific_settings.gecko.id);
  assert.deepEqual(firefox.permissions, chrome.permissions);
  assert.deepEqual(firefox.host_permissions, chrome.host_permissions);
  assert.deepEqual(firefox.optional_host_permissions, chrome.optional_host_permissions);
  assert.equal(firefox.options_ui.page, chrome.options_page);
  for (const name of ['background.js', 'content.js', 'config.js', 'options.js', 'popup.js', 'styles.css']) {
    assert.equal(fs.readFileSync(path.join(output, name), 'utf8'), fs.readFileSync(path.join(source, name), 'utf8'));
  }
  assert.equal(fs.existsSync(path.join(output, 'manifest.firefox.json')), false);
  assert.equal(build('--output', output).status, 0, 'rebuilding a generated directory succeeds');
});

test('packaging rejects missing arguments, invalid versions, source overlap, and unrelated files', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-firefox-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  fs.writeFileSync(path.join(directory, 'keep.txt'), 'keep');
  for (const args of [
    ['--output'], ['--unknown', 'value'], ['--version', '1.2.3-beta'],
    ['--output', root], ['--output', source], ['--output', path.join(source, 'nested')],
    ['--output', directory],
  ]) {
    assert.notEqual(build(...args).status, 0, args.join(' '));
  }
  assert.equal(fs.readFileSync(path.join(directory, 'keep.txt'), 'utf8'), 'keep');
});

for (const browser of ['Chrome', 'Firefox']) {
  test(`${browser} loads the shared background configuration`, () => {
    const listener = { addListener() {} };
    const context = vm.createContext({
      chrome: { runtime: { onInstalled: listener, onStartup: listener, onMessage: listener },
        storage: { onChanged: listener } },
    });
    const loadConfig = () => vm.runInContext(fs.readFileSync(path.join(source, 'config.js'), 'utf8'), context);
    if (browser === 'Chrome') context.importScripts = loadConfig;
    else loadConfig();
    vm.runInContext(fs.readFileSync(path.join(source, 'background.js'), 'utf8'), context);
    assert.equal(vm.runInContext('DEFAULT_API_BASE', context), 'http://localhost:8080');
  });
}


test('release version synchronization updates both browser manifests', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-version-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  fs.mkdirSync(path.join(directory, 'src/scripts'), { recursive: true });
  fs.mkdirSync(path.join(directory, 'src/frontend'), { recursive: true });
  fs.mkdirSync(path.join(directory, 'chrome-extension'));
  const script = path.join(directory, 'src/scripts/sync-manifest-version.js');
  fs.copyFileSync(path.join(root, 'src/scripts/sync-manifest-version.js'), script);
  fs.writeFileSync(path.join(directory, 'src/frontend/package.json'), JSON.stringify({ version: '2.3.4' }));
  for (const name of ['manifest.json', 'manifest.firefox.json']) {
    fs.writeFileSync(path.join(directory, 'chrome-extension', name), JSON.stringify({ version: '1.0.0', name: 'Kiji' }));
  }
  assert.equal(spawnSync(process.execPath, [script]).status, 0);
  for (const name of ['manifest.json', 'manifest.firefox.json']) {
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(directory, 'chrome-extension', name))),
      { version: '2.3.4', name: 'Kiji' });
  }
  assert.equal(spawnSync(process.execPath, [script]).status, 0, 'repeated synchronization succeeds');
});
