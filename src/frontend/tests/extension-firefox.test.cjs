const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../../..');
const source = path.join(root, 'browser-extension');
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
  assert.equal(firefox.version, '2.3.4.400000000');
  assert.equal(firefox.version_name, '2.3.4');
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
    ['--output'], ['--unknown', 'value'], ['--version', '1.2.3-preview.1'],
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
  fs.mkdirSync(path.join(directory, 'browser-extension'));
  const script = path.join(directory, 'src/scripts/sync-manifest-version.js');
  fs.copyFileSync(path.join(root, 'src/scripts/sync-manifest-version.js'), script);
  fs.writeFileSync(path.join(directory, 'src/frontend/package.json'), JSON.stringify({ version: '2.3.4' }));
  for (const name of ['manifest.json', 'manifest.firefox.json']) {
    fs.writeFileSync(path.join(directory, 'browser-extension', name), JSON.stringify({ version: '1.0.0', name: 'Kiji' }));
  }
  assert.equal(spawnSync(process.execPath, [script]).status, 0);
  for (const name of ['manifest.json', 'manifest.firefox.json']) {
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(directory, 'browser-extension', name))),
      { version: '2.3.4', name: 'Kiji' });
  }
  assert.equal(spawnSync(process.execPath, [script]).status, 0, 'repeated synchronization succeeds');
});


const releaseVersions = [
  ['1.2.3-alpha', '1.2.3.100000000'],
  ['1.2.3-alpha.0', '1.2.3.100000001'],
  ['1.2.3-alpha.1', '1.2.3.100000002'],
  ['1.2.3-alpha.99999998', '1.2.3.199999999'],
  ['1.2.3-beta', '1.2.3.200000000'],
  ['1.2.3-beta.0', '1.2.3.200000001'],
  ['1.2.3-beta.9', '1.2.3.200000010'],
  ['1.2.3-beta.10', '1.2.3.200000011'],
  ['1.2.3-beta.99999998', '1.2.3.299999999'],
  ['1.2.3-rc', '1.2.3.300000000'],
  ['1.2.3-rc.1', '1.2.3.300000002'],
  ['1.2.3-rc.99999998', '1.2.3.399999999'],
  ['1.2.3', '1.2.3.400000000'],
  ['1.2.4-alpha', '1.2.4.100000000'],
];

for (const [version, expected] of releaseVersions) {
  test(`Firefox packages release ${version} with an ordered numeric version`, (t) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-release-test-'));
    t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
    const result = build('--output', directory, '--version', version);
    assert.equal(result.status, 0, result.stderr);
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json')));
    assert.equal(manifest.version, expected);
    assert.equal(manifest.version_name, version);
    assert.match(manifest.version, /^(0|[1-9]\d{0,8})(\.(0|[1-9]\d{0,8})){0,3}$/);
    assert.equal(build('--output', directory, '--version', '1.2.3').status, 0);
    const stable = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json')));
    assert.equal(stable.version, '1.2.3.400000000');
    assert.equal(stable.version_name, '1.2.3');
  });
}

test('numeric version ordering follows alpha, beta, rc, stable and the next release', () => {
  const compare = (left, right) => {
    const a = left.split('.').map(Number);
    const b = right.split('.').map(Number);
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      const difference = (a[i] || 0) - (b[i] || 0);
      if (difference) return difference;
    }
    return 0;
  };
  for (let i = 1; i < releaseVersions.length; i++) {
    assert.ok(compare(releaseVersions[i - 1][1], releaseVersions[i][1]) < 0);
  }
});

test('build metadata keeps the same update version and preserves the release label', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-metadata-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const [version, expected] of [
    ['1.2.3-beta.2+build.7', '1.2.3.200000003'],
    ['1.2.3+build.7', '1.2.3.400000000'],
    ['1', '1'], ['1.2', '1.2'], ['1.2.3.4', '1.2.3.4'],
  ]) {
    const result = build('--output', directory, '--version', version);
    assert.equal(result.status, 0, result.stderr);
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json')));
    assert.equal(manifest.version, expected);
    assert.equal(manifest.version_name, version);
  }
});

test('malformed and overflowing release versions fail without writing output', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'kiji-invalid-release-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const output = path.join(directory, 'extension');
  for (const version of ['1.2.3-beta.', '1.2.3-beta.01', '1.2.3-rc.-1', '1.2.3.4-beta.1',
    '1.2.3-beta.99999999', '01.2.3', '1.2.3.4.5', '1000000000.2.3', '1.2.3+']) {
    assert.notEqual(build('--output', output, '--version', version).status, 0, version);
    assert.equal(fs.existsSync(output), false);
  }
});


for (const manifest of [
  { version: '1.2.3.200000002', version_name: '1.2.3-beta.1' },
  { version: '1.2.3.400000000', version_name: '1.2.3' },
  { version: '1.2.3' },
]) {
  test(`popup displays ${manifest.version_name || manifest.version}`, () => {
    const element = { textContent: '' };
    const context = vm.createContext({
      chrome: { runtime: { getManifest: () => manifest } },
      document: { addEventListener() {}, getElementById: () => element },
    });
    vm.runInContext(fs.readFileSync(path.join(source, 'popup.js'), 'utf8'), context);
    vm.runInContext('renderVersion()', context);
    assert.equal(element.textContent, manifest.version_name || manifest.version);
  });
}
