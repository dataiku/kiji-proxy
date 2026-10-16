const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const tick = () => new Promise(resolve => setImmediate(resolve));

async function setup() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', textContent: '', checked: true, disabled: false,
      listeners: {}, addEventListener(type, callback) { this.listeners[type] = callback; },
      querySelector: () => null,
    });
    return elements.get(id);
  };
  let ready;
  let policy = { enabled: true, ttl_seconds: 300 };
  const posts = [];
  const context = {
    document: { addEventListener: (_, callback) => { ready = callback; }, getElementById: element },
    CONFIG: { DEFAULT_API_BASE: 'http://localhost:8080', DEFAULT_DOMAINS: [] },
    chrome: {
      runtime: { getManifest: () => ({ version: 'test' }) },
      storage: { sync: { get: async (defaults, callback) => {
        if (callback) callback(defaults);
        return defaults;
      } } },
    },
    fetch: async (url, init) => {
      if (!url.endsWith('/mapping-expiration')) return { ok: true, json: async () => ({}) };
      if (init?.method === 'POST') {
        const body = JSON.parse(init.body);
        posts.push(body);
        if (JSON.stringify(body.expected) !== JSON.stringify(policy)) {
          return { status: 409, ok: false, json: async () => policy };
        }
        policy = { enabled: body.enabled, ttl_seconds: body.ttl_seconds };
      }
      return { status: 200, ok: true, json: async () => policy };
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../../chrome-extension/options.js'), 'utf8'), context);
  ready();
  await tick();
  return {
    element, posts, remote: p => { policy = p; }, current: () => policy,
    submit: () => element('expiration-form').listeners.submit({ preventDefault() {} }),
  };
}

test('extension sends its loaded baseline and updates it after a successful save', async () => {
  const f = await setup();
  f.element('expiration-seconds').value = '120';
  await f.submit();
  assert.deepEqual(f.posts[0].expected, { enabled: true, ttl_seconds: 300 });
  f.element('expiration-enabled').checked = false;
  await f.submit();
  assert.deepEqual(f.posts[1].expected, { enabled: true, ttl_seconds: 120 });
  assert.equal(f.element('expiration-status').textContent, 'Saved.');
});

test('extension preserves a conflicting draft until explicit reload', async () => {
  const f = await setup();
  f.element('expiration-seconds').value = '90';
  f.remote({ enabled: false, ttl_seconds: 60 });
  await f.submit();
  assert.deepEqual(f.current(), { enabled: false, ttl_seconds: 60 });
  assert.equal(f.element('expiration-seconds').value, '90');
  assert.match(f.element('expiration-status').textContent, /changed in another client/);
  assert.equal(f.element('expiration-fields').disabled, false);
  await f.element('expiration-reload').listeners.click();
  assert.equal(f.element('expiration-seconds').value, 60);
  assert.equal(f.element('expiration-enabled').checked, false);
});
