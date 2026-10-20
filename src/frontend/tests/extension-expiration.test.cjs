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
      replaceChildren(...children) { this.children = children; },
      appendChild(child) { (this.children ??= []).push(child); },
    });
    return elements.get(id);
  };
  let ready;
  let policy = { ttl_seconds: 300 };
  const posts = [];
  const context = {
    Option: function Option(text, value) { this.text = text; this.value = value; },
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
        policy = { ttl_seconds: body.ttl_seconds };
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
  assert.deepEqual(f.posts[0].expected, { ttl_seconds: 300 });
  f.element('expiration-seconds').value = '0';
  await f.submit();
  assert.deepEqual(f.posts[1].expected, { ttl_seconds: 120 });
  assert.equal(f.element('expiration-status').textContent, 'Saved.');
});

test('extension preserves a conflicting draft until explicit reload', async () => {
  const f = await setup();
  f.element('expiration-seconds').value = '90';
  f.remote({ ttl_seconds: 0 });
  await f.submit();
  assert.deepEqual(f.current(), { ttl_seconds: 0 });
  assert.equal(f.element('expiration-seconds').value, '90');
  assert.match(f.element('expiration-status').textContent, /changed in another client/);
  assert.equal(f.element('expiration-fields').disabled, false);
  await f.element('expiration-reload').listeners.click();
  assert.equal(f.element('expiration-seconds').value, '0');
  assert.equal(f.element('expiration-seconds').disabled, false);
});

test('extension defaults to 300 and uses zero to disable then a positive duration to enable', async () => {
  const f = await setup();
  assert.equal(f.element('expiration-seconds').value, '300');
  f.element('expiration-seconds').value = '0';
  await f.submit();
  assert.deepEqual(f.current(), { ttl_seconds: 0 });
  assert.equal(f.element('expiration-seconds').value, '0');
  assert.equal(f.element('expiration-seconds').disabled, false);
  f.element('expiration-seconds').value = '300';
  await f.submit();
  assert.deepEqual(f.posts[1].expected, { ttl_seconds: 0 });
  assert.deepEqual(f.current(), { ttl_seconds: 300 });
});

test('extension rejects empty, negative, fractional and excessive durations', async () => {
  const f = await setup();
  for (const value of ['', ' ', '-1', '1.5', '2147483648']) {
    f.element('expiration-seconds').value = value;
    await f.submit();
    assert.match(f.element('expiration-status').textContent, /whole number from 0/);
  }
  assert.equal(f.posts.length, 0);
});

test('extension preserves a zero draft when another client changes the lifetime', async () => {
  const f = await setup();
  f.element('expiration-seconds').value = '0';
  f.remote({ ttl_seconds: 45 });
  await f.submit();
  assert.equal(f.element('expiration-seconds').value, '0');
  assert.match(f.element('expiration-status').textContent, /changed in another client/);
  assert.deepEqual(f.current(), { ttl_seconds: 45 });
});
