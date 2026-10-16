const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

// Use the project's compiler in memory; works on the Node 20 CI runtime and
// tests the same controller used by React without a second implementation.
const source = fs.readFileSync(path.join(__dirname, '../src/components/settings/mappingExpirationStore.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } });
const exportsUnderTest = {};
new Function('exports', compiled.outputText)(exportsUnderTest);
const { createMappingExpirationStore } = exportsUnderTest;
const policy = (ttl_seconds, enabled = true) => ({ enabled, ttl_seconds });
const response = (value, status = 200) => ({ ok: status === 200, status, json: async () => value });
const tick = () => new Promise(resolve => setImmediate(resolve));
const deferred = () => {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
};
function fixture() {
  let current = policy(300);
  const posts = [];
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init?.method === 'POST') {
      const body = JSON.parse(init.body);
      posts.push(body);
      if (JSON.stringify(body.expected) !== JSON.stringify(current)) return response(current, 409);
      current = policy(body.ttl_seconds, body.enabled);
    }
    return response(current);
  });
  return { store, posts, remote: p => { current = p; }, current: () => current };
}

test('focus and visibility refresh a clean form without navigating away', async () => {
  const f = fixture();
  const focus = new EventTarget();
  const visibility = Object.assign(new EventTarget(), { visibilityState: 'visible' });
  const stop = f.store.watch(focus, visibility);
  await tick();
  f.remote(policy(120));
  focus.dispatchEvent(new Event('focus'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '120');
  f.remote(policy(60, false));
  visibility.visibilityState = 'hidden';
  visibility.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '120');
  visibility.visibilityState = 'visible';
  visibility.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.equal(f.store.getSnapshot().enabled, false);
  assert.equal(f.store.getSnapshot().seconds, '60');
  stop();
  f.remote(policy(900));
  focus.dispatchEvent(new Event('focus'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '60');
});

test('refresh keeps a draft when the server has not changed', async () => {
  const { store } = fixture();
  await store.refresh();
  store.edit({ seconds: '90', enabled: false });
  await store.refresh();
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().enabled, false);
  assert.equal(store.getSnapshot().status, null);
});

test('external change preserves draft and blocks saving until resolved', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '90' });
  f.remote(policy(120, false));
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().seconds, '90');
  assert.equal(f.store.getSnapshot().enabled, true);
  assert.equal(f.store.getSnapshot().status, 'conflict');
  await f.store.save();
  assert.equal(f.posts.length, 0);
  f.store.useServerSettings();
  assert.equal(f.store.getSnapshot().seconds, '120');
  assert.equal(f.store.getSnapshot().enabled, false);
  assert.equal(f.store.getSnapshot().status, null);
});

test('keeping edits explicitly rebases the conditional save on the latest policy', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '90' });
  f.remote(policy(120));
  await f.store.refresh();
  f.store.keepEdits();
  assert.equal(f.store.getSnapshot().seconds, '90');
  await f.store.save();
  assert.deepEqual(f.posts[0].expected, policy(120));
  assert.deepEqual(f.current(), policy(90));
  assert.equal(f.store.getSnapshot().status, 'saved');
});

test('a server change after the last refresh returns a conflict without losing edits', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '90' });
  f.remote(policy(60));
  await f.store.save();
  assert.deepEqual(f.current(), policy(60));
  assert.equal(f.store.getSnapshot().seconds, '90');
  assert.equal(f.store.getSnapshot().status, 'conflict');
  assert.equal(f.store.getSnapshot().saving, false);
  // A further edit must not accidentally clear the conflict.
  f.store.edit({ seconds: '80' });
  await f.store.save();
  assert.equal(f.posts.length, 1);
});

test('edits made while a refresh is in flight are preserved', async () => {
  let read = async () => response(policy(300));
  const store = createMappingExpirationStore('/policy', () => read());
  await store.refresh();
  const pending = deferred();
  read = () => pending.promise;
  const refresh = store.refresh();
  store.edit({ seconds: '90' });
  pending.resolve(response(policy(120)));
  await refresh;
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, 'conflict');
});

test('out-of-order refresh responses cannot replace newer settings', async () => {
  const reads = [deferred(), deferred()];
  let index = 0;
  const store = createMappingExpirationStore('/policy', () => reads[index++].promise);
  const older = store.refresh();
  const newer = store.refresh();
  reads[1].resolve(response(policy(120)));
  await newer;
  reads[0].resolve(response(policy(300)));
  await older;
  assert.equal(store.getSnapshot().seconds, '120');
});

test('a pre-save GET and focus events during a save cannot overwrite the saved value', async () => {
  let calls = 0;
  const oldRead = deferred();
  const saveReply = deferred();
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    calls++;
    if (init?.method === 'POST') return saveReply.promise;
    return calls === 1 ? response(policy(300)) : oldRead.promise;
  });
  await store.refresh();
  const refresh = store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  await store.refresh();
  assert.equal(calls, 3);
  saveReply.resolve(response(policy(90)));
  await save;
  oldRead.resolve(response(policy(300)));
  await refresh;
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, 'saved');
});

test('failed saves preserve the draft and can be retried', async () => {
  let fail = true;
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init?.method !== 'POST') return response(policy(300));
    if (fail) throw new Error('offline');
    return response(policy(90));
  });
  await store.refresh();
  store.edit({ seconds: '90' });
  await store.save();
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, 'error');
  fail = false;
  await store.save();
  assert.equal(store.getSnapshot().status, 'saved');
});

test('invalid values never reach the server', async () => {
  const f = fixture();
  await f.store.refresh();
  for (const seconds of ['', '0', '-1', '1.5', '2147483648']) {
    f.store.edit({ seconds });
    await f.store.save();
    assert.equal(f.store.getSnapshot().status, 'invalid');
  }
  assert.equal(f.posts.length, 0);
});

test('visible settings poll for side-by-side changes and stop polling when hidden or unmounted', async () => {
  const f = fixture();
  const visibility = Object.assign(new EventTarget(), { visibilityState: 'visible' });
  let poll;
  let stopped = false;
  const stop = f.store.watch(new EventTarget(), visibility, callback => {
    poll = callback;
    return () => { stopped = true; };
  });
  await tick();
  f.remote(policy(120));
  poll();
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '120');
  f.store.edit({ seconds: '90' });
  f.remote(policy(60));
  poll();
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '90');
  assert.equal(f.store.getSnapshot().status, 'conflict');
  visibility.visibilityState = 'hidden';
  f.remote(policy(30));
  poll();
  await tick();
  assert.equal(f.store.getSnapshot().latest.ttl_seconds, 60);
  stop();
  assert.equal(stopped, true);
});
