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
const policy = ttl_seconds => ({ ttl_seconds });
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
      current = policy(body.ttl_seconds);
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
  f.remote(policy(0));
  visibility.visibilityState = 'hidden';
  visibility.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '120');
  visibility.visibilityState = 'visible';
  visibility.dispatchEvent(new Event('visibilitychange'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '0');
  stop();
  f.remote(policy(900));
  focus.dispatchEvent(new Event('focus'));
  await tick();
  assert.equal(f.store.getSnapshot().seconds, '0');
});

test('refresh keeps a draft when the server has not changed', async () => {
  const { store } = fixture();
  await store.refresh();
  store.edit({ seconds: '90' });
  await store.refresh();
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, null);
});

test('external change preserves draft and blocks saving until resolved', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '90' });
  f.remote(policy(0));
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().seconds, '90');
  assert.equal(f.store.getSnapshot().status, 'conflict');
  await f.store.save();
  assert.equal(f.posts.length, 0);
  f.store.useServerSettings();
  assert.equal(f.store.getSnapshot().seconds, '0');
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
    if (fail) return response({}, 500);
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
  for (const seconds of ['', '   ', '-1', '1.5', '2147483648']) {
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

function requestDeadlines() {
  const callbacks = new Set();
  return {
    schedule(callback) {
      callbacks.add(callback);
      return () => callbacks.delete(callback);
    },
    expire() { for (const callback of [...callbacks]) callback(); },
    size: () => callbacks.size,
  };
}

for (const stalledPart of ['headers', 'body']) {
  test(`polling recovers from stalled ${stalledPart} without losing a draft or accepting late data`, async () => {
    const deadlines = requestDeadlines();
    const late = deferred();
    let calls = 0, stalledSignal, poll;
    const store = createMappingExpirationStore('/policy', async (_, init) => {
      calls++;
      if (calls === 2) {
        stalledSignal = init.signal;
        return stalledPart === 'headers' ? late.promise : { ok: true, json: () => late.promise };
      }
      return response(policy(calls === 1 ? 300 : 120));
    }, deadlines.schedule);
    const stop = store.watch(new EventTarget(), Object.assign(new EventTarget(), { visibilityState: 'visible' }), callback => {
      poll = callback;
      return () => {};
    });
    await tick();
    store.edit({ seconds: '90' });
    poll();
    await tick();
    poll();
    assert.equal(calls, 2);
    deadlines.expire();
    await tick();
    assert.equal(stalledSignal.aborted, true);
    assert.equal(store.getSnapshot().status, 'error');
    assert.equal(store.getSnapshot().seconds, '90');
    poll();
    await tick();
    assert.equal(calls, 3);
    assert.equal(store.getSnapshot().status, 'conflict');
    assert.equal(store.getSnapshot().latest.ttl_seconds, 120);
    assert.equal(store.getSnapshot().seconds, '90');
    late.resolve(stalledPart === 'headers' ? response(policy(600)) : policy(600));
    await tick();
    assert.equal(store.getSnapshot().latest.ttl_seconds, 120);
    stop();
    assert.equal(deadlines.size(), 0);
  });
}

test('superseding a refresh and unmounting abort outstanding requests and clear deadlines', async () => {
  const deadlines = requestDeadlines();
  const signals = [];
  const store = createMappingExpirationStore('/policy', (_, init) => {
    signals.push(init.signal);
    return new Promise(() => {});
  }, deadlines.schedule);
  const stop = store.watch(new EventTarget(), Object.assign(new EventTarget(), { visibilityState: 'visible' }));
  const refresh = store.refresh();
  await tick();
  assert.equal(signals[0].aborted, true);
  assert.equal(deadlines.size(), 1);
  stop();
  await refresh;
  assert.equal(signals[1].aborted, true);
  assert.equal(deadlines.size(), 0);
  assert.equal(store.getSnapshot().status, null);
});

test('a timed-out save retains edits, releases the form, and reconciles a server-side success', async () => {
  const deadlines = requestDeadlines();
  let current = policy(300), saveSignal;
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') {
      current = policy(90); // The server commits, but the reply never arrives.
      saveSignal = init.signal;
      return new Promise(() => {});
    }
    return response(current);
  }, deadlines.schedule);
  await store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  deadlines.expire();
  await save;
  assert.equal(saveSignal.aborted, true);
  assert.equal(store.getSnapshot().saving, false);
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, 'uncertain');
  await store.refresh();
  assert.equal(store.getSnapshot().status, 'saved');
  assert.deepEqual(store.getSnapshot().latest, current);
  assert.equal(deadlines.size(), 0);
});

test('unchanged refreshes retain save confirmations and failed-save messages', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '90' });
  await f.store.save();
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().status, 'saved');
  f.remote(policy(120));
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().status, null);

  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') return response({}, 500);
    return response(policy(300));
  });
  await store.refresh();
  store.edit({ seconds: '90' });
  await store.save();
  await store.refresh();
  assert.equal(store.getSnapshot().status, 'error');
  assert.equal(store.getSnapshot().seconds, '90');
});

for (const retry of [false, true]) {
  test(`uncertain save adopts server success and preserves newer edits (${retry ? 'retry' : 'refresh'})`, async () => {
    const deadlines = requestDeadlines();
    let current = policy(300), posts = 0;
    const store = createMappingExpirationStore('/policy', async (_, init) => {
      if (init.method === 'POST') {
        posts++;
        const body = JSON.parse(init.body);
        if (posts === 1) {
          current = policy(90);
          return new Promise(() => {});
        }
        assert.deepEqual(body.expected, policy(90));
        current = policy(body.ttl_seconds);
      }
      return response(current);
    }, deadlines.schedule);
    await store.refresh();
    store.edit({ seconds: '90' });
    const save = store.save();
    deadlines.expire();
    await save;
    store.edit({ seconds: '0' });
    if (retry) await store.save();
    else await store.refresh();
    assert.equal(store.getSnapshot().seconds, '0');
    assert.equal(store.getSnapshot().status, retry ? 'saved' : null);
    assert.deepEqual(store.getSnapshot().baseline, retry ? policy(0) : policy(90));
    assert.equal(posts, retry ? 2 : 1);
    assert.equal(deadlines.size(), 0);
  });
}

test('retry checks an uncertain save and avoids a redundant POST if already applied', async () => {
  const deadlines = requestDeadlines();
  let current = policy(300), posts = 0;
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') {
      posts++;
      current = policy(90);
      return new Promise(() => {});
    }
    return response(current);
  }, deadlines.schedule);
  await store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  deadlines.expire();
  await save;
  await store.save();
  assert.equal(posts, 1);
  assert.equal(store.getSnapshot().status, 'saved');
});

test('a save completing after retry preflight is recognized from its matching 409', async () => {
  const deadlines = requestDeadlines();
  let posts = 0;
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') {
      if (++posts === 1) return new Promise(() => {});
      return response(policy(90), 409);
    }
    return response(policy(300));
  }, deadlines.schedule);
  await store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  deadlines.expire();
  await save;
  await store.save();
  assert.equal(posts, 2);
  assert.equal(store.getSnapshot().status, 'saved');
  assert.deepEqual(store.getSnapshot().baseline, policy(90));
});

test('uncertain saves still protect drafts from genuinely different remote changes', async () => {
  const deadlines = requestDeadlines();
  let current = policy(300), posts = 0;
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') { posts++; return new Promise(() => {}); }
    return response(current);
  }, deadlines.schedule);
  await store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  deadlines.expire();
  await save;
  current = policy(120);
  await store.save();
  assert.equal(posts, 1);
  assert.equal(store.getSnapshot().seconds, '90');
  assert.equal(store.getSnapshot().status, 'conflict');
});

test('reverting a draft to the old baseline during an uncertain save is preserved', async () => {
  const deadlines = requestDeadlines();
  let current = policy(300);
  const store = createMappingExpirationStore('/policy', async (_, init) => {
    if (init.method === 'POST') { current = policy(90); return new Promise(() => {}); }
    return response(current);
  }, deadlines.schedule);
  await store.refresh();
  store.edit({ seconds: '90' });
  const save = store.save();
  deadlines.expire();
  await save;
  store.edit({ seconds: '300' });
  await store.refresh();
  assert.equal(store.getSnapshot().seconds, '300');
  assert.deepEqual(store.getSnapshot().baseline, policy(90));
  assert.equal(store.getSnapshot().status, null);
});

for (const newerDraft of [false, true]) {
  test(`a definite retry failure stays an error after an earlier timeout (${newerDraft ? 'new' : 'same'} draft)`, async () => {
    const deadlines = requestDeadlines();
    let current = policy(300), posts = 0, failReads = false;
    const store = createMappingExpirationStore('/policy', async (_, init) => {
      if (init.method === 'POST') {
        posts++;
        if (posts === 1) return new Promise(() => {});
        if (posts === 2) return response({}, 503);
        const body = JSON.parse(init.body);
        assert.deepEqual(body.expected, policy(90));
        current = policy(body.ttl_seconds);
      }
      if (failReads) throw new Error('offline');
      return response(current);
    }, deadlines.schedule);
    await store.refresh();
    store.edit({ seconds: '90' });
    const first = store.save();
    deadlines.expire();
    await first;
    assert.equal(store.getSnapshot().status, 'uncertain');
    if (newerDraft) store.edit({ seconds: '120' });
    await store.save();
    assert.equal(store.getSnapshot().status, 'error');
    await store.refresh();
    assert.equal(store.getSnapshot().status, 'error');
    failReads = true;
    await store.refresh();
    assert.equal(store.getSnapshot().status, 'error');
    failReads = false;
    // The first POST eventually commits: retain its identity for reconciliation.
    current = policy(90);
    await store.refresh();
    assert.deepEqual(store.getSnapshot().baseline, policy(90));
    assert.equal(store.getSnapshot().seconds, newerDraft ? '120' : '90');
    assert.equal(store.getSnapshot().status, newerDraft ? 'error' : 'saved');
    if (newerDraft) {
      await store.refresh();
      assert.equal(store.getSnapshot().status, 'error');
      await store.save();
      assert.equal(store.getSnapshot().status, 'saved');
    }
    assert.equal(deadlines.size(), 0);
  });
}

test('zero disables expiration and a positive lifetime re-enables it', async () => {
  const f = fixture();
  assert.equal(f.store.getSnapshot().seconds, '300');
  await f.store.refresh();
  f.store.edit({ seconds: '0' });
  await f.store.save();
  assert.deepEqual(f.current(), policy(0));
  assert.equal(f.store.getSnapshot().seconds, '0');
  assert.equal(f.store.getSnapshot().status, 'saved');
  f.store.edit({ seconds: '300' });
  await f.store.save();
  assert.deepEqual(f.posts[1].expected, policy(0));
  assert.deepEqual(f.current(), policy(300));
});

test('a zero draft survives a remote change and can be saved after conflict resolution', async () => {
  const f = fixture();
  await f.store.refresh();
  f.store.edit({ seconds: '0' });
  f.remote(policy(45));
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().seconds, '0');
  assert.equal(f.store.getSnapshot().status, 'conflict');
  await f.store.save();
  assert.equal(f.posts.length, 0);
  f.store.keepEdits();
  await f.store.save();
  assert.deepEqual(f.posts[0].expected, policy(45));
  assert.deepEqual(f.current(), policy(0));
});

test('zero from the server is a clean draft and can be changed to a positive lifetime', async () => {
  const f = fixture();
  f.remote(policy(0));
  await f.store.refresh();
  assert.equal(f.store.getSnapshot().seconds, '0');
  assert.equal(exportsUnderTest.hasExpirationEdits(f.store.getSnapshot()), false);
  f.store.edit({ seconds: '60' });
  await f.store.save();
  assert.deepEqual(f.posts[0].expected, policy(0));
  assert.deepEqual(f.current(), policy(60));
});
