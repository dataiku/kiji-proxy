const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const source = fs.readFileSync(path.join(__dirname, '../src/components/settings/profileSaveQueue.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } });
const exported = {};
new Function('exports', compiled.outputText)(exported);
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  let shown = { id: 'default', name: 'Default', confidence: 0.25, disabled_entities: ['EMAIL'], custom_regexes: [] };
  let stored = shown;
  const requests = [];
  const save = exported.createProfileSaveQueue(() => shown, p => { shown = p; }, p => new Promise(resolve => {
    requests.push({ p, finish(ok = true) {
      if (ok) stored = p;
      resolve({ ok, status: ok ? 200 : 500, json: async () => p });
    } });
  }));
  return { save, requests, shown: () => shown, stored: () => stored, select: p => { shown = p; } };
}
test('overlapping full-profile saves cannot arrive out of order or undo email masking', async () => {
  const f = fixture();
  const a = f.save({ confidence: 0.1 });
  const b = f.save({ disabled_entities: [] });
  assert.deepEqual(f.shown().disabled_entities, []);
  assert.equal(f.requests.length, 1);
  f.requests[0].finish(); await a; await tick();
  assert.equal(f.requests.length, 2);
  assert.equal(f.requests[1].p.confidence, 0.1);
  f.requests[1].finish(); await b;
  assert.deepEqual(f.stored(), f.shown());
  assert.deepEqual(f.stored().disabled_entities, []);
});
test('failed save is not resurrected by a queued edit', async () => {
  const f = fixture();
  const a = f.save({ confidence: 0.1 });
  const b = f.save({ disabled_entities: [] });
  f.requests[0].finish(false); assert.equal((await a).ok, false); await tick();
  assert.equal(f.requests[1].p.confidence, 0.25);
  f.requests[1].finish(); await b;
  assert.deepEqual(f.stored(), f.shown());
});
test('failed last edit rolls back to confirmed state and subsequent saves still work', async () => {
  const f = fixture();
  const a = f.save({ confidence: 0.1 });
  const b = f.save({ disabled_entities: [] });
  f.requests[0].finish(); await a; await tick();
  f.requests[1].finish(false); await b;
  assert.deepEqual(f.shown(), f.stored());
  const c = f.save({ disabled_entities: [] });
  f.requests[2].finish(); await c;
  assert.deepEqual(f.stored().disabled_entities, []);
});
test('completion for an earlier profile does not replace another selected profile', async () => {
  const f = fixture();
  const a = f.save({ confidence: 0.1 });
  const other = { ...f.shown(), id: 'other' };
  f.select(other);
  f.requests[0].finish(); await a;
  assert.equal(f.shown(), other);
});
