const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = name => fs.readFileSync(path.join(__dirname, '../../../chrome-extension', name), 'utf8');

// Execute the production background checker and content-script submission
// handlers with transport/DOM boundaries mocked. No browser or backend needed.
function fixture(status) {
  let sent = 0;
  const notices = [];
  const bodies = [];
  const chrome = { storage: {
    sync: { get: async () => ({ backendUrl: 'http://localhost:8080', piiProfile: 'deleted' }) },
    local: { get: (_, cb) => cb({ checksTotal: 0 }), set() {} },
  }, runtime: {} };
  const context = { chrome, DEFAULT_API_BASE: 'http://localhost:8080', backendUrl: '', AbortSignal,
    console: { log() {}, error() {}, warn() {} },
    fetch: async (_, init) => {
      bodies.push(JSON.parse(init.body));
      return { ok: status === 200, status, statusText: 'Test',
        text: async () => '{"error":"unknown PII profile: deleted"}',
        json: async () => ({ pii_found: false }) };
    },
    isChecking: false, maskedTextPending: null, lastCheckError: null, blockSubmission: false,
    contextInvalidated: false, isExtensionAlive: () => true,
    window: { location: { hostname: 'chatgpt.com' } },
    getInputText: () => 'Synthetic private content',
    showToast: message => notices.push(message), triggerSubmit: () => sent++,
  };
  vm.createContext(context);
  const background = read('background.js');
  vm.runInContext(background.slice(background.indexOf('async function handlePIICheck('), background.indexOf('chrome.runtime.onMessage.addListener')), context);
  chrome.runtime.sendMessage = message => context.handlePIICheck(message.text, message.site);
  const content = read('content.js');
  vm.runInContext(content.slice(content.indexOf('  async function checkPII('), content.indexOf('  // ChatGPT mounts/unmounts')), context);
  return { bodies, notices, sent: () => sent, setStatus: s => { status = s; },
    submit: () => context.handleSubmit({ preventDefault() {}, stopPropagation() {}, stopImmediatePropagation() {} }) };
}
test('deleted selected profile blocks original-text submission on every retry', async () => {
  const f = fixture(400);
  await f.submit(); await f.submit();
  assert.equal(f.bodies[0].profile, 'deleted');
  assert.equal(f.sent(), 0);
  assert.equal(f.notices.length, 2);
  assert.match(f.notices[0], /Message not sent/);
  assert.match(f.notices[0], /Choose a valid PII profile/);
});
test('a successful check after repairing profile selection can submit', async () => {
  const f = fixture(400);
  await f.submit();
  f.setStatus(200);
  await f.submit();
  assert.equal(f.sent(), 1);
});
test('server errors retain the existing explicit unchecked-send warning', async () => {
  const f = fixture(503);
  await f.submit();
  assert.equal(f.sent(), 1);
  assert.match(f.notices[0], /Message sent without PII check/);
});
test('a second submit is blocked while a profile check is pending', async () => {
  const f = fixture(400);
  await Promise.all([f.submit(), f.submit()]);
  assert.equal(f.bodies.length, 1);
  assert.equal(f.sent(), 0);
});
