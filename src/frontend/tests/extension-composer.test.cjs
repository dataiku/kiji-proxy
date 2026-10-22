const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Minimal DOM fixture: hidden composer precedes the visible, current ChatGPT
// composer, with a nested icon in its accessible-name-only Send button. `split`
// wraps it in a form whose Send button sits outside [data-composer-body].
function fixture({ legacy = false, split = false, unrelatedForm = false, response } = {}) {
  const listeners = {};
  const requests = [];
  const sent = [];
  class Element {
    constructor(tag = 'div', attrs = {}, parent = null) {
      Object.assign(this, { tag, attrs, parent, children: [], visible: true,
        isConnected: true, innerText: '', value: '', disabled: false });
      this.classList = new Set();
      this.classList.remove = this.classList.delete;
      parent?.children.push(this);
    }
    matches(selector) {
      return selector.split(',').some(part => {
        const s = part.trim();
        if (s.startsWith('#')) return this.attrs.id === s.slice(1);
        const m = s.match(/^(\w+)?(?:\[([^=\]]+)(?:="([^"]*)")?\])?$/);
        return m && (!m[1] || m[1] === this.tag) &&
          (!m[2] || (m[3] === undefined ? m[2] in this.attrs : this.attrs[m[2]] === m[3]));
      });
    }
    closest(s) { return this.matches(s) ? this : this.parent?.closest(s); }
    querySelector(s) { return this.querySelectorAll(s)[0] || null; }
    querySelectorAll(s) {
      return this.children.flatMap(c => [...(c.matches(s) ? [c] : []), ...c.querySelectorAll(s)]);
    }
    getAttribute(key) { return this.attrs[key] ?? null; }
    getClientRects() { return this.visible ? [{}] : []; }
    appendChild(c) { this.children.push(c); }
    replaceChildren(...children) { this.children = children; }
    dispatchEvent() {}
    click() { const e = emit('click', this); if (!e.defaultPrevented && this === send) sent.push(input.innerText); }
  }
  const root = new Element();
  const hidden = new Element('div', { contenteditable: 'true' }, root);
  hidden.visible = false;
  hidden.innerText = 'hidden draft';
  const form = split ? new Element('form', {}, root) : root;
  const composer = new Element('div', { 'data-composer-body': '' }, form);
  const input = new Element('div', { contenteditable: 'true' }, composer);
  input.innerText = 'Ticket QA-123456';
  const paragraph = new Element('p', {}, input);
  const send = new Element('button', legacy ? { 'data-testid': 'send-button' } : { 'aria-label': 'Send' },
    split ? form : composer);
  const icon = new Element('svg', {}, send);
  const otherScope = unrelatedForm ? new Element('form', {}, root) : root;
  const unrelated = new Element('button', { 'aria-label': 'Send' }, otherScope);
  const other = new Element('textarea', {}, otherScope);
  other.value = 'Be concise.';
  const ids = {};
  for (const name of ['overlay', 'entities', 'masked', 'sub', 'cancel', 'use-masked', 'send-anyway']) {
    ids[`kiji-pii-${name}`] = new Element();
  }
  function emit(type, target, extras = {}) {
    const event = { target, defaultPrevented: false, preventDefault() { this.defaultPrevented = true; },
      stopPropagation() {}, stopImmediatePropagation() {}, ...extras };
    listeners[type]?.(event);
    return event;
  }
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../../chrome-extension/content.js'), 'utf8'), {
    document: { readyState: 'complete', activeElement: null, body: root,
      getElementById: id => ids[id], createElement: tag => new Element(tag),
      querySelectorAll: s => root.querySelectorAll(s),
      addEventListener: (type, callback) => { listeners[type] = callback; },
      removeEventListener: type => { delete listeners[type]; } },
    chrome: { runtime: { id: 'test', sendMessage: async message => {
      requests.push(message);
      return message.type === 'check-pii-text' ? (response ? response(message) : {
        success: true, data: { pii_found: true, masked_message: 'Ticket [MASKED]',
          entities: { '[MASKED]': 'QA-123456' } } }) : undefined;
    } } },
    window: { location: { hostname: 'chatgpt.com' } },
    getComputedStyle: () => ({ visibility: 'visible' }),
    Event: class {}, console: { log() {}, warn() {}, error: console.error }, setTimeout,
  });
  return { input, hidden, paragraph, send, icon, unrelated, other, requests, sent, ids, emit,
    checks: () => requests.filter(r => r.type === 'check-pii-text') };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

for (const method of ['Send', 'Enter']) {
  test(`${method} checks the visible editor and blocks submission for review`, async () => {
    const f = fixture();
    const event = method === 'Send' ? f.emit('click', f.icon) : f.emit('keydown', f.paragraph, { key: 'Enter' });
    assert.equal(event.defaultPrevented, true);
    await settle();
    assert.equal(f.checks()[0].text, 'Ticket QA-123456');
    assert.equal(f.ids['kiji-pii-overlay'].classList.has('is-open'), true);
    assert.deepEqual(f.sent, []);
    f.ids['kiji-pii-use-masked'].onclick();
    assert.equal(f.input.innerText, 'Ticket [MASKED]');
    assert.equal(f.hidden.innerText, 'hidden draft');
    f.send.click();
    await settle();
    assert.deepEqual(f.sent, ['Ticket [MASKED]']);
    assert.equal(f.checks().length, 1);
  });
}
for (const legacy of [false, true]) {
  test(`clean text submits exactly once (${legacy ? 'legacy' : 'current'} button)`, async () => {
    const f = fixture({ legacy, response: () => ({ success: true, data: { pii_found: false } }) });
    f.send.click();
    await settle();
    assert.equal(f.checks().length, 1);
    assert.deepEqual(f.sent, ['Ticket QA-123456']);
  });
}
test('Shift+Enter, IME composition and unrelated Send are not intercepted', () => {
  const f = fixture();
  for (const extra of [{ shiftKey: true }, { isComposing: true }, { keyCode: 229 }]) {
    assert.equal(f.emit('keydown', f.input, { key: 'Enter', ...extra }).defaultPrevented, false);
  }
  assert.equal(f.emit('click', f.unrelated).defaultPrevented, false);
  assert.equal(f.checks().length, 0);
});
test('duplicate sends are blocked while checking; edited text is not auto-submitted', async () => {
  let resolve;
  const f = fixture({ response: () => new Promise(r => { resolve = r; }) });
  f.send.click();
  f.send.click();
  assert.equal(f.checks().length, 1);
  assert.deepEqual(f.sent, []);
  f.input.innerText = 'Changed while checking';
  resolve({ success: true, data: { pii_found: false } });
  await settle();
  assert.deepEqual(f.sent, []);
});
test('editing masked text requires another check', async () => {
  const f = fixture();
  f.send.click();
  await settle();
  f.ids['kiji-pii-use-masked'].onclick();
  f.input.innerText += ' QA-654321';
  f.send.click();
  await settle();
  assert.equal(f.checks().length, 2);
  assert.deepEqual(f.sent, []);
});
test('Enter accepts reviewed masked text without another check', async () => {
  const f = fixture();
  f.emit('keydown', f.input, { key: 'Enter' });
  await settle();
  f.ids['kiji-pii-use-masked'].onclick();
  assert.equal(f.emit('keydown', f.input, { key: 'Enter' }).defaultPrevented, false);
  assert.equal(f.checks().length, 1);
});
test('Cancel preserves draft; Send as-is submits once', async () => {
  const f = fixture();
  f.send.click();
  await settle();
  f.ids['kiji-pii-cancel'].onclick();
  assert.deepEqual(f.sent, []);
  assert.equal(f.input.innerText, 'Ticket QA-123456');
  f.send.click();
  await settle();
  f.ids['kiji-pii-send-anyway'].onclick();
  assert.deepEqual(f.sent, ['Ticket QA-123456']);
  assert.equal(f.checks().length, 2);
});
test('Send outside [data-composer-body] but inside the form is checked', async () => {
  const f = fixture({ split: true });
  assert.equal(f.emit('click', f.icon).defaultPrevented, true);
  await settle();
  assert.equal(f.checks()[0].text, 'Ticket QA-123456');
  assert.deepEqual(f.sent, []);
});
test('Send as-is submits when the button sits outside [data-composer-body]', async () => {
  const f = fixture({ split: true });
  f.emit('keydown', f.paragraph, { key: 'Enter' });
  await settle();
  f.ids['kiji-pii-send-anyway'].onclick();
  assert.deepEqual(f.sent, ['Ticket QA-123456']);
});
test('Enter in a textarea outside any composer is not intercepted', async () => {
  const f = fixture({ legacy: true, response: () => ({ success: true, data: { pii_found: false } }) });
  assert.equal(f.emit('keydown', f.other, { key: 'Enter' }).defaultPrevented, false);
  await settle();
  assert.equal(f.checks().length, 0);
  assert.deepEqual(f.sent, []);
});

test('an unrelated form does not intercept Enter or its Send button', async () => {
  const f = fixture({ unrelatedForm: true,
    response: () => ({ success: true, data: { pii_found: false } }) });
  let unrelatedClicks = 0;
  f.unrelated.click = () => { unrelatedClicks++; };
  assert.equal(f.emit('keydown', f.other, { key: 'Enter' }).defaultPrevented, false);
  assert.equal(f.emit('click', f.unrelated).defaultPrevented, false);
  await settle();
  assert.equal(f.checks().length, 0);
  assert.equal(unrelatedClicks, 0);
  assert.equal(f.other.value, 'Be concise.');
  assert.deepEqual(f.sent, []);
});
