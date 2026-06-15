import './style.css';
import './app.css';

import { CancelRequest, CheckReady, ImageDataURL, OpenDocs, OpenText, Recognize, SaveText, SelectImage, SelectImages } from '../wailsjs/go/main/App';

const defaults = {
  apiUrl: localStorage.getItem('paddleocrvl.apiUrl') || 'http://127.0.0.1:8080/v1/ocr',
  apiKey: localStorage.getItem('paddleocrvl.apiKey') || '',
  task: localStorage.getItem('paddleocrvl.task') || 'ocr',
  maxNewTokens: boundedInt(localStorage.getItem('paddleocrvl.maxNewTokens'), 1024, 1, 4096),
  timeoutSecs: boundedInt(localStorage.getItem('paddleocrvl.timeoutSecs'), 600, 1, 3600),
  continueOnError: localStorage.getItem('paddleocrvl.continueOnError') === 'true',
  imagePath: '',
};
const historyKey = 'paddleocrvl.history';
const historyMaxItems = 10;
const historyMaxChars = 512 * 1024;
const historyPreviewChars = 8000;

document.querySelector('#app').innerHTML = `
  <main class="shell">
    <section class="panel controls">
      <div class="titlebar">
        <h1>PaddleOCR-VL Client</h1>
        <span id="status" class="status idle">Idle</span>
      </div>

      <label>
        <span>API URL</span>
        <input id="apiUrl" autocomplete="off" spellcheck="false" value="${escapeHtml(defaults.apiUrl)}" />
      </label>

      <label>
        <span>API Key</span>
        <input id="apiKey" type="password" autocomplete="off" spellcheck="false" value="${escapeHtml(defaults.apiKey)}" />
      </label>

      <div class="row">
        <label>
          <span>Task</span>
          <select id="task">
            ${option('ocr', 'OCR', defaults.task)}
            ${option('table', 'Table', defaults.task)}
            ${option('formula', 'Formula', defaults.task)}
            ${option('chart', 'Chart', defaults.task)}
          </select>
        </label>
        <label>
          <span>Max Tokens</span>
          <input id="maxNewTokens" type="number" min="1" max="4096" step="1" value="${defaults.maxNewTokens}" />
        </label>
        <label>
          <span>Timeout</span>
          <input id="timeoutSecs" type="number" min="1" max="3600" step="1" value="${defaults.timeoutSecs}" />
        </label>
      </div>

      <div class="filebox">
        <button id="pickImage" class="secondary" type="button">Choose Image</button>
        <button id="pickImages" class="secondary" type="button">Choose Batch</button>
        <div id="imagePath" class="path">No image selected</div>
      </div>

      <div class="preview" id="preview">
        <span>No preview</span>
      </div>
      <div id="queueList" class="queue-list"></div>
      <label class="checkline">
        <input id="continueOnError" type="checkbox" ${defaults.continueOnError ? 'checked' : ''} />
        <span>Continue batch on error</span>
      </label>

      <div class="button-row">
        <button id="checkReady" class="secondary" type="button">Check Connection</button>
        <button id="openDocs" class="secondary" type="button">API Docs</button>
        <button id="cancelRequest" class="danger" type="button" disabled>Cancel</button>
      </div>
      <div class="config-row">
        <button id="importConfig" class="secondary" type="button">Import Config</button>
        <button id="exportConfig" class="secondary" type="button">Export Config</button>
      </div>
      <button id="submit" class="primary" type="button">Upload And Recognize</button>
      <div id="error" class="error" hidden></div>
    </section>

    <section class="panel output">
      <div class="result-head">
        <h2>Result</h2>
        <div class="actions">
          <button id="copyText" class="tiny" type="button">Copy</button>
          <button id="saveText" class="tiny" type="button">Save</button>
          <button id="exportBatch" class="tiny" type="button">Export</button>
          <button id="clearAll" class="tiny" type="button">Clear</button>
        </div>
      </div>
      <div id="meta" class="meta"></div>
      <textarea id="textResult" readonly placeholder="Recognition text appears here"></textarea>
      <details>
        <summary>Raw JSON</summary>
        <pre id="rawResult">{}</pre>
      </details>
      <div class="history-head">
        <h2>Batch Results</h2>
      </div>
      <div id="batchList" class="batch-list"></div>
      <div class="history-head">
        <h2>History</h2>
        <button id="clearHistory" class="tiny" type="button">Clear</button>
      </div>
      <div id="historyList" class="history-list"></div>
    </section>
  </main>
`;

const fields = {
  apiUrl: document.getElementById('apiUrl'),
  apiKey: document.getElementById('apiKey'),
  task: document.getElementById('task'),
  maxNewTokens: document.getElementById('maxNewTokens'),
  timeoutSecs: document.getElementById('timeoutSecs'),
  continueOnError: document.getElementById('continueOnError'),
  imagePath: document.getElementById('imagePath'),
  preview: document.getElementById('preview'),
  queueList: document.getElementById('queueList'),
  pickImage: document.getElementById('pickImage'),
  pickImages: document.getElementById('pickImages'),
  checkReady: document.getElementById('checkReady'),
  openDocs: document.getElementById('openDocs'),
  cancelRequest: document.getElementById('cancelRequest'),
  importConfig: document.getElementById('importConfig'),
  exportConfig: document.getElementById('exportConfig'),
  submit: document.getElementById('submit'),
  copyText: document.getElementById('copyText'),
  saveText: document.getElementById('saveText'),
  exportBatch: document.getElementById('exportBatch'),
  clearAll: document.getElementById('clearAll'),
  clearHistory: document.getElementById('clearHistory'),
  status: document.getElementById('status'),
  error: document.getElementById('error'),
  textResult: document.getElementById('textResult'),
  rawResult: document.getElementById('rawResult'),
  meta: document.getElementById('meta'),
  batchList: document.getElementById('batchList'),
  historyList: document.getElementById('historyList'),
};

let selectedImage = defaults.imagePath;
let selectedImages = [];
let lastBatchResults = [];
let queueStatuses = [];
let busyState = false;

fields.pickImage.addEventListener('click', async () => {
  clearError();
  const path = await SelectImage();
  if (!path) return;
  selectedImage = path;
  selectedImages = [path];
  queueStatuses = ['pending'];
  fields.imagePath.textContent = path;
  await loadPreview(path);
  renderQueue();
});

fields.pickImages.addEventListener('click', async () => {
  clearError();
  const paths = await SelectImages();
  if (!paths || !paths.length) return;
  selectedImages = paths;
  selectedImage = paths[0];
  queueStatuses = paths.map(() => 'pending');
  fields.imagePath.textContent = `${paths.length} images selected`;
  await loadPreview(paths[0]);
  renderQueue();
});

fields.copyText.addEventListener('click', async () => {
  const text = currentResultText();
  if (!text.trim()) return;
  await navigator.clipboard?.writeText(text);
  setStatus('Copied', 'ok');
});

fields.saveText.addEventListener('click', async () => {
  clearError();
  const text = currentResultText();
  if (!text.trim()) return;
  try {
    const path = await SaveText('paddleocrvl-result.txt', text);
    if (path) setStatus('Saved', 'ok');
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  }
});

fields.exportBatch.addEventListener('click', async () => {
  clearError();
  if (!lastBatchResults.length) return;
  try {
    const path = await SaveText('paddleocrvl-batch-results.json', JSON.stringify(lastBatchResults, null, 2));
    if (path) setStatus('Exported', 'ok');
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  }
});

fields.clearAll.addEventListener('click', () => {
  selectedImage = '';
  selectedImages = [];
  queueStatuses = [];
  lastBatchResults = [];
  fields.imagePath.textContent = 'No image selected';
  fields.preview.innerHTML = '<span>No preview</span>';
  fields.queueList.innerHTML = '';
  fields.batchList.innerHTML = '';
  fields.textResult.value = '';
  fields.rawResult.textContent = '{}';
  fields.meta.textContent = '';
  clearError();
  setStatus('Idle', 'idle');
});

fields.clearHistory.addEventListener('click', () => {
  localStorage.removeItem(historyKey);
  renderHistory();
  setStatus('Cleared', 'idle');
});

fields.cancelRequest.addEventListener('click', async () => {
  await CancelRequest();
  setStatus('Canceling', 'busy');
});

fields.openDocs.addEventListener('click', async () => {
  clearError();
  saveConfig();
  try {
    await OpenDocs(fields.apiUrl.value.trim());
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  }
});

fields.exportConfig.addEventListener('click', async () => {
  clearError();
  saveConfig();
  try {
    const path = await SaveText('paddleocrvl-client-config.json', JSON.stringify(currentConfig(), null, 2));
    if (path) setStatus('Exported', 'ok');
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  }
});

fields.importConfig.addEventListener('click', async () => {
  clearError();
  try {
    const raw = await OpenText('Import config');
    if (!raw) return;
    applyConfig(JSON.parse(raw));
    saveConfig();
    setStatus('Imported', 'ok');
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  }
});

fields.checkReady.addEventListener('click', async () => {
  clearError();
  setBusy(true);
  saveConfig();
  try {
    const result = await CheckReady(fields.apiUrl.value.trim(), fields.apiKey.value.trim());
    fields.rawResult.textContent = formatJSON(result.raw);
    fields.meta.textContent = [
      result.backend && `backend ${result.backend}`,
      result.quantization && `quant ${result.quantization}`,
      Number.isFinite(result.availableSlots) && `slots ${result.availableSlots}/${result.concurrency}`,
      `vision ${result.visionLoaded ? 'loaded' : 'lazy'}`,
    ].filter(Boolean).join(' / ');
    setStatus(result.status || 'Ready', 'ok');
  } catch (err) {
    setError(err);
    setStatus('Failed', 'bad');
  } finally {
    setBusy(false);
  }
});

fields.submit.addEventListener('click', async () => {
  clearError();
  setBusy(true);
  saveConfig();
  const batch = [];
  try {
    const images = selectedImages.length ? selectedImages : (selectedImage ? [selectedImage] : []);
    if (!images.length) throw new Error('select an image');
    queueStatuses = images.map(() => 'pending');
    lastBatchResults = batch;
    renderBatchResults();
    renderQueue();
    let result = null;
    let ok = 0;
    let failed = 0;
    for (let i = 0; i < images.length; i += 1) {
      queueStatuses[i] = 'running';
      renderQueue();
      fields.meta.textContent = `processing ${i + 1}/${images.length} / ${fileName(images[i])}`;
      try {
        result = await Recognize(requestForImage(images[i]));
        const item = {
          status: 'ok',
          task: fields.task.value,
          imagePath: images[i],
          text: result.text || '',
          raw: result.raw || '{}',
          promptTokens: result.promptTokens,
          generatedTokens: result.generatedTokens,
          createdAt: new Date().toISOString(),
        };
        ok += 1;
        batch.push(item);
        lastBatchResults = batch;
        renderBatchResults();
        addHistory(item);
        queueStatuses[i] = 'ok';
        renderQueue();
      } catch (err) {
        queueStatuses[i] = 'error';
        renderQueue();
        failed += 1;
        batch.push({
          status: 'error',
          task: fields.task.value,
          imagePath: images[i],
          error: String(err),
          createdAt: new Date().toISOString(),
        });
        lastBatchResults = batch;
        renderBatchResults();
        if (isCancelError(err) || !fields.continueOnError.checked) throw err;
      }
    }
    lastBatchResults = batch;
    renderBatchResults();
    if (result) {
      fields.textResult.value = result.text || '';
      fields.rawResult.textContent = formatJSON(result.raw);
      fields.meta.textContent = `${ok} ok / ${failed} failed / last generated ${result.generatedTokens}`;
    } else {
      fields.textResult.value = '';
      fields.rawResult.textContent = JSON.stringify(batch, null, 2);
      fields.meta.textContent = `0 ok / ${failed} failed`;
    }
    setStatus(failed ? 'Partial' : 'Done', failed ? 'busy' : 'ok');
  } catch (err) {
    if (batch.length) {
      lastBatchResults = batch;
      renderBatchResults();
    }
    const ok = batch.filter((item) => item.status === 'ok').length;
    const failed = batch.filter((item) => item.status === 'error').length;
    if (batch.length) {
      fields.meta.textContent = `${ok} ok / ${failed} failed`;
    }
    setError(err);
    setStatus('Failed', 'bad');
  } finally {
    setBusy(false);
  }
});

function requestForImage(imagePath) {
  return {
    apiUrl: fields.apiUrl.value.trim(),
    apiKey: fields.apiKey.value.trim(),
    task: fields.task.value,
    maxNewTokens: boundedInt(fields.maxNewTokens.value, 1024, 1, 4096),
    timeoutSecs: boundedInt(fields.timeoutSecs.value, 600, 1, 3600),
    imagePath,
  };
}

renderHistory();
renderQueue();
renderBatchResults();

function saveConfig() {
  const config = currentConfig();
  localStorage.setItem('paddleocrvl.apiUrl', config.apiUrl);
  localStorage.setItem('paddleocrvl.apiKey', config.apiKey);
  localStorage.setItem('paddleocrvl.task', config.task);
  localStorage.setItem('paddleocrvl.maxNewTokens', String(config.maxNewTokens));
  localStorage.setItem('paddleocrvl.timeoutSecs', String(config.timeoutSecs));
  localStorage.setItem('paddleocrvl.continueOnError', String(config.continueOnError));
}

function currentConfig() {
  return {
    apiUrl: fields.apiUrl.value.trim(),
    apiKey: fields.apiKey.value.trim(),
    task: fields.task.value,
    maxNewTokens: boundedInt(fields.maxNewTokens.value, 1024, 1, 4096),
    timeoutSecs: boundedInt(fields.timeoutSecs.value, 600, 1, 3600),
    continueOnError: fields.continueOnError.checked,
  };
}

function applyConfig(config) {
  if (!config || typeof config !== 'object') throw new Error('invalid config file');
  fields.apiUrl.value = String(config.apiUrl || defaults.apiUrl);
  fields.apiKey.value = String(config.apiKey || '');
  fields.task.value = ['ocr', 'table', 'formula', 'chart'].includes(config.task) ? config.task : 'ocr';
  fields.maxNewTokens.value = boundedInt(config.maxNewTokens, 1024, 1, 4096);
  fields.timeoutSecs.value = boundedInt(config.timeoutSecs, 600, 1, 3600);
  fields.continueOnError.checked = Boolean(config.continueOnError);
}

async function loadPreview(path) {
  fields.preview.innerHTML = '<span>Loading preview</span>';
  try {
    const src = await ImageDataURL(path);
    fields.preview.innerHTML = `<img src="${src}" alt="Selected image preview" />`;
  } catch (err) {
    fields.preview.innerHTML = '<span>Preview unavailable</span>';
    setError(err);
  }
}

function currentResultText() {
  return fields.textResult.value || fields.rawResult.textContent || '';
}

function renderQueue() {
  if (!selectedImages.length) {
    fields.queueList.innerHTML = '';
    return;
  }
  fields.queueList.innerHTML = selectedImages.map((path, index) => `
    <div class="queue-item ${path === selectedImage ? 'active' : ''}" data-index="${index}">
      <button class="queue-name" type="button" data-action="preview" ${busyState ? 'disabled' : ''}>${index + 1}. ${escapeHtml(fileName(path))}</button>
      <span class="queue-status ${escapeHtml(queueStatuses[index] || 'pending')}">${escapeHtml(queueStatuses[index] || 'pending')}</span>
      <button class="queue-remove" type="button" data-action="remove" ${busyState ? 'disabled' : ''}>Remove</button>
    </div>
  `).join('');
  for (const item of fields.queueList.querySelectorAll('.queue-item')) {
    item.addEventListener('click', async (event) => {
      if (busyState) return;
      const index = Number(item.dataset.index);
      const action = event.target.dataset.action;
      if (action === 'remove') {
        const removed = selectedImages[index];
        selectedImages.splice(index, 1);
        queueStatuses.splice(index, 1);
        if (selectedImage === removed) {
          selectedImage = selectedImages[0] || '';
        }
        fields.imagePath.textContent = selectedImages.length ? `${selectedImages.length} images selected` : 'No image selected';
        if (selectedImage) {
          await loadPreview(selectedImage);
        } else {
          fields.preview.innerHTML = '<span>No preview</span>';
        }
        renderQueue();
        return;
      }
      selectedImage = selectedImages[index];
      await loadPreview(selectedImage);
      renderQueue();
    });
  }
}

function renderBatchResults() {
  if (!lastBatchResults.length) {
    fields.batchList.innerHTML = '<div class="empty">No batch results</div>';
    return;
  }
  fields.batchList.innerHTML = lastBatchResults.map((item, index) => `
    <button class="batch-item ${item.status === 'error' ? 'error-row' : ''}" type="button" data-index="${index}">
      <span class="batch-status ${escapeHtml(item.status || 'ok')}">${escapeHtml(item.status || 'ok')}</span>
      <span class="batch-file">${escapeHtml(fileName(item.imagePath || 'image'))}</span>
      <span class="batch-tokens">${item.status === 'error' ? 'error' : Number(item.generatedTokens || 0)}</span>
      <span class="batch-preview">${escapeHtml((item.text || item.error || item.raw || '').slice(0, 120))}</span>
    </button>
  `).join('');
  for (const button of fields.batchList.querySelectorAll('.batch-item')) {
    button.addEventListener('click', () => restoreBatchResult(Number(button.dataset.index)));
  }
}

function restoreBatchResult(index) {
  const item = lastBatchResults[index];
  if (!item) return;
  fields.textResult.value = item.text || '';
  fields.rawResult.textContent = item.status === 'error' ? JSON.stringify(item, null, 2) : formatJSON(item.raw || '{}');
  fields.meta.textContent = item.status === 'error'
    ? `error / ${fileName(item.imagePath || 'image')}`
    : `prompt ${Number(item.promptTokens || 0)} / generated ${Number(item.generatedTokens || 0)}`;
  setStatus(item.status === 'error' ? 'Error Item' : 'Loaded', item.status === 'error' ? 'bad' : 'ok');
}

function addHistory(item) {
  const items = loadHistory();
  items.unshift(compactHistoryItem(item));
  saveHistory(items);
  renderHistory();
}

function saveHistory(items) {
  const trimmed = trimHistory(items);
  try {
    localStorage.setItem(historyKey, JSON.stringify(trimmed));
  } catch {
    localStorage.setItem(historyKey, JSON.stringify(trimmed.map(minimalHistoryItem).slice(0, 3)));
  }
}

function loadHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(historyKey) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function trimHistory(items) {
  const kept = [];
  for (const item of items.slice(0, historyMaxItems)) {
    kept.push(item);
    while (JSON.stringify(kept).length > historyMaxChars && kept.length) {
      kept.pop();
    }
    if (JSON.stringify(kept).length > historyMaxChars) break;
  }
  return kept;
}

function compactHistoryItem(item) {
  const compact = { ...item };
  if (compact.text && compact.text.length > historyPreviewChars) {
    compact.text = `${compact.text.slice(0, historyPreviewChars)}\n[truncated for history]`;
  }
  if (compact.raw && compact.raw.length > historyPreviewChars) {
    compact.raw = JSON.stringify({
      truncated: true,
      note: 'Full raw JSON omitted from local history. Use export for full batch output.',
      preview: compact.raw.slice(0, historyPreviewChars),
    }, null, 2);
  }
  return compact;
}

function minimalHistoryItem(item) {
  return {
    status: item.status,
    task: item.task,
    imagePath: item.imagePath,
    text: item.text ? item.text.slice(0, 1000) : '',
    promptTokens: item.promptTokens,
    generatedTokens: item.generatedTokens,
    createdAt: item.createdAt,
  };
}

function renderHistory() {
  const items = loadHistory();
  if (!items.length) {
    fields.historyList.innerHTML = '<div class="empty">No saved runs</div>';
    return;
  }
  fields.historyList.innerHTML = items.map((item, index) => `
    <button class="history-item" type="button" data-index="${index}">
      <span class="history-title">${escapeHtml(item.task || 'ocr')} / ${escapeHtml(fileName(item.imagePath || 'image'))}</span>
      <span class="history-meta">${escapeHtml(shortTime(item.createdAt))} / generated ${Number(item.generatedTokens || 0)}</span>
      <span class="history-preview">${escapeHtml((item.text || item.raw || '').slice(0, 110))}</span>
    </button>
  `).join('');
  for (const button of fields.historyList.querySelectorAll('.history-item')) {
    button.addEventListener('click', () => restoreHistory(Number(button.dataset.index)));
  }
}

function restoreHistory(index) {
  const item = loadHistory()[index];
  if (!item) return;
  fields.textResult.value = item.text || '';
  fields.rawResult.textContent = formatJSON(item.raw || '{}');
  fields.meta.textContent = `prompt ${Number(item.promptTokens || 0)} / generated ${Number(item.generatedTokens || 0)}`;
  setStatus('Loaded', 'ok');
}

function fileName(path) {
  return String(path).split(/[\\/]/).pop() || path;
}

function shortTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString();
}

function isCancelError(err) {
  const message = String(err).toLowerCase();
  return message.includes('request canceled') || message.includes('canceling');
}

function boundedInt(value, fallback, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

function setBusy(busy) {
  busyState = busy;
  fields.submit.disabled = busy;
  fields.pickImage.disabled = busy;
  fields.pickImages.disabled = busy;
  fields.checkReady.disabled = busy;
  fields.openDocs.disabled = busy;
  fields.importConfig.disabled = busy;
  fields.exportConfig.disabled = busy;
  fields.copyText.disabled = busy;
  fields.saveText.disabled = busy;
  fields.exportBatch.disabled = busy;
  fields.clearAll.disabled = busy;
  fields.clearHistory.disabled = busy;
  fields.continueOnError.disabled = busy;
  fields.cancelRequest.disabled = !busy;
  renderQueue();
  if (busy) setStatus('Running', 'busy');
}

function setStatus(text, mode) {
  fields.status.textContent = text;
  fields.status.className = `status ${mode}`;
}

function setError(err) {
  fields.error.hidden = false;
  fields.error.textContent = String(err);
}

function clearError() {
  fields.error.hidden = true;
  fields.error.textContent = '';
}

function formatJSON(raw) {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw || '{}';
  }
}

function option(value, label, selected) {
  return `<option value="${value}" ${value === selected ? 'selected' : ''}>${label}</option>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
