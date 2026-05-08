// Frontend script to interact with the Flask backend API.
// - Sends parse requests (/api/parse)
// - Sends execute requests (/api/execute)
// - Uploads audio files (/api/upload-audio)
// - Basic browser recording using MediaRecorder (may produce webm).

const output = document.getElementById('output');
const spinner = document.getElementById('spinner');
// legacy elements removed from markup; keep functions but don't assume these elements exist
const uploadAudioBtn = document.getElementById('uploadAudioBtn');
// const player = document.getElementById('player');

let mediaRecorder;
let recordedBlobs = [];

function log(obj) {
  try {
    if (output) output.textContent = JSON.stringify(obj, null, 2);
    else console.log(obj);
  }
  catch(e) { console.log(obj); }
}

function showParsed(obj) {
  try {
    const el = document.getElementById('recParsed');
    if (el) {
      el.textContent = JSON.stringify(obj, null, 2);
    }
  } catch(e) {
    console.error('showParsed error', e);
  }
}

function setLoading(isLoading) {
  if (uploadAudioBtn) uploadAudioBtn.disabled = isLoading;
  if (spinner) {
    if (isLoading) spinner.removeAttribute('hidden'); else spinner.setAttribute('hidden', '');
  }
}

async function postJSON(path, data) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

// Text-command handlers removed (audio-only UI)

if (uploadAudioBtn) {
  uploadAudioBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('audioFile');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      showToast('info', 'Please choose an audio file first (wav recommended).');
      return;
    }
    const file = fileInput.files[0];
    try { setLoading(true); await uploadAudioFile(file); }
    finally { setLoading(false); }
  });
}

async function uploadAudioFile(file) {
  const autoExecute = document.getElementById('autoExecute')?.checked || false;
  const filename = document.getElementById('recFilename')?.value || 'data.xlsx';

  const fd = new FormData();
  fd.append('file', file, file.name);
  fd.append('auto_execute', autoExecute ? 'true' : 'false');
  fd.append('filename', filename);
  fd.append('language', document.getElementById('speechLang')?.value || 'en-US');

  const res = await fetch('/api/upload-audio', { method: 'POST', body: fd });
  const data = await res.json();
  log(data);
  showParsed(data.action || {});
  renderActionPreview(data.action);
  if (cleanedTextEl) cleanedTextEl.textContent = data.cleaned || 'No cleaned command.';
  if (cleanedTextParsedEl) cleanedTextParsedEl.textContent = data.cleaned || 'No cleaned command.';
  renderSuggestions(data.suggestions || []);
  if (data.action) addToHistory({ text: data.transcript || transcriptText.value || '', action: data.action });
  if (data.download_url) {
    showDownloadLink(data.download_url);
  }
}

// legacy inline recorder removed; improved recorder component used below

/* ---------------- Improved Recorder Component ---------------- */
// Elements
const startRec = document.getElementById('startRec');
const stopRec = document.getElementById('stopRec');
const playRec = document.getElementById('playRec');
const reRecord = document.getElementById('reRecord');
const recPlayer = document.getElementById('recPlayer');
const transcriptText = document.getElementById('transcriptText');
const recStatus = document.getElementById('recStatus');
const sendTranscript = document.getElementById('sendTranscript');
const sendAudio = document.getElementById('sendAudio');
const recParsed = document.getElementById('recParsed');
const executeParsed = document.getElementById('executeParsed');
const recDownload = document.getElementById('recDownload');

function showDownloadLink(url) {
  if (!url || !recDownload) return;
  const parts = url.split(/[\/]/);
  const fname = parts[parts.length - 1] || 'data.xlsx';
  recDownload.href = url;
  recDownload.textContent = 'Download ' + fname;
  recDownload.hidden = false;
}

if (recDownload) {
  recDownload.addEventListener('click', (e) => {
    e.preventDefault();
    const href = recDownload.getAttribute('href');
    if (!href || href === '#') {
      showToast('info', 'No generated file available yet.');
      return;
    }
    window.open(href, '_blank');
  });
}

// New UI elements
const actionPreviewEl = document.getElementById('actionPreview');
const suggestionsListEl = document.getElementById('suggestionsList');
const commandHistoryEl = document.getElementById('commandHistory');
const clearHistoryBtn = document.getElementById('clearHistory');
const cleanedTextEl = document.getElementById('cleanedText');
const cleanedTextParsedEl = document.getElementById('cleanedTextParsed');

// Confirmation modal elements
const confirmModal = document.getElementById('confirmModal');
const modalMessage = document.getElementById('modalMessage');
const modalConfirmBtn = document.getElementById('modalConfirm');
const modalCancelBtn = document.getElementById('modalCancel');
const modalBackdrop = document.getElementById('modalBackdrop');

let recStream = null;
let recRecorder = null;
let recChunks = [];
let recBlob = null;
let recognition = null;
let finalTranscript = '';

// Setup SpeechRecognition if available
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition || null;
if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = document.getElementById('speechLang')?.value || 'en-US';
  recognition.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interim += event.results[i][0].transcript;
      }
    }
    transcriptText.value = (finalTranscript + ' ' + interim).trim();
  };
  recognition.onerror = (e) => {
    console.warn('SpeechRecognition error', e);
    recStatus.textContent = 'Recognition error';
  };
  recognition.onend = () => { recStatus.textContent = 'Recognition ended'; };
}

function updateRecControls(state) {
  // states: idle, recording, recorded
  if (state === 'idle') {
    if (startRec) startRec.disabled = false;
    if (stopRec) stopRec.disabled = true;
    if (playRec) playRec.disabled = true;
    if (reRecord) reRecord.disabled = true;
    if (sendTranscript) sendTranscript.disabled = true;
    if (sendAudio) sendAudio.disabled = true;
    if (executeParsed) executeParsed.disabled = true;
    if (recStatus) recStatus.textContent = 'Idle';
  } else if (state === 'recording') {
    if (startRec) startRec.disabled = true;
    if (stopRec) stopRec.disabled = false;
    if (playRec) playRec.disabled = true;
    if (reRecord) reRecord.disabled = true;
    if (sendTranscript) sendTranscript.disabled = true;
    if (sendAudio) sendAudio.disabled = true;
    if (executeParsed) executeParsed.disabled = true;
    if (recStatus) recStatus.textContent = 'Recording...';
  } else if (state === 'recorded') {
    if (startRec) startRec.disabled = false;
    if (stopRec) stopRec.disabled = true;
    if (playRec) playRec.disabled = false;
    if (reRecord) reRecord.disabled = false;
    if (sendTranscript) sendTranscript.disabled = false;
    if (sendAudio) sendAudio.disabled = false;
    if (recStatus) recStatus.textContent = 'Recorded';
  }
}

// showConfirm: display modal and resolve true/false based on user action
function showConfirm(message) {
  // Fallback to native confirm if modal elements are missing
  if (!confirmModal || !modalMessage || !modalConfirmBtn || !modalCancelBtn) {
    return Promise.resolve(window.confirm(message));
  }

  modalMessage.textContent = message;
  confirmModal.setAttribute('aria-hidden', 'false');
  // focus confirm button
  modalConfirmBtn.focus();

  return new Promise((resolve) => {
    function cleanup() {
      confirmModal.setAttribute('aria-hidden', 'true');
      modalConfirmBtn.removeEventListener('click', onConfirm);
      modalCancelBtn.removeEventListener('click', onCancel);
      modalBackdrop.removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
    }

    function onConfirm() { cleanup(); resolve(true); }
    function onCancel() { cleanup(); resolve(false); }
    function onKey(e) { if (e.key === 'Escape') { onCancel(); } }

    modalConfirmBtn.addEventListener('click', onConfirm);
    modalCancelBtn.addEventListener('click', onCancel);
    modalBackdrop.addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
  });
}

  // Toast notification helper
  function showToast(type, message, timeout = 3500) {
    const container = document.getElementById('toastContainer');
    if (!container) return window.alert(message);
    const toast = document.createElement('div');
    toast.className = 'toast ' + (type || 'info');
    toast.innerHTML = '<svg class="t-icon" viewBox="0 0 24 24" aria-hidden><use href="icons.svg#icon-info"></use></svg><div class="t-msg">' + message + '</div>';
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    const hide = () => {
      toast.classList.remove('show');
      toast.classList.add('hide');
      setTimeout(() => { try { container.removeChild(toast); } catch(e) {} }, 220);
    };
    if (timeout > 0) setTimeout(hide, timeout);
    toast.addEventListener('click', hide);
    return toast;
  }

async function startRecording() {
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recRecorder = new MediaRecorder(recStream);
    recChunks = [];
    recRecorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
    recRecorder.onstop = () => {
      recBlob = new Blob(recChunks, { type: recChunks[0]?.type || 'audio/webm' });
      recPlayer.src = URL.createObjectURL(recBlob);
      recPlayer.hidden = false;
      updateRecControls('recorded');
    };
    recRecorder.start();
    finalTranscript = '';
    transcriptText.value = '';
    if (recognition) {
      recognition.lang = document.getElementById('speechLang')?.value || 'en-US';
      recognition.start();
    }
    updateRecControls('recording');
    if (startRec) startRec.classList.add('recording');
  } catch (e) {
    console.error('startRecording failed', e);
    recStatus.textContent = 'Microphone access denied';
  }
}

/* ----------------- Action preview, suggestions, history ----------------- */

function describeAction(action) {
  if (!action || typeof action !== 'object') return 'No action parsed.';
  if (Array.isArray(action.steps) && action.steps.length > 0) {
    const labels = action.steps.map((s, i) => `${i + 1}. ${s.type || 'unknown'}`);
    return 'Execute multiple steps: ' + labels.join(' | ');
  }
  const t = action.type || action.action || '';
  switch (t) {
    case 'create_excel': {
      const hdrs = (action.headers || []).join(', ') || 'none';
      const name = action.filename ? ` "${action.filename}.xlsx"` : '';
      return `Create Excel file${name} with headers: ${hdrs}`;
    }
    case 'add_row':
      return `Add row with values: ${(action.values || []).join(', ')}`;
    case 'create_sheet_and_add_row':
      return `Create sheet with headers ${(action.headers || []).join(', ')} and add row ${(action.values || []).join(', ')}`;
    case 'delete_row':
      return `Delete row ${action.index ?? 'unspecified'}`;
    case 'delete_column':
      return `Delete column ${action.column ?? 'unspecified'}`;
    case 'add_column':
      return `Add column "${action.name ?? 'unspecified'}"`;
    case 'write_cell':
      return `Write "${action.value ?? ''}" to cell ${action.cell ?? 'unspecified'}`;
    case 'calculate_row_sum':
      return `Calculate sum of row ${action.index ?? 'unspecified'}`;
    case 'save':
      return 'Save the workbook';
    case 'unknown':
      return `Unknown action: ${action.raw || JSON.stringify(action)}`;
    default:
      try { return JSON.stringify(action, null, 2); } catch (e) { return String(action); }
  }
}

function validateAction(action) {
  if (!action || typeof action !== 'object') return { valid: false, reason: 'No action' };
  const allowed = ['create_excel','create_sheet_and_add_row','add_row','delete_row','delete_column','add_column','write_cell','calculate_row_sum','save','create_sheet','switch_sheet','delete_sheet','list_sheets','insert_data','format_cells','calculate_sum','calculate_average','create_chart','open_excel_file','create_excel_file','multi_step','workflow'];
  if (Array.isArray(action.steps) && action.steps.length > 0) {
    const bad = action.steps.find(s => !s || typeof s !== 'object' || !allowed.includes(s.type));
    return bad ? { valid: false, reason: 'One or more steps are invalid' } : { valid: true };
  }
  const t = action.type || action.action || '';
  if (!t) return { valid: false, reason: 'Unknown action type' };
  if (!allowed.includes(t)) return { valid: false, reason: 'Action type not in allow-list' };
  return { valid: true };
}

function renderActionPreview(action) {
  const text = describeAction(action);
  if (actionPreviewEl) actionPreviewEl.textContent = text;
  if (recParsed) recParsed.textContent = JSON.stringify(action || {}, null, 2);
  const v = validateAction(action);
  if (executeParsed) {
    executeParsed.disabled = !v.valid;
    executeParsed._lastAction = action;
  }
  // Rich multi-step plan preview
  if (action && Array.isArray(action.steps) && action.steps.length > 1) {
    renderMultiStepPreview(action.steps);
  }
  if (!v.valid) {
    showToast('info', 'Action requires confirmation or refinement: ' + (v.reason || ''));
  }
}

/** Render a detailed multi-step plan preview with numbered steps */
function renderMultiStepPreview(steps) {
  if (!actionPreviewEl) return;
  const lines = ['\ud83d\udccb Multi-step plan:'];
  steps.forEach((step, i) => {
    const stype = (step.type || 'unknown').replace(/_/g, ' ');
    const details = [];
    for (const key of ['values', 'headers', 'column', 'range', 'sheet_name', 'filename']) {
      if (step[key] != null) details.push(`${key}=${JSON.stringify(step[key])}`);
    }
    const detailStr = details.length ? ` (${details.join(', ')})` : '';
    lines.push(`  ${i + 1}. ${stype}${detailStr}`);
  });
  actionPreviewEl.textContent = lines.join('\n');
}

function renderSuggestions(suggestions) {
  if (!suggestionsListEl) return;
  if (!suggestions || suggestions.length === 0) {
    suggestionsListEl.innerHTML = '<div class="muted">No suggestions available. Try rephrasing the command.</div>';
    return;
  }
  suggestionsListEl.innerHTML = '';
  suggestions.forEach(s => {
    const div = document.createElement('div');
    div.className = 's-item';
    div.textContent = s.text || s;
    div.addEventListener('click', async () => {
      transcriptText.value = s.text || s;
      showToast('info', 'Loaded suggestion into transcript');
    });
    suggestionsListEl.appendChild(div);
  });
}

// Generate lightweight suggestions from free text when parser returns unknown
function generateSuggestionsFromText(text) {
  if (!text || typeof text !== 'string') return [];
  const t = text.toLowerCase();
  const out = new Set();
  // simple heuristics
  if (t.includes('create') || t.includes('new') || t.includes('make')) out.add('create excel');
  if (t.includes('row') || t.includes('add row') || t.includes('insert row')) out.add('add row');
  if (t.match(/delete.*row|remove.*row|delete row \d+|remove row \d+/)) {
    const m = t.match(/(delete|remove) row\s*(\d+)/);
    if (m && m[2]) out.add(`delete row ${m[2]}`); else out.add('delete row 2');
  }
  if (t.includes('column') || t.includes('add column') || t.includes('insert column')) {
    // try to guess column name
    const m = t.match(/column\s+(\w+)/);
    if (m && m[1]) out.add(`add column ${m[1]}`); else out.add('add column score');
  }
  if (t.includes('sheet')) out.add('create sheet Sheet1');
  if (out.size === 0) {
    // fallback generic suggestions
    out.add('create excel'); out.add('add row'); out.add('add column score');
  }
  return Array.from(out).slice(0, 6);
}

// Show suggestion(s) as toasts so user notices them quickly
function showSuggestionToasts(suggestions) {
  if (!suggestions || suggestions.length === 0) return;
  const max = Math.min(3, suggestions.length);
  for (let i = 0; i < max; i++) {
    const s = suggestions[i];
    const toast = showToast('info', `Suggestion: ${s}`, 8000);
    if (!toast) continue;
    toast.style.cursor = 'pointer';
    toast.title = 'Click to load suggestion into transcript';
    toast.addEventListener('click', () => {
      transcriptText.value = s;
      showToast('success', 'Suggestion loaded into transcript', 2000);
      // optionally focus and allow user to edit
      transcriptText.focus();
    });
  }
}

function loadHistory() {
  try { return JSON.parse(localStorage.getItem('cmdHistory') || '[]'); } catch(e) { return []; }
}

function saveHistory(h) { localStorage.setItem('cmdHistory', JSON.stringify(h)); }

function addToHistory(entry) {
  if (!entry) return;
  const h = loadHistory();
  h.unshift({ ...entry, ts: Date.now() });
  while (h.length > 10) h.pop();
  saveHistory(h);
  renderHistory();
}

function renderHistory() {
  const h = loadHistory();
  if (!commandHistoryEl) return;
  if (!h || h.length === 0) { commandHistoryEl.textContent = 'No history yet.'; return; }
  commandHistoryEl.innerHTML = '';
  h.forEach((it, idx) => {
    const div = document.createElement('div');
    div.className = 'h-item';
    const left = document.createElement('div');
    left.textContent = (it.text || describeAction(it.action)).slice(0, 80);
    const right = document.createElement('small');
    const date = new Date(it.ts || Date.now());
    right.textContent = date.toLocaleTimeString();
    div.appendChild(left);
    div.appendChild(right);
    div.addEventListener('click', () => {
      renderActionPreview(it.action);
      transcriptText.value = it.text || '';
      showToast('info', 'Loaded history entry');
    });
    commandHistoryEl.appendChild(div);
  });
}

if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', () => { localStorage.removeItem('cmdHistory'); renderHistory(); showToast('success', 'History cleared'); });

// initialize
renderHistory();


function stopRecording() {
  if (recRecorder && recRecorder.state === 'recording') recRecorder.stop();
  if (recStream) {
    recStream.getTracks().forEach(t => t.stop());
    recStream = null;
  }
  if (recognition) try { recognition.stop(); } catch(e) {}
  if (startRec) startRec.classList.remove('recording');
  if (playRec) playRec.classList.remove('playing');
}

function reRecordAction() {
  recBlob = null; recChunks = []; recPlayer.src = ''; recPlayer.hidden = true; transcriptText.value = ''; finalTranscript = ''; updateRecControls('idle');
}

if (startRec) startRec.addEventListener('click', () => startRecording());
if (stopRec) stopRec.addEventListener('click', () => stopRecording());
if (reRecord) reRecord.addEventListener('click', () => reRecordAction());
if (playRec) playRec.addEventListener('click', () => { if (recPlayer && recPlayer.src) recPlayer.play(); });
// animate play button while audio is playing
if (recPlayer) {
  recPlayer.addEventListener('play', () => { if (playRec) playRec.classList.add('playing'); });
  recPlayer.addEventListener('pause', () => { if (playRec) playRec.classList.remove('playing'); });
  recPlayer.addEventListener('ended', () => { if (playRec) playRec.classList.remove('playing'); });
}
// Send edited transcript to /api/parse (confirm before sending)
if (sendTranscript) sendTranscript.addEventListener('click', async () => {
  const text = transcriptText.value.trim();
  if (!text) { showToast('info', 'Transcript is empty.'); return; }
  const ok = await showConfirm('Send this command to the server for parsing?\n\n"' + text + '"');
  if (!ok) return;
  setLoading(true);
  try {
    const resp = await postJSON('/api/parse', { text });
    renderActionPreview(resp.action);
    if (cleanedTextEl) cleanedTextEl.textContent = resp.cleaned || 'No cleaned command.';
    if (cleanedTextParsedEl) cleanedTextParsedEl.textContent = resp.cleaned || 'No cleaned command.';
    // if parser returned unknown/invalid action, generate suggestions from the text
    const isUnknown = !resp.action || (resp.action.type === 'unknown' || resp.action.action === 'unknown');
    if (isUnknown) {
      const gens = generateSuggestionsFromText(text);
      renderSuggestions(gens);
      showSuggestionToasts(gens);
    } else {
      renderSuggestions(resp.suggestions || []);
    }
    if (resp.action) addToHistory({ text, action: resp.action });
    log(resp);
  } catch (e) { console.error(e); showToast('error', 'Failed to send transcript'); }
  finally { setLoading(false); }
});

// Send audio file to server for transcription (fallback)
if (sendAudio) sendAudio.addEventListener('click', async () => {
  if (!recBlob) { showToast('info', 'No recording available'); return; }
  const ok = await showConfirm('Send recorded audio to server for transcription?');
  if (!ok) return;
  setLoading(true);
  speakText('Working on your Excel command');
  try {
    const fd = new FormData();
    fd.append('file', new File([recBlob], 'recording.webm', { type: recBlob.type }));
    fd.append('auto_execute', 'false');
    fd.append('filename', document.getElementById('recFilename').value || 'data.xlsx');
    fd.append('language', document.getElementById('speechLang')?.value || 'en-US');
    const res = await fetch('/api/upload-audio', { method: 'POST', body: fd });
    const data = await res.json();
    transcriptText.value = data.transcript || transcriptText.value;
    if (cleanedTextEl) cleanedTextEl.textContent = data.cleaned || 'No cleaned command.';
    if (cleanedTextParsedEl) cleanedTextParsedEl.textContent = data.cleaned || 'No cleaned command.';
    renderActionPreview(data.action);
    const isUnknown = !data.action || (data.action.type === 'unknown' || data.action.action === 'unknown');
    if (isUnknown) {
      const gens = generateSuggestionsFromText(data.transcript || transcriptText.value || '');
      renderSuggestions(gens);
      showSuggestionToasts(gens);
    } else {
      renderSuggestions(data.suggestions || []);
    }
    if (data.action) addToHistory({ text: data.transcript || transcriptText.value || '', action: data.action });
    log(data);
  } catch (e) { console.error(e); showToast('error', 'Failed to upload audio'); }
  finally { setLoading(false); }
});

// Execute parsed action
if (executeParsed) executeParsed.addEventListener('click', async () => {
  const action = executeParsed._lastAction;
  if (!action) { showToast('info', 'No parsed action available'); return; }
  const validation = validateAction(action);
  if (!validation.valid) {
    const typed = window.prompt('Action appears unsafe or unknown. To execute anyway, type CONFIRM');
    if (typed !== 'CONFIRM') { showToast('info', 'Execution cancelled'); return; }
  }
  const ok = await showConfirm('Execute this action on the Excel file?');
  if (!ok) return;
  setLoading(true);
  speakText('Working on your Excel command');
  try {
    // Use filename from parsed action if present, otherwise from the input field
    let filename = document.getElementById('recFilename').value || 'data.xlsx';
    if (action.filename) {
      filename = action.filename.replace(/\.xlsx$/i, '') + '.xlsx';
      document.getElementById('recFilename').value = filename;
    }
    const resp = await postJSON('/api/execute', { action, filename });
    log(resp);
    // If server asks for confirmation, show modal and call /api/confirm
    if (resp && resp.requires_confirmation) {
      const ok = await showConfirm(resp.message || 'Please confirm this action');
      if (!ok) {
        // notify server of cancellation
        await postJSON('/api/confirm', { confirmation_id: resp.confirmation_id, confirm: false });
        showToast('info', 'Action cancelled');
        return;
      }
      setLoading(true);
      try {
        const confirmed = await postJSON('/api/confirm', { confirmation_id: resp.confirmation_id, confirm: true });
        log(confirmed);
        if (confirmed.speech_text) speakText(confirmed.speech_text);
        if (confirmed.download_url) {
          showDownloadLink(confirmed.download_url);
          window.open(confirmed.download_url, '_blank');
        }
      } finally { setLoading(false); }
      return;
    }
    // Show success result
    if (resp.result && resp.result.message) {
      showToast('success', resp.result.message);
    }
    // Show recovery suggestions on failure
    if (resp.success === false) {
      const sugg = resp.suggestions || (resp.recovery && resp.recovery.suggestions) || [];
      if (sugg.length) {
        renderSuggestions(sugg);
        showSuggestionToasts(sugg);
      }
    }
    recParsed.textContent = JSON.stringify(action, null, 2);
    if (resp.download_url) {
      showDownloadLink(resp.download_url);
      window.open(resp.download_url, '_blank');
      showToast('success', 'File ready — click Download to save it.');
    }
    if (resp.speech_text) speakText(resp.speech_text);
  } catch (e) { console.error(e); showToast('error', 'Execute failed'); }
  finally { setLoading(false); }
});

// Speak text using browser TTS if available
function speakText(text) {
  if (!text) return;
  if ('speechSynthesis' in window) {
    try {
      const ut = new SpeechSynthesisUtterance(text);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(ut);
    } catch (e) { console.warn('TTS failed', e); }
  } else {
    // no-op or fallback
    console.log('TTS:', text);
  }
}

// Undo button handler
const undoBtn = document.getElementById('undoBtn');
const undoSteps = document.getElementById('undoSteps');
if (undoBtn) undoBtn.addEventListener('click', async () => {
  const filename = document.getElementById('recFilename')?.value || 'data.xlsx';
  const steps = parseInt(undoSteps?.value || '1', 10);
  const ok = await showConfirm(`Undo ${steps} step(s) for "${filename}"?`);
  if (!ok) return;
  setLoading(true);
  try {
    const resp = await postJSON('/api/undo', { filename, steps });
    log(resp);
    if (resp.success) {
      showToast('success', resp.message || 'Undo successful');
      if (resp.speech_text) speakText(resp.speech_text);
      if (resp.download_url && recDownload) {
        showDownloadLink(resp.download_url);
      }
    } else {
      showToast('error', resp.error || 'Undo failed');
    }
  } catch (e) { console.error(e); showToast('error', 'Undo request failed'); }
  finally { setLoading(false); }
});

// Initialize recorder UI
updateRecControls('idle');

/* ================================================================
   Workflow Panel – fetch, render, filter, click-to-insert
   ================================================================ */

// Example trigger phrases per workflow id (static map – keeps the UI
// self-contained without needing an extra backend field).
const WORKFLOW_EXAMPLES = {
  create_excel_file:    'Create an excel sheet for student scores',
  open_excel_file:      'Open the file data.xlsx',
  insert_data:          'Add row Tom 90',
  add_column:           'Add column revenue',
  delete_column:        'Delete column B',
  calculate_sum:        'Calculate sum of row 2',
  calculate_average:    'Calculate average of column B',
  format_cells:         'Format cells A1:C5 bold',
  create_chart:         'Create chart from revenue data',
  create_sales_report:  'Create a sales report',
  save:                 'Save the file',
};

const wfGrid      = document.getElementById('workflowGrid');
const wfSearch    = document.getElementById('wfSearch');
const wfCatFilter = document.getElementById('wfCategoryFilter');
let   _workflows  = [];   // cached response

function humanize(id) {
  return id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function renderWorkflows(list) {
  if (!wfGrid) return;
  wfGrid.innerHTML = '';
  if (!list || list.length === 0) {
    wfGrid.innerHTML = '<div class="wf-empty">No workflows match your filter.</div>';
    return;
  }
  list.forEach(wf => {
    const card = document.createElement('div');
    card.className = 'wf-card';
    card.title = 'Click to load example command';

    const title = document.createElement('div');
    title.className = 'wf-card-title';
    title.textContent = humanize(wf.id);

    const cat = document.createElement('div');
    cat.className = 'wf-card-cat';
    cat.setAttribute('data-cat', wf.category);
    cat.textContent = wf.category.replace(/_/g, ' ');

    const desc = document.createElement('div');
    desc.className = 'wf-card-desc';
    desc.textContent = wf.description || '';

    const flags = document.createElement('div');
    flags.className = 'wf-card-flags';
    if (wf.requires_file) {
      const f = document.createElement('span');
      f.className = 'wf-flag'; f.textContent = 'requires file';
      flags.appendChild(f);
    }
    if (wf.destructive) {
      const f = document.createElement('span');
      f.className = 'wf-flag destructive'; f.textContent = 'destructive';
      flags.appendChild(f);
    }

    const example = document.createElement('div');
    example.className = 'wf-card-example';
    const exText = WORKFLOW_EXAMPLES[wf.id] || (wf.action_types || [])[0] || '';
    example.textContent = exText ? `Example: "${exText}"` : '';

    card.append(title, cat, desc, flags, example);

    // Click → insert example into transcript input
    card.addEventListener('click', () => {
      const cmd = WORKFLOW_EXAMPLES[wf.id] || humanize(wf.id);
      if (transcriptText) {
        transcriptText.value = cmd;
        transcriptText.focus();
      }
      showToast('info', `Loaded: "${cmd}"`);
    });

    wfGrid.appendChild(card);
  });
}

function filterWorkflows() {
  const q   = (wfSearch ? wfSearch.value : '').toLowerCase();
  const cat = wfCatFilter ? wfCatFilter.value : '';
  const filtered = _workflows.filter(wf => {
    if (cat && wf.category !== cat) return false;
    if (q) {
      const haystack = (wf.id + ' ' + wf.description + ' ' + wf.category).toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    return true;
  });
  renderWorkflows(filtered);
}

async function loadWorkflows() {
  try {
    const res  = await fetch('/api/workflows');
    const data = await res.json();
    if (!data.success || !Array.isArray(data.workflows)) {
      if (wfGrid) wfGrid.innerHTML = '<div class="wf-empty">Failed to load workflows.</div>';
      return;
    }
    _workflows = data.workflows;

    // Populate category dropdown
    if (wfCatFilter) {
      const cats = [...new Set(_workflows.map(w => w.category))].sort();
      cats.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.textContent = c.replace(/_/g, ' ');
        wfCatFilter.appendChild(opt);
      });
    }

    renderWorkflows(_workflows);
  } catch (e) {
    console.error('loadWorkflows failed', e);
    if (wfGrid) wfGrid.innerHTML = '<div class="wf-empty">Could not reach backend.</div>';
  }
}

if (wfSearch)    wfSearch.addEventListener('input', filterWorkflows);
if (wfCatFilter) wfCatFilter.addEventListener('change', filterWorkflows);

// Kick off on page load
loadWorkflows();

// ── Metrics panel ──────────────────────────────────────────────

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (!data.success) return;
    document.getElementById('mCommands').textContent  = data.commands_processed;
    document.getElementById('mLatency').textContent   = data.avg_latency.toFixed(1);
    document.getElementById('mGroq').textContent      = data.groq_usage;
    document.getElementById('mOllama').textContent    = data.ollama_usage;
    document.getElementById('mHeuristic').textContent = data.heuristic_usage;
    document.getElementById('mSuccess').textContent   = (data.success_rate * 100).toFixed(1) + '%';
  } catch (e) {
    console.error('loadMetrics failed', e);
  }
}

const refreshMetricsBtn = document.getElementById('refreshMetrics');
if (refreshMetricsBtn) refreshMetricsBtn.addEventListener('click', loadMetrics);
loadMetrics();


// Helpful: attempt to download the last created file when user clicks file path shown.
// (Not implemented: backend returns file path to download via /files/<filename>)
