'use strict';
const names = [...Array.from({length: 30}, (_, i) => `face-${String(i + 1).padStart(2, '0')}.png`), ...Array.from({length: 10}, (_, i) => `negative-${String(i + 1).padStart(2, '0')}.png`)];
const byId = id => document.getElementById(id);
const label = name => name === 'face-01.png' ? 'enrollment' : name.startsWith('face-') ? 'intended same' : 'intended different';
let savedResults = null;
function showSaved(filter) {
  if (!savedResults) return;
  const faceRows = new Map(savedResults.face.rows.map(row => [row.image, row]));
  const sfaceRows = new Map(savedResults.sface.rows.map(row => [row.image, row]));
  const grid = byId('saved-grid'); grid.replaceChildren();
  const selected = names.slice(1).filter(name => filter === 'all' || (filter === 'same' ? name.startsWith('face-') : name.startsWith('negative-')));
  for (const name of selected) {
    const card = document.createElement('article'); card.className = 'card';
    const image = document.createElement('img'); image.src = `/images/${name}`; image.alt = `Synthetic probe ${name}`; image.loading = 'lazy'; card.append(image);
    const heading = document.createElement('h3'); heading.textContent = name; card.append(heading);
    const expected = document.createElement('p'); expected.textContent = label(name); card.append(expected);
    for (const [engine, description, row] of [['Face-API WASM', 'Distance (lower is closer; match ≤ 0.600)', faceRows.get(name)], ['SFace', 'Similarity (higher is closer; match ≥ 0.363)', sfaceRows.get(name)]]) {
      const line = document.createElement('p');
      const outcome = row?.status || 'missing result';
      line.textContent = `${engine}: ${outcome}`;
      if (name.startsWith('negative-') && outcome === 'matched') line.className = 'matched';
      else if (name.startsWith('negative-') && outcome === 'not matched') line.className = 'unmatched';
      card.append(line);
      const detail = document.createElement('p'); detail.className = 'score-detail';
      detail.textContent = `${description}: ${Number.isFinite(row?.score) ? row.score.toFixed(4) : 'unavailable'}`;
      card.append(detail);
    }
    grid.append(card);
  }
  byId('saved-status').textContent = `Showing ${selected.length} probes. Red highlights a match on an intended different-person sample.`;
  for (const button of document.querySelectorAll('[data-filter]')) button.setAttribute('aria-pressed', String(button.dataset.filter === filter));
}
function resultRate(rows) {
  const completed = rows.filter(row => row && (row.status === 'matched' || row.status === 'not matched'));
  const matched = completed.filter(row => row.status === 'matched').length;
  const rate = completed.length ? `${(matched * 100 / completed.length).toFixed(1).replace(/\.0$/, '')}%` : '—';
  return {matched, completed: completed.length, total: rows.length, rate};
}
async function loadSaved() {
  try {
    const response = await fetch('/api/saved-results', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    savedResults = await response.json();
    if (savedResults.face.backend !== 'wasm') throw new Error('Saved Face-API result is not WASM');
    for (const [id, result, engine] of [['wasm-metric', savedResults.face, 'Face-API WASM'], ['sface-metric', savedResults.sface, 'YuNet + SFace']]) {
      const indexed = new Map(result.rows.map(row => [row.image, row]));
      const negatives = resultRate(names.filter(name => name.startsWith('negative-')).map(name => indexed.get(name)));
      const positives = resultRate(names.slice(1, 30).map(name => indexed.get(name)));
      const metric = byId(id); metric.replaceChildren();
      const heading = document.createElement('h3'); heading.textContent = engine; metric.append(heading);
      for (const [label, data] of [['Intended same-person matched', positives], ['Intended different-person matched', negatives]]) {
        const line = document.createElement('p'); line.className = 'metric-line';
        const title = document.createElement('span'); title.textContent = label; line.append(title);
        const value = document.createElement('strong'); value.textContent = `${data.rate} (${data.matched}/${data.completed})`; line.append(value);
        if (data.completed !== data.total) {
          const missing = document.createElement('small'); missing.textContent = `${data.total - data.completed} of ${data.total} probes unavailable or unclassified`; line.append(missing);
        }
        metric.append(line);
      }
    }
    showSaved('different');
  } catch (error) { byId('saved-status').textContent = `Could not load saved results: ${error.message}`; }
}
for (const button of document.querySelectorAll('[data-filter]')) button.addEventListener('click', () => showSaved(button.dataset.filter));
loadSaved();
function showRows(id, rows) {
  const target = byId(id);
  target.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement('tr');
    for (const value of [row.image, label(row.image), row.error ? `${row.status}: ${row.error}` : row.status, row.score == null ? '—' : row.score.toFixed(4), row.detection_score == null ? '—' : row.detection_score.toFixed(3), row.ms]) {
      const td = document.createElement('td'); td.textContent = value; tr.append(td);
    }
    target.append(tr);
  }
}
function summary(rows, elapsed, runtime) {
  const lines = [`${runtime}; elapsed ${elapsed.toFixed(1)} ms (includes model initialization).`];
  for (const group of ['intended same', 'intended different']) {
    const selected = rows.filter(r => label(r.image) === group);
    const count = status => selected.filter(r => r.status === status).length;
    lines.push(`${group}: matched ${count('matched')}, not matched ${count('not matched')}; no face ${count('no face detected')}, multiple faces ${count('multiple faces detected')}; missing ${count('missing')}, errors ${count('processing error')}, enrollment unavailable ${count('enrollment unavailable')}.`);
  }
  const ref = rows.find(r => label(r.image) === 'enrollment');
  lines.push(`Enrollment: ${ref.status}${ref.error ? ': ' + ref.error : ''}.`);
  return lines.join('\n');
}
function loadScript(url) {
  return new Promise((resolve, reject) => { const script = document.createElement('script'); script.src = url; script.onload = resolve; script.onerror = () => reject(new Error(`Could not load ${url}. Run download.py first.`)); document.head.append(script); });
}
async function runFace() {
  const start = performance.now();
  if (!window.faceapi) await loadScript('/vendor/face-api.js');
  await faceapi.tf.setBackend('cpu'); await faceapi.tf.ready();
  if (faceapi.tf.getBackend() !== 'cpu') throw new Error('Face-API CPU backend unavailable');
  await Promise.all([faceapi.nets.ssdMobilenetv1.loadFromUri('/vendor/models'), faceapi.nets.faceLandmark68Net.loadFromUri('/vendor/models'), faceapi.nets.faceRecognitionNet.loadFromUri('/vendor/models')]);
  const rows = []; let enrollment = null;
  for (const name of names) {
    byId('progress').textContent = `Face-API: ${name}`;
    const t = performance.now(); const row = {image: name};
    try {
      const response = await fetch(`/images/${name}`);
      if (response.status === 404) row.status = 'missing';
      else {
        if (!response.ok) throw new Error(`Image HTTP ${response.status}`);
        const url = URL.createObjectURL(await response.blob());
        try {
          const image = new Image(); image.src = url; await image.decode();
          const faces = await faceapi.detectAllFaces(image, new faceapi.SsdMobilenetv1Options({minConfidence: 0.5})).withFaceLandmarks().withFaceDescriptors();
          if (faces.length === 0) row.status = 'no face detected';
          else if (faces.length !== 1) row.status = 'multiple faces detected';
          else {
            row.detection_score = faces[0].detection.score;
            if (name === names[0]) { enrollment = faces[0].descriptor; row.status = 'enrolled'; }
            else if (!enrollment) row.status = 'enrollment unavailable';
            else { row.score = faceapi.euclideanDistance(enrollment, faces[0].descriptor); row.status = row.score <= 0.6 ? 'matched' : 'not matched'; }
          }
        } finally { URL.revokeObjectURL(url); }
      }
    } catch (error) { row.status = 'processing error'; row.error = String(error.message || error); }
    row.ms = +(performance.now() - t).toFixed(1); rows.push(row); showRows('face-results', rows);
  }
  byId('face-summary').textContent = summary(rows, performance.now() - start, `Face-API 1.7.15, TF.js ${faceapi.tf.version.tfjs}, backend ${faceapi.tf.getBackend()}`);
}
async function runSface() {
  byId('progress').textContent = 'Running YuNet + SFace locally…';
  const response = await fetch('/api/sface', {cache: 'no-store'});
  if (!response.ok) throw new Error(`SFace HTTP ${response.status}: ${await response.text()}`);
  const result = await response.json();
  showRows('sface-results', result.rows);
  byId('sface-summary').textContent = summary(result.rows, result.elapsed_ms, `OpenCV ${result.opencv}, CPU`);
}
byId('run').addEventListener('click', async () => {
  byId('run').disabled = true;
  for (const id of ['face-results', 'sface-results']) byId(id).replaceChildren();
  for (const id of ['face-summary', 'sface-summary']) byId(id).textContent = 'Running…';
  let errors = 0;
  for (const [fn, id] of [[runFace, 'face-summary'], [runSface, 'sface-summary']]) {
    try { await fn(); } catch (error) { errors++; byId(id).textContent = `Pipeline failed: ${error.message || error}`; }
  }
  byId('progress').textContent = errors ? `Finished with ${errors} pipeline error(s).` : 'Finished.';
  byId('run').disabled = false;
});
