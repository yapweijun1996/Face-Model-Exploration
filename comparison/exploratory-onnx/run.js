'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { performance } = require('node:perf_hooks');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');
const manifest = path.resolve(root, '../multi-person/manifest.csv');
const priorPath = path.join(root, 'multi-person-results/multi-comparison-2026-10-27T09-34-58-670808Z.json');
const models = {
  edgeface: ['edgeface_xs_gamma_06.onnx', '9206e2eb13a2761d7b5b76e13016d4b9acd3fa3535a9a09939f3adacd139a5ff'],
  auraface: ['glintr100.onnx', 'a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60'],
  yunet: ['face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'],
};
function sha(buffer) { return crypto.createHash('sha256').update(buffer).digest('hex'); }
function cosine(a, b) {
  if (!a || !b || a.length !== b.length || !a.length) return null;
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] ** 2; bb += b[i] ** 2; }
  const value = dot / Math.sqrt(aa * bb);
  return Number.isFinite(value) ? value : null;
}
function tensorData(rgb) {
  if (rgb.length !== 112 * 112 * 3) throw Error('Invalid crop size');
  const data = new Float32Array(rgb.length), pixels = 112 * 112;
  for (let i = 0; i < pixels; i++) for (let c = 0; c < 3; c++) data[c * pixels + i] = (rgb[i * 3 + c] - 127.5) / 127.5;
  return data;
}
function distribution(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const quantile = q => {
    if (!sorted.length) return null;
    const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  };
  return { count: sorted.length, min: quantile(0), p25: quantile(.25), median: quantile(.5), p75: quantile(.75), max: quantile(1) };
}
function combine(prior, rows, features) {
  const lineIndex = new Map(prior.manifest.map((r, i) => [r.line, i]));
  return prior.comparisons.map(p => {
    const a = lineIndex.get(p.probe_line), b = lineIndex.get(p.enrollment_line);
    const scores = {};
    for (const name of ['edgeface', 'auraface']) scores[name] = cosine(features[name][a], features[name][b]);
    return { probe_line: p.probe_line, enrollment_line: p.enrollment_line, probe_image: p.probe_image,
      enrollment_image: p.enrollment_image, probe_person_id: p.probe_person_id,
      enrollment_person_id: p.enrollment_person_id, relation: p.relation, scores,
      prior_scores: { face_wasm: p.engines.face_wasm.score ?? null, sface: p.engines.sface.score ?? null },
      errors: { probe: rows[a] ?? null, enrollment: rows[b] ?? null } };
  });
}

async function main() {
  const priorBytes = fs.readFileSync(priorPath), prior = JSON.parse(priorBytes);
  if (sha(fs.readFileSync(manifest)) !== prior.manifest_sha256) throw Error('Manifest differs from prior results');
  for (const [name, [file, expected]] of Object.entries(models)) {
    const modelPath = path.join(root, 'vendor/models', file);
    if (!fs.existsSync(modelPath)) throw Error(`Missing ${name} model: ${modelPath}`);
    if (sha(fs.readFileSync(modelPath)) !== expected) throw Error(`SHA-256 mismatch: ${modelPath}`);
  }
  let ort;
  try { ort = require('onnxruntime-node'); } catch { throw Error('Missing onnxruntime-node@1.22.0; see README.md'); }
  const installed = require('onnxruntime-node/package.json').version;
  if (installed !== '1.22.0') throw Error(`Expected onnxruntime-node 1.22.0, found ${installed}`);
  const sessions = {};
  for (const name of ['edgeface', 'auraface']) {
    const session = await ort.InferenceSession.create(path.join(root, 'vendor/models', models[name][0]), { executionProviders: ['cpu'] });
    if (session.inputNames.length !== 1 || session.outputNames.length !== 1) throw Error(`${name}: unexpected model input/output count`);
    sessions[name] = session;
  }
  const prep = spawnSync('python', ['-B', path.join(__dirname, 'prepare.py')], {
    input: JSON.stringify({ manifest, expected_sha256: prior.manifest_sha256 }), encoding: 'utf8', maxBuffer: 32 * 1024 * 1024,
  });
  if (prep.status !== 0) throw Error(`Image preparation failed: ${prep.error?.message || prep.stderr}`);
  const prepared = JSON.parse(prep.stdout);
  if (prepared.rows.length !== prior.manifest.length) throw Error('Manifest row count changed');
  const features = { edgeface: [], auraface: [] }, rows = [];
  for (const row of prepared.rows) {
    const result = { status: row.status, error: row.error ?? null, detection_score: row.detection_score ?? null, inference_errors: {}, inference_ms: {} };
    for (const name of ['edgeface', 'auraface']) {
      let feature = null;
      if (row.status === 'detected') {
        try {
          const session = sessions[name];
          const input = new ort.Tensor('float32', tensorData(Buffer.from(row.rgb, 'base64')), [1, 3, 112, 112]);
          const start = performance.now();
          const output = (await session.run({ [session.inputNames[0]]: input }))[session.outputNames[0]];
          result.inference_ms[name] = performance.now() - start;
          feature = Array.from(output.data);
          if (feature.length !== 512 || !feature.every(Number.isFinite) || !feature.some(x => x !== 0)) throw Error('invalid 512-element embedding');
        } catch (err) { feature = null; result.inference_errors[name] = String(err.message || err); }
      }
      features[name].push(feature);
    }
    rows.push(result);
  }
  const comparisons = combine(prior, rows, features);
  const distributions = {};
  for (const name of ['edgeface', 'auraface']) {
    distributions[name] = {};
    for (const relation of ['intended same', 'intended different']) distributions[name][relation] = distribution(comparisons.filter(p => p.relation === relation).map(p => p.scores[name]));
  }
  const inference_ms = Object.fromEntries(['edgeface', 'auraface'].map(name => [name, distribution(rows.map(row => row.inference_ms[name]))]));
  const result = { description: 'Exploratory only; intended identity labels are unverified; no match thresholds.',
    metadata: { manifest_sha256: prior.manifest_sha256, prior_result_sha256: sha(priorBytes), model_sha256: Object.fromEntries(Object.entries(models).map(([k, v]) => [k, v[1]])),
      node: process.version, onnxruntime_node: installed, provider: 'cpu', python: prepared.python, opencv: prepared.opencv, numpy: prepared.numpy,
      detection: 'YuNet score 0.9, NMS 0.3, top-k 5000, maximum image dimension 640; exactly one face',
      preprocessing: 'five-point similarity alignment to ArcFace 112; RGB CHW float32 (pixel-127.5)/127.5', warnings: prepared.warnings },
    manifest: prior.manifest, rows, distributions, inference_ms, comparisons };
  const outputRoot = path.join(root, 'exploratory-runs');
  fs.mkdirSync(outputRoot, { recursive: true });
  const output = fs.mkdtempSync(path.join(outputRoot, 'run-'));
  const imageBase = path.relative(output, path.dirname(manifest)).split(path.sep).map(encodeURIComponent).join('/');
  const template = fs.readFileSync(path.join(__dirname, 'report.html'), 'utf8');
  const data = JSON.stringify({ ...result, imageBase }).replace(/</g, '\\u003c');
  fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(result, null, 2), { flag: 'wx' });
  fs.writeFileSync(path.join(output, 'report.html'), template.replace('/* DATA */', `const DATA = ${data};`), { flag: 'wx' });
  console.log(`Report: ${path.join(output, 'report.html')}`);
  if (rows.some(r => r.status !== 'detected' || Object.keys(r.inference_errors).length)) process.exitCode = 2;
}
if (require.main === module) main().catch(err => { console.error(err.message); process.exitCode = 1; });
module.exports = { cosine, tensorData, distribution, combine, sha };
