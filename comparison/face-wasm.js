'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');

const root = __dirname;
const names = [
  ...Array.from({ length: 30 }, (_, i) => `face-${String(i + 1).padStart(2, '0')}.png`),
  ...Array.from({ length: 10 }, (_, i) => `negative-${String(i + 1).padStart(2, '0')}.png`),
];
const threshold = 0.6;

async function main() {
  const tf = require('@tensorflow/tfjs');
  const wasm = require('@tensorflow/tfjs-backend-wasm');
  const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');
  const { PNG } = require('pngjs');
  const started = performance.now();
  const wasmDir = path.dirname(require.resolve('@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm'));
  wasm.setWasmPaths(`${wasmDir}${path.sep}`);
  if (!await tf.setBackend('wasm')) throw new Error('Could not initialize the WASM backend');
  await tf.ready();
  if (tf.getBackend() !== 'wasm' || faceapi.tf.getBackend() !== 'wasm' || tf.engine() !== faceapi.tf.engine()) {
    throw new Error('Face-API and TensorFlow.js do not share the WASM backend');
  }
  console.log(`tf.getBackend(): ${tf.getBackend()}`);
  const modelDir = path.join(root, 'vendor', 'models');
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(modelDir),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelDir),
    faceapi.nets.faceRecognitionNet.loadFromDisk(modelDir),
  ]);
  const rows = [];
  let enrollment = null;
  for (const name of names) {
    const start = performance.now();
    const row = { image: name, intended: name === names[0] ? 'enrollment' : name.startsWith('face-') ? 'same' : 'different' };
    let tensor;
    try {
      const image = PNG.sync.read(fs.readFileSync(path.join(root, '..', name)));
      const rgb = new Int32Array(image.width * image.height * 3);
      for (let src = 0, dst = 0; src < image.data.length; src += 4) {
        rgb[dst++] = image.data[src];
        rgb[dst++] = image.data[src + 1];
        rgb[dst++] = image.data[src + 2];
      }
      tensor = tf.tensor3d(rgb, [image.height, image.width, 3], 'int32');
      const faces = await faceapi.detectAllFaces(tensor, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
        .withFaceLandmarks().withFaceDescriptors();
      if (faces.length === 0) row.status = 'no face detected';
      else if (faces.length !== 1) { row.status = 'multiple faces detected'; row.faces = faces.length; }
      else {
        row.detection_score = faces[0].detection.score;
        if (name === names[0]) { enrollment = faces[0].descriptor; row.status = 'enrolled'; }
        else if (!enrollment) row.status = 'enrollment unavailable';
        else {
          row.score = faceapi.euclideanDistance(enrollment, faces[0].descriptor);
          if (!Number.isFinite(row.score)) throw new Error('Non-finite descriptor distance');
          row.status = row.score <= threshold ? 'matched' : 'not matched';
        }
      }
    } catch (error) {
      row.status = error.code === 'ENOENT' ? 'missing' : 'processing error';
      row.error = error.message;
    } finally {
      if (tensor) tensor.dispose();
    }
    row.ms = Math.round((performance.now() - start) * 10) / 10;
    rows.push(row);
    console.log(`${name}: ${row.status}`);
  }
  if (tf.getBackend() !== 'wasm' || faceapi.tf.getBackend() !== 'wasm') throw new Error('Backend changed during inference');
  const summary = {};
  for (const group of ['enrollment', 'same', 'different']) {
    summary[group] = {};
    for (const row of rows.filter(item => item.intended === group)) {
      summary[group][row.status] = (summary[group][row.status] || 0) + 1;
    }
  }
  const result = {
    backend: tf.getBackend(), tfjs: tf.version.tfjs, face_api: '1.7.15',
    reference: names[0], distance_threshold: threshold, elapsed_ms: Math.round((performance.now() - started) * 10) / 10,
    summary, rows,
  };
  const output = path.join(root, `results-face-wasm-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
  fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`, { flag: 'wx' });
  console.log(`Saved ${output}\n${JSON.stringify(summary, null, 2)}`);
  if (!enrollment || rows.some(row => ['missing', 'processing error'].includes(row.status))) process.exitCode = 1;
}

main().catch(error => { console.error(`Face-API WASM run failed: ${error.message}`); process.exitCode = 1; });
