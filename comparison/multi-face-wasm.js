'use strict';

// Internal worker: one JSON request on stdin, one tagged JSON response on stdout.
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const request = JSON.parse(fs.readFileSync(0, 'utf8'));
  const tf = require('@tensorflow/tfjs');
  const wasm = require('@tensorflow/tfjs-backend-wasm');
  const faceapi = require('@vladmandic/face-api/dist/face-api.node-wasm.js');
  const { PNG } = require('pngjs');
  const wasmDir = path.dirname(require.resolve('@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm'));
  wasm.setWasmPaths(`${wasmDir}${path.sep}`);
  if (!await tf.setBackend('wasm')) throw new Error('WASM initialization failed');
  await tf.ready();
  const check = () => {
    if (tf.getBackend() !== 'wasm' || faceapi.tf.getBackend() !== 'wasm' || tf.engine() !== faceapi.tf.engine()) {
      throw new Error('Face-API and TensorFlow.js do not share the WASM engine');
    }
  };
  check();
  const modelDir = path.join(__dirname, 'vendor', 'models');
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(modelDir),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelDir),
    faceapi.nets.faceRecognitionNet.loadFromDisk(modelDir),
  ]);
  const rows = [];
  for (const item of request.images) {
    const row = {};
    let tensor;
    try {
      if (!item.path) throw new Error(item.error || 'Invalid image path');
      const image = PNG.sync.read(fs.readFileSync(item.path));
      const rgb = new Int32Array(image.width * image.height * 3);
      for (let src = 0, dst = 0; src < image.data.length; src += 4) {
        rgb[dst++] = image.data[src]; rgb[dst++] = image.data[src + 1]; rgb[dst++] = image.data[src + 2];
      }
      tensor = tf.tensor3d(rgb, [image.height, image.width, 3], 'int32');
      const faces = await faceapi.detectAllFaces(tensor, new faceapi.SsdMobilenetv1Options({minConfidence: 0.5}))
        .withFaceLandmarks().withFaceDescriptors();
      if (!faces.length) row.status = 'no face detected';
      else if (faces.length !== 1) { row.status = 'multiple faces detected'; row.faces = faces.length; }
      else {
        row.status = 'detected'; row.detection_score = faces[0].detection.score;
        row.feature = Array.from(faces[0].descriptor);
        if (!row.feature.every(Number.isFinite)) throw new Error('Non-finite descriptor');
      }
    } catch (error) {
      row.status = error.code === 'ENOENT' ? 'missing' : 'processing error';
      row.error = error.message; delete row.feature;
    } finally { if (tensor) tensor.dispose(); }
    rows.push(row);
  }
  check();
  process.stdout.write(`MULTI_RESULT:${JSON.stringify({backend: tf.getBackend(), tfjs: tf.version.tfjs, node: process.version, face_api: '1.7.15', rows})}\n`);
}
main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
