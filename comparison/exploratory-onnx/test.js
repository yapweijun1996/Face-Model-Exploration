'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { cosine, tensorData, distribution, combine } = require('./run');

test('RGB pixels become normalized CHW input', () => {
  const rgb = Buffer.alloc(112 * 112 * 3); rgb[0] = 255; rgb[1] = 128; rgb[2] = 0;
  const data = tensorData(rgb), n = 112 * 112;
  assert.equal(data[0], 1); assert.ok(Math.abs(data[n] - 0.5 / 127.5) < 1e-7); assert.equal(data[2*n], -1);
  assert.throws(() => tensorData(Buffer.alloc(3)));
});
test('cosine and distributions reject missing or invalid values', () => {
  assert.equal(cosine([1,0],[0,1]), 0); assert.ok(Math.abs(cosine([1,1],[1,1]) - 1) < 1e-12);
  assert.equal(cosine([0],[0]), null); assert.equal(cosine([1],null), null);
  assert.deepEqual(distribution([null, 3, 1, 2]), {count:3,min:1,p25:1.5,median:2,p75:2.5,max:3});
});
test('prior scores remain unchanged and unavailable new features stay unavailable', () => {
  const prior = {manifest:[{line:2},{line:3}], comparisons:[{probe_line:3,enrollment_line:2,relation:'intended same',engines:{face_wasm:{score:0.4},sface:{score:0.8}}}]};
  const rows=[{status:'detected',inference_errors:{}},{status:'error',error:'no face',inference_errors:{}}];
  const p=combine(prior,rows,{edgeface:[[1,0],null],auraface:[[1,0],[1,0]]})[0];
  assert.deepEqual(p.prior_scores,{face_wasm:0.4,sface:0.8}); assert.equal(p.scores.edgeface,null); assert.equal(p.scores.auraface,1); assert.equal(p.errors.probe.error,'no face');
});
