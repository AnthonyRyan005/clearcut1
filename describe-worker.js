/* Clearcut "describe as prompt" worker (iPad and web app builds only).
   Runs Florence-2 PromptGen v2.0 (MIT) with the same ONNX Runtime as the rest
   of the app. The model files are part of this app (m/p-*.bin). The weights are
   8-bit; each layer is expanded just before it runs, so they stay small in memory.
   The photo never leaves the device. */
'use strict';
const ORT_BASE = new URL('vendor/ort/', self.location.href).href;
const SIDE = 768, LAYERS = 6, HEADS = 12, HEAD_DIM = 64, BEAMS = 3, NGRAM = 3;
const EOS = 2, BOS = 0;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
/* Style -> PromptGen task and the most tokens it may write. */
const TASKS = {
  prompt: ['<MORE_DETAILED_CAPTION>', 320],
  detailed: ['<DETAILED_CAPTION>', 220],
  short: ['<CAPTION>', 60],
};

let rt = null, model = null, stopRequested = false, plainOrt = false;
const post = (m) => self.postMessage(m);

async function gunzip(blob) {
  return new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
let F16 = null;
function f16Table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    F16[h] = s * (e === 0 ? f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * Math.pow(2, e - 15));
  }
  return F16;
}
/* Expand a weight pack: raw segments are copied, fp16 segments become fp32. */
function buildWeights(packBuf, man) {
  const pack = new Uint8Array(packBuf);
  const out = new ArrayBuffer(man.weightsBytes);
  const u8 = new Uint8Array(out), lut = f16Table();
  for (const seg of man.segments) {
    const o = seg[0], len = seg[1], enc = seg[2], s = seg[3], n = seg[4];
    if (enc === 0) u8.set(pack.subarray(s, s + len), o);
    else {
      const dst = new Float32Array(out, o, n);
      const lo = pack.subarray(s, s + n), hi = pack.subarray(s + n, s + 2 * n);
      for (let i = 0; i < n; i++) dst[i] = lut[lo[i] | (hi[i] << 8)];
    }
  }
  return u8;
}
/* plain: ONNX Runtime's processor-only build, for Safari (see worker.js). */
async function loadOrt(runtimeBlob, plain) {
  if (rt) return;
  plainOrt = !!plain;
  importScripts(ORT_BASE + (plainOrt ? 'ort.wasm.min.js' : 'ort.webgpu.min.js'));
  rt = self.ort;
  rt.env.wasm.numThreads = 1;
  rt.env.wasm.proxy = false;
  rt.env.wasm.wasmPaths = { mjs: ORT_BASE + (plainOrt ? 'ort-wasm-simd-threaded.js' : 'ort-wasm-simd-threaded.jsep.js') };
  rt.env.wasm.wasmBinary = await gunzip(runtimeBlob);
  rt.env.logLevel = 'error';
  if (rt.env.webgpu) rt.env.webgpu.powerPreference = 'high-performance';
}

const crossNames = [], pastNames = [], presentNames = [];
for (let l = 0; l < LAYERS; l++) for (const kv of ['key', 'value']) {
  crossNames.push(`cross_${kv}.${l}`); pastNames.push(`past_${kv}.${l}`); presentNames.push(`present_${kv}.${l}`);
}

async function createSessions(files, dev) {
  // No per-session memory arena: the two sessions then share one pool of
  // memory instead of each keeping its own high-water mark.
  const opts = (weights, extra) => Object.assign({
    executionProviders: [dev === 'gpu' && !plainOrt ? 'webgpu' : 'wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false,
    externalData: [{ path: 'weights.bin', data: weights }],
  }, extra || {});
  const onGpu = (names) => Object.fromEntries(names.map((n) => [n, 'gpu-buffer']));
  const encGraph = new Uint8Array(await gunzip(files.enc.graph));
  const enc = await rt.InferenceSession.create(encGraph, opts(buildWeights(await gunzip(files.enc.pack), files.enc.man),
    dev === 'gpu' ? { preferredOutputLocation: onGpu(crossNames) } : null));
  const decGraph = new Uint8Array(await gunzip(files.dec.graph));
  const dec = await rt.InferenceSession.create(decGraph, opts(buildWeights(await gunzip(files.dec.pack), files.dec.man),
    dev === 'gpu' ? { preferredOutputLocation: onGpu(presentNames) } : null));
  return { enc, dec };
}

async function load(m) {
  await loadOrt(m.runtime, m.plain);
  post({ type: 'status', text: 'Starting the description model' });
  const info = m.info;
  const embBuf = buildWeights(await gunzip(m.emb.pack), m.emb.man).buffer;
  const rows = info.rows, dim = info.dim, nb = dim / info.block;
  const vocab = JSON.parse(new TextDecoder().decode(await gunzip(m.vocab)));
  let device = m.device === 'gpu' && !plainOrt && self.navigator && self.navigator.gpu ? 'gpu' : 'cpu';
  let s = null;
  if (device === 'gpu') { try { s = await createSessions(m, 'gpu'); } catch (e) { device = 'cpu'; } }
  if (!s) s = await createSessions(m, 'cpu');
  model = {
    s, device, files: m, info, vocab, block: info.block, dim, nb,
    embQ: new Int8Array(embBuf, 0, rows * dim), embS: new Float32Array(embBuf, rows * dim, rows * nb),
  };
  post({ type: 'ready', device });
}

/* Token embeddings for the last token of each beam: int8 rows with one scale per 64 values. */
function embed(tokens) {
  const { embQ, embS, dim, nb, block } = model;
  const out = new Float32Array(tokens.length * dim);
  tokens.forEach((t, b) => {
    const q = t * dim, so = t * nb, o = b * dim;
    for (let j = 0; j < nb; j++) {
      const sc = embS[so + j];
      for (let k = j * block; k < (j + 1) * block; k++) out[o + k] = embQ[q + k] * sc;
    }
  });
  return out;
}

/* RGBA image -> 768 x 768 CHW, normalised, with a bicubic filter like the
   model's own image processor (stretched to a square, as in training). */
function pixels(rgba, w, h) {
  const cubic = (x) => {
    const a = -0.5; x = Math.abs(x);
    return x < 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0;
  };
  const kern = (inS, outS) => {
    const scale = inS / outS, fs = Math.max(1, scale), support = 2 * fs, out = [];
    for (let i = 0; i < outS; i++) {
      const c = (i + 0.5) * scale, lo = Math.max(0, Math.floor(c - support)), hi = Math.min(inS, Math.ceil(c + support));
      const ws = []; let sum = 0;
      for (let j = lo; j < hi; j++) { const v = cubic((j + 0.5 - c) / fs); ws.push(v); sum += v; }
      out.push([lo, Float32Array.from(ws, (v) => v / (sum || 1))]);
    }
    return out;
  };
  const kx = kern(w, SIDE), ky = kern(h, SIDE), P = SIDE * SIDE;
  const tmp = new Float32Array(SIDE * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < SIDE; x++) {
    const [lo, ws] = kx[x]; let r = 0, g = 0, b = 0;
    for (let k = 0; k < ws.length; k++) { const p = (y * w + lo + k) * 4; r += rgba[p] * ws[k]; g += rgba[p + 1] * ws[k]; b += rgba[p + 2] * ws[k]; }
    const o = (y * SIDE + x) * 3; tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
  }
  const out = new Float32Array(3 * P);
  const clamp = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
  for (let y = 0; y < SIDE; y++) {
    const [lo, ws] = ky[y];
    for (let x = 0; x < SIDE; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < ws.length; k++) { const o = ((lo + k) * SIDE + x) * 3; r += tmp[o] * ws[k]; g += tmp[o + 1] * ws[k]; b += tmp[o + 2] * ws[k]; }
      const i = y * SIDE + x;
      out[i] = (Math.round(clamp(r)) / 255 - MEAN[0]) / STD[0];
      out[P + i] = (Math.round(clamp(g)) / 255 - MEAN[1]) / STD[1];
      out[2 * P + i] = (Math.round(clamp(b)) / 255 - MEAN[2]) / STD[2];
    }
  }
  return out;
}

/* Tokens -> text, like the tokenizer's decode with skip_special_tokens. */
function decode(ids) {
  const v = model.vocab;
  let s = '';
  for (const id of ids) s += v[id] || '';
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new TextDecoder('utf-8').decode(bytes)
    .replace(/ \./g, '.').replace(/ \?/g, '?').replace(/ !/g, '!').replace(/ ,/g, ',')
    .replace(/ ' /g, "'").replace(/ n't/g, "n't").replace(/ '(m|s|ve|re)\b/g, "'$1");
}

/* Tokens that would repeat a 3-token run already in the sequence. */
function banned(seq) {
  if (seq.length < NGRAM - 1) return [];
  const a = seq[seq.length - 2], b = seq[seq.length - 1], out = [];
  for (let i = 0; i + 2 < seq.length; i++) if (seq[i] === a && seq[i + 1] === b) out.push(seq[i + 2]);
  return out;
}

const zeros = (dims) => new rt.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
const dispose = (t) => { if (t && t.location === 'gpu-buffer' && t.dispose) t.dispose(); };

/* Beam search as the model was tuned for: 3 beams, no repeated 3-token runs,
   stop once 3 finished candidates exist. Calls onBest with the leading beam. */
async function generate(cross, maxNew, onBest) {
  const { dec } = model.s;
  const B = BEAMS, V = model.info.rows;
  let seqs = Array.from({ length: B }, () => [EOS]);
  let scores = [0, ...new Array(B - 1).fill(-1e9)];
  let beamIdx = Array.from({ length: B }, (_, i) => i);
  let past = pastNames.map(() => zeros([B, HEADS, 1, HEAD_DIM]));
  const done = [];
  const lp = new Float32Array(V);
  for (let step = 0; step < maxNew && !stopRequested; step++) {
    const T1 = step + 2;
    const mask = new Float32Array(B * T1);
    for (let b = 0; b < B; b++) mask[b * T1] = -1e4;
    const feeds = {
      inputs_embeds: new rt.Tensor('float32', embed(seqs.map((s) => s[s.length - 1])), [B, 1, model.dim]),
      position: new rt.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]),
      self_mask: new rt.Tensor('float32', mask, [B, T1]),
      beam_idx: new rt.Tensor('int64', BigInt64Array.from(beamIdx.map(BigInt)), [B]),
    };
    pastNames.forEach((n, i) => { feeds[n] = past[i]; });
    crossNames.forEach((n, i) => { feeds[n] = cross[i]; });
    const r = await dec.run(feeds);
    for (const t of past) dispose(t);
    past = presentNames.map((n) => r[n]);
    const logits = r.logits.data;
    // best 2B candidates over all beams
    const top = [];   // [score, beam, token], sorted high to low
    for (let b = 0; b < B; b++) {
      if (scores[b] <= -1e8 && step > 0) continue;
      const row = b * V;
      let mx = -Infinity;
      for (let t = 0; t < V; t++) if (logits[row + t] > mx) mx = logits[row + t];
      let sum = 0;
      for (let t = 0; t < V; t++) sum += Math.exp(logits[row + t] - mx);
      const lse = mx + Math.log(sum);
      for (let t = 0; t < V; t++) lp[t] = logits[row + t] - lse;
      if (step === 0) { lp.fill(-Infinity); lp[BOS] = 0; }
      for (const t of banned(seqs[b])) lp[t] = -Infinity;
      for (let t = 0; t < V; t++) {
        const c = scores[b] + lp[t];
        if (top.length === 2 * B && c <= top[top.length - 1][0]) continue;
        let k = top.length;
        while (k > 0 && top[k - 1][0] < c) k--;
        top.splice(k, 0, [c, b, t]);
        if (top.length > 2 * B) top.pop();
      }
    }
    const nextSeqs = [], nextScores = [], nextIdx = [];
    for (let rank = 0; rank < top.length && nextSeqs.length < B; rank++) {
      const [c, b, t] = top[rank];
      if (t === EOS) {
        if (rank < B) { const seq = seqs[b].concat([EOS]); done.push([c / (seq.length - 1), seq]); }
        continue;
      }
      nextSeqs.push(seqs[b].concat([t])); nextScores.push(c); nextIdx.push(b);
    }
    if (done.length >= B || !nextSeqs.length) break;
    seqs = nextSeqs; scores = nextScores; beamIdx = nextIdx;
    if (onBest && step % 2 === 1) onBest(seqs[0]);
  }
  for (const t of past) dispose(t);
  if (!done.length) for (let b = 0; b < seqs.length; b++) done.push([scores[b] / seqs[b].length, seqs[b]]);
  done.sort((a, b) => b[0] - a[0]);
  const best = done[0][1];
  return { tokens: best, finished: best[best.length - 1] === EOS };
}

/* Sentence case, and drop sentences about watermarks or JPEG artefacts, which
   the model picked up from its training captions. */
function tidy(text, final, finished) {
  let t = text.replace(/^\s+/, '').replace(/\s+/g, ' ');
  if (final && !finished) {
    const end = Math.max(t.lastIndexOf('. '), t.lastIndexOf('.'));
    if (end > 40) t = t.slice(0, end + 1);
  }
  let parts = t.split(/(?<=[.!?])\s+/);
  parts = parts.filter((p, i) => !(i > 0 && /\bwatermark|\bjpe?g artifact/i.test(p)));
  t = parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
  return final ? t.trim() : t;
}

async function describe(m) {
  const t0 = performance.now();
  const [task, maxNew] = TASKS[m.style] || TASKS.prompt;
  const idx = model.info.prompts[task];
  post({ type: 'status', text: 'Looking at the image' });
  const enc = await model.s.enc.run({
    pixel_values: new rt.Tensor('float32', pixels(new Uint8ClampedArray(m.rgba), m.w, m.h), [1, 3, SIDE, SIDE]),
    prompt_idx: new rt.Tensor('int64', BigInt64Array.from(idx.map(BigInt)), [1, idx.length]),
  });
  const cross = crossNames.map((n) => enc[n]);
  const tEnc = performance.now() - t0;
  post({ type: 'status', text: 'Writing' });
  let last = 0;
  const r = await generate(cross, maxNew, (seq) => {
    const now = performance.now();
    if (now - last < 120) return;
    last = now;
    post({ type: 'text', text: tidy(decode(seq), false, false) });
  });
  for (const t of cross) dispose(t);
  return { text: tidy(decode(r.tokens), true, r.finished), ms: performance.now() - t0, encMs: tEnc, tokens: r.tokens.length };
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'stop') { stopRequested = true; return; }
    if (m.type === 'load') { if (!model) await load(m); else post({ type: 'ready', device: model.device }); return; }
    if (m.type === 'describe') {
      stopRequested = false;
      if (!model) throw new Error('The description model is not loaded.');
      let r;
      try { r = await describe(m); }
      catch (e) {
        if (model.device !== 'gpu') throw e;
        // The graphics chip ran out of room or failed: start again on the processor.
        post({ type: 'status', text: 'Retrying on the processor. This takes longer.' });
        model.s = await createSessions(model.files, 'cpu'); model.device = 'cpu';
        r = await describe(m);
      }
      post({ type: 'done', text: r.text, ms: r.ms, encMs: r.encMs, tokens: r.tokens, device: model.device });
    }
  } catch (e) {
    post({ type: 'error', error: (e && e.message) ? e.message : String(e) });
  }
};
