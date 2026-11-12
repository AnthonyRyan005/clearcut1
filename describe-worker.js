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
/* What each look at the picture asks the model, and the most tokens it may write.
   subject: a close-up of the subject, for its details. The mixed task writes a
   caption, then tags, then short notes such as the action and which way someone
   faces, which help with people. scene: the whole picture, for the setting.
   short: one line about the whole picture. */
const TASKS = {
  subject: ['<MIXED_CAPTION_PLUS>', 448],
  scene: ['<DETAILED_CAPTION>', 220],
  short: ['<CAPTION>', 60],
};
/* Longest prompt per style, in words. People get a little more room for pose and expression. */
const MAX_WORDS = { prompt: 120, detailed: 230, short: 40 };
const PEOPLE_WORDS = { prompt: 145, detailed: 250, short: 40 };

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

/* ---------- turning the model's captions into a prompt ----------
   PromptGen writes from habits in its training captions: it adds watermarks and
   "high quality" that aren't there, guesses at things it can't see, uses stock
   openings ("A photo-realistic shoot from a profile camera angle about ...") and
   repeats itself. The captions are split into sentences, those habits are
   removed or rewritten, and only sentences that add something new are kept. */
const STOP = new Set(('a an the and or of in on at to with without from by for as is are was were be been being it its this that these those there their ' +
  'his her she he they them which who while into onto over under above below near behind front left right top bottom side middle centre center image ' +
  'picture appears appear seems giving gives give creating adds adding has have having also very slightly some few small large one two shown visible positioned').split(' '));
const words = (s) => s.toLowerCase().replace(/[^a-z0-9 -]/g, ' ').split(/\s+/).filter((w) => w.length > 2 && !STOP.has(w));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const JUNK = [
  /\bwatermark|\bjpe?g artifact|\bsignature\b|\blogo in the (bottom|top)/i,
  /\b(is|are) not (specified|visible|clear)\b|\bno other (elements|objects|details)\b/i,
  /\bno (pants|shoes|humans|accessor\w*)\b/i,
  // The model learned from tagged artwork and sometimes adds body or sexual details that aren't there.
  /\b(breasts?|cleavage|nsfw|uncensored|sexy|seductive|nude|naked|nudity|nipples?|topless|lingerie|panties|underwear|crotch|groin|saliva|drool\w*|curves|curvy|voluptuous)\b/i,
  /^the image (is|has) (of )?(a )?(high[- ]quality|high resolution|good quality)/i,
  /^the image (has|is) (a )?(simple|clean|minimalist)\b/i,
  /^(the )?overall,? (the )?(mood|atmosphere|aesthetic|image)\b/i,
  /\bgiving (it|the image) a (sense of depth|three-dimensional)/i,
];
const FIX = [
  [/\bsleeveled\b/gi, 'sleeveless'], [/\b(\d+) years old (wo)?man\b/gi, '$1-year-old $2man'],
  [/\ban? (\d+) years? old,? (?:1 ?)?(girl|boy)\b/gi, 'a $1-year-old $2'],
  // tag words that leak into captions: "a 1boy" -> "a boy", "2girls" -> "two girls"
  [/\b1(boy|girl)\b/gi, '$1'], [/\b2(boys|girls)\b/gi, 'two $1'], [/\b3(boys|girls)\b/gi, 'three $1'],
  [/\bmultiple (boys|girls)\b/gi, 'several $1'],
  [/,\s*(giving|creating|adding) (it |the image |the scene |the photo )?an? (sense|feeling) of (depth|perspective|dimension)( and (depth|perspective|dimension))?/gi, ''],
  [/\b(her|his|their) eye colou?r is (unknown|not specified|not visible|unclear),?\s*(and\s+)?/gi, ''],
  [/\bphoto realistic\b/gi, 'photo-realistic'], [/\s+,/g, ','], [/\s{2,}/g, ' '],
];
// "on a plain white background" and the like, removed when the background is described separately
const BG_PHRASE = /\s*,?\s*\b(in|on|with|against|set against|over) (a |an |the )?(plain|simple|solid|clean|blank|pure)?\s*(white|black|grey|gray|light|dark|neutral|colou?red|\w+)?\s*(background|backdrop)\b/gi;
const BACKGROUND = /\b(background|backdrop|setting|scenery|landscape|in the distance|horizon)\b|^the sky\b/i;
function sentences(text) {
  let t = text.replace(/\s+/g, ' ').trim();
  for (const [a, b] of FIX) t = t.replace(a, b);
  const parts = t.split(/(?<=[.!?])\s+/).map((p) => p.trim()).filter(Boolean).map((p) => {
    // "A photo-realistic shoot from a low angle about X" -> "Photo-realistic low angle shot of X."
    let m = p.match(/^an? ([\w -]+?) shoot from (?:an? |the )?([\w -]+?)(?: camera angle)? about (.+?)\.?$/i);
    if (m) return `${cap(m[1])} ${m[2]} shot of ${m[3]}.`;
    m = p.match(/^an? ([\w -]+?) (?:shoot|depiction) (?:about|of) (.+?)\.?$/i);
    if (m) return `${cap(m[1])} photo of ${m[2]}.`;
    p = p.replace(/^(on|in) the (middle|center|centre|top|bottom)( side)? of the (image|frame),\s*/i, '');
    p = p.replace(/^(on|in) the (left|right)( side)? of the (image|frame),\s*/i, 'On the $2, ');
    p = p.replace(/^(the|this) image (shows|depicts|features|is of)\s+/i, '');
    p = p.replace(/^the image also shows\s+/i, '');
    return cap(p.replace(/\s+\./g, '.'));
  });
  return parts.filter((p) => p.length > 3 && !JUNK.some((re) => re.test(p)));
}
const jac = (a, b) => {
  const A = new Set(words(a)), B = new Set(words(b));
  let i = 0;
  for (const x of A) if (B.has(x)) i++;
  return i / Math.max(1, A.size + B.size - i);
};
/* Does cand add at least minNew words the kept sentences don't have, without repeating one of them? */
function adds(kept, cand, minNew) {
  if (kept.some((k) => jac(k, cand) >= 0.45)) return false;
  const have = new Set(kept.flatMap(words));
  const w = words(cand), fresh = w.filter((x) => !have.has(x));
  return fresh.length >= minNew && fresh.length >= 0.3 * w.length;
}
function capWords(parts, max) {
  const out = [];
  let n = 0;
  for (const p of parts) { const c = p.split(/\s+/).length; if (n && n + c > max) break; out.push(p); n += c; }
  return out;
}
/* ---------- people: posture, movement, expression and intent ----------
   The mixed task writes a caption, then comma-separated tags, then notes such
   as "action: catching a ball, facing_direction: facing away", with blank
   lines between them. */
function parseMixed(text) {
  const blocks = text.split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const out = { caption: blocks[0] || '', tags: [], notes: {} };
  for (const b of blocks.slice(1)) {
    if (/^[a-z_]+\s*[:;]/i.test(b)) {
      for (const part of b.split(/,\s*(?=[a-z_]+\s*[:;])/i)) {
        const m = part.match(/^([a-z_]+)\s*[:;]\s*(.*)$/i);
        if (!m) continue;
        const vals = m[2].split(';').map((v) => v.trim().toLowerCase()).filter((v) => v && v !== 'na' && v !== 'none');
        if (vals.length) out.notes[m[1].toLowerCase()] = [...new Set(vals)];
      }
    } else out.tags.push(...b.split(',').map((t) => t.trim().toLowerCase().replace(/;.*/, '')).filter(Boolean));
  }
  return out;
}
const PERSON = /\b(man|men|woman|women|boy|boys|girl|girls|child|children|kid|kids|baby|toddler|person|people|player|players|lady|guy|teenager|couple|family|athlete|skier|surfer|dancer|bride|groom|student|soldier|cadet|worker|chef|catcher|batter|he|she)\b/i;
const PERSON_TAG = /^(\d+|multiple )?(boys?|girls?|others?)$|^(solo|male focus)$/;
/* Tags that say how someone holds themselves, moves, looks or feels, as words
   for the prompt, each with what counts as already said in the caption.
   Only these tags are used: the model's other tags are often wrong. */
const POSE_TAGS = [
  // posture
  ['standing', 'standing', /\bstand/], ['sitting', 'sitting', /\bsit|\bseated/], ['kneeling', 'kneeling', /\bkneel/],
  ['squatting', 'squatting', /\bsquat|\bcrouch/], ['crouching', 'crouching', /\bcrouch|\bsquat/], ['lying', 'lying down', /\blying|\blies\b/],
  ['leaning forward', 'leaning forward', /\blean/], ['leaning back', 'leaning back', /\blean/], ['bent over', 'bent over', /\bbent|\bbend/],
  ['arms up', 'arms raised', /\barms (are )?(raised|up)/], ['arm up', 'one arm raised', /\barm (is )?(raised|up)/], ['hand up', 'hand raised', /\bhands? (is |are )?(raised|up)/],
  ['crossed arms', 'arms crossed', /\barms (are )?(crossed|folded)/], ['hand on hip', 'hand on hip', /\bon (his|her|their) hips?/], ['hands on hips', 'hands on hips', /\bon (his|her|their) hips/],
  ['hand in pocket', 'hand in pocket', /\bpocket/], ['hands in pockets', 'hands in pockets', /\bpocket/],
  ['outstretched arm', 'arm outstretched', /\boutstretched|\bextended/], ['outstretched arms', 'arms outstretched', /\boutstretched|\bextended/],
  ['spread arms', 'arms spread wide', /\bspread/], ['arms behind back', 'arms behind the back', /\bbehind (his|her|their) back/],
  ['head tilt', 'head tilted', /\btilt/], ['hand on own chin', 'hand on chin', /\bchin/], ['hand on own face', 'hand on face', /\bhand on (his|her|their) face/],
  ['crossed legs', 'legs crossed', /\blegs (are )?crossed/], ['pointing', 'pointing', /\bpoint/], ['waving', 'waving', /\bwav/],
  ['salute', 'saluting', /\bsalut/], ['thumbs up', 'thumbs up', /\bthumbs?[- ]up/], ['v', 'making a peace sign', /\bpeace sign/],
  ['clenched hand', 'fist clenched', /\bclench|\bfist/], ['hug', 'hugging', /\bhug/], ['holding hands', 'holding hands', /\bholding hands/],
  // movement
  ['walking', 'walking', /\bwalk/], ['running', 'running', /\brun(ning|s)?\b/], ['jumping', 'jumping', /\bjump/], ['dancing', 'dancing', /\bdanc/],
  ['kicking', 'kicking', /\bkick/], ['throwing', 'throwing', /\bthrow/], ['swinging', 'swinging', /\bswing/], ['climbing', 'climbing', /\bclimb/],
  ['swimming', 'swimming', /\bswim/], ['skiing', 'skiing', /\bski(ing|s)?\b/], ['riding', 'riding', /\bridi?ng?\b|\brides\b/], ['falling', 'falling', /\bfall/],
  ['midair', 'in mid-air', /\bmid-?air/], ['motion blur', 'caught in motion', /\bmotion|\bmoving|\bmovement/], ['motion lines', 'caught in motion', /\bmotion|\bmoving|\bmovement/],
  ['fighting stance', 'in a fighting stance', /\bstance/],
  // what they are doing
  ['talking on the phone', 'talking on the phone', /\b(talking|speaking) on (a|the|his|her|their) (cell)?phone/], ['phone', null, null],
  ['taking picture', 'taking a photo', /\btaking a (photo|picture|selfie)/], ['selfie', 'taking a selfie', /\bselfie/],
  ['eating', 'eating', /\beat/], ['drinking', 'drinking', /\bdrink/], ['reading', 'reading', /\bread(ing|s)\b/], ['singing', 'singing', /\bsing/],
  ['playing instrument', 'playing an instrument', /\bplaying (a|an|the) /], ['writing', 'writing', /\bwriting/], ['sleeping', 'asleep', /\bsleep|\basleep/],
  // where they look
  ['looking at viewer', 'looking at the camera', /\b(looking|gazing|staring|directed|looks) (directly |straight )?(at|into) (the )?(viewer|camera|lens|them)/],
  ['looking away', 'looking away', /\blook\w* (away|off)/], ['looking to the side', 'looking to the side', /\blook\w* (to|over) (the|his|her|their) (side|left|right|shoulder)/],
  ['looking down', 'looking down', /\blook\w* down/], ['looking up', 'looking up', /\blook\w* up/], ['looking back', 'looking back', /\blook\w* back|over (his|her|their) shoulder/],
  ['looking at another', 'looking at someone else', /\blook\w* at (each other|one another|another|someone|the other)/],
  // ("closed eyes" is left out: the model tags it whenever someone looks down)
  ['wink', 'winking', /\bwink/], ['one eye closed', 'winking', /\bwink/],
  // face and feeling
  ['smile', 'smiling', /\bsmil/], ['grin', 'grinning', /\bgrin/], ['smirk', 'smirking', /\bsmirk/], ['laughing', 'laughing', /\blaugh/], [':d', 'laughing', /\blaugh/],
  ['open mouth', 'mouth open', /\bmouth (is |wide )?open|\bopen mouth|\blips (are )?parted/], ['parted lips', 'lips parted', /\blips (are )?parted|\bmouth (is |wide )?open/],
  ['frown', 'frowning', /\bfrown/], ['serious', 'serious expression', /\bserious/], ['expressionless', 'blank expression', /\bexpressionless|\bblank/],
  ['angry', 'angry', /\bangr/], ['annoyed', 'annoyed', /\bannoy/], ['sad', 'sad', /\bsad/], ['crying', 'crying', /\bcry/], ['tears', 'in tears', /\btears?\b/],
  ['surprised', 'surprised', /\bsurpris/], ['shouting', 'shouting', /\bshout|\byell/], ['screaming', 'screaming', /\bscream/], ['scared', 'scared', /\bscared|\bfright|\bafraid/],
  ['nervous', 'nervous', /\bnervous/], ['embarrassed', 'embarrassed', /\bembarrass/], ['blush', 'blushing', /\bblush/], ['happy', 'happy', /\bhapp/], ['excited', 'excited', /\bexcit/],
  ['tongue out', 'tongue out', /\btongue/], ['pout', 'pouting', /\bpout/], ['yawning', 'yawning', /\byawn/], ['thinking', 'thoughtful', /\bthink|\bthought/],
  ['determined', 'determined', /\bdetermin/], ['confused', 'confused', /\bconfus/], ['worried', 'worried', /\bworr/], ['tired', 'tired', /\btired/],
  // from where we see them
  ['from side', 'seen from the side', /\bfrom the side|\bprofile|\bside view/], ['profile', 'in profile', /\bprofile/], ['from behind', 'seen from behind', /\bfrom behind|\bback to the (camera|viewer)/],
  ['from above', 'seen from above', /\bfrom above|\bhigh angle|\bbird/], ['from below', 'seen from below', /\bfrom below|\blow angle|\blooking up at/],
];
const GAZE_AT_VIEWER = /,?\s*(and\s+)?(with\s+)?(his|her|their)\s+eyes\s+(are\s+)?(looking|directed|fixed)\s+(directly\s+)?at\s+(the\s+)?(viewer|camera)|,?\s*(and\s+)?(is\s+)?looking\s+(directly|straight)\s+(at|into)\s+(the\s+)?(viewer|camera|them)/gi;
const NEUTRAL = /^(her|his|their) (facial )?expression is neutral,?\s*(and\s+)?|,?\s*(with|and)\s+(a|an)\s+neutral\s+(facial\s+)?expression/gi;
const MANY = /^(\d+|multiple )(boys|girls|others)$/;
const GROUP = /\b(two|three|four|five|several|many|group of|couple|crowd|team|family|friends)\b/i;
const EXPRESSION_TAGS = new Set(['smile', 'grin', 'smirk', 'laughing', ':d', 'frown', 'serious', 'expressionless', 'angry', 'annoyed', 'sad', 'crying', 'tears',
  'surprised', 'shouting', 'screaming', 'scared', 'nervous', 'embarrassed', 'blush', 'happy', 'excited', 'pout', 'yawning', 'thinking', 'determined', 'confused', 'worried', 'tired']);
const VIEWS = ['seen from the side', 'in profile', 'seen from behind', 'seen from above', 'seen from below'];
/* A sentence with a clause taken out: no stray commas or leading "and". */
function tidy(p) {
  p = p.replace(/\s+,/g, ',').replace(/,\s*\./g, '.').replace(/,\s*,/g, ',').replace(/^\s*,?\s*(and|but)\s+/i, '').replace(/\s{2,}/g, ' ').trim();
  if (p && !/[.!?]$/.test(p)) p += '.';
  return cap(p);
}
/* Pose, movement, gaze and expression for a prompt about people. Returns the
   caption, cleaned, and a sentence of short phrases from the task's tags and
   notes that the caption doesn't already say. The caption's "looking directly
   at the viewer" is dropped when the tags don't agree, because the model writes
   it about nearly everyone; so is a "neutral expression" when the tags see a
   smile or a frown. With several people the tags mix up who does what, so then
   only the caption is used. */
function peopleNotes(parsed) {
  const { tags, notes } = parsed;
  const isPeople = PERSON.test(parsed.caption) || tags.some((t) => PERSON_TAG.test(t) || MANY.test(t));
  if (!isPeople) return { caption: parsed.caption, sentence: '', people: false };
  let sens = sentences(parsed.caption);
  const tagSet = new Set(tags);
  const many = tags.some((t) => MANY.test(t)) || GROUP.test(sens[0] || '');
  if (tags.length && !tagSet.has('looking at viewer') && !tagSet.has('eye contact')) sens = sens.map((p) => tidy(p.replace(GAZE_AT_VIEWER, '')));
  if (!many && tags.some((t) => EXPRESSION_TAGS.has(t))) sens = sens.map((p) => tidy(p.replace(NEUTRAL, '')));
  sens = sens.filter((p) => p.split(/\s+/).length >= 4);
  const caption = sens.join(' ');
  if (many) return { caption, sentence: '', people: true };
  const lower = caption.toLowerCase(), have = (re) => re && re.test(lower);
  // Does the caption say this already? The verb's stem and the last word both appear.
  const said = (phrase) => {
    const w = phrase.split(' '), stem = w[0].replace(/[^a-z]/g, '').slice(0, 5), noun = w[w.length - 1].replace(/[^a-z]/g, '');
    if (w[0] === 'holding' && w.length > 1) return have(new RegExp('\\b' + noun));
    return have(new RegExp('\\b' + stem)) && (w.length < 2 || have(new RegExp('\\b' + noun)));
  };
  let out = [];
  const push = (p) => { if (p && !out.includes(p) && !JUNK.some((re) => re.test(p))) out.push(p); };
  for (const a of notes.action || []) if (a.length < 40 && /^[a-z ,'-]+$/.test(a) && !said(a)) push(a);
  for (const f of notes.facing_direction || []) {
    const p = /away/.test(f) ? 'facing away' : /viewer|camera|front|forward/.test(f) ? 'facing the camera' : /left|right|side/.test(f) ? 'turned to the side' : '';
    if (p && !have(/\bfacing|\bturned/)) push(p);
  }
  for (const t of tags) {
    const hit = POSE_TAGS.find((e) => e[0] === t);
    if (hit) { if (hit[1] && !have(hit[2])) push(hit[1]); }
    else if (/^holding [a-z ]{3,24}$/.test(t) && !said(t)) push(t);
  }
  if (out.includes('seen from behind') && out.includes('looking at the camera')) {
    out = out.filter((p) => p !== 'seen from behind' && p !== 'looking at the camera');
    out.push(have(/\bshoulder/) ? 'looking at the camera' : 'looking back over the shoulder at the camera');
  }
  let view = false;
  out = out.filter((p) => !VIEWS.includes(p) || (!view && (view = true)));
  // A person can't be both: keep the first.
  for (const [a, b] of [['standing', 'sitting'], ['looking at the camera', 'looking away'], ['looking at the camera', 'eyes closed'], ['smiling', 'frowning'], ['facing away', 'facing the camera']]) {
    if (out.includes(a) && out.includes(b)) out.splice(out.indexOf(b), 1);
  }
  const list = out.slice(0, 7);
  return { caption, sentence: list.length ? `Pose and expression: ${list.join(', ')}.` : '', people: true };
}
/* Sentences about how people hold themselves, move, look, feel or what they're about to do. */
const PEOPLE_CUE = /\b(smil\w*|laugh\w*|grin\w*|express\w*|emotion\w*|happ\w*|sad|angry|serious|focus\w*|determin\w*|calm|relax\w*|excit\w*|surpris\w*|joy\w*|frown\w*|look(s|ing)?( directly| straight)? (at|into|away|down|up|back|to|over)|gaz\w*|facing|posture|pos(e|es|ed|ing)|stand\w*|sitting|seated|crouch\w*|kneel\w*|lean\w*|bend\w*|bent|walk\w*|run\w*|jump\w*|swing\w*|catch\w*|throw\w*|reach\w*|point\w*|wav\w*|danc\w*|ready to|about to|as if|trying to|preparing to|in mid-?air|motion|moving|gestur\w*|outstretched|raised)\b/i;
/* Keep to the word limit by dropping sentences from the end, those that don't satisfy keep first. */
function capKeep(parts, max, keep) {
  const n = (p) => p.split(/\s+/).length, out = parts.slice();
  let total = out.reduce((a, p) => a + n(p), 0);
  for (let i = out.length - 1; i > 0 && total > max; i--) if (!keep(out[i])) { total -= n(out[i]); out.splice(i, 1); }
  for (let i = out.length - 1; i > 0 && total > max; i--) { total -= n(out[i]); out.splice(i, 1); }
  return out;
}

/* One prompt from up to two captions. scene: the whole picture, or null.
   subject: a close-up of the subject. plain: the colour of a plain background.
   people: a sentence on pose and expression, or '', and whether there are people
   at all. With people, sentences about pose, movement and expression are the
   last to go when the prompt is too long. */
function merge(scene, subject, plain, maxWords, people = '', hasPeople = !!people) {
  const sub = sentences(subject);
  const fit = hasPeople ? (parts, max) => capKeep(parts, max, (p) => p === people || PEOPLE_CUE.test(p)) : capWords;
  const out = [];
  // The pose sentence goes after the subject's details, before the setting.
  const place = (n) => { if (people) out.splice(n, 0, people); };
  if (scene) {
    // The whole picture leads, the close-up adds details of the subject, then the rest of the scene.
    const sc = sentences(scene);
    if (sc[0]) out.push(sc[0]);
    for (const p of sub.slice(sc[0] ? 1 : 0)) if (!BACKGROUND.test(p) && adds(out, p, 3)) out.push(p);
    const n = out.length;
    // With people, the close-up is the better judge of pose and expression; the
    // whole picture's own guesses about them often disagree, so they're left out.
    for (const p of sc.slice(1)) if (!(hasPeople && PEOPLE_CUE.test(p) && !BACKGROUND.test(p)) && adds(out, p, 2)) out.push(p);
    place(n);
    return fit(out, maxWords).join(' ');
  }
  if (plain) {
    // The subject alone; the model's own words about the plain background are replaced with the real colour.
    const first = (sub[0] || '').replace(BG_PHRASE, '').replace(/\s+\./, '.').replace(/,\s*\./, '.');
    if (first) out.push(first);
    for (const p of sub.slice(1)) if (!BACKGROUND.test(p) && adds(out, p, 2)) out.push(p);
    let last = 0;
    out.forEach((p, i) => { if (PEOPLE_CUE.test(p)) last = i; });
    place(last + 1);
    return fit(out, maxWords - 4).concat(`Plain ${plain} background.`).join(' ');
  }
  for (const p of sub) if (!out.length || adds(out, p, 2)) out.push(p);
  let last = 0;   // after the last sentence about pose, movement or expression
  out.forEach((p, i) => { if (PEOPLE_CUE.test(p) && !BACKGROUND.test(p)) last = i; });
  place(last + 1);
  return fit(out, maxWords).join(' ');
}

/* One look: encode the picture with a task, then write. */
async function look(view, role, onText) {
  const [task, maxNew] = TASKS[role];
  const idx = model.info.prompts[task];
  const t0 = performance.now();
  const enc = await model.s.enc.run({
    pixel_values: new rt.Tensor('float32', pixels(new Uint8ClampedArray(view.rgba), view.w, view.h), [1, 3, SIDE, SIDE]),
    prompt_idx: new rt.Tensor('int64', BigInt64Array.from(idx.map(BigInt)), [1, idx.length]),
  });
  const cross = crossNames.map((n) => enc[n]);
  const encMs = performance.now() - t0;
  let last = 0;
  const r = await generate(cross, maxNew, (seq) => {
    const now = performance.now();
    if (!onText || now - last < 120) return;
    last = now;
    onText(decode(seq));
  });
  for (const t of cross) dispose(t);
  let text = decode(r.tokens).replace(/^\s+/, '');
  if (!r.finished) {
    // Cut short: end at the last full sentence, or the last full tag or note when it got that far.
    const blocks = text.split(/\n\s*\n/);
    if (blocks.length > 1) {
      const last = blocks.pop(), end = last.lastIndexOf(',');
      text = blocks.concat(end > 0 ? [last.slice(0, end)] : []).join('\n\n');
    } else {
      const end = text.lastIndexOf('.');
      if (end > 40) text = text.slice(0, end + 1);
    }
  }
  return { text, encMs, tokens: r.tokens.length };
}

/* views: [{ rgba, w, h, role }] with role subject, scene or short. Looks at each in turn
   (the close-up of the subject first), then combines what was written. */
async function describe(m) {
  const t0 = performance.now();
  const views = m.views, n = views.length, maxWords = MAX_WORDS[m.style] || MAX_WORDS.prompt;
  const got = {};
  let encMs = 0, tokens = 0;
  for (let i = 0; i < n && !stopRequested; i++) {
    const v = views[i];
    post({ type: 'status', text: n > 1 ? (v.role === 'subject' ? `Looking at the subject close up (1 of ${n})` : `Looking at the whole picture (2 of ${n})`) : 'Looking at the image' });
    const r = await look(v, v.role, (text) => {
      // Show the first look as it is written (its caption only); later looks only update the status.
      if (i === 0) post({ type: 'text', text: sentences(text.split(/\n\s*\n/)[0]).join(' ') });
      else post({ type: 'status', text: `Writing about the whole picture (2 of ${n})` });
    });
    got[v.role] = r.text; encMs += r.encMs; tokens += r.tokens;
  }
  let text, people = '';
  if (got.short !== undefined) text = capWords(sentences(got.short), maxWords).join(' ');
  else {
    let subject = got.subject;
    let hasPeople = false;
    if (subject !== undefined) {
      const notes = peopleNotes(parseMixed(subject));
      subject = notes.caption; people = notes.sentence; hasPeople = notes.people;
    }
    text = merge(got.scene || null, subject || got.scene || '', m.plain || null, hasPeople ? (PEOPLE_WORDS[m.style] || maxWords) : maxWords, people, hasPeople);
  }
  return { text, ms: performance.now() - t0, encMs, tokens, people: !!people };
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
