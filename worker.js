'use strict';
/* Clearcut worker: model runtime, object picking, image resampling, edge
   refinement, colour decontamination and PNG encoding. Runs off the main thread. */
const ORT_LOCAL = self.location && self.location.protocol !== 'blob:' ? new URL('vendor/ort/', self.location.href).href : null;
const ORT_SOURCES = (ORT_LOCAL ? [ORT_LOCAL] : []).concat(['https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/', 'https://unpkg.com/onnxruntime-web@1.22.0/dist/']);
let ORT_BASE = ORT_SOURCES[0];
const SIZE = 1024;      // BiRefNet and EfficientSAM input size
const LOW = 256;        // EfficientSAM mask size, also the grid used for object logic
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ortReady = false;
let plainOrt = false;   // ONNX Runtime's plain processor build (Safari), see loadOrt
let main = null;        // { session, device, graph, pack, man, weights }
let sam = null;         // { enc, dec, encDevice, encGraph, encPack, encMan }
let img = null;         // per-photo state, see storeImage()

const progress = (id, data) => self.postMessage({ id, progress: data });

self.onmessage = async (ev) => {
  const msg = ev.data;
  const fn = handlers[msg.type];
  try {
    if (!fn) throw new Error('Unknown request ' + msg.type);
    const [res, transfer] = await fn(msg);
    self.postMessage(Object.assign({ id: msg.id, ok: true }, res), transfer || []);
  } catch (err) {
    self.postMessage({ id: msg.id, ok: false, error: (err && err.message) ? err.message : String(err) });
  }
};

/* ---------- runtime + weights ---------- */
async function gunzip(blob) {
  const stream = blob.stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}
let F16 = null;
function f16Table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    let v;
    if (e === 0) v = f * 5.960464477539063e-8;
    else if (e === 31) v = f ? NaN : Infinity;
    else v = (1 + f / 1024) * Math.pow(2, e - 15);
    F16[h] = s * v;
  }
  return F16;
}
function buildWeights(packBuf, man) {
  const pack = new Uint8Array(packBuf);
  const out = new ArrayBuffer(man.weightsBytes);
  const u8 = new Uint8Array(out);
  const lut = f16Table();
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
/* plain: load ONNX Runtime's processor-only build. Safari 26 needs gigabytes of
   memory to compile the graphics-chip build, which crashes iPhones and iPads. */
async function loadOrt(runtimeBlob, plain) {
  if (ortReady) return;
  plainOrt = !!plain;
  let lastErr = null;
  for (const base of ORT_SOURCES) {
    try { importScripts(base + (plainOrt ? 'ort.wasm.min.js' : 'ort.webgpu.min.js')); ORT_BASE = base; ortReady = true; break; }
    catch (e) { lastErr = e; }
  }
  if (!ortReady) throw new Error('Could not load the model runtime (' + (lastErr && lastErr.message ? lastErr.message : 'network error') + ').');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  // Self-hosted copies ship the runtime glue as .js, because some static hosts serve .mjs with a non-script type.
  ort.env.wasm.wasmPaths = ORT_BASE === ORT_LOCAL ? { mjs: ORT_LOCAL + (plainOrt ? 'ort-wasm-simd-threaded.js' : 'ort-wasm-simd-threaded.jsep.js') } : ORT_BASE;
  ort.env.wasm.wasmBinary = await gunzip(runtimeBlob);
  ort.env.logLevel = 'error';
  if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance';
}
async function makeSession(graph, weights, dev) {
  return ort.InferenceSession.create(graph, Object.assign({
    executionProviders: [dev === 'gpu' && !plainOrt ? 'webgpu' : 'wasm'],
    graphOptimizationLevel: 'all',
    externalData: [{ path: 'weights.bin', data: weights }],
  }));
}

/* ---------- resampling ---------- */
/* Separable weights with a triangle kernel. When shrinking, the kernel widens
   with the scale factor, so the result is antialiased. */
function kernel(inSize, outSize) {
  const scale = inSize / outSize;
  const support = Math.max(1, scale);
  const taps = Math.ceil(support) * 2 + 2;
  const start = new Int32Array(outSize), count = new Int32Array(outSize);
  const w = new Float32Array(outSize * taps);
  for (let i = 0; i < outSize; i++) {
    const c = (i + 0.5) * scale;
    const lo = Math.max(0, Math.floor(c - support)), hi = Math.min(inSize, Math.ceil(c + support));
    let sum = 0, n = 0;
    for (let j = lo; j < hi && n < taps; j++) {
      const v = Math.max(0, 1 - Math.abs((j + 0.5 - c) / support));
      w[i * taps + n] = v; sum += v; n++;
    }
    if (sum > 0) for (let k = 0; k < n; k++) w[i * taps + k] /= sum;
    start[i] = lo; count[i] = n;
  }
  return { start, count, w, taps };
}
function resizePlane(src, sw, sh, dw, dh) {
  const kx = kernel(sw, dw), ky = kernel(sh, dh);
  const tmp = new Float32Array(dw * sh);
  for (let y = 0; y < sh; y++) {
    const row = y * sw, orow = y * dw;
    for (let x = 0; x < dw; x++) {
      const st = kx.start[x], n = kx.count[x], b = x * kx.taps;
      let acc = 0;
      for (let k = 0; k < n; k++) acc += src[row + st + k] * kx.w[b + k];
      tmp[orow + x] = acc;
    }
  }
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const st = ky.start[y], n = ky.count[y], b = y * ky.taps, orow = y * dw;
    for (let k = 0; k < n; k++) {
      const wt = ky.w[b + k], srow = (st + k) * dw;
      for (let x = 0; x < dw; x++) out[orow + x] += tmp[srow + x] * wt;
    }
  }
  return out;
}
/* RGBA bytes -> 1024x1024 CHW tensor. normalise=true gives the BiRefNet input,
   false gives plain 0..1 values for EfficientSAM (it normalises internally). */
function toModelInput(rgba, w, h, normalise) {
  const kx = kernel(w, SIZE), ky = kernel(h, SIZE);
  const tmp = new Float32Array(SIZE * h * 3);
  for (let y = 0; y < h; y++) {
    const row = y * w * 4, orow = y * SIZE * 3;
    for (let x = 0; x < SIZE; x++) {
      const st = kx.start[x], n = kx.count[x], b = x * kx.taps;
      let r = 0, g = 0, bl = 0;
      for (let k = 0; k < n; k++) {
        const p = row + (st + k) * 4, wt = kx.w[b + k];
        r += rgba[p] * wt; g += rgba[p + 1] * wt; bl += rgba[p + 2] * wt;
      }
      const o = orow + x * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = bl;
    }
  }
  const plane = SIZE * SIZE;
  const out = new Float32Array(3 * plane);
  const acc = new Float32Array(SIZE * 3);
  for (let y = 0; y < SIZE; y++) {
    acc.fill(0);
    const st = ky.start[y], n = ky.count[y], b = y * ky.taps;
    for (let k = 0; k < n; k++) {
      const wt = ky.w[b + k], srow = (st + k) * SIZE * 3;
      for (let i = 0; i < SIZE * 3; i++) acc[i] += tmp[srow + i] * wt;
    }
    const orow = y * SIZE;
    for (let x = 0; x < SIZE; x++) {
      for (let c = 0; c < 3; c++) {
        const v = acc[x * 3 + c] / 255;
        out[c * plane + orow + x] = normalise ? (v - MEAN[c]) / STD[c] : v;
      }
    }
  }
  return out;
}
function boxMean(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = 0; x <= Math.min(r, w - 1); x++) s += src[row + x];
    for (let x = 0; x < w; x++) {
      const lo = x - r < 0 ? 0 : x - r, hi = x + r > w - 1 ? w - 1 : x + r;
      tmp[row + x] = s / (hi - lo + 1);
      if (x + r + 1 < w) s += src[row + x + r + 1];
      if (x - r >= 0) s -= src[row + x - r];
    }
  }
  const out = new Float32Array(w * h);
  const col = new Float64Array(w);
  for (let y = 0; y <= Math.min(r, h - 1); y++) { const row = y * w; for (let x = 0; x < w; x++) col[x] += tmp[row + x]; }
  for (let y = 0; y < h; y++) {
    const lo = y - r < 0 ? 0 : y - r, hi = y + r > h - 1 ? h - 1 : y + r;
    const inv = 1 / (hi - lo + 1), row = y * w;
    for (let x = 0; x < w; x++) out[row + x] = col[x] * inv;
    if (y + r + 1 < h) { const a = (y + r + 1) * w; for (let x = 0; x < w; x++) col[x] += tmp[a + x]; }
    if (y - r >= 0) { const d = (y - r) * w; for (let x = 0; x < w; x++) col[x] -= tmp[d + x]; }
  }
  return out;
}
function sampleBilinear(arr, w, h, fx, fy) {
  const x = fx < 0 ? 0 : (fx > w - 1 ? w - 1 : fx);
  const y = fy < 0 ? 0 : (fy > h - 1 ? h - 1 : fy);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
  const ax = x - x0, ay = y - y0;
  const a = arr[y0 * w + x0], b = arr[y0 * w + x1], c = arr[y1 * w + x0], d = arr[y1 * w + x1];
  return (a * (1 - ax) + b * ax) * (1 - ay) + (c * (1 - ax) + d * ax) * ay;
}
function grayOf(rgba, n) {
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) { const p = i * 4; g[i] = (0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2]) / 255; }
  return g;
}

/* Fast guided filter with a grey guide, applied only inside a band of
   uncertain pixels. Snaps soft or coarse edges to real edges in the photo.
   bandOf(P, ws, hs) may return a custom low-resolution band. */
function guidedRefine(alpha, gray, w, h, rFull, bandOf) {
  const scale = Math.max(w, h) / SIZE;
  const s = Math.max(1, Math.floor(scale));
  const ws = Math.max(1, Math.round(w / s)), hs = Math.max(1, Math.round(h / s));
  const I = s > 1 ? resizePlane(gray, w, h, ws, hs) : gray;
  const P = s > 1 ? resizePlane(alpha, w, h, ws, hs) : alpha;
  const r = Math.max(1, Math.round(rFull / s));
  const n = ws * hs;
  const II = new Float32Array(n), IP = new Float32Array(n);
  for (let i = 0; i < n; i++) { II[i] = I[i] * I[i]; IP[i] = I[i] * P[i]; }
  let unc = bandOf ? bandOf(P, ws, hs) : null;
  if (!unc) { unc = new Float32Array(n); for (let i = 0; i < n; i++) unc[i] = P[i] > 0.01 && P[i] < 0.99 ? 1 : 0; }
  const mI = boxMean(I, ws, hs, r), mP = boxMean(P, ws, hs, r);
  const mII = boxMean(II, ws, hs, r), mIP = boxMean(IP, ws, hs, r);
  const band = boxMean(unc, ws, hs, r + 1);
  const eps = 1e-4;
  const A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = mII[i] - mI[i] * mI[i], c = mIP[i] - mI[i] * mP[i];
    A[i] = c / (v + eps); B[i] = mP[i] - A[i] * mI[i];
  }
  const mA = boxMean(A, ws, hs, r), mB = boxMean(B, ws, hs, r);
  const sx = ws / w, sy = hs / h;
  for (let y = 0; y < h; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    const by = Math.min(hs - 1, Math.floor((y + 0.5) * sy));
    for (let x = 0; x < w; x++) {
      const bx = Math.min(ws - 1, Math.floor((x + 0.5) * sx));
      if (band[by * ws + bx] <= 0) continue;
      const fx = (x + 0.5) * sx - 0.5, i = y * w + x;
      const q = sampleBilinear(mA, ws, hs, fx, fy) * gray[i] + sampleBilinear(mB, ws, hs, fx, fy);
      alpha[i] = q < 0 ? 0 : (q > 1 ? 1 : q);
    }
  }
}

/* Multi-level foreground estimation (Germer et al. 2020). Solves for the
   foreground colour F and background colour B behind each soft edge pixel. */
function estimateFB(img3, alpha, w0, h0) {
  const reg = 1e-5, gw = 1.0;
  const levels = Math.max(1, Math.ceil(Math.log2(Math.max(w0, h0))));
  let Fp = new Float32Array(3), Bp = new Float32Array(3), wp = 1, hp = 1;
  for (let lv = 0; lv <= levels; lv++) {
    const w = Math.max(1, Math.round(Math.pow(w0, lv / levels)));
    const h = Math.max(1, Math.round(Math.pow(h0, lv / levels)));
    const n = w * h;
    const I = new Float32Array(n * 3), A = new Float32Array(n);
    const F = new Float32Array(n * 3), B = new Float32Array(n * 3);
    for (let y = 0; y < h; y++) {
      const sy = Math.min(h0 - 1, Math.floor(y * h0 / h)), py = Math.min(hp - 1, Math.floor(y * hp / h));
      for (let x = 0; x < w; x++) {
        const sx = Math.min(w0 - 1, Math.floor(x * w0 / w)), px = Math.min(wp - 1, Math.floor(x * wp / w));
        const i = y * w + x, si = sy * w0 + sx, pi = py * wp + px;
        A[i] = alpha[si];
        I[3 * i] = img3[3 * si]; I[3 * i + 1] = img3[3 * si + 1]; I[3 * i + 2] = img3[3 * si + 2];
        F[3 * i] = Fp[3 * pi]; F[3 * i + 1] = Fp[3 * pi + 1]; F[3 * i + 2] = Fp[3 * pi + 2];
        B[3 * i] = Bp[3 * pi]; B[3 * i + 1] = Bp[3 * pi + 1]; B[3 * i + 2] = Bp[3 * pi + 2];
      }
    }
    const iters = (w <= 32 && h <= 32) ? 10 : 2;
    for (let it = 0; it < iters; it++) {
      for (let y = 0; y < h; y++) {
        const yu = y > 0 ? y - 1 : 0, yd = y < h - 1 ? y + 1 : h - 1;
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          const a0 = A[i], a1 = 1 - a0;
          let a00 = a0 * a0, a11 = a1 * a1;
          const a01 = a0 * a1;
          const ir = I[3 * i], ig = I[3 * i + 1], ib = I[3 * i + 2];
          let f0 = a0 * ir, f1 = a0 * ig, f2 = a0 * ib;
          let b0 = a1 * ir, b1 = a1 * ig, b2 = a1 * ib;
          const xl = x > 0 ? x - 1 : 0, xr = x < w - 1 ? x + 1 : w - 1;
          for (let k = 0; k < 4; k++) {
            const j = k === 0 ? y * w + xl : k === 1 ? y * w + xr : k === 2 ? yu * w + x : yd * w + x;
            const da = reg + gw * Math.abs(a0 - A[j]);
            const j3 = 3 * j;
            a00 += da; a11 += da;
            f0 += da * F[j3]; f1 += da * F[j3 + 1]; f2 += da * F[j3 + 2];
            b0 += da * B[j3]; b1 += da * B[j3 + 1]; b2 += da * B[j3 + 2];
          }
          const inv = 1 / (a00 * a11 - a01 * a01);
          const c00 = inv * a11, c01 = -inv * a01, c11 = inv * a00;
          let v;
          v = c00 * f0 + c01 * b0; F[3 * i] = v < 0 ? 0 : (v > 1 ? 1 : v);
          v = c00 * f1 + c01 * b1; F[3 * i + 1] = v < 0 ? 0 : (v > 1 ? 1 : v);
          v = c00 * f2 + c01 * b2; F[3 * i + 2] = v < 0 ? 0 : (v > 1 ? 1 : v);
          v = c01 * f0 + c11 * b0; B[3 * i] = v < 0 ? 0 : (v > 1 ? 1 : v);
          v = c01 * f1 + c11 * b1; B[3 * i + 1] = v < 0 ? 0 : (v > 1 ? 1 : v);
          v = c01 * f2 + c11 * b2; B[3 * i + 2] = v < 0 ? 0 : (v > 1 ? 1 : v);
        }
      }
    }
    Fp = F; Bp = B; wp = w; hp = h;
  }
  return { F: Fp, B: Bp };
}

/* Model mask (1024x1024) -> refined full-resolution alpha. */
function baseAlpha(rgba, w, h, mask, gray) {
  const alpha = resizePlane(mask, SIZE, SIZE, w, h);
  for (let i = 0; i < alpha.length; i++) { const a = alpha[i]; alpha[i] = a < 0 ? 0 : (a > 1 ? 1 : a); }
  guidedRefine(alpha, gray, w, h, Math.max(2, Math.round(2 * Math.max(w, h) / SIZE)));
  return alpha;
}

/* Recover clean foreground colours behind soft edges and pack the cutout as
   straight-alpha RGBA. */
function finishCutout(rgba, w, h, alpha) {
  const n = w * h;
  const lim = 1600;
  const sc = Math.min(1, lim / Math.max(w, h));
  const lw = Math.max(1, Math.round(w * sc)), lh = Math.max(1, Math.round(h * sc));
  let lowA, lowI;
  if (sc < 1) {
    lowA = resizePlane(alpha, w, h, lw, lh);
    const ch = new Float32Array(n);
    lowI = new Float32Array(lw * lh * 3);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < n; i++) ch[i] = rgba[i * 4 + c] / 255;
      const r = resizePlane(ch, w, h, lw, lh);
      for (let i = 0; i < lw * lh; i++) lowI[i * 3 + c] = r[i];
    }
  } else {
    lowA = alpha;
    lowI = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { lowI[i * 3] = rgba[i * 4] / 255; lowI[i * 3 + 1] = rgba[i * 4 + 1] / 255; lowI[i * 3 + 2] = rgba[i * 4 + 2] / 255; }
  }
  const fb = estimateFB(lowI, lowA, lw, lh);
  const Fl = fb.F, Bl = fb.B;
  const sx = lw / w, sy = lh / h;
  const out = new Uint8ClampedArray(n * 4);
  const smp = (arr, fx, fy, c) => {
    const x = fx < 0 ? 0 : (fx > lw - 1 ? lw - 1 : fx), y = fy < 0 ? 0 : (fy > lh - 1 ? lh - 1 : fy);
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = x0 + 1 < lw ? x0 + 1 : x0, y1 = y0 + 1 < lh ? y0 + 1 : y0;
    const ax = x - x0, ay = y - y0;
    const a = arr[(y0 * lw + x0) * 3 + c], b = arr[(y0 * lw + x1) * 3 + c];
    const cc = arr[(y1 * lw + x0) * 3 + c], d = arr[(y1 * lw + x1) * 3 + c];
    return (a * (1 - ax) + b * ax) * (1 - ay) + (cc * (1 - ax) + d * ax) * ay;
  };
  for (let y = 0; y < h; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    for (let x = 0; x < w; x++) {
      const i = y * w + x, p = i * 4, a = alpha[i];
      if (a >= 0.996) { out[p] = rgba[p]; out[p + 1] = rgba[p + 1]; out[p + 2] = rgba[p + 2]; out[p + 3] = 255; continue; }
      if (a <= 0.004) continue;
      const fx = (x + 0.5) * sx - 0.5;
      let t = (a - 0.15) / 0.45; t = t < 0 ? 0 : (t > 1 ? 1 : t); t = t * t * (3 - 2 * t);
      for (let c = 0; c < 3; c++) {
        const I = rgba[p + c] / 255;
        const fe = smp(Fl, fx, fy, c), be = smp(Bl, fx, fy, c);
        let fs = (I - (1 - a) * be) / a;
        fs = fs < 0 ? 0 : (fs > 1 ? 1 : fs);
        out[p + c] = Math.round((t * fs + (1 - t) * fe) * 255);
      }
      out[p + 3] = Math.round(a * 255);
    }
  }
  return out;
}

/* ---------- per-photo state and object picking ---------- */
/* Label connected foreground blobs on the 256 grid, then give every grid cell
   the label of its nearest blob, so soft hair and fur stay with their owner. */
function blobs(a256) {
  const N = LOW * LOW;
  const comp = new Int32Array(N).fill(-1);
  const queue = new Int32Array(N);
  let count = 0;
  for (let s = 0; s < N; s++) {
    if (comp[s] !== -1 || a256[s] < 0.5) continue;
    let head = 0, tail = 0;
    queue[tail++] = s; comp[s] = count;
    while (head < tail) {
      const p = queue[head++], px = p % LOW, py = (p - px) / LOW;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || ny < 0 || nx >= LOW || ny >= LOW) continue;
        const q = ny * LOW + nx;
        if (comp[q] === -1 && a256[q] >= 0.5) { comp[q] = count; queue[tail++] = q; }
      }
    }
    count++;
  }
  const cell = Int32Array.from(comp);
  let head = 0, tail = 0;
  for (let p = 0; p < N; p++) if (cell[p] >= 0) queue[tail++] = p;
  while (head < tail) {
    const p = queue[head++], px = p % LOW, py = (p - px) / LOW;
    const nb = [px > 0 ? p - 1 : -1, px < LOW - 1 ? p + 1 : -1, py > 0 ? p - LOW : -1, py < LOW - 1 ? p + LOW : -1];
    for (const q of nb) if (q >= 0 && cell[q] === -1) { cell[q] = cell[p]; queue[tail++] = q; }
  }
  return { comp, cell, count };
}
function dilate(mask, r) {
  if (r <= 0) return mask;
  const tmp = new Uint8Array(LOW * LOW), out = new Uint8Array(LOW * LOW);
  for (let y = 0; y < LOW; y++) for (let x = 0; x < LOW; x++) {
    let v = 0;
    for (let k = Math.max(0, x - r); k <= Math.min(LOW - 1, x + r) && !v; k++) v = mask[y * LOW + k];
    tmp[y * LOW + x] = v;
  }
  for (let y = 0; y < LOW; y++) for (let x = 0; x < LOW; x++) {
    let v = 0;
    for (let k = Math.max(0, y - r); k <= Math.min(LOW - 1, y + r) && !v; k++) v = tmp[k * LOW + x];
    out[y * LOW + x] = v;
  }
  return out;
}
function storeImage(rgba, w, h, mask, alpha) {
  const a8 = new Uint8Array(w * h);
  for (let i = 0; i < a8.length; i++) a8[i] = Math.round(alpha[i] * 255);
  const a256 = resizePlane(mask, SIZE, SIZE, LOW, LOW);
  const b = blobs(a256);
  img = {
    w, h, rgba: new Uint8ClampedArray(rgba), alpha: a8, a256, comp: b.comp, cell: b.cell, emb: null, objs: new Map(),
    edited: new Uint8Array(LOW * LOW),   // grid cells changed by object removal
    embStale: new Uint8Array(LOW * LOW), // ... and changed since the last embedding
  };
}
/* Note which grid cells a photo edit touched. */
function markEdited(x, y, w, h) {
  const gx0 = Math.max(0, Math.floor(x * LOW / img.w)), gx1 = Math.min(LOW - 1, Math.floor((x + w - 1) * LOW / img.w));
  const gy0 = Math.max(0, Math.floor(y * LOW / img.h)), gy1 = Math.min(LOW - 1, Math.floor((y + h - 1) * LOW / img.h));
  for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) { img.edited[gy * LOW + gx] = 1; img.embStale[gy * LOW + gx] = 1; }
}

async function ensureEmbedding(id) {
  if (img.emb) return 0;
  if (!sam) throw new Error('The picking model is not loaded yet.');
  const t0 = performance.now();
  const input = new ort.Tensor('float32', toModelInput(img.rgba, img.w, img.h, false), [1, 3, SIZE, SIZE]);
  let res;
  try { res = await sam.enc.run({ batched_images: input }); }
  catch (e) {
    if (sam.encDevice !== 'gpu') throw e;
    if (id) progress(id, { fallback: true });
    try { await sam.enc.release(); } catch (err) { /* ignore */ }
    sam.enc = await makeSession(sam.encGraph, buildWeights(await gunzip(sam.encPack), sam.encMan), 'cpu');
    sam.encDevice = 'cpu';
    res = await sam.enc.run({ batched_images: input });
  }
  const data = res.image_embeddings.data;
  img.emb = new ort.Tensor('float32', new Float32Array(data), [1, 256, 64, 64]);
  return performance.now() - t0;
}

/* EfficientSAM at one point: the whole object rather than a small part, so of
   the masks that contain the point and score close to the best, the largest.
   A point inside an area that object removal changed gets a fresh embedding
   first, so it doesn't select what used to be there. */
async function samPoint(x, y, id, whole) {
  const px = x * LOW / img.w, py = y * LOW / img.h;
  const ci = Math.min(LOW - 1, Math.max(0, Math.floor(py))) * LOW + Math.min(LOW - 1, Math.max(0, Math.floor(px)));
  if (img.emb && img.embStale[ci]) { img.emb = null; img.embStale.fill(0); }
  await ensureEmbedding(id);
  const res = await sam.dec.run({
    image_embeddings: img.emb,
    batched_point_coords: new ort.Tensor('float32', new Float32Array([px, py]), [1, 1, 1, 2]),
    batched_point_labels: new ort.Tensor('float32', new Float32Array([1]), [1, 1, 1]),
  });
  const masks = res.output_masks.data, iou = res.iou_predictions.data;
  const N = LOW * LOW;
  let cands = [0, 1, 2].filter((k) => masks[k * N + ci] > 0);
  if (!cands.length) cands = [[0, 1, 2].reduce((a, b) => (iou[b] > iou[a] ? b : a), 0)];
  const best = Math.max(...cands.map((k) => iou[k]));
  const area = (k) => { let c = 0; for (let j = 0; j < N; j++) if (masks[k * N + j] > 0) c++; return c; };
  let k = cands[0], kArea = -1;
  for (const c of cands) if (iou[c] >= best - 0.15) { const a = area(c); if (a > kArea) { k = c; kArea = a; } }
  if (whole) k = wholeObject(masks, iou, cands, ci, k);
  return { masks, iou, k, ci };
}
/* Removing: a tap on someone's shirt means the whole person. On a subject the
   background model found, take the mask that covers most of that subject while
   staying inside it. Elsewhere, take the largest mask the model is reasonably
   sure of, unless it is huge. */
function wholeObject(masks, iou, cands, ci, k0) {
  const N = LOW * LOW;
  const L = img.a256[ci] >= 0.5 && !img.edited[ci] ? img.comp[ci] : -1;
  if (L >= 0) {
    let sizeC = 0;
    for (let j = 0; j < N; j++) if (img.comp[j] === L) sizeC++;
    let pick = -1, bestCover = 0;
    for (const c of cands) {
      let a = 0, inter = 0;
      for (let j = 0; j < N; j++) if (masks[c * N + j] > 0) { a++; if (img.comp[j] === L) inter++; }
      const inside = inter / Math.max(1, a), cover = inter / Math.max(1, sizeC);
      if (inside >= 0.75 && cover > bestCover) { bestCover = cover; pick = c; }
    }
    if (pick >= 0) return pick;
  }
  const best = Math.max(...cands.map((c) => iou[c]));
  const sized = cands.map((c) => { let a = 0; for (let j = 0; j < N; j++) if (masks[c * N + j] > 0) a++; return [c, a]; }).sort((a, b) => b[1] - a[1]);
  for (const [c, a] of sized) if (iou[c] >= Math.max(0.5, best - 0.35) && a <= 0.15 * N) return c;
  return k0;
}

/* One click -> one kept object. Clicks on detected foreground keep that blob
   with its fine matte (or a confident part of it); clicks elsewhere keep the
   object EfficientSAM finds there. */
async function analyseClick(x, y, id) {
  const { masks, iou, k, ci } = await samPoint(x, y, id);
  const N = LOW * LOW;
  const L = img.comp[ci];
  if (img.a256[ci] >= 0.5 && L >= 0 && !img.edited[ci]) {
    const M = new Uint8Array(N), C = new Uint8Array(N);
    let inter = 0, sizeC = 0;
    for (let j = 0; j < N; j++) {
      M[j] = masks[k * N + j] > 0 ? 1 : 0;
      C[j] = img.comp[j] === L ? 1 : 0;
      sizeC += C[j]; inter += M[j] & C[j];
    }
    const region = new Uint8Array(N);
    if (iou[k] >= 0.7 && inter / Math.max(1, sizeC) < 0.7) {
      const Cd = dilate(C, 3), part = new Uint8Array(N);
      for (let j = 0; j < N; j++) part[j] = M[j] & Cd[j];
      const pd = dilate(part, 2);
      for (let j = 0; j < N; j++) region[j] = pd[j] && img.cell[j] === L ? 1 : 0;
      return { kind: 'part', region };
    }
    for (let j = 0; j < N; j++) region[j] = img.cell[j] === L ? 1 : 0;
    return { kind: 'blob', region };
  }
  return { kind: 'new', logits: new Float32Array(masks.subarray(k * N, (k + 1) * N)) };
}

/* Combine the picked objects into one full-resolution alpha. */
function compose(objs) {
  const { w, h, alpha, rgba } = img;
  const n = w * h;
  const out = new Float32Array(n);
  if (!objs.length) { for (let i = 0; i < n; i++) out[i] = alpha[i] / 255; return out; }
  const N = LOW * LOW;
  const gate = new Float32Array(N), logit = new Float32Array(N).fill(-30);
  let anyGate = false, anyNew = false;
  for (const o of objs) {
    if (o.kind === 'new') { anyNew = true; for (let j = 0; j < N; j++) if (o.logits[j] > logit[j]) logit[j] = o.logits[j]; }
    else { anyGate = true; for (let j = 0; j < N; j++) if (o.region[j]) gate[j] = 1; }
  }
  const sx = LOW / w, sy = LOW / h;
  const gray = grayOf(rgba, n);
  const rCoarse = Math.max(2, Math.round(1.5 * Math.max(w, h) / LOW));
  if (anyGate) {
    let cut = false;
    for (let y = 0; y < h; y++) {
      const fy = (y + 0.5) * sy - 0.5;
      for (let x = 0; x < w; x++) {
        const i = y * w + x, a = alpha[i];
        if (!a) continue;
        const g = sampleBilinear(gate, LOW, LOW, (x + 0.5) * sx - 0.5, fy);
        out[i] = (a / 255) * g;
        if (g > 0.02 && g < 0.98 && a > 128) cut = true;
      }
    }
    // Where a pick boundary crosses solid foreground, snap the cut to real edges.
    if (cut) {
      guidedRefine(out, gray, w, h, rCoarse, (P, ws, hs) => {
        const unc = new Float32Array(ws * hs);
        for (let y = 0; y < hs; y++) for (let x = 0; x < ws; x++) {
          const g = sampleBilinear(gate, LOW, LOW, (x + 0.5) * LOW / ws - 0.5, (y + 0.5) * LOW / hs - 0.5);
          unc[y * ws + x] = g > 0.02 && g < 0.98 && P[y * ws + x] > 0.02 ? 1 : 0;
        }
        return unc;
      });
    }
  }
  if (anyNew) {
    const sa = new Float32Array(n);
    for (let y = 0; y < h; y++) {
      const fy = (y + 0.5) * sy - 0.5;
      for (let x = 0; x < w; x++) sa[y * w + x] = 1 / (1 + Math.exp(-sampleBilinear(logit, LOW, LOW, (x + 0.5) * sx - 0.5, fy)));
    }
    guidedRefine(sa, gray, w, h, rCoarse);
    for (let i = 0; i < n; i++) if (sa[i] > out[i]) out[i] = sa[i];
  }
  return out;
}

/* ---------- object removal: MI-GAN fills the gap ---------- */
const FILL = 512;       // both fill models work on 512 x 512 windows
/* Fill models by kind: 'lama' (Big-LaMa, best) and 'migan' (MI-GAN, fast).
   Each is { session, device, graph, pack, man, inName, outName }. */
const fills = {};

/* Chamfer (3-4) distance, in pixels, from every pixel to the nearest pixel
   whose value is `target`. Close to true distance and runs in two passes. */
function chamfer(m, w, h, target) {
  const d = new Float32Array(w * h);
  for (let i = 0; i < d.length; i++) d[i] = m[i] === target ? 0 : 1e9;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = d[i];
      if (v === 0) continue;
      if (x > 0 && d[i - 1] + 3 < v) v = d[i - 1] + 3;
      if (y > 0) {
        if (d[i - w] + 3 < v) v = d[i - w] + 3;
        if (x > 0 && d[i - w - 1] + 4 < v) v = d[i - w - 1] + 4;
        if (x < w - 1 && d[i - w + 1] + 4 < v) v = d[i - w + 1] + 4;
      }
      d[i] = v;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let v = d[i];
      if (v === 0) continue;
      if (x < w - 1 && d[i + 1] + 3 < v) v = d[i + 1] + 3;
      if (y < h - 1) {
        if (d[i + w] + 3 < v) v = d[i + w] + 3;
        if (x < w - 1 && d[i + w + 1] + 4 < v) v = d[i + w + 1] + 4;
        if (x > 0 && d[i + w - 1] + 4 < v) v = d[i + w - 1] + 4;
      }
      d[i] = v;
    }
  }
  for (let i = 0; i < d.length; i++) d[i] /= 3;
  return d;
}
/* Crop a window of the photo and resample it to D x D planes (0..1). */
function resampleRGBA(rgba, W, x0, y0, ww, wh, D) {
  const kx = kernel(ww, D), ky = kernel(wh, D);
  const tmp = new Float32Array(D * wh * 3);
  for (let y = 0; y < wh; y++) {
    const row = ((y0 + y) * W + x0) * 4, orow = y * D * 3;
    for (let x = 0; x < D; x++) {
      const st = kx.start[x], n = kx.count[x], b = x * kx.taps;
      let r = 0, g = 0, bl = 0;
      for (let k = 0; k < n; k++) {
        const p = row + (st + k) * 4, wt = kx.w[b + k];
        r += rgba[p] * wt; g += rgba[p + 1] * wt; bl += rgba[p + 2] * wt;
      }
      const o = orow + x * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = bl;
    }
  }
  const P = D * D, out = new Float32Array(3 * P), acc = new Float32Array(D * 3);
  for (let y = 0; y < D; y++) {
    acc.fill(0);
    const st = ky.start[y], n = ky.count[y], b = y * ky.taps;
    for (let k = 0; k < n; k++) {
      const wt = ky.w[b + k], srow = (st + k) * D * 3;
      for (let i = 0; i < D * 3; i++) acc[i] += tmp[srow + i] * wt;
    }
    for (let x = 0; x < D; x++) for (let c = 0; c < 3; c++) out[c * P + y * D + x] = acc[x * 3 + c] / 255;
  }
  return out;
}
/* Coverage of a 0/1 mask after resampling to D x D. */
function resampleMask(m, ww, wh, D) {
  const kx = kernel(ww, D), ky = kernel(wh, D);
  const tmp = new Float32Array(D * wh);
  for (let y = 0; y < wh; y++) {
    const row = y * ww, orow = y * D;
    for (let x = 0; x < D; x++) {
      const st = kx.start[x], n = kx.count[x], b = x * kx.taps;
      let a = 0;
      for (let k = 0; k < n; k++) a += m[row + st + k] * kx.w[b + k];
      tmp[orow + x] = a;
    }
  }
  const out = new Float32Array(D * D);
  for (let y = 0; y < D; y++) {
    const st = ky.start[y], n = ky.count[y], b = y * ky.taps, orow = y * D;
    for (let k = 0; k < n; k++) {
      const wt = ky.w[b + k], srow = (st + k) * D;
      for (let x = 0; x < D; x++) out[orow + x] += tmp[srow + x] * wt;
    }
  }
  return out;
}
function copyRect(src, W, ch, x0, y0, w, h) {
  const out = ch === 4 ? new Uint8ClampedArray(w * h * 4) : new Uint8Array(w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray(((y0 + y) * W + x0) * ch, ((y0 + y) * W + x0 + w) * ch), y * w * ch);
  return out;
}
function writeRect(dst, W, ch, x0, y0, w, h, data) {
  for (let y = 0; y < h; y++) dst.set(data.subarray(y * w * ch, (y + 1) * w * ch), ((y0 + y) * W + x0) * ch);
}

async function runFill(kind, input) {
  const fill = fills[kind];
  const tensor = new ort.Tensor('float32', input, [1, 4, FILL, FILL]);
  let res;
  try { res = await fill.session.run({ [fill.inName]: tensor }); }
  catch (e) {
    if (fill.device !== 'gpu') throw e;
    try { await fill.session.release(); } catch (err) { /* ignore */ }
    fill.session = await makeSession(fill.graph, buildWeights(await gunzip(fill.pack), fill.man), 'cpu');
    fill.device = 'cpu';
    res = await fill.session.run({ [fill.inName]: tensor });
  }
  const t = res[fill.outName];
  const out = new Float32Array(t.data);
  if (t.dispose) t.dispose();
  // Both models end up as 0..1 colour planes.
  if (kind === 'migan') for (let i = 0; i < out.length; i++) out[i] = out[i] * 0.5 + 0.5;
  return out;
}

/* What a tap removes: the object EfficientSAM finds there. When that object is
   a whole foreground blob, its fine matte joins in, so hair and soft edges go
   too. The area then grows a little to catch halos and blurred edges. */
async function removalArea(x, y, id) {
  const { w, h } = img;
  const { masks, k, ci } = await samPoint(x, y, id, true);
  const N = LOW * LOW;
  const lg = masks.subarray(k * N, (k + 1) * N);
  let L = -1;
  if (img.a256[ci] >= 0.5 && img.comp[ci] >= 0 && !img.edited[ci]) {
    const C = img.comp[ci];
    let inter = 0, size = 0;
    for (let j = 0; j < N; j++) if (img.comp[j] === C) { size++; if (lg[j] > 0) inter++; }
    if (inter / Math.max(1, size) >= 0.6) L = C;
  }
  let gx0 = LOW, gy0 = LOW, gx1 = -1, gy1 = -1;
  for (let j = 0; j < N; j++) {
    if (lg[j] > 0 || (L >= 0 && img.cell[j] === L && img.a256[j] > 0.02)) {
      const gx = j % LOW, gy = (j - gx) / LOW;
      if (gx < gx0) gx0 = gx; if (gx > gx1) gx1 = gx; if (gy < gy0) gy0 = gy; if (gy > gy1) gy1 = gy;
    }
  }
  if (gx1 < 0) return null;
  const sx = w / LOW, sy = h / LOW;
  const grow = Math.round(Math.min(40, Math.max(3, Math.max(w, h) * 0.005)) + (L >= 0 ? 0 : 0.6 * Math.max(sx, sy)));
  const x0 = Math.max(0, Math.floor((gx0 - 1) * sx) - grow - 1), y0 = Math.max(0, Math.floor((gy0 - 1) * sy) - grow - 1);
  const x1 = Math.min(w, Math.ceil((gx1 + 2) * sx) + grow + 1), y1 = Math.min(h, Math.ceil((gy1 + 2) * sy) + grow + 1);
  const rw = x1 - x0, rh = y1 - y0;
  const hole = new Uint8Array(rw * rh);
  let any = 0;
  for (let y = y0; y < y1; y++) {
    const fy = (y + 0.5) / sy - 0.5, cy = Math.min(LOW - 1, Math.floor(y / sy));
    for (let x = x0; x < x1; x++) {
      let on = sampleBilinear(lg, LOW, LOW, (x + 0.5) / sx - 0.5, fy) > 0;
      if (!on && L >= 0) on = img.cell[cy * LOW + Math.min(LOW - 1, Math.floor(x / sx))] === L && img.alpha[y * w + x] > 25;
      if (on) { hole[(y - y0) * rw + (x - x0)] = 1; any++; }
    }
  }
  if (!any) return null;
  const d = chamfer(hole, rw, rh, 1);
  let bx0 = rw, by0 = rh, bx1 = -1, by1 = -1, area = 0;
  for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
    if (d[y * rw + x] <= grow) { area++; if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y; }
  }
  const mw = bx1 - bx0 + 1, mh = by1 - by0 + 1;
  const mask = new Uint8Array(mw * mh);
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) mask[y * mw + x] = d[(y + by0) * rw + x + bx0] <= grow ? 1 : 0;
  return { rect: { x: x0 + bx0, y: y0 + by0, w: mw, h: mh }, mask, area };
}

/* Fill the masked area of the photo. The model sees a square-ish window with
   plenty of context, resampled to 512 x 512. Its result is blended back with a
   soft edge that sits just inside the masked area. */
async function fillArea(rect, src, kind) {
  const W = img.w, H = img.h;
  let bx0 = W, by0 = H, bx1 = -1, by1 = -1;
  for (let y = 0; y < rect.h; y++) for (let x = 0; x < rect.w; x++) {
    if (src[y * rect.w + x]) {
      const ix = rect.x + x, iy = rect.y + y;
      if (ix < bx0) bx0 = ix; if (ix > bx1) bx1 = ix; if (iy < by0) by0 = iy; if (iy > by1) by1 = iy;
    }
  }
  if (bx1 < 0) return null;
  const big = Math.max(bx1 - bx0 + 1, by1 - by0 + 1);
  const side = Math.max(FILL, big + 2 * Math.max(128, Math.round(0.3 * big)));
  const ww = Math.min(side, W), wh = Math.min(side, H);
  const wx = Math.round(clampN((bx0 + bx1 + 1) / 2 - ww / 2, 0, W - ww)), wy = Math.round(clampN((by0 + by1 + 1) / 2 - wh / 2, 0, H - wh));
  const scale = Math.max(ww, wh) / FILL;
  const fw = Math.max(2, Math.round(1.5 * Math.max(1, scale)));
  const gx0 = Math.max(wx, bx0 - fw - 1), gy0 = Math.max(wy, by0 - fw - 1);
  const gx1 = Math.min(wx + ww, bx1 + fw + 2), gy1 = Math.min(wy + wh, by1 + fw + 2);
  const gw = gx1 - gx0, gh = gy1 - gy0;
  const hole = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    const sy = gy0 + y - rect.y;
    if (sy < 0 || sy >= rect.h) continue;
    for (let x = 0; x < gw; x++) {
      const sx = gx0 + x - rect.x;
      if (sx >= 0 && sx < rect.w && src[sy * rect.w + sx]) hole[y * gw + x] = 1;
    }
  }
  const dOut = chamfer(hole, gw, gh, 1);
  const grown = new Uint8Array(gw * gh);
  for (let i = 0; i < grown.length; i++) grown[i] = dOut[i] <= fw ? 1 : 0;
  const dIn = chamfer(grown, gw, gh, 0);
  const wgt = new Uint8Array(gw * gh);
  for (let i = 0; i < wgt.length; i++) {
    if (!grown[i]) continue;
    let t = dIn[i] / fw; t = t > 1 ? 1 : t;
    wgt[i] = Math.round(t * t * (3 - 2 * t) * 255);
  }
  // MI-GAN input: 0.5 for known pixels, -0.5 for the gap, then the photo (-1..1) with the gap blanked.
  // LaMa input: the photo (0..1) with the gap blanked, then the gap as 1.
  const winHole = new Uint8Array(ww * wh);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) if (grown[y * gw + x]) winHole[(gy0 - wy + y) * ww + (gx0 - wx + x)] = 1;
  const planes = resampleRGBA(img.rgba, W, wx, wy, ww, wh, FILL);
  const cov = resampleMask(winHole, ww, wh, FILL);
  const P = FILL * FILL, inp = new Float32Array(4 * P);
  for (let i = 0; i < P; i++) {
    const known = cov[i] > 0.01 ? 0 : 1;
    if (kind === 'lama') {
      inp[i] = planes[i] * known; inp[P + i] = planes[P + i] * known; inp[2 * P + i] = planes[2 * P + i] * known;
      inp[3 * P + i] = 1 - known;
    } else {
      inp[i] = known - 0.5;
      inp[P + i] = (planes[i] * 2 - 1) * known;
      inp[2 * P + i] = (planes[P + i] * 2 - 1) * known;
      inp[3 * P + i] = (planes[2 * P + i] * 2 - 1) * known;
    }
  }
  const t0 = performance.now();
  const out = await runFill(kind, inp);
  const ms = performance.now() - t0;
  const before = copyRect(img.rgba, W, 4, gx0, gy0, gw, gh);
  const alphaBefore = copyRect(img.alpha, W, 1, gx0, gy0, gw, gh);
  // Upscale the result to the window, but only where it lands in the gap.
  const kx = kernel(FILL, ww), ky = kernel(FILL, wh);
  const tmp = new Float32Array(FILL * gw * 3);
  for (let r = 0; r < FILL; r++) {
    for (let xi = 0; xi < gw; xi++) {
      const x = gx0 - wx + xi, st = kx.start[x], n = kx.count[x], b = x * kx.taps;
      let cr = 0, cg = 0, cb = 0;
      for (let k = 0; k < n; k++) {
        const q = r * FILL + st + k, wt = kx.w[b + k];
        cr += out[q] * wt; cg += out[P + q] * wt; cb += out[2 * P + q] * wt;
      }
      const o = (r * gw + xi) * 3;
      tmp[o] = cr; tmp[o + 1] = cg; tmp[o + 2] = cb;
    }
  }
  const rgba = img.rgba, alpha = img.alpha;
  for (let yi = 0; yi < gh; yi++) {
    const y = gy0 - wy + yi, st = ky.start[y], n = ky.count[y], b = y * ky.taps;
    for (let xi = 0; xi < gw; xi++) {
      const wv = wgt[yi * gw + xi];
      if (!wv) continue;
      let cr = 0, cg = 0, cb = 0;
      for (let k = 0; k < n; k++) {
        const o = ((st + k) * gw + xi) * 3, wt = ky.w[b + k];
        cr += tmp[o] * wt; cg += tmp[o + 1] * wt; cb += tmp[o + 2] * wt;
      }
      const a = wv / 255, i = (gy0 + yi) * W + gx0 + xi, p = i * 4;
      rgba[p] = rgba[p] * (1 - a) + cr * 255 * a;
      rgba[p + 1] = rgba[p + 1] * (1 - a) + cg * 255 * a;
      rgba[p + 2] = rgba[p + 2] * (1 - a) + cb * 255 * a;
      alpha[i] = Math.round(alpha[i] * (1 - a));
    }
  }
  if (scale > 1.3) matchGrain(rgba, W, gx0, gy0, gw, gh, wgt, dOut, fw, Math.max(4, Math.round(0.02 * Math.max(ww, wh))));
  markEdited(gx0, gy0, gw, gh);
  return {
    rect: { x: gx0, y: gy0, w: gw, h: gh }, wgt, before, alphaBefore,
    after: copyRect(rgba, W, 4, gx0, gy0, gw, gh), alphaAfter: copyRect(alpha, W, 1, gx0, gy0, gw, gh),
    ms, scale,
  };
}
function clampN(v, a, b) { return v < a ? a : (v > b ? b : v); }

/* Remove without a fill: the area becomes transparent, in the photo and in the
   cutout. A soft edge about a pixel and a half wide sits just inside it. */
function clearArea(rect, src) {
  const W = img.w, H = img.h;
  const x0 = Math.max(0, rect.x - 1), y0 = Math.max(0, rect.y - 1);
  const x1 = Math.min(W, rect.x + rect.w + 1), y1 = Math.min(H, rect.y + rect.h + 1);
  const gw = x1 - x0, gh = y1 - y0;
  const hole = new Uint8Array(gw * gh);
  let any = 0;
  for (let y = 0; y < rect.h; y++) for (let x = 0; x < rect.w; x++) {
    if (src[y * rect.w + x]) { hole[(rect.y + y - y0) * gw + (rect.x + x - x0)] = 1; any++; }
  }
  if (!any) return null;
  const dIn = chamfer(hole, gw, gh, 0);
  const wgt = new Uint8Array(gw * gh), fw = 1.5;
  for (let i = 0; i < wgt.length; i++) {
    if (!hole[i]) continue;
    let t = dIn[i] / fw; t = t > 1 ? 1 : t;
    wgt[i] = Math.round(t * t * (3 - 2 * t) * 255);
  }
  const before = copyRect(img.rgba, W, 4, x0, y0, gw, gh);
  const alphaBefore = copyRect(img.alpha, W, 1, x0, y0, gw, gh);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const wv = wgt[y * gw + x];
    if (!wv) continue;
    const i = (y0 + y) * W + x0 + x, k = 1 - wv / 255;
    img.alpha[i] = Math.round(img.alpha[i] * k);
    img.rgba[i * 4 + 3] = Math.round(img.rgba[i * 4 + 3] * k);
  }
  markEdited(x0, y0, gw, gh);
  return {
    rect: { x: x0, y: y0, w: gw, h: gh }, wgt, before, alphaBefore,
    after: copyRect(img.rgba, W, 4, x0, y0, gw, gh), alphaAfter: copyRect(img.alpha, W, 1, x0, y0, gw, gh),
  };
}

/* A fill made at 512 px and enlarged is smoother than the photo around it.
   Measure the fine grain just outside the gap and inside the fill (robustly,
   so edges don't count), then add the missing grain as soft luma noise. */
function matchGrain(rgba, W, x0, y0, w, h, wgt, dOut, fw, ring) {
  const n = w * h, Y = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = ((y0 + y) * W + x0 + x) * 4;
    Y[y * w + x] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  const m = boxMean(Y, w, h, 1);
  const BINS = 256, hRing = new Uint32Array(BINS), hFill = new Uint32Array(BINS);
  let nRing = 0, nFill = 0;
  for (let i = 0; i < n; i++) {
    const b = Math.min(BINS - 1, Math.round(Math.abs(Y[i] - m[i]) * 8));
    if (wgt[i] === 255) { hFill[b]++; nFill++; }
    else if (!wgt[i] && dOut[i] > fw && dOut[i] <= fw + ring) { hRing[b]++; nRing++; }
  }
  if (nRing < 200 || nFill < 200) return;
  const median = (hist, total) => { let c = 0; for (let b = 0; b < BINS; b++) { c += hist[b]; if (c >= total / 2) return (b + 0.5) / 8; } return BINS / 8; };
  const sRing = 1.4826 * median(hRing, nRing), sFill = 1.4826 * median(hFill, nFill);
  // The residual of a 3x3 mean keeps 8/9 of white noise power.
  const add = Math.sqrt(Math.max(0, sRing * sRing - sFill * sFill) * 9 / 8);
  if (add < 0.4) return;
  let seed = (x0 * 73856093) ^ (y0 * 19349663) ^ (w * 83492791) ^ h;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let i = 0; i < n; i++) {
    const wv = wgt[i];
    if (!wv) continue;
    const g = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(6.283185307 * rnd());
    const v = g * add * (wv / 255);
    const p = ((y0 + Math.floor(i / w)) * W + x0 + (i % w)) * 4;
    rgba[p] += v; rgba[p + 1] += v; rgba[p + 2] += v;
  }
}

/* ---------- PNG encoder (straight alpha, no premultiply loss) ---------- */
let CRC = null;
function crc32(buf, crc) {
  if (!CRC) {
    CRC = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c >>> 0; }
  }
  let c = (crc ^ 0xffffffff) >>> 0;
  for (let i = 0; i < buf.length; i++) c = CRC[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function pngChunk(type, data) {
  const head = new Uint8Array(8);
  new DataView(head.buffer).setUint32(0, data.length);
  for (let i = 0; i < 4; i++) head[4 + i] = type.charCodeAt(i);
  const crc = crc32(data, crc32(head.subarray(4, 8), 0));
  const tail = new Uint8Array(4); new DataView(tail.buffer).setUint32(0, crc);
  return [head, data, tail];
}
async function encodePNG(rgba, w, h, crop) {
  const cx = crop ? crop.x : 0, cy = crop ? crop.y : 0, cw = crop ? crop.w : w, ch = crop ? crop.h : h;
  const stride = cw * 4;
  const raw = new Uint8Array((stride + 1) * ch);
  let prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  for (let y = 0; y < ch; y++) {
    const src = ((cy + y) * w + cx) * 4;
    cur.set(rgba.subarray(src, src + stride));
    for (let x = 0; x < stride; x += 4) if (cur[x + 3] === 0) { cur[x] = 0; cur[x + 1] = 0; cur[x + 2] = 0; }
    const o = y * (stride + 1);
    raw[o] = 4; // Paeth
    for (let x = 0; x < stride; x++) {
      const a = x >= 4 ? cur[x - 4] : 0, b = prev[x], c = x >= 4 ? prev[x - 4] : 0;
      const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
      const pr = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      raw[o + 1 + x] = (cur[x] - pr) & 255;
    }
    const t = prev; prev = cur; cur = t;
  }
  const z = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, cw); dv.setUint32(4, ch);
  ihdr[8] = 8; ihdr[9] = 6;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  return new Blob([sig, ...pngChunk('IHDR', ihdr), ...pngChunk('IDAT', z), ...pngChunk('IEND', new Uint8Array(0))], { type: 'image/png' });
}

/* ---------- requests ---------- */
async function runMain(input, id) {
  const tensor = new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
  let res;
  try { res = await main.session.run({ input_image: tensor }); }
  catch (e) {
    if (main.device !== 'gpu') throw e;
    progress(id, { step: 'segment', fallback: true });
    try { await main.session.release(); } catch (err) { /* ignore */ }
    if (!main.weights) main.weights = buildWeights(await gunzip(main.pack), main.man);
    main.session = await makeSession(main.graph, main.weights, 'cpu');
    main.device = 'cpu';
    res = await main.session.run({ input_image: tensor });
  }
  const logits = res.output_image.data;
  const mask = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < mask.length; i++) mask[i] = 1 / (1 + Math.exp(-logits[i]));
  if (res.output_image.dispose) res.output_image.dispose();
  return mask;
}

const handlers = {
  async init(m) {
    progress(m.id, { step: 'unpack' });
    await loadOrt(m.runtime, m.plain);
    const graph = new Uint8Array(await gunzip(m.graph));
    main = { session: null, device: null, graph, pack: m.pack, man: m.manifest, weights: buildWeights(await gunzip(m.pack), m.manifest) };
    progress(m.id, { step: 'start' });
    let dev = m.device, note = null;
    if (dev === 'gpu') {
      try { main.session = await makeSession(graph, main.weights, 'gpu'); main.device = 'gpu'; }
      catch (e) { note = 'gpu-failed: ' + (e && e.message ? e.message : e); dev = 'cpu'; }
    }
    if (dev === 'cpu') { main.session = await makeSession(graph, main.weights, 'cpu'); main.device = 'cpu'; }
    // The session keeps its own copy. If the graphics chip fails later, runMain rebuilds them from the pack.
    main.weights = null;
    return [{ device: main.device, note, plain: plainOrt }];
  },

  async cutout(m) {
    const rgba = new Uint8ClampedArray(m.rgba);
    const t0 = performance.now();
    progress(m.id, { step: 'segment' });
    const mask = await runMain(toModelInput(rgba, m.w, m.h, true), m.id);
    main.weights = null; // free the expanded weights once a run has succeeded
    const t1 = performance.now();
    progress(m.id, { step: 'refine' });
    const gray = grayOf(rgba, m.w * m.h);
    const alpha = baseAlpha(rgba, m.w, m.h, mask, gray);
    storeImage(rgba, m.w, m.h, mask, alpha);
    const cut = finishCutout(rgba, m.w, m.h, alpha);
    const t2 = performance.now();
    const maskCopy = mask.slice();   // lets the page restore this state in a fresh worker
    return [{ cut: cut.buffer, rgba: rgba.buffer, mask: maskCopy.buffer, device: main.device, ms: { model: t1 - t0, refine: t2 - t1 } }, [cut.buffer, rgba.buffer, maskCopy.buffer]];
  },

  /* Example photo: the model mask ships with the page. */
  async refine(m) {
    const rgba = new Uint8ClampedArray(m.rgba);
    const mask = new Float32Array(m.mask);
    const alpha = baseAlpha(rgba, m.w, m.h, mask, grayOf(rgba, m.w * m.h));
    storeImage(rgba, m.w, m.h, mask, alpha);
    const cut = finishCutout(rgba, m.w, m.h, alpha);
    return [{ cut: cut.buffer, rgba: rgba.buffer }, [cut.buffer, rgba.buffer]];
  },

  async samInit(m) {
    if (sam) return [{ device: sam.encDevice }];
    await loadOrt(m.runtime, m.plain);
    const encGraph = new Uint8Array(await gunzip(m.enc.graph));
    const decGraph = new Uint8Array(await gunzip(m.dec.graph));
    const decWeights = buildWeights(await gunzip(m.dec.pack), m.dec.man);
    const dec = await makeSession(decGraph, decWeights, 'cpu');
    let enc = null, encDevice = m.device;
    const encWeights = buildWeights(await gunzip(m.enc.pack), m.enc.man);
    if (encDevice === 'gpu') {
      try { enc = await makeSession(encGraph, encWeights, 'gpu'); } catch (e) { encDevice = 'cpu'; }
    }
    if (!enc) { enc = await makeSession(encGraph, encWeights, 'cpu'); encDevice = 'cpu'; }
    sam = { enc, dec, encDevice, encGraph, encPack: m.enc.pack, encMan: m.enc.man };
    return [{ device: encDevice }];
  },

  async samEmbed(m) {
    if (!img) throw new Error('No photo is loaded.');
    const ms = await ensureEmbedding(m.id);
    return [{ ms, device: sam.encDevice }];
  },

  /* Rebuild the cutout from the current list of picked objects. */
  async select(m) {
    if (!img) throw new Error('No photo is loaded.');
    const objs = [];
    if (m.objects.length) {
      await ensureEmbedding(m.id);
      for (const o of m.objects) {
        let c = img.objs.get(o.id);
        if (!c) { c = await analyseClick(o.x, o.y, m.id); img.objs.set(o.id, c); }
        objs.push(c);
      }
    }
    for (const id of Array.from(img.objs.keys())) if (!m.objects.some((o) => o.id === id)) img.objs.delete(id);
    const cut = finishCutout(img.rgba, img.w, img.h, compose(objs));
    return [{ cut: cut.buffer, kinds: objs.map((o) => o.kind) }, [cut.buffer]];
  },

  async fillInit(m) {
    const kind = m.kind || 'migan';
    if (fills[kind]) return [{ device: fills[kind].device }];
    await loadOrt(m.runtime, m.plain);
    const graph = new Uint8Array(await gunzip(m.graph));
    const weights = buildWeights(await gunzip(m.pack), m.man);
    let session = null, device = m.device;
    if (device === 'gpu') {
      try { session = await makeSession(graph, weights, 'gpu'); } catch (e) { device = 'cpu'; }
    }
    if (!session) { session = await makeSession(graph, weights, 'cpu'); device = 'cpu'; }
    fills[kind] = { session, device, graph, pack: m.pack, man: m.man, inName: session.inputNames[0], outName: session.outputNames[0] };
    return [{ device }];
  },

  /* A tap with the remove tool: which area to remove. */
  async removeSelect(m) {
    if (!img) throw new Error('No photo is loaded.');
    if (!sam) throw new Error('The picking model is not loaded yet.');
    const r = await removalArea(m.x, m.y, m.id);
    if (!r) return [{ empty: true }];
    return [{ rect: r.rect, mask: r.mask.buffer, frac: r.area / (img.w * img.h) }, [r.mask.buffer]];
  },

  /* Fill an area of the photo with new background. */
  async removeFill(m) {
    if (!img) throw new Error('No photo is loaded.');
    const kind = m.kind || 'migan';
    if (!fills[kind]) throw new Error('The fill model is not loaded yet.');
    const r = await fillArea(m.rect, new Uint8Array(m.mask), kind);
    if (!r) return [{ empty: true }];
    const bufs = [r.wgt.buffer, r.before.buffer, r.after.buffer, r.alphaBefore.buffer, r.alphaAfter.buffer];
    return [{
      rect: r.rect, wgt: r.wgt.buffer, before: r.before.buffer, after: r.after.buffer,
      alphaBefore: r.alphaBefore.buffer, alphaAfter: r.alphaAfter.buffer, ms: r.ms, scale: r.scale, device: fills[kind].device, kind,
    }, bufs];
  },

  /* Remove an area and leave it transparent. */
  async removeClear(m) {
    if (!img) throw new Error('No photo is loaded.');
    const r = clearArea(m.rect, new Uint8Array(m.mask));
    if (!r) return [{ empty: true }];
    const bufs = [r.wgt.buffer, r.before.buffer, r.after.buffer, r.alphaBefore.buffer, r.alphaAfter.buffer];
    return [{
      rect: r.rect, wgt: r.wgt.buffer, before: r.before.buffer, after: r.after.buffer,
      alphaBefore: r.alphaBefore.buffer, alphaAfter: r.alphaAfter.buffer, ms: 0, scale: 1, device: 'cpu', kind: 'clear',
    }, bufs];
  },

  /* Undo and redo of a removal: put a saved area of the photo back. */
  async patch(m) {
    if (!img) throw new Error('No photo is loaded.');
    const { x, y, w, h } = m.rect;
    writeRect(img.rgba, img.w, 4, x, y, w, h, new Uint8ClampedArray(m.rgba));
    writeRect(img.alpha, img.w, 1, x, y, w, h, new Uint8Array(m.alpha));
    markEdited(x, y, w, h);
    return [{}];
  },

  async png(m) {
    const rgba = new Uint8ClampedArray(m.rgba);
    let blob = null, error = null;
    try { blob = await encodePNG(rgba, m.w, m.h, m.crop); }
    catch (e) { error = (e && e.message) ? e.message : String(e); }
    return [{ blob, error, rgba: rgba.buffer }, [rgba.buffer]];
  },
};
