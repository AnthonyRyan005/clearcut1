/* Clearcut offline support. The app shell and runtime come from this cache.
   Model weights are stored by the app itself in IndexedDB, so this worker
   passes those large chunk downloads straight to the network. */
const VERSION = 'clearcut-af01db23123e';
const SHELL = ["./", "index.html", "worker.js", "describe-worker.js", "manifest.webmanifest", "icons/apple-touch-icon.png", "icons/icon-192.png", "icons/icon-512.png", "icons/favicon-64.png", "vendor/ort/ort.webgpu.min.js", "vendor/ort/ort-wasm-simd-threaded.jsep.js", "vendor/ort/ort.wasm.min.js", "vendor/ort/ort-wasm-simd-threaded.js", "m/manifest.json", "m/graph.bin", "m/rt.bin", "m/rt-cpu.bin", "m/s-enc-graph.bin", "m/s-dec-graph.bin", "m/f-graph.bin"];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(VERSION).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('clearcut-') && k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

const isChunk = (url) => /\/m\/([a-z]+-)*w\d+\.bin$/.test(url.pathname);
// The page, its worker and the model list always come from the network when online, so they never mix versions.
const isFresh = (url, req) => req.mode === 'navigate' || /\/(index\.html|worker\.js|describe-worker\.js|m\/manifest\.json)$/.test(url.pathname) || url.pathname.endsWith('/');
const isFont = (url) => url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com';

async function networkFirst(req) {
  const cache = await caches.open(VERSION);
  try {
    const res = await fetch(req);
    if (res && res.ok) cache.put(req, res.clone());
    return res;
  } catch (e) {
    const hit = await cache.match(req, { ignoreSearch: true }) || (req.mode === 'navigate' ? await cache.match('index.html') : null);
    if (hit) return hit;
    throw e;
  }
}
async function cacheFirst(req) {
  const cache = await caches.open(VERSION);
  const hit = await cache.match(req, { ignoreSearch: true });
  if (hit) return hit;
  const res = await fetch(req);
  if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone());
  return res;
}
async function staleWhileRevalidate(req) {
  const cache = await caches.open(VERSION);
  const hit = await cache.match(req);
  const update = fetch(req).then((res) => { if (res && (res.ok || res.type === 'opaque')) cache.put(req, res.clone()); return res; }).catch(() => hit);
  return hit || update;
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (isFont(url)) { event.respondWith(staleWhileRevalidate(req)); return; }
  if (url.origin !== self.location.origin) return;
  if (isChunk(url)) return;
  if (isFresh(url, req)) { event.respondWith(networkFirst(req)); return; }
  event.respondWith(cacheFirst(req));
});
