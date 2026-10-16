// Thin wrappers over the REST + WebSocket API.

const KEY = "tsukumo-key";
const savedKey = () => { try { return localStorage.getItem(KEY); } catch { return null; } };

async function request(method, path, body, retried = false) {
  const headers = body ? { "Content-Type": "application/json" } : {};
  if (savedKey()) headers["X-Tsukumo-Key"] = savedKey();
  const res = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
  const data = await res.json().catch(() => ({}));
  // A workshop with TSUKUMO_KEY set asks for it once, the first time it's needed.
  if (res.status === 403 && /workshop key/.test(data.detail) && !retried) {
    const key = prompt("Workshop key");
    if (key) {
      try { localStorage.setItem(KEY, key); } catch {}
      return request(method, path, body, true);
    }
  }
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body = {}) => request("POST", path, body),
  del: (path) => request("DELETE", path),
};

export function socket(path, onMessage, { onOpen, onClose } = {}) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  if (onOpen) ws.onopen = onOpen;
  if (onClose) ws.onclose = onClose;
  return ws;
}

let infoPromise;
export const info = () => (infoPromise ??= api.get("/api/info"));
