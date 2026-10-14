// Thin wrappers over the REST + WebSocket API.

async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
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
