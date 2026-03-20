const DEFAULT_API_BASE = "http://localhost:5000";

function getApiBase() {
  return (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE).replace(/\/$/, "");
}

async function request(path, options = {}) {
  const response = await fetch(`${getApiBase()}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.error || payload.detail || "Request failed");
  }

  return payload;
}

export function getApiBaseUrl() {
  return getApiBase();
}

export function fetchHealth() {
  return request("/health");
}

export function createIncident(body) {
  return request("/incidents", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchIncident(incidentId) {
  return request(`/incidents/${incidentId}`);
}

export function fetchSimilarIncidents(incidentId, topN = 5) {
  return request(`/incidents/${incidentId}/similar?top_n=${topN}`);
}

export function resolveIncident(incidentId, body = {}) {
  return request(`/incidents/${incidentId}/resolve`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
