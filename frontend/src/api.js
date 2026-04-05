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
    const detail = payload.error || payload.detail || "Request failed";
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg || item.message || JSON.stringify(item)).join("; ")
      : typeof detail === "object"
        ? detail.message || JSON.stringify(detail)
        : detail;
    throw new Error(message);
  }

  return payload;
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

export function checkDuplicates(body) {
  return request("/incidents/check-duplicates", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function submitDuplicateReview(body) {
  return request("/duplicate-reviews", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function fetchIncident(incidentId, includeSimilar = false) {
  return request(`/incidents/${incidentId}?include_similar=${includeSimilar}`);
}

export function fetchIncidentByTicket(ticketId, includeSimilar = false) {
  return request(`/incidents/by-ticket/${encodeURIComponent(ticketId)}?include_similar=${includeSimilar}`);
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
