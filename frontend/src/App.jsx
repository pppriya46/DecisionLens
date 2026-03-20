import { useEffect, useState } from "react";
import {
  createIncident,
  fetchHealth,
  fetchIncident,
  fetchSimilarIncidents,
  getApiBaseUrl,
  resolveIncident,
} from "./api";

const initialIncidentForm = {
  ticket_id: "",
  initial_message: "",
  customer_id: "",
  customer_segment: "",
  channel: "web",
  product_area: "",
  issue_type: "",
  priority: "medium",
  platform: "",
  region: "",
  has_attachment: false,
};

function StatCard({ label, value, tone = "default" }) {
  return (
    <div className={`stat-card stat-card--${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Pill({ children, tone = "default" }) {
  return <span className={`pill pill--${tone}`}>{children}</span>;
}

function SimilarIncidentCard({ incident }) {
  const score = incident?.scores?.final ?? incident?.similarity_score ?? 0;

  return (
    <article className="result-card">
      <div className="result-card__topline">
        <div>
          <h4>{incident.ticket_id}</h4>
          <p>{incident.issue_type || "Uncategorized"} • {incident.product_area || "Unknown area"}</p>
        </div>
        <Pill tone="accent">{Math.round(score * 100)}% match</Pill>
      </div>

      <div className="result-card__meta">
        <Pill>{incident.status || "unknown"}</Pill>
        <Pill>{incident.priority || "medium"}</Pill>
        {incident.created_at ? <Pill>{new Date(incident.created_at).toLocaleDateString()}</Pill> : null}
      </div>

      <p className="result-card__body">{incident.description}</p>

      {incident.resolution ? (
        <div className="result-card__resolution">
          <span>Resolution</span>
          <p>{incident.resolution}</p>
        </div>
      ) : null}
    </article>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [incidentForm, setIncidentForm] = useState(initialIncidentForm);
  const [createState, setCreateState] = useState({ loading: false, message: "", error: "" });
  const [selectedIncidentId, setSelectedIncidentId] = useState("1");
  const [activeIncident, setActiveIncident] = useState(null);
  const [incidentError, setIncidentError] = useState("");
  const [incidentLoading, setIncidentLoading] = useState(false);
  const [similarState, setSimilarState] = useState({ loading: false, data: null, error: "" });
  const [resolveState, setResolveState] = useState({ loading: false, data: null, error: "" });

  useEffect(() => {
    let ignore = false;

    fetchHealth()
      .then((data) => {
        if (!ignore) {
          setHealth(data);
        }
      })
      .catch((error) => {
        if (!ignore) {
          setHealthError(error.message);
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  async function handleCreateIncident(event) {
    event.preventDefault();
    setCreateState({ loading: true, message: "", error: "" });

    try {
      const payload = {
        ...incidentForm,
        customer_id: incidentForm.customer_id || null,
        customer_segment: incidentForm.customer_segment || null,
        product_area: incidentForm.product_area || null,
        issue_type: incidentForm.issue_type || null,
        platform: incidentForm.platform || null,
        region: incidentForm.region || null,
      };

      const response = await createIncident(payload);
      setCreateState({
        loading: false,
        message: `Created ${response.ticket_id} as incident ${response.incident_id}.`,
        error: "",
      });
      setSelectedIncidentId(String(response.incident_id));
      setIncidentForm(initialIncidentForm);
    } catch (error) {
      setCreateState({ loading: false, message: "", error: error.message });
    }
  }

  async function handleLoadIncident(event) {
    event?.preventDefault();
    if (!selectedIncidentId) {
      setIncidentError("Enter an incident ID to inspect.");
      return;
    }

    setIncidentLoading(true);
    setIncidentError("");
    setResolveState({ loading: false, data: null, error: "" });

    try {
      const data = await fetchIncident(selectedIncidentId);
      setActiveIncident(data);
    } catch (error) {
      setActiveIncident(null);
      setIncidentError(error.message);
    } finally {
      setIncidentLoading(false);
    }
  }

  async function handleFindSimilar() {
    if (!selectedIncidentId) {
      setSimilarState({ loading: false, data: null, error: "Load an incident first." });
      return;
    }

    setSimilarState({ loading: true, data: null, error: "" });

    try {
      const data = await fetchSimilarIncidents(selectedIncidentId, 5);
      setSimilarState({ loading: false, data, error: "" });
    } catch (error) {
      setSimilarState({ loading: false, data: null, error: error.message });
    }
  }

  async function handleResolveIncident() {
    if (!selectedIncidentId) {
      setResolveState({ loading: false, data: null, error: "Load an incident first." });
      return;
    }

    setResolveState({ loading: true, data: null, error: "" });

    try {
      const data = await resolveIncident(selectedIncidentId, { force_regenerate: true });
      setResolveState({ loading: false, data, error: "" });
    } catch (error) {
      setResolveState({ loading: false, data: null, error: error.message });
    }
  }

  const incident = activeIncident?.incident;
  const predictions = activeIncident?.predictions;
  const inlineSimilar = activeIncident?.similar_incidents || [];
  const searchedSimilar = similarState.data?.results || [];

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero__copy">
          <Pill tone="accent">DecisionLens Console</Pill>
          <h1>Minimal incident intelligence for search, triage, and guided resolution.</h1>
          <p>
            A React workspace over the FastAPI backend for creating incidents, reviewing semantic
            matches, and generating AI-assisted troubleshooting notes.
          </p>
        </div>

        <div className="hero__panel">
          <div className="hero__panel-header">
            <span>Backend status</span>
            <Pill tone={health?.status === "healthy" ? "success" : "default"}>
              {health?.status || (healthError ? "offline" : "checking")}
            </Pill>
          </div>
          <div className="stat-grid">
            <StatCard
              label="API base"
              value={getApiBaseUrl().replace(/^https?:\/\//, "")}
            />
            <StatCard
              label="Incidents"
              value={health?.database?.incidents_count ?? "—"}
              tone="soft"
            />
            <StatCard
              label="Embeddings"
              value={health?.database?.embeddings_count ?? "—"}
              tone="soft"
            />
            <StatCard
              label="OpenAI"
              value={health?.openai_api || "—"}
              tone="soft"
            />
          </div>
          {healthError ? <p className="helper-text helper-text--error">{healthError}</p> : null}
        </div>
      </header>

      <main className="dashboard">
        <section className="panel panel--form">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">Create Incident</span>
              <h2>Add a new support case</h2>
            </div>
            <Pill>{incidentForm.priority}</Pill>
          </div>

          <form className="incident-form" onSubmit={handleCreateIncident}>
            <label>
              Ticket ID
              <input
                value={incidentForm.ticket_id}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, ticket_id: event.target.value }))
                }
                placeholder="TKT-2026-1001"
                required
              />
            </label>

            <label className="incident-form__wide">
              Problem Description
              <textarea
                value={incidentForm.initial_message}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, initial_message: event.target.value }))
                }
                placeholder="Users report failed VPN authentication after the latest SSO rollout."
                rows={5}
                required
              />
            </label>

            <label>
              Product Area
              <input
                value={incidentForm.product_area}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, product_area: event.target.value }))
                }
                placeholder="Identity Platform"
              />
            </label>

            <label>
              Issue Type
              <input
                value={incidentForm.issue_type}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, issue_type: event.target.value }))
                }
                placeholder="SSO Login Failure"
              />
            </label>

            <label>
              Priority
              <select
                value={incidentForm.priority}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, priority: event.target.value }))
                }
              >
                <option value="low">low</option>
                <option value="medium">medium</option>
                <option value="high">high</option>
                <option value="critical">critical</option>
              </select>
            </label>

            <label>
              Channel
              <input
                value={incidentForm.channel}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, channel: event.target.value }))
                }
                placeholder="web"
              />
            </label>

            <label>
              Platform
              <input
                value={incidentForm.platform}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, platform: event.target.value }))
                }
                placeholder="Windows 11"
              />
            </label>

            <label>
              Region
              <input
                value={incidentForm.region}
                onChange={(event) =>
                  setIncidentForm((current) => ({ ...current, region: event.target.value }))
                }
                placeholder="US-West"
              />
            </label>

            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={incidentForm.has_attachment}
                onChange={(event) =>
                  setIncidentForm((current) => ({
                    ...current,
                    has_attachment: event.target.checked,
                  }))
                }
              />
              Incident includes an attachment
            </label>

            <button className="button button--primary" type="submit" disabled={createState.loading}>
              {createState.loading ? "Creating..." : "Create incident"}
            </button>
          </form>

          {createState.message ? <p className="helper-text">{createState.message}</p> : null}
          {createState.error ? <p className="helper-text helper-text--error">{createState.error}</p> : null}
        </section>

        <section className="panel panel--viewer">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">Incident Workspace</span>
              <h2>Inspect, retrieve, and resolve</h2>
            </div>
          </div>

          <form className="lookup-bar" onSubmit={handleLoadIncident}>
            <label>
              Incident ID
              <input
                value={selectedIncidentId}
                onChange={(event) => setSelectedIncidentId(event.target.value)}
                placeholder="1"
              />
            </label>
            <button className="button button--ghost" type="submit" disabled={incidentLoading}>
              {incidentLoading ? "Loading..." : "Load incident"}
            </button>
            <button className="button button--ghost" type="button" onClick={handleFindSimilar} disabled={similarState.loading}>
              {similarState.loading ? "Searching..." : "Find similar"}
            </button>
            <button className="button button--primary" type="button" onClick={handleResolveIncident} disabled={resolveState.loading}>
              {resolveState.loading ? "Generating..." : "Generate resolution"}
            </button>
          </form>

          {incidentError ? <p className="helper-text helper-text--error">{incidentError}</p> : null}

          {incident ? (
            <div className="incident-detail">
              <div className="incident-detail__header">
                <div>
                  <h3>{incident.ticket_id}</h3>
                  <p>{incident.issue_type || "Uncategorized"} • {incident.product_area || "Unknown area"}</p>
                </div>
                <div className="result-card__meta">
                  <Pill>{incident.status}</Pill>
                  <Pill>{incident.priority || "medium"}</Pill>
                </div>
              </div>

              <p className="incident-detail__message">{incident.initial_message}</p>

              <div className="stat-grid">
                <StatCard label="Predicted priority" value={predictions?.predicted_priority || "—"} />
                <StatCard
                  label="Model confidence"
                  value={predictions ? `${Math.round(predictions.confidence * 100)}%` : "—"}
                  tone="soft"
                />
                <StatCard label="Region" value={incident.region || "—"} tone="soft" />
                <StatCard label="Platform" value={incident.platform || "—"} tone="soft" />
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <h3>No incident loaded yet</h3>
              <p>Load an incident ID to inspect predictions, similar cases, and generated resolutions.</p>
            </div>
          )}

          <div className="result-section">
            <div className="result-section__heading">
              <h3>Similar incidents</h3>
              <Pill tone="soft">{searchedSimilar.length || inlineSimilar.length} results</Pill>
            </div>

            {(searchedSimilar.length ? searchedSimilar : inlineSimilar).length ? (
              <div className="results-grid">
                {(searchedSimilar.length ? searchedSimilar : inlineSimilar).map((item) => (
                  <SimilarIncidentCard key={`${item.ticket_id}-${item.id || item.ticket_id}`} incident={item} />
                ))}
              </div>
            ) : (
              <p className="helper-text">Run similar-incident search to see semantic matches here.</p>
            )}

            {similarState.error ? <p className="helper-text helper-text--error">{similarState.error}</p> : null}
          </div>

          <div className="result-section">
            <div className="result-section__heading">
              <h3>Generated resolution</h3>
              {resolveState.data?.confidence ? <Pill tone="accent">{resolveState.data.confidence}</Pill> : null}
            </div>

            {resolveState.data ? (
              <div className="resolution-panel">
                <p>{resolveState.data.answer}</p>
                <div className="stat-grid">
                  <StatCard
                    label="Average similarity"
                    value={`${Math.round(resolveState.data.avg_similarity * 100)}%`}
                  />
                  <StatCard
                    label="Sources used"
                    value={resolveState.data.source_incidents.length}
                    tone="soft"
                  />
                </div>
              </div>
            ) : (
              <p className="helper-text">Generate a resolution to view the compact RAG output and supporting signals.</p>
            )}

            {resolveState.error ? <p className="helper-text helper-text--error">{resolveState.error}</p> : null}
          </div>
        </section>
      </main>
    </div>
  );
}
