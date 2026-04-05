import { useState } from "react";

const AFFECTED_AREAS = [
  "Identity Platform",
  "Billing",
  "Analytics Dashboard",
  "Network Services",
  "Collaboration Tools",
  "Security",
];

const ISSUE_CATEGORIES = [
  "SSO Login Failure",
  "Duplicate Charge",
  "Dashboard Performance",
  "VPN Access",
  "Password Reset",
  "Export Failure",
];

const INITIAL_INCIDENTS = [
  {
    id: 100001,
    ticketId: "TKT-20260405-7C3BD8",
    title: "Password reset still fails after SSO rollout",
    category: "SSO Login Failure",
    area: "Identity Platform",
    status: "resolved",
    date: "Apr 5, 2026",
    description: "Users cannot log in after the latest SSO rollout. Password reset also fails.",
    guidance: {
      likelyIssue: "The SSO rollout likely introduced an authentication mapping or reset-flow mismatch.",
      steps: [
        "Verify that the identity provider is returning the expected user identifier for the affected tenant.",
        "Check whether the password reset flow is still pointing to the pre-rollout callback or redirect URL.",
        "Validate the issue with one affected account and compare the login trace against a known-good tenant.",
      ],
      escalateIf: "Escalate to the identity engineering team if reset links still fail after callback and mapping checks.",
    },
    sourceCases: [
      {
        ticketId: "TCKT_074545",
        title: "SSO redirect loop after auth changes",
        category: "SSO Login Failure",
        status: "resolved",
        date: "Oct 21, 2025",
        summary: "Users were redirected back to login because the callback URL changed during rollout.",
      },
      {
        ticketId: "TCKT_049613",
        title: "Password reset email succeeds but login still fails",
        category: "Password Reset",
        status: "resolved",
        date: "Apr 4, 2026",
        summary: "A stale identity mapping prevented reset-complete accounts from authenticating successfully.",
      },
    ],
  },
  {
    id: 100009,
    ticketId: "TCKT_000009",
    title: "Customer charged twice for same subscription",
    category: "Duplicate Charge",
    area: "Billing",
    status: "open",
    date: "Apr 3, 2026",
    description: "I was charged twice for my subscription this month.",
    guidance: {
      likelyIssue: "Billing retry logic or duplicate invoice creation is likely causing repeat charges.",
      steps: [
        "Review the payment timeline to confirm whether the second charge came from a retry or a new invoice.",
        "Check for duplicate subscription records or replayed payment webhooks on the affected account.",
        "Pause any additional billing attempts until the duplicate-charge path is understood.",
      ],
      escalateIf: "Escalate to the billing platform owner if duplicate invoices are being created across multiple accounts.",
    },
    sourceCases: [
      {
        ticketId: "TCKT_062016",
        title: "Duplicate invoice generated after payment retry",
        category: "Duplicate Charge",
        status: "resolved",
        date: "Dec 23, 2025",
        summary: "Retry logic created a second invoice instead of reusing the failed-payment record.",
      },
    ],
  },
  {
    id: 100065,
    ticketId: "TCKT_000065",
    title: "Analytics dashboard queries are timing out",
    category: "Dashboard Performance",
    area: "Analytics Dashboard",
    status: "escalated",
    date: "Apr 1, 2026",
    description: "Queries in the analytics dashboard module are timing out.",
    guidance: {
      likelyIssue: "A slow dashboard query path or undersized compute pool is degrading request completion.",
      steps: [
        "Identify the slowest dashboard query and compare its execution time to the baseline from the previous release.",
        "Check whether the analytics worker pool or cache tier is saturated during the timeout window.",
        "Temporarily route affected traffic to a lighter query path if one is available.",
      ],
      escalateIf: "Escalate if timeout volume continues to rise after query tuning or traffic mitigation.",
    },
    sourceCases: [
      {
        ticketId: "TCKT_092722",
        title: "Dashboard query timeout during peak usage",
        category: "Dashboard Performance",
        status: "resolved",
        date: "Dec 20, 2025",
        summary: "A cache invalidation bug caused repeated full queries and exhausted the analytics pool.",
      },
      {
        ticketId: "TCKT_081877",
        title: "Dashboard render failures after schema update",
        category: "Dashboard Performance",
        status: "pending",
        date: "Nov 21, 2025",
        summary: "A recent schema migration increased payload size and slowed dashboard requests significantly.",
      },
    ],
  },
];

const EMPTY_GUIDANCE = {
  likelyIssue: "",
  steps: [],
  escalateIf: "",
};

function buildTicketId() {
  return `TKT-${new Date().toISOString().slice(0, 10).replaceAll("-", "")}-${Math.random()
    .toString(16)
    .slice(2, 8)
    .toUpperCase()}`;
}

function getStatusTone(status) {
  if (status === "resolved") return "success";
  if (status === "escalated") return "danger";
  if (status === "pending") return "pending";
  return "warning";
}

function scoreSimilarity(query, incident) {
  const normalizedQuery = query.toLowerCase();
  let score = 0;

  if (normalizedQuery.includes("login") || normalizedQuery.includes("password")) {
    if (incident.category.includes("Login") || incident.category.includes("Password")) score += 3;
  }

  if (normalizedQuery.includes("charged") || normalizedQuery.includes("billing")) {
    if (incident.area === "Billing") score += 3;
  }

  if (normalizedQuery.includes("vpn")) {
    if (incident.category === "VPN Access" || incident.area === "Network Services") score += 3;
  }

  if (normalizedQuery.includes("dashboard") || normalizedQuery.includes("analytics")) {
    if (incident.area === "Analytics Dashboard") score += 3;
  }

  const words = normalizedQuery.split(/\s+/).filter(Boolean);
  for (const word of words) {
    if (incident.description.toLowerCase().includes(word)) score += 1;
  }

  return score;
}

function buildGuidance(description, area, category) {
  const lower = description.toLowerCase();

  if (lower.includes("login") || lower.includes("password")) {
    return {
      likelyIssue: "Authentication flow changes or reset-path failures are preventing users from completing login.",
      steps: [
        "Validate the login flow with one affected account and confirm where the authentication handoff is failing.",
        "Check that the reset-password experience still routes to the active identity provider configuration.",
        "Compare the failing login trace with a known-good account to isolate callback or mapping differences.",
      ],
      escalateIf: "Escalate if password reset succeeds but users still cannot authenticate after identity checks.",
    };
  }

  if (lower.includes("charged") || area === "Billing" || category === "Duplicate Charge") {
    return {
      likelyIssue: "Duplicate billing activity or invoice retries are likely generating repeated charges.",
      steps: [
        "Confirm whether the second charge was triggered by a retry event or a duplicate invoice.",
        "Review account-level billing history and payment webhook activity for duplicates.",
        "Pause automated retries on the affected account until the charge path is confirmed.",
      ],
      escalateIf: "Escalate if repeated charges are affecting multiple customers or duplicate invoices continue to appear.",
    };
  }

  if (lower.includes("vpn") || area === "Network Services") {
    return {
      likelyIssue: "VPN client configuration or authentication handoff is blocking successful connection setup.",
      steps: [
        "Confirm the client is using the latest VPN profile and endpoint configuration.",
        "Check whether authentication succeeds but tunnel setup fails during the handshake stage.",
        "Test the connection on a known-good network to separate client issues from gateway issues.",
      ],
      escalateIf: "Escalate if authentication succeeds but the tunnel still fails across multiple networks or devices.",
    };
  }

  if (lower.includes("dashboard") || lower.includes("analytics")) {
    return {
      likelyIssue: "Slow dashboard query execution or resource saturation is causing requests to fail or time out.",
      steps: [
        "Review the slowest dashboard query and compare it against the expected execution baseline.",
        "Check whether cache misses or worker saturation increased during the incident window.",
        "Reduce query scope or route affected users to a lighter reporting path while the issue is investigated.",
      ],
      escalateIf: "Escalate if timeout rates continue to increase after query tuning or temporary traffic mitigation.",
    };
  }

  return {
    likelyIssue: "The issue appears new or too broad to match a known resolution pattern confidently.",
    steps: [
      "Capture a more specific reproduction path, affected scope, and any error details from the request.",
      "Check whether the issue aligns with a recent release, infrastructure change, or policy update.",
      "Use similar-case review to gather adjacent incidents before handing off for deeper triage.",
    ],
    escalateIf: "Escalate if the issue impacts multiple users, blocks a core workflow, or lacks a clear owner.",
  };
}

function FieldHint({ text }) {
  return (
    <span className="field-hint">
      <span className="field-hint__icon">?</span>
      <span className="field-hint__tooltip">{text}</span>
    </span>
  );
}

function StatusBadge({ status }) {
  return <span className={`status-badge status-badge--${getStatusTone(status)}`}>{status}</span>;
}

function SourceSkeletonCard() {
  return (
    <article className="source-card source-card--skeleton" aria-hidden="true">
      <div className="skeleton skeleton--title" />
      <div className="skeleton skeleton--meta" />
      <div className="skeleton skeleton--line" />
      <div className="skeleton skeleton--line short" />
    </article>
  );
}

function SourceCaseCard({ item }) {
  return (
    <article className="source-card">
      <div className="source-card__top">
        <div>
          <strong>{item.ticketId}</strong>
          <p>{item.title}</p>
        </div>
        <StatusBadge status={item.status} />
      </div>
      <div className="source-card__meta">
        <span>{item.category}</span>
        <span>{item.date}</span>
      </div>
      <p className="source-card__summary">{item.summary}</p>
    </article>
  );
}

export default function App() {
  const [incidents, setIncidents] = useState(INITIAL_INCIDENTS);
  const [description, setDescription] = useState("");
  const [affectedArea, setAffectedArea] = useState("");
  const [issueCategory, setIssueCategory] = useState("");
  const [intakeMatches, setIntakeMatches] = useState([]);
  const [intakeLoading, setIntakeLoading] = useState(false);
  const [intakeChecked, setIntakeChecked] = useState(false);
  const [workspaceQuery, setWorkspaceQuery] = useState(INITIAL_INCIDENTS[0].ticketId);
  const [activeIncident, setActiveIncident] = useState(INITIAL_INCIDENTS[0]);
  const [guidance, setGuidance] = useState(INITIAL_INCIDENTS[0].guidance);
  const [guidanceLoading, setGuidanceLoading] = useState(false);
  const [sourceCases, setSourceCases] = useState(INITIAL_INCIDENTS[0].sourceCases);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [guidanceVisible, setGuidanceVisible] = useState(true);

  const sourceCountLabel = sourceLoading ? "Loading" : `${sourceCases.length} cases`;

  function runIntakeCheck() {
    if (!description.trim()) return;

    setIntakeLoading(true);
    setIntakeChecked(true);

    window.setTimeout(() => {
      const matches = incidents
        .map((incident) => ({ ...incident, score: scoreSimilarity(description, incident) }))
        .filter((incident) => incident.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
      setIntakeMatches(matches);
      setIntakeLoading(false);
    }, 550);
  }

  function createIncidentFromIntake() {
    if (!description.trim()) return;

    const newIncident = {
      id: 100000 + incidents.length + 1,
      ticketId: buildTicketId(),
      title: description.slice(0, 64),
      category: issueCategory || "General request",
      area: affectedArea || "Unassigned",
      status: "open",
      date: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
      description: description.trim(),
      guidance: buildGuidance(description, affectedArea, issueCategory),
      sourceCases: [],
    };

    setIncidents((current) => [newIncident, ...current]);
    setActiveIncident(newIncident);
    setWorkspaceQuery(newIncident.ticketId);
    setGuidance(newIncident.guidance);
    setSourceCases([]);
    setDescription("");
    setAffectedArea("");
    setIssueCategory("");
    setIntakeMatches([]);
    setIntakeChecked(false);
    setGuidanceVisible(false);
    window.setTimeout(() => setGuidanceVisible(true), 30);
  }

  function openIncident() {
    const query = workspaceQuery.trim().toLowerCase();
    const found = incidents.find(
      (incident) =>
        String(incident.id) === query ||
        incident.ticketId.toLowerCase() === query,
    );

    if (!found) return;

    setActiveIncident(found);
    setGuidance(found.guidance);
    setSourceCases(found.sourceCases);
    setGuidanceLoading(false);
    setSourceLoading(false);
    setGuidanceVisible(false);
    window.setTimeout(() => setGuidanceVisible(true), 30);
  }

  function findSimilarCases() {
    if (!activeIncident) return;

    setSourceLoading(true);
    window.setTimeout(() => {
      const similar = incidents
        .filter((incident) => incident.ticketId !== activeIncident.ticketId)
        .map((incident) => ({ ...incident, score: scoreSimilarity(activeIncident.description, incident) }))
        .filter((incident) => incident.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 4)
        .map((incident) => ({
          ticketId: incident.ticketId,
          title: incident.title,
          category: incident.category,
          status: incident.status,
          date: incident.date,
          summary: incident.description,
        }));

      setSourceCases(similar);
      setSourceLoading(false);
    }, 700);
  }

  function regenerateGuidance() {
    if (!activeIncident) return;

    setGuidanceLoading(true);
    setGuidanceVisible(false);

    window.setTimeout(() => {
      setGuidance(buildGuidance(activeIncident.description, activeIncident.area, activeIncident.category));
      setGuidanceLoading(false);
      setGuidanceVisible(true);
    }, 680);
  }

  return (
    <div className="workspace-shell">
      <header className="workspace-header">
        <div className="workspace-header__identity">
          <span className="workspace-header__eyebrow">DecisionLens</span>
          <div className="workspace-header__title-row">
            <h1>Support incident workspace</h1>
            <span className="workspace-header__status">Demo ready</span>
          </div>
          <p>Search historical cases, triage new incidents, and generate guided next steps.</p>
        </div>
        <div className="workspace-header__actions">
          <button
            className="button button--ghost button--compact"
            type="button"
            onClick={() => {
              setDescription("");
              setAffectedArea("");
              setIssueCategory("");
            }}
          >
            New incident
          </button>
          <button
            className="button button--secondary button--compact"
            type="button"
            onClick={findSimilarCases}
          >
            Recent cases
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="panel panel--intake">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">New Incident</span>
              <h2>Intake</h2>
            </div>
          </div>

          <div className="field-group">
            <label htmlFor="incident-description">
              What happened?
              <FieldHint text="Describe the issue in plain language so we can surface related incidents." />
            </label>
            <textarea
              id="incident-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Describe the incident, impact, and any symptoms the support analyst should know."
            />
          </div>

          <div className="intake-row">
            <div className="field-group">
              <label htmlFor="affected-area">
                Affected area
                <FieldHint text="Optional. Add this if you know it. We use it as a hint, not a requirement." />
              </label>
              <select id="affected-area" value={affectedArea} onChange={(event) => setAffectedArea(event.target.value)}>
                <option value="">Select if known</option>
                {AFFECTED_AREAS.map((area) => (
                  <option key={area} value={area}>
                    {area}
                  </option>
                ))}
              </select>
            </div>

            <div className="field-group">
              <label htmlFor="issue-category">
                Issue category
                <FieldHint text="Optional. Helpful when you're confident, but the description still drives matching." />
              </label>
              <select id="issue-category" value={issueCategory} onChange={(event) => setIssueCategory(event.target.value)}>
                <option value="">Select if known</option>
                {ISSUE_CATEGORIES.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="button-row">
            <button className="button button--secondary" type="button" onClick={runIntakeCheck}>
              {intakeLoading ? "Checking..." : "Check similar cases"}
            </button>
            <button className="button button--primary" type="button" onClick={createIncidentFromIntake}>
              Create incident
            </button>
          </div>

          <div className="intake-review">
            <div className="intake-review__header">
              <span className="panel__eyebrow">Similar cases</span>
              {intakeChecked ? <span className="panel__meta">{intakeMatches.length} found</span> : null}
            </div>

            {intakeLoading ? (
              <div className="intake-loading">
                <div className="skeleton skeleton--line" />
                <div className="skeleton skeleton--line short" />
              </div>
            ) : intakeMatches.length ? (
              <div className="intake-results">
                {intakeMatches.map((item) => (
                  <article className="intake-result" key={item.ticketId}>
                    <div className="intake-result__top">
                      <strong>{item.ticketId}</strong>
                      <StatusBadge status={item.status} />
                    </div>
                    <p>{item.title}</p>
                    <span>{item.category} • {item.area}</span>
                  </article>
                ))}
              </div>
            ) : (
              <p className="panel__hint">
                {intakeChecked
                  ? "No close matches yet. You can still create the incident and continue from the workspace."
                  : "Use similar-case review when you want extra context before creating a new incident."}
              </p>
            )}
          </div>
        </section>

        <section className="panel panel--workspace">
          <div className="panel__header">
            <div>
              <span className="panel__eyebrow">Incident Workspace</span>
              <h2>Inspect and resolve</h2>
            </div>
          </div>

          <div className="workspace-toolbar">
            <div className="inline-input">
              <input
                value={workspaceQuery}
                onChange={(event) => setWorkspaceQuery(event.target.value)}
                placeholder="Incident ID or Ticket ID"
              />
              <button className="button button--ghost" type="button" onClick={openIncident}>
                Open incident
              </button>
            </div>

            <div className="workspace-actions">
              <button className="button button--primary" type="button" onClick={findSimilarCases}>
                Find similar
              </button>
              <button className="button button--secondary" type="button" onClick={regenerateGuidance}>
                Regenerate guidance
              </button>
            </div>
          </div>

          <div className="incident-meta">
            <div className="incident-meta__main">
              <span className="incident-meta__id">{activeIncident.ticketId}</span>
              <h3>{activeIncident.title}</h3>
            </div>
            <div className="incident-meta__row">
              <span>{activeIncident.category}</span>
              <span>{activeIncident.area}</span>
              <StatusBadge status={activeIncident.status} />
              <span>{activeIncident.date}</span>
            </div>
          </div>

          <div className="workspace-content">
            <section className="guidance-card">
              <div className="section-title">
                <h3>Guided next steps</h3>
              </div>

              {guidanceLoading ? (
                <div className="guidance-skeleton">
                  <div className="skeleton skeleton--label" />
                  <div className="skeleton skeleton--line" />
                  <div className="skeleton skeleton--label" />
                  <div className="skeleton skeleton--line" />
                  <div className="skeleton skeleton--line short" />
                  <div className="skeleton skeleton--line short" />
                </div>
              ) : (
                <div className={`guidance-card__body ${guidanceVisible ? "fade-in" : ""}`}>
                  <div className="guidance-section">
                    <span className="section-kicker">Likely issue</span>
                    <p className="guidance-muted">{guidance.likelyIssue || EMPTY_GUIDANCE.likelyIssue}</p>
                  </div>

                  <div className="guidance-section">
                    <span className="section-kicker">Next steps</span>
                    <ol className="guidance-list">
                      {guidance.steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ol>
                  </div>

                  <div className="guidance-section guidance-section--warning">
                    <span className="section-kicker">Escalate if</span>
                    <p>{guidance.escalateIf}</p>
                  </div>
                </div>
              )}
            </section>

            <aside className="source-panel">
              <div className="section-title">
                <h3>Source cases</h3>
                <span className="panel__meta">{sourceCountLabel}</span>
              </div>

              <div className="source-panel__stack">
                {sourceLoading
                  ? Array.from({ length: 3 }).map((_, index) => <SourceSkeletonCard key={index} />)
                  : sourceCases.map((item) => <SourceCaseCard item={item} key={item.ticketId} />)}
              </div>
            </aside>
          </div>
        </section>
      </main>
    </div>
  );
}
