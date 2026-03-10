CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS incidents (
    id                    SERIAL PRIMARY KEY,
    ticket_id             VARCHAR(20) UNIQUE NOT NULL,
    created_at            TIMESTAMP,
    customer_id           VARCHAR(20),
    customer_segment      VARCHAR(50),
    channel               VARCHAR(50),
    product_area          VARCHAR(100),
    issue_type            VARCHAR(100),
    priority              VARCHAR(20),
    status                VARCHAR(50),
    sla_plan              VARCHAR(20),
    initial_message       TEXT,
    agent_first_reply     TEXT,
    resolution_summary    TEXT,
    resolution_time_hours FLOAT,
    reopened              BOOLEAN,
    customer_sentiment    VARCHAR(20),
    csat_score            INTEGER,
    has_attachment        BOOLEAN,
    platform              VARCHAR(50),
    region                VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS incident_embeddings (
    id            SERIAL PRIMARY KEY,
    incident_id   INTEGER REFERENCES incidents(id) ON DELETE CASCADE,
    embedding_vector vector(1536),
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS severity_predictions (
    id            SERIAL PRIMARY KEY,
    incident_id   INTEGER REFERENCES incidents(id) ON DELETE CASCADE,
    predicted_severity VARCHAR(20),
    confidence    FLOAT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incidents_ticket_id    ON incidents(ticket_id);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at   ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_issue_type   ON incidents(issue_type);
CREATE INDEX IF NOT EXISTS idx_incidents_priority     ON incidents(priority);
CREATE INDEX IF NOT EXISTS idx_incidents_status       ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_product_area ON incidents(product_area);
CREATE INDEX IF NOT EXISTS idx_embeddings_incident_id ON incident_embeddings(incident_id);