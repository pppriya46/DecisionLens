-- db/schema.sql
-- Decision Lens Database Schema
-- Creates all tables needed for the incident intelligence platform

-- Enable pgvector extension for storing embeddings later
CREATE EXTENSION IF NOT EXISTS vector;

-- Main incidents table
CREATE TABLE IF NOT EXISTS incidents (
    id                      SERIAL PRIMARY KEY,
    number                  VARCHAR(50) UNIQUE NOT NULL,
    incident_state          VARCHAR(50),
    active                  BOOLEAN,
    reassignment_count      INTEGER DEFAULT 0,
    reopen_count            INTEGER DEFAULT 0,
    sys_mod_count           INTEGER DEFAULT 0,
    made_sla                BOOLEAN,
    contact_type            VARCHAR(100),
    location                VARCHAR(100),
    category                VARCHAR(100),
    subcategory             VARCHAR(100),
    u_symptom               VARCHAR(255),
    impact                  VARCHAR(50),
    urgency                 VARCHAR(50),
    priority                VARCHAR(50),
    assignment_group        VARCHAR(100),
    knowledge               BOOLEAN,
    notify                  VARCHAR(100),
    closed_code             VARCHAR(100),
    opened_at               TIMESTAMP,
    resolved_at             TIMESTAMP,
    closed_at               TIMESTAMP,
    created_at              TIMESTAMP DEFAULT NOW()
);

-- Table to store ML severity predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id                  SERIAL PRIMARY KEY,
    incident_id         INTEGER REFERENCES incidents(id),
    predicted_severity  VARCHAR(50),
    confidence          FLOAT,
    model_version       VARCHAR(50),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Table to store embeddings for RAG system
CREATE TABLE IF NOT EXISTS incident_embeddings (
    id              SERIAL PRIMARY KEY,
    incident_id     INTEGER REFERENCES incidents(id),
    embedding_vector vector(1536),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_incidents_priority 
    ON incidents(priority);

CREATE INDEX IF NOT EXISTS idx_incidents_category 
    ON incidents(category);

CREATE INDEX IF NOT EXISTS idx_incidents_state 
    ON incidents(incident_state);

CREATE INDEX IF NOT EXISTS idx_incidents_opened_at 
    ON incidents(opened_at);