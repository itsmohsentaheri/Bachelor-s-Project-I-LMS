-- Phase 0, step 4: core tables for the LLM tutor
-- Assumes PostgreSQL + pgvector and an existing `chunks` table.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ---------- Graph versions ----------
CREATE TABLE graph_versions (
    version     integer PRIMARY KEY,
    status      text NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'frozen', 'archived')),
    notes       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    frozen_at   timestamptz
);

-- ---------- Nodes (atoms) ----------
CREATE TABLE nodes (
    node_id             text NOT NULL,   -- stable hash of normalized learning_objective + source_ref
    graph_version       integer NOT NULL REFERENCES graph_versions(version),
    title               text NOT NULL,
    learning_objective  text NOT NULL,
    type                text NOT NULL
                        CHECK (type IN ('concept', 'practical', 'test', 'reading')),
    difficulty_weight   smallint NOT NULL CHECK (difficulty_weight BETWEEN 1 AND 5),
    estimated_time_min  smallint NOT NULL CHECK (estimated_time_min > 0),
    source_ref          text[] NOT NULL DEFAULT '{}',   -- chunk_ids from the chunks table
    misconceptions      jsonb NOT NULL DEFAULT '[]',    -- [{id, description}]
    needs_review        boolean NOT NULL DEFAULT false,
    parent_topic        text,                           -- syllabus section (for coverage check)
    PRIMARY KEY (node_id, graph_version)
);

-- ---------- Edges (prerequisites): from_id is a prerequisite of to_id ----------
CREATE TABLE edges (
    graph_version  integer NOT NULL,
    from_id        text NOT NULL,
    to_id          text NOT NULL,
    confidence     real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    rationale      text,
    origin         text NOT NULL DEFAULT 'local'
                   CHECK (origin IN ('local', 'cross_branch', 'manual')),
    PRIMARY KEY (graph_version, from_id, to_id),
    FOREIGN KEY (from_id, graph_version) REFERENCES nodes(node_id, graph_version),
    FOREIGN KEY (to_id, graph_version)   REFERENCES nodes(node_id, graph_version),
    CHECK (from_id <> to_id)
);
CREATE INDEX edges_to_idx ON edges (graph_version, to_id);

-- ---------- Items (questions) ----------
CREATE TABLE items (
    item_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id         text NOT NULL,
    graph_version   integer NOT NULL,
    style           text NOT NULL CHECK (style IN ('conceptual', 'application')),
    kind            text NOT NULL CHECK (kind IN ('mcq', 'open')),
    stem            text NOT NULL,
    options         jsonb,            -- mcq: [{key, text}]
    correct_answer  text,             -- never sent to the teacher LLM
    rubric          jsonb,            -- open: [{criterion, max_points}]
    distractor_map  jsonb,            -- mcq: {option_key: misconception_id}
    difficulty_init real NOT NULL,    -- from difficulty_weight (logit or Elo scale)
    elo_rating      real NOT NULL DEFAULT 1000,
    n_answers       integer NOT NULL DEFAULT 0,
    verified        boolean NOT NULL DEFAULT false,   -- passed solver check / human review
    created_at      timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (node_id, graph_version) REFERENCES nodes(node_id, graph_version)
);
CREATE INDEX items_node_idx ON items (graph_version, node_id);

-- ---------- Learners ----------
CREATE TABLE learners (
    learner_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at     timestamptz NOT NULL DEFAULT now(),
    graph_version  integer NOT NULL REFERENCES graph_versions(version),
    profile        jsonb NOT NULL DEFAULT '{}',   -- onboarding answers (minimal data only)
    goal           jsonb NOT NULL DEFAULT '{}',   -- target nodes / free-text goal
    elo_rating     real NOT NULL DEFAULT 1000
);

-- ---------- Per-learner, per-node state ----------
CREATE TABLE node_state (
    learner_id         uuid NOT NULL REFERENCES learners(learner_id),
    node_id            text NOT NULL,
    graph_version      integer NOT NULL,
    p_l                real NOT NULL CHECK (p_l BETWEEN 0.01 AND 0.99),   -- BKT mastery
    n_observations     integer NOT NULL DEFAULT 0,
    consecutive_fails  integer NOT NULL DEFAULT 0,
    last_attempt_at    timestamptz,
    -- FSRS card state (null until the node is first mastered)
    fsrs_state         smallint,
    fsrs_stability     real,
    fsrs_difficulty    real,
    fsrs_due           timestamptz,
    fsrs_last_review   timestamptz,
    PRIMARY KEY (learner_id, node_id, graph_version),
    FOREIGN KEY (node_id, graph_version) REFERENCES nodes(node_id, graph_version)
);
CREATE INDEX node_state_due_idx ON node_state (learner_id, fsrs_due)
    WHERE fsrs_due IS NOT NULL;

-- ---------- Events (append-only log) ----------
CREATE TABLE events (
    event_id       bigserial PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    learner_id     uuid REFERENCES learners(learner_id),   -- null for system/graph-build events
    type           text NOT NULL,       -- e.g. answer, diagnosis, decision, llm_teach, llm_judge
    payload        jsonb NOT NULL,
    graph_version  integer,
    model_version  text,
    prompt_version text
);
CREATE INDEX events_learner_ts_idx ON events (learner_id, ts);
CREATE INDEX events_type_idx ON events (type);

-- Enforce append-only: block UPDATE and DELETE on events
CREATE FUNCTION events_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'events table is append-only';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER events_no_update_delete
    BEFORE UPDATE OR DELETE ON events
    FOR EACH ROW EXECUTE FUNCTION events_append_only();