CREATE TABLE IF NOT EXISTS uploads (
 id text PRIMARY KEY, tenant_id text NOT NULL, object_key text NOT NULL UNIQUE,
 filename text NOT NULL, size_bytes bigint NOT NULL CHECK(size_bytes BETWEEN 1 AND 209715200),
 content_type text NOT NULL, checksum text, expires_at timestamptz NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
 id text PRIMARY KEY, tenant_id text NOT NULL, upload_id text NOT NULL REFERENCES uploads(id),
 input_key text NOT NULL, idempotency_key text NOT NULL, request_hash text NOT NULL,
 status text NOT NULL CHECK(status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED','NEEDS_REVIEW','CANCEL_REQUESTED','CANCELED')),
 stage text CHECK(stage IN ('probe','detect','mask','inpaint','encode','publish')),
 attempt integer NOT NULL DEFAULT 0 CHECK(attempt BETWEEN 0 AND 3),
 lease_token text, lease_until timestamptz, options jsonb NOT NULL,
 pipeline_version text NOT NULL, error_code text, error_message text,
 result_key text, result_meta jsonb, result_expires_at timestamptz,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(tenant_id,idempotency_key)
);
CREATE INDEX IF NOT EXISTS jobs_lease ON jobs(lease_until) WHERE status IN ('RUNNING','CANCEL_REQUESTED');
CREATE TABLE IF NOT EXISTS outbox (
 id bigserial PRIMARY KEY, job_id text NOT NULL REFERENCES jobs(id), dispatch_no integer NOT NULL,
 available_at timestamptz NOT NULL DEFAULT now(), sent_at timestamptz,
 UNIQUE(job_id,dispatch_no)
);
CREATE INDEX IF NOT EXISTS outbox_ready ON outbox(available_at) WHERE sent_at IS NULL;
CREATE TABLE IF NOT EXISTS attempts (
 job_id text NOT NULL REFERENCES jobs(id), attempt_no integer NOT NULL,
 worker_id text NOT NULL, started_at timestamptz NOT NULL DEFAULT now(), ended_at timestamptz,
 config_hash text NOT NULL, peak_vram_mib integer, phase_ms jsonb NOT NULL DEFAULT '{}', metrics jsonb NOT NULL DEFAULT '{}', failure_code text,
 PRIMARY KEY(job_id,attempt_no)
);
