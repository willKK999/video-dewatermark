-- Idempotent hardening merged from the parallel V1.1 implementation.
-- 001 stays the original cloud schema. Re-running AUTO_MIGRATE applies this file
-- on databases that were created before the merge.

ALTER TABLE uploads DROP CONSTRAINT IF EXISTS uploads_content_type_check;
ALTER TABLE uploads ADD CONSTRAINT uploads_content_type_check
 CHECK (content_type IN ('video/mp4', 'video/quicktime'));

CREATE UNIQUE INDEX IF NOT EXISTS uploads_tenant_id_uidx ON uploads (tenant_id, id);

DO $$ BEGIN
 ALTER TABLE jobs ADD CONSTRAINT jobs_upload_tenant_fk
  FOREIGN KEY (tenant_id, upload_id) REFERENCES uploads (tenant_id, id);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
 ALTER TABLE jobs ADD CONSTRAINT jobs_running_lease_chk CHECK (
  (status = 'RUNNING' AND lease_token IS NOT NULL AND lease_until IS NOT NULL)
  OR status <> 'RUNNING'
 );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
 ALTER TABLE jobs ADD CONSTRAINT jobs_result_ptr_chk CHECK (
  result_key IS NULL OR status = 'SUCCEEDED'
 );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE attempts ADD COLUMN IF NOT EXISTS image_digest text;
ALTER TABLE attempts ADD COLUMN IF NOT EXISTS model_versions jsonb;
UPDATE attempts SET image_digest = 'unrecorded' WHERE image_digest IS NULL;
UPDATE attempts SET model_versions = '{}'::jsonb WHERE model_versions IS NULL;
ALTER TABLE attempts ALTER COLUMN image_digest SET DEFAULT 'unrecorded';
ALTER TABLE attempts ALTER COLUMN model_versions SET DEFAULT '{}'::jsonb;
ALTER TABLE attempts ALTER COLUMN image_digest SET NOT NULL;
ALTER TABLE attempts ALTER COLUMN model_versions SET NOT NULL;

DO $$ BEGIN
 ALTER TABLE attempts ADD CONSTRAINT attempts_no_chk CHECK (attempt_no BETWEEN 1 AND 3);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Terminal rows cannot be rewritten. QUEUED can cancel immediately because no
-- worker holds a lease; RUNNING must pass through CANCEL_REQUESTED.
CREATE OR REPLACE FUNCTION guard_job() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.status IN ('SUCCEEDED', 'FAILED', 'NEEDS_REVIEW', 'CANCELED') THEN
  RAISE EXCEPTION 'terminal job immutable';
 END IF;
 IF NEW.status <> OLD.status AND NOT (
  (OLD.status = 'QUEUED' AND NEW.status IN ('RUNNING', 'CANCEL_REQUESTED', 'CANCELED')) OR
  (OLD.status = 'RUNNING' AND NEW.status IN ('QUEUED', 'SUCCEEDED', 'FAILED', 'NEEDS_REVIEW', 'CANCEL_REQUESTED')) OR
  (OLD.status = 'CANCEL_REQUESTED' AND NEW.status = 'CANCELED')
 ) THEN
  RAISE EXCEPTION 'invalid state transition % -> %', OLD.status, NEW.status;
 END IF;
 RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS jobs_guard ON jobs;
CREATE TRIGGER jobs_guard BEFORE UPDATE ON jobs FOR EACH ROW EXECUTE FUNCTION guard_job();
