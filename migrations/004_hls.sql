ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress jsonb NOT NULL DEFAULT '{}';
ALTER TABLE uploads DROP CONSTRAINT IF EXISTS uploads_content_type_check;
ALTER TABLE uploads ADD CONSTRAINT uploads_content_type_check CHECK(content_type IN ('video/mp4','video/quicktime','application/vnd.apple.mpegurl'));

-- 仅为 HLS 失败块重试开放狭窄的终态转换；成功和取消任务仍不可改写。
CREATE OR REPLACE FUNCTION guard_job() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.status IN ('SUCCEEDED', 'FAILED', 'NEEDS_REVIEW', 'CANCELED') THEN
  IF OLD.status IN ('FAILED','NEEDS_REVIEW') AND OLD.options->>'hls'='true'
    AND NEW.status='QUEUED' AND OLD.attempt<3 AND NEW.attempt=OLD.attempt
    AND NEW.stage IS NULL AND NEW.error_code IS NULL AND NEW.error_message IS NULL
    AND (to_jsonb(NEW)-ARRAY['status','stage','error_code','error_message','updated_at'])
      = (to_jsonb(OLD)-ARRAY['status','stage','error_code','error_message','updated_at']) THEN
   RETURN NEW;
  END IF;
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
