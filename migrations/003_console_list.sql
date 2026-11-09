CREATE INDEX IF NOT EXISTS jobs_tenant_created ON jobs(tenant_id,created_at DESC,id DESC);
