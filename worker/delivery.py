"""Redis is a hint. Database state decides whether a message may be acknowledged."""


def reclaim_batch(previous, response):
    """Advance the XAUTOCLAIM cursor instead of always restarting at 0-0."""
    cursor = previous or "0-0"
    if not response:
        return cursor, []
    nxt = response[0] or "0-0"
    messages = response[1] or []
    return str(nxt), messages


def ack_after_miss(job_row, worker_version):
    """Keep a queued message pending when this worker cannot run its pipeline version.

    A missing job, a terminal job, or a version this worker understands is safe to
    acknowledge: the database already absorbed the duplicate, or the outbox
    reconciler will republish a still-queued compatible job.
    """
    if not job_row:
        return True
    if job_row.get("status") == "QUEUED" and job_row.get("pipeline_version") != worker_version:
        return False
    return True
