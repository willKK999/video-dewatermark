package control

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/minio/minio-go/v7"
	"github.com/redis/go-redis/v9"
)

type Server struct {
	DB             *pgxpool.Pool
	Redis          *redis.Client
	Store          *minio.Client
	Signer         *minio.Client
	Bucket         string
	Tokens         map[string]string // bearer -> tenant, populated from secret configuration
	Pipeline       string
	MaxPendingJobs int
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/uploads", s.upload)
	mux.HandleFunc("POST /v1/jobs", s.create)
	mux.HandleFunc("GET /v1/jobs/{id}", s.get)
	mux.HandleFunc("POST /v1/jobs/{id}/cancel", s.cancel)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if s.DB.Ping(ctx) != nil || s.Redis.Ping(ctx).Err() != nil {
			reply(w, 503, map[string]string{"status": "unavailable"})
			return
		}
		reply(w, 200, map[string]string{"status": "ok"})
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Header.Set("X-Request-ID", id("req_"))
		w.Header().Set("X-Request-ID", r.Header.Get("X-Request-ID"))
		if r.URL.Path == "/healthz" {
			mux.ServeHTTP(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		tenant := ""
		if strings.HasPrefix(auth, "Bearer ") {
			for token, t := range s.Tokens {
				if subtle.ConstantTimeCompare([]byte(strings.TrimPrefix(auth, "Bearer ")), []byte(token)) == 1 {
					tenant = t
				}
			}
		}
		if tenant == "" {
			fail(w, r, 401, "UNAUTHORIZED", "valid Bearer token required")
			return
		}
		ctx := context.WithValue(r.Context(), tenantContext{}, tenant)
		mux.ServeHTTP(w, r.WithContext(ctx))
	})
}

type tenantContext struct{}

func tenant(r *http.Request) string { return r.Context().Value(tenantContext{}).(string) }
func (s *Server) upload(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Filename    string `json:"filename"`
		Size        int64  `json:"size_bytes"`
		ContentType string `json:"content_type"`
	}
	if decode(w, r, &in) != nil {
		fail(w, r, 400, "INVALID_ARGUMENT", "invalid JSON")
		return
	}
	if in.Size > MaxUpload {
		fail(w, r, 413, "UPLOAD_TOO_LARGE", "maximum input is 200 MiB")
		return
	}
	ext := strings.ToLower(filepath.Ext(in.Filename))
	if in.Size <= 0 || len(in.Filename) > 255 || (ext != ".mp4" && ext != ".mov") || (in.ContentType != "video/mp4" && in.ContentType != "video/quicktime") {
		fail(w, r, 400, "INVALID_ARGUMENT", "MP4/MOV, positive size and video content_type required")
		return
	}
	uid := id("upl_")
	key := tenant(r) + "/uploads/" + uid + "/source"
	expires := time.Now().UTC().Add(15 * time.Minute)
	// Sign Content-Type as well as the object key. Size is checked again with HEAD.
	headers := http.Header{}
	headers.Set("Content-Type", in.ContentType)
	headers.Set("X-Amz-Tagging", "kind=intermediate")
	u, err := s.Signer.PresignHeader(r.Context(), http.MethodPut, s.Bucket, key, 15*time.Minute, url.Values{}, headers)
	if err != nil {
		fail(w, r, 503, "STORAGE_UNAVAILABLE", "cannot sign upload")
		return
	}
	_, err = s.DB.Exec(r.Context(), `INSERT INTO uploads(id,tenant_id,object_key,filename,size_bytes,content_type,expires_at) VALUES($1,$2,$3,$4,$5,$6,$7)`, uid, tenant(r), key, in.Filename, in.Size, in.ContentType, expires)
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot create upload")
		return
	}
	reply(w, 201, map[string]any{"upload_id": uid, "put_url": u.String(), "required_headers": map[string]string{"Content-Type": in.ContentType, "X-Amz-Tagging": "kind=intermediate"}, "expires_at": expires})
}
func (s *Server) load(ctx context.Context, q interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}, jid, t string) (Job, error) {
	var j Job
	var key, code, message *string
	var meta []byte
	var expires *time.Time
	err := q.QueryRow(ctx, `SELECT id,status,stage,attempt,pipeline_version,created_at,updated_at,result_key,result_meta,result_expires_at,error_code,error_message FROM jobs WHERE id=$1 AND tenant_id=$2`, jid, t).Scan(&j.ID, &j.Status, &j.Stage, &j.Attempt, &j.PipelineVersion, &j.CreatedAt, &j.UpdatedAt, &key, &meta, &expires, &code, &message)
	if err != nil {
		return j, err
	}
	if code != nil {
		j.Error = map[string]any{"code": *code, "message": message, "retryable": j.Status == "QUEUED"}
	}
	if j.Status == "SUCCEEDED" && key != nil {
		result := map[string]any{}
		if err = json.Unmarshal(meta, &result); err != nil {
			return j, err
		}
		expired := expires == nil || !expires.After(time.Now())
		if !expired {
			_, err = s.Store.StatObject(ctx, s.Bucket, *key, minio.StatObjectOptions{})
			if err != nil {
				e := minio.ToErrorResponse(err)
				if e.Code == "NoSuchKey" || e.Code == "NoSuchObject" {
					expired = true
				} else {
					return j, err
				}
			}
		}
		result["expired"] = expired
		if !expired {
			ttl := 10 * time.Minute
			if left := time.Until(*expires); left < ttl {
				ttl = left
			}
			u, e := s.Signer.PresignedGetObject(ctx, s.Bucket, *key, ttl, url.Values{})
			if e != nil {
				return j, e
			}
			result["download_url"] = u.String()
			result["expires_at"] = time.Now().UTC().Add(ttl)
		}
		j.Result = result
	}
	return j, nil
}
func (s *Server) create(w http.ResponseWriter, r *http.Request) {
	var in struct {
		UploadID string  `json:"upload_id"`
		Options  Options `json:"options"`
	}
	ik := r.Header.Get("Idempotency-Key")
	if ik == "" || len(ik) > 200 || decode(w, r, &in) != nil || in.UploadID == "" {
		fail(w, r, 400, "INVALID_ARGUMENT", "valid body and Idempotency-Key required")
		return
	}
	if err := in.Options.Validate(); err != nil {
		fail(w, r, 400, "INVALID_ARGUMENT", err.Error())
		return
	}
	body, _ := json.Marshal(in)
	sum := sha256.Sum256(body)
	hash := hex.EncodeToString(sum[:])
	ctx := r.Context()
	t := tenant(r)
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot create job")
		return
	}
	defer tx.Rollback(ctx)
	if _, err = tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, "tenant:"+t); err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot lock job")
		return
	}
	var existing, oldHash string
	err = tx.QueryRow(ctx, `SELECT id,request_hash FROM jobs WHERE tenant_id=$1 AND idempotency_key=$2`, t, ik).Scan(&existing, &oldHash)
	if err == nil {
		if oldHash != hash {
			fail(w, r, 409, "IDEMPOTENCY_CONFLICT", "key used with a different request")
			return
		}
		j, e := s.load(ctx, tx, existing, t)
		if e != nil {
			fail(w, r, 503, "SERVICE_UNAVAILABLE", "cannot query job")
			return
		}
		reply(w, 202, j)
		return
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot query idempotency key")
		return
	}
	if s.MaxPendingJobs > 0 {
		var pending int
		err = tx.QueryRow(ctx, `SELECT count(*) FROM jobs WHERE tenant_id=$1 AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')`, t).Scan(&pending)
		if err != nil {
			fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot check quota")
			return
		}
		if pending >= s.MaxPendingJobs {
			w.Header().Set("Retry-After", "3")
			fail(w, r, 429, "QUOTA_EXCEEDED", "tenant pending job quota reached")
			return
		}
	}
	var source, ct string
	var size int64
	var expires time.Time
	err = tx.QueryRow(ctx, `SELECT object_key,size_bytes,content_type,expires_at FROM uploads WHERE id=$1 AND tenant_id=$2`, in.UploadID, t).Scan(&source, &size, &ct, &expires)
	if errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 404, "NOT_FOUND", "upload not found")
		return
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot read upload")
		return
	}
	if time.Now().After(expires) {
		fail(w, r, 422, "UPLOAD_EXPIRED", "upload reservation expired")
		return
	}
	info, err := s.Store.StatObject(ctx, s.Bucket, source, minio.StatObjectOptions{})
	if err != nil {
		e := minio.ToErrorResponse(err)
		if e.Code == "NoSuchKey" || e.Code == "NoSuchObject" {
			fail(w, r, 422, "UPLOAD_INVALID", "upload missing")
		} else {
			fail(w, r, 503, "STORAGE_UNAVAILABLE", "cannot verify upload")
		}
		return
	}
	if info.Size != size || info.ContentType != ct {
		fail(w, r, 422, "UPLOAD_INVALID", "uploaded size or content type differs")
		return
	}
	jid := id("job_")
	input := t + "/inputs/" + jid + "/source"
	_, err = s.Store.CopyObject(ctx, minio.CopyDestOptions{Bucket: s.Bucket, Object: input, UserTags: map[string]string{"kind": "intermediate"}, ReplaceTags: true}, minio.CopySrcOptions{Bucket: s.Bucket, Object: source, MatchETag: info.ETag})
	if err != nil {
		fail(w, r, 503, "STORAGE_UNAVAILABLE", "cannot freeze input")
		return
	}
	opts, _ := json.Marshal(in.Options)
	_, err = tx.Exec(ctx, `INSERT INTO jobs(id,tenant_id,upload_id,input_key,idempotency_key,request_hash,status,options,pipeline_version) VALUES($1,$2,$3,$4,$5,$6,'QUEUED',$7,$8)`, jid, t, in.UploadID, input, ik, hash, opts, s.Pipeline)
	if err == nil {
		_, err = tx.Exec(ctx, `INSERT INTO outbox(job_id,dispatch_no) VALUES($1,0)`, jid)
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot save job")
		return
	}
	j, err := s.load(ctx, tx, jid, t)
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot read job")
		return
	}
	if err = tx.Commit(ctx); err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot commit job")
		return
	}
	reply(w, 202, j)
}
func (s *Server) get(w http.ResponseWriter, r *http.Request) {
	j, err := s.load(r.Context(), s.DB, r.PathValue("id"), tenant(r))
	if errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 404, "NOT_FOUND", "job not found")
		return
	}
	if err != nil {
		fail(w, r, 503, "SERVICE_UNAVAILABLE", "cannot query job")
		return
	}
	reply(w, 200, j)
}
func (s *Server) cancel(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	jid := r.PathValue("id")
	t := tenant(r)
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot cancel")
		return
	}
	defer tx.Rollback(ctx)
	var status string
	err = tx.QueryRow(ctx, `SELECT status FROM jobs WHERE id=$1 AND tenant_id=$2 FOR UPDATE`, jid, t).Scan(&status)
	if errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 404, "NOT_FOUND", "job not found")
		return
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot cancel")
		return
	}
	changed := true
	switch status {
	case "QUEUED":
		status = "CANCELED"
	case "RUNNING":
		status = "CANCEL_REQUESTED"
	case "CANCEL_REQUESTED", "CANCELED":
		// Repeat cancel returns the current row and must not rewrite a terminal state.
		changed = false
	default:
		fail(w, r, 409, "NOT_CANCELABLE", "job is terminal")
		return
	}
	if changed {
		_, err = tx.Exec(ctx, `UPDATE jobs SET status=$3,updated_at=now(),stage=CASE WHEN $3='CANCELED' THEN NULL ELSE stage END WHERE id=$1 AND tenant_id=$2`, jid, t, status)
		if err != nil {
			fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot commit cancellation")
			return
		}
	}
	if tx.Commit(ctx) != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "cannot commit cancellation")
		return
	}
	s.getWithStatus(w, r, 202)
}
func (s *Server) getWithStatus(w http.ResponseWriter, r *http.Request, status int) {
	j, err := s.load(r.Context(), s.DB, r.PathValue("id"), tenant(r))
	if err != nil {
		fail(w, r, 503, "SERVICE_UNAVAILABLE", "cannot query job")
		return
	}
	reply(w, status, j)
}

// XADD may repeat after a database commit failure. Claims remain guarded by SQL.
func (s *Server) Tick(ctx context.Context) error {
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	rows, err := tx.Query(ctx, `SELECT id,job_id FROM outbox WHERE sent_at IS NULL AND available_at<=now() ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 100`)
	if err != nil {
		return err
	}
	type item struct {
		id  int64
		job string
	}
	items := []item{}
	for rows.Next() {
		var x item
		if err = rows.Scan(&x.id, &x.job); err != nil {
			rows.Close()
			return err
		}
		items = append(items, x)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	for _, x := range items {
		if err = s.Redis.XAdd(ctx, &redis.XAddArgs{Stream: "jobs", Values: map[string]any{"job_id": x.job}}).Err(); err != nil {
			return err
		}
		if _, err = tx.Exec(ctx, `UPDATE outbox SET sent_at=now() WHERE id=$1`, x.id); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}
func (s *Server) Reap(ctx context.Context) error {
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	rows, err := tx.Query(ctx, `SELECT id,status,attempt FROM jobs WHERE status IN ('RUNNING','CANCEL_REQUESTED') AND lease_until<now() FOR UPDATE SKIP LOCKED`)
	if err != nil {
		return err
	}
	type stale struct {
		id, status string
		attempt    int
	}
	jobs := []stale{}
	for rows.Next() {
		var j stale
		if err = rows.Scan(&j.id, &j.status, &j.attempt); err != nil {
			rows.Close()
			return err
		}
		jobs = append(jobs, j)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	for _, j := range jobs {
		next := "QUEUED"
		if j.status == "CANCEL_REQUESTED" {
			next = "CANCELED"
		} else if j.attempt >= 3 {
			next = "FAILED"
		}
		_, err = tx.Exec(ctx, `UPDATE jobs SET status=$2,stage=NULL,lease_token=NULL,lease_until=NULL,error_code='LEASE_EXPIRED',error_message='worker lease expired',updated_at=now() WHERE id=$1`, j.id, next)
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE attempts SET ended_at=now(),failure_code='LEASE_EXPIRED' WHERE job_id=$1 AND attempt_no=$2`, j.id, j.attempt)
		if err != nil {
			return err
		}
		if next == "QUEUED" {
			_, err = tx.Exec(ctx, `INSERT INTO outbox(job_id,dispatch_no,available_at) VALUES($1,$2,now()+interval '5 seconds') ON CONFLICT DO NOTHING`, j.id, j.attempt)
			if err != nil {
				return err
			}
		}
	}
	// Recover lost Redis messages, including a Redis reset after XADD was committed.
	_, err = tx.Exec(ctx, `UPDATE outbox o SET sent_at=NULL,available_at=now() FROM jobs j WHERE o.job_id=j.id AND j.status='QUEUED' AND o.dispatch_no=j.attempt AND o.sent_at<now()-interval '60 seconds'`)
	if err != nil {
		return err
	}
	return tx.Commit(ctx)
}
func (s *Server) String() string { return fmt.Sprintf("pipeline=%s", s.Pipeline) }
