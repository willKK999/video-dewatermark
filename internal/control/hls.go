package control

import (
	"errors"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/minio/minio-go/v7"
)

func validHLSInput(input string) bool {
	if len(input) > 4096 || strings.ContainsAny(input, "\r\n\x00") {
		return false
	}
	u, err := url.Parse(input)
	if err != nil || !strings.HasSuffix(strings.ToLower(u.Path), ".m3u8") || u.Fragment != "" {
		return false
	}
	if u.Scheme != "" {
		return (u.Scheme == "https" || u.Scheme == "http") && u.Hostname() != "" && u.User == nil
	}
	return !strings.HasPrefix(input, "/") && path.Clean(u.Path) == u.Path && !strings.HasPrefix(path.Clean(input), "../") && u.RawQuery == ""
}

var hlsName = regexp.MustCompile(`^(index\.m3u8|[0-9]{6}\.ts)$`)

// 分片地址保持同源；每次请求都检查租户和有效期，不在播放列表暴露签名链接。
func (s *Server) hlsMedia(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if !hlsName.MatchString(name) {
		http.NotFound(w, r)
		return
	}
	var prefix string
	err := s.DB.QueryRow(r.Context(), `SELECT result_meta->>'hls_prefix' FROM jobs WHERE id=$1 AND tenant_id=$2 AND status='SUCCEEDED' AND result_expires_at>now() AND result_meta->>'hls_prefix' IS NOT NULL`, r.PathValue("id"), tenant(r)).Scan(&prefix)
	if err != nil {
		fail(w, r, 404, "NOT_FOUND", "HLS 成品不存在或已过期")
		return
	}
	obj, err := s.Store.GetObject(r.Context(), s.Bucket, prefix+name, minio.GetObjectOptions{})
	if err != nil {
		fail(w, r, 503, "STORAGE_UNAVAILABLE", "无法读取 HLS 成品")
		return
	}
	defer obj.Close()
	info, err := obj.Stat()
	if err != nil {
		fail(w, r, 404, "NOT_FOUND", "HLS 分片不存在")
		return
	}
	w.Header().Set("Content-Type", info.ContentType)
	http.ServeContent(w, r, name, time.Time{}, obj)
}

// 重试保留成功块的检查点；未完成块会重新执行完整流水线。
func (s *Server) retryHLS(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	tx, err := s.DB.Begin(ctx)
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法重试")
		return
	}
	defer tx.Rollback(ctx)
	if _, err = tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtextextended($1,0))`, "tenant:"+tenant(r)); err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法重试")
		return
	}
	if s.MaxPendingJobs > 0 {
		var pending int
		if err = tx.QueryRow(ctx, `SELECT count(*) FROM jobs WHERE tenant_id=$1 AND status IN ('QUEUED','RUNNING','CANCEL_REQUESTED')`, tenant(r)).Scan(&pending); err != nil {
			fail(w, r, 503, "DATABASE_UNAVAILABLE", "无法检查任务配额")
			return
		}
		if pending >= s.MaxPendingJobs {
			fail(w, r, 429, "QUOTA_EXCEEDED", "当前等待处理的任务过多")
			return
		}
	}
	var attempt int
	err = tx.QueryRow(ctx, `UPDATE jobs SET status='QUEUED',stage=NULL,error_code=NULL,error_message=NULL,updated_at=now()
 WHERE id=$1 AND tenant_id=$2 AND options->>'hls'='true' AND status IN ('FAILED','NEEDS_REVIEW') AND attempt<3 RETURNING attempt`, r.PathValue("id"), tenant(r)).Scan(&attempt)
	if errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 409, "NOT_RETRYABLE", "仅可重试失败的 HLS 任务，最多执行三次")
		return
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法重试任务")
		return
	}
	_, err = tx.Exec(ctx, `INSERT INTO outbox(job_id,dispatch_no) VALUES($1,$2) ON CONFLICT(job_id,dispatch_no) DO UPDATE SET sent_at=NULL,available_at=now()`, r.PathValue("id"), attempt)
	if err != nil || tx.Commit(ctx) != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "重试提交失败")
		return
	}
	s.getWithStatus(w, r, 202)
}
