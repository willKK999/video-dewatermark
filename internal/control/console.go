package control

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/minio/minio-go/v7"
)

//go:embed web/*
var consoleFiles embed.FS

func consolePage(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	b, _ := consoleFiles.ReadFile("web/index.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
	w.Write(b)
}

func (s *Server) list(w http.ResponseWriter, r *http.Request) {
	rows, err := s.DB.Query(r.Context(), `SELECT j.id,j.status,j.stage,j.attempt,j.pipeline_version,j.created_at,j.updated_at,j.progress,j.result_meta,j.result_expires_at,j.error_code,j.error_message,u.filename FROM jobs j JOIN uploads u ON u.id=j.upload_id WHERE j.tenant_id=$1 ORDER BY j.created_at DESC,j.id DESC LIMIT 100`, tenant(r))
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法读取任务")
		return
	}
	defer rows.Close()
	type entry struct {
		Job      Job    `json:"job"`
		Filename string `json:"filename"`
	}
	entries := []entry{}
	for rows.Next() {
		var e entry
		var meta []byte
		var expires *time.Time
		var code, message *string
		j := &e.Job
		if err = rows.Scan(&j.ID, &j.Status, &j.Stage, &j.Attempt, &j.PipelineVersion, &j.CreatedAt, &j.UpdatedAt, &j.Progress, &meta, &expires, &code, &message, &e.Filename); err != nil {
			break
		}
		if code != nil {
			j.Error = map[string]any{"code": *code, "message": message, "retryable": j.Status == "QUEUED"}
		}
		// 列表轮询只访问数据库；播放时再检查存储，避免逐条签名和查询对象。
		if j.Status == "SUCCEEDED" {
			result := map[string]any{}
			if len(meta) > 0 {
				if err = json.Unmarshal(meta, &result); err != nil {
					break
				}
			}
			if result == nil {
				result = map[string]any{}
			}
			result["expired"] = expires == nil || !expires.After(time.Now())
			j.Result = result
		}
		entries = append(entries, e)
	}
	if err == nil {
		err = rows.Err()
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法读取任务")
		return
	}
	reply(w, 200, map[string]any{"items": entries})
}

// 上传到临时文件后检查实际媒体格式，再写入对象存储；不需要浏览器访问对象存储。
func (s *Server) proxyUpload(w http.ResponseWriter, r *http.Request) {
	var key, ct string
	var size int64
	var expires time.Time
	err := s.DB.QueryRow(r.Context(), `SELECT object_key,content_type,size_bytes,expires_at FROM uploads WHERE id=$1 AND tenant_id=$2`, r.PathValue("id"), tenant(r)).Scan(&key, &ct, &size, &expires)
	if errors.Is(err, pgx.ErrNoRows) {
		fail(w, r, 404, "NOT_FOUND", "上传记录不存在")
		return
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法上传")
		return
	}
	if time.Now().After(expires) {
		fail(w, r, 422, "UPLOAD_EXPIRED", "上传已过期，请重新选择视频")
		return
	}
	if r.ContentLength != size || r.Header.Get("Content-Type") != ct {
		fail(w, r, 422, "UPLOAD_INVALID", "文件大小或类型不匹配")
		return
	}
	f, err := os.CreateTemp("", "console-video-*")
	if err != nil {
		fail(w, r, 503, "SERVICE_UNAVAILABLE", "无法暂存视频")
		return
	}
	defer os.Remove(f.Name())
	defer f.Close()
	n, err := io.Copy(f, http.MaxBytesReader(w, r.Body, size))
	if err != nil || n != size {
		fail(w, r, 413, "UPLOAD_INVALID", "上传不完整或超过大小限制")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()
	output, err := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams", "-show_format", "-of", "json", f.Name()).Output()
	var probe struct {
		Streams []struct {
			Type  string `json:"codec_type"`
			Codec string `json:"codec_name"`
			Pixel string `json:"pix_fmt"`
		}
		Format struct {
			Duration string `json:"duration"`
			Name     string `json:"format_name"`
		}
	}
	if err != nil || json.Unmarshal(output, &probe) != nil || !validConsoleVideo(probe.Streams, probe.Format.Duration, probe.Format.Name) {
		fail(w, r, 422, "UNSUPPORTED_VIDEO", "仅支持不超过 30 秒的 H.264 8-bit 4:2:0 MP4/MOV 视频，请先转换格式")
		return
	}
	f.Seek(0, 0)
	opts := minio.PutObjectOptions{ContentType: ct}
	if s.DisableObjectTagging {
		opts.UserMetadata = map[string]string{"kind": "intermediate"}
	} else {
		opts.UserTags = map[string]string{"kind": "intermediate"}
	}
	_, err = s.Store.PutObject(r.Context(), s.Bucket, key, f, size, opts)
	if err != nil {
		fail(w, r, 503, "STORAGE_UNAVAILABLE", "存储暂时不可用，请重试")
		return
	}
	reply(w, 200, map[string]bool{"uploaded": true})
}

func (s *Server) media(w http.ResponseWriter, r *http.Request) {
	var key string
	var expires *time.Time
	query := `SELECT CASE WHEN options->>'hls'='true' THEN result_meta->>'source_key' ELSE input_key END,NULL::timestamptz FROM jobs WHERE id=$1 AND tenant_id=$2`
	if r.PathValue("kind") == "result" {
		query = `SELECT result_key,result_expires_at FROM jobs WHERE id=$1 AND tenant_id=$2 AND status='SUCCEEDED' AND result_key IS NOT NULL AND result_expires_at>now()`
	} else if r.PathValue("kind") == "bundle" {
		query = `SELECT result_meta->>'bundle_key',result_expires_at FROM jobs WHERE id=$1 AND tenant_id=$2 AND status='SUCCEEDED' AND result_meta->>'bundle_key' IS NOT NULL AND result_expires_at>now()`
	} else if r.PathValue("kind") != "source" {
		http.NotFound(w, r)
		return
	}
	err := s.DB.QueryRow(r.Context(), query, r.PathValue("id"), tenant(r)).Scan(&key, &expires)
	if errors.Is(err, pgx.ErrNoRows) || (expires != nil && !expires.After(time.Now())) {
		fail(w, r, 404, "NOT_FOUND", "视频不存在或已过期")
		return
	}
	if err != nil {
		fail(w, r, 503, "DATABASE_UNAVAILABLE", "暂时无法读取视频")
		return
	}
	obj, err := s.Store.GetObject(r.Context(), s.Bucket, key, minio.GetObjectOptions{})
	if err != nil {
		fail(w, r, 503, "STORAGE_UNAVAILABLE", "暂时无法读取视频")
		return
	}
	defer obj.Close()
	info, err := obj.Stat()
	if err != nil {
		fail(w, r, 404, "NOT_FOUND", "视频不存在或已过期")
		return
	}
	w.Header().Set("Content-Type", info.ContentType)
	name := "视频.mp4"
	if r.PathValue("kind") == "bundle" {
		name = "HLS视频.zip"
	}
	http.ServeContent(w, r, name, info.LastModified, obj)
}

func validConsoleVideo(streams []struct {
	Type  string `json:"codec_type"`
	Codec string `json:"codec_name"`
	Pixel string `json:"pix_fmt"`
}, duration, format string) bool {
	d, err := strconv.ParseFloat(duration, 64)
	if err != nil || !(d > 0 && d <= 30) || !strings.Contains(format, "mp4") {
		return false
	}
	count := 0
	for _, s := range streams {
		if s.Type == "video" {
			count++
			if s.Codec != "h264" || (s.Pixel != "yuv420p" && s.Pixel != "yuvj420p") {
				return false
			}
		}
	}
	return count == 1
}
