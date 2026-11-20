package control

import (
	"context"
	"encoding/json"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestConsoleAccessBoundary(t *testing.T) {
	s := &Server{Tokens: map[string]string{"test-session-token": "kelin"}}
	for _, path := range []string{"/v1/jobs", "/v1/jobs/other/media/source", "/v1/jobs/other/media/result", "/v1/uploads/other/content", "/v1/jobs/other/hls/index.m3u8", "/v1/jobs/other/hls/000000.ts", "/v1/jobs/other/media/bundle", "/v1/jobs/other/retry"} {
		method := "GET"
		if strings.Contains(path, "uploads") {
			method = "PUT"
		}
		if strings.HasSuffix(path, "/retry") {
			method = "POST"
		}
		w := httptest.NewRecorder()
		s.Handler().ServeHTTP(w, httptest.NewRequest(method, path, nil))
		if w.Code != 401 {
			t.Fatalf("%s: %d", path, w.Code)
		}
		var body map[string]any
		if json.Unmarshal(w.Body.Bytes(), &body) != nil || body["request_id"] == nil {
			t.Fatal("缺少错误结构")
		}
	}
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/", nil))
	if w.Code != 200 || !strings.Contains(w.Body.String(), "视频去水印") || strings.Contains(w.Body.String(), "test-session-token") {
		t.Fatal("登录页不正确")
	}
	if w.Header().Get("Cache-Control") != "no-store" || !strings.Contains(w.Header().Get("Content-Security-Policy"), "frame-ancestors 'none'") {
		t.Fatal("缺少安全响应头")
	}
	w = httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/missing", nil)
	r.Header.Set("Authorization", "Bearer test-session-token")
	s.Handler().ServeHTTP(w, r)
	if w.Code != 404 {
		t.Fatal("未知路径应返回不存在")
	}
}

func TestConsoleVideoValidation(t *testing.T) {
	for _, tc := range []struct {
		codec, pixel, duration, format string
		want                           bool
	}{
		{"h264", "yuv420p", "30", "mov,mp4,m4a,3gp,3g2,mj2", true},
		{"h264", "yuv420p", "8", "mov,mp4,m4a,3gp,3g2,mj2", true},
		{"h264", "yuv420p", "30.01", "mp4", false},
		{"hevc", "yuv420p", "8", "mp4", false},
		{"h264", "yuv420p10le", "8", "mp4", false},
		{"h264", "yuv444p", "8", "mp4", false},
		{"h264", "yuv420p", "NaN", "mp4", false},
		{"h264", "yuv420p", "0", "mp4", false},
		{"h264", "yuv420p", "8", "matroska", false},
	} {
		streams := []struct {
			Type  string `json:"codec_type"`
			Codec string `json:"codec_name"`
			Pixel string `json:"pix_fmt"`
		}{{"video", tc.codec, tc.pixel}}
		if got := validConsoleVideo(streams, tc.duration, tc.format); got != tc.want {
			t.Errorf("%+v: %v", tc, got)
		}
	}
	if validConsoleVideo(nil, "8", "mp4") {
		t.Fatal("不能接受无视频流文件")
	}
}

func TestCopyStorageCapability(t *testing.T) {
	for _, disabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "使用标签", true: "关闭标签"}[disabled], func(t *testing.T) {
			var headers http.Header
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				headers = r.Header.Clone()
				w.Header().Set("Content-Type", "application/xml")
				w.Write([]byte(`<CopyObjectResult><ETag>"test"</ETag><LastModified>2026-01-01T00:00:00Z</LastModified></CopyObjectResult>`))
			}))
			defer backend.Close()
			u, _ := url.Parse(backend.URL)
			store, err := minio.New(u.Host, &minio.Options{Creds: credentials.NewStaticV4("test", "test-secret", ""), Region: "us-east-1"})
			if err != nil {
				t.Fatal(err)
			}
			s := Server{Store: store, Bucket: "console-test", DisableObjectTagging: disabled}
			_, err = store.CopyObject(context.Background(), s.inputCopyOptions("tenant/input"), minio.CopySrcOptions{Bucket: s.Bucket, Object: "tenant/upload", MatchETag: "source-etag"})
			if err != nil {
				t.Fatal(err)
			}
			if headers.Get("X-Amz-Copy-Source-If-Match") != "source-etag" {
				t.Fatal("必须保留冻结输入的对象校验")
			}
			if disabled {
				if headers.Get("X-Amz-Tagging") != "" || headers.Get("X-Amz-Tagging-Directive") != "" {
					t.Fatal("不支持标签的存储不能收到标签请求")
				}
			} else if headers.Get("X-Amz-Tagging") != "kind=intermediate" {
				t.Fatal("默认存储应保留对象标签")
			}
			if headers.Get("X-Amz-Metadata-Directive") == "REPLACE" {
				t.Fatal("不能丢弃原片媒体类型和元数据")
			}
		})
	}
}

func TestHLSPlayerAssetPublic(t *testing.T) {
	s := &Server{}
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/assets/hls.min.js", nil))
	if w.Code != 200 || !strings.HasPrefix(w.Header().Get("Content-Type"), "text/javascript") || w.Body.Len() < 100000 {
		t.Fatal("登录页应能从同源读取实际播放器依赖")
	}
	if w.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Fatal("缺少静态资源响应保护")
	}
}
