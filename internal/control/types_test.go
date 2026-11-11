package control

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOptions(t *testing.T) {
	o := Options{}
	if o.Validate() != nil || o.MaxShortEdge != 720 {
		t.Fatal("default")
	}
	o.MaxShortEdge = 1080
	if o.Validate() == nil {
		t.Fatal("unsupported resolution")
	}
	o.MaxShortEdge = 720
	o.Prompts = []Prompt{{Frame: 0, Box: [4]float64{30, 20, 10, 40}}}
	if o.Validate() == nil {
		t.Fatal("invalid box")
	}
}
func TestStrictJSON(t *testing.T) {
	for _, body := range []string{`{"unknown":1}`, `{} {}`, `[]`} {
		r := httptest.NewRequest("POST", "/", strings.NewReader(body))
		if decode(httptest.NewRecorder(), r, &Options{}) == nil {
			t.Fatalf("accepted %s", body)
		}
	}
}
func TestAuthentication(t *testing.T) {
	s := &Server{Tokens: map[string]string{"long-secret-token": "a"}}
	r := httptest.NewRequest("GET", "/v1/jobs/job_1", nil)
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, r)
	if w.Code != 401 || !strings.Contains(w.Body.String(), "request_id") {
		t.Fatal(w.Code, w.Body.String())
	}
}

func Test原生交付开关(t *testing.T) {
	for _, mode := range []string{"", "speedup-v2", "v7"} {
		for _, encoder := range []string{"", "libx264", "h264_nvenc"} {
			o := Options{HLS: true, HLSNativePath: mode, HLSEncoder: encoder}
			if err := o.Validate(); err != nil {
				t.Fatal("合法原生交付选项被拒绝", err)
			}
		}
	}
	for _, o := range []Options{
		{HLS: true, HLSNativePath: "invalid"},
		{HLS: true, HLSEncoder: "invalid"},
		{HLSNativePath: "v7"},
		{HLSEncoder: "h264_nvenc"},
	} {
		if o.Validate() == nil {
			t.Fatal("非法原生交付选项未被拒绝")
		}
	}
	r := httptest.NewRequest("POST", "/", strings.NewReader(`{"hls":true,"hls_native_path":"v7","hls_encoder":"libx264"}`))
	var o Options
	if decode(httptest.NewRecorder(), r, &o) != nil || o.Validate() != nil || o.HLSNativePath != "v7" {
		t.Fatal("回退开关未能完整通过接口解析")
	}
}
