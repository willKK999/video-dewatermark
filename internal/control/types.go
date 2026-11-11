package control

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"time"
)

const MaxUpload = 200 * 1024 * 1024

type Prompt struct {
	Frame int        `json:"frame"`
	Box   [4]float64 `json:"box"`
}
type Options struct {
	HLS            bool     `json:"hls,omitempty"`
	HLSBandwidth   int      `json:"hls_bandwidth,omitempty"`
	HLSNativePath  string   `json:"hls_native_path,omitempty"`
	HLSEncoder     string   `json:"hls_encoder,omitempty"`
	MaxShortEdge   int      `json:"max_short_edge"`
	AllowDowngrade bool     `json:"allow_downgrade"`
	Prompts        []Prompt `json:"prompts,omitempty"`
}

func (o *Options) Validate() error {
	if (o.HLSNativePath != "" && o.HLSNativePath != "speedup-v2" && o.HLSNativePath != "v7") || (o.HLSEncoder != "" && o.HLSEncoder != "libx264" && o.HLSEncoder != "h264_nvenc") {
		return errors.New("整集原生路径或编码器无效")
	}
	if !o.HLS && (o.HLSNativePath != "" || o.HLSEncoder != "") {
		return errors.New("原生交付开关仅用于整集任务")
	}
	if o.HLSBandwidth < 0 || (o.HLS && len(o.Prompts) > 0) {
		return errors.New("HLS 码率必须非负，且只支持自动检测")
	}
	if o.MaxShortEdge == 0 {
		o.MaxShortEdge = 720
	}
	if o.MaxShortEdge != 720 && o.MaxShortEdge != 540 {
		return errors.New("max_short_edge must be 720 or 540")
	}
	for _, p := range o.Prompts {
		if p.Frame < 0 || p.Frame >= 900 || p.Box[0] < 0 || p.Box[1] < 0 || p.Box[2] <= p.Box[0] || p.Box[3] <= p.Box[1] || p.Box[2] > 1280 || p.Box[3] > 1280 {
			return errors.New("invalid prompt: use normalized-frame pixel coordinates")
		}
	}
	return nil
}

type Job struct {
	ID              string          `json:"id"`
	Status          string          `json:"status"`
	Stage           *string         `json:"stage"`
	Attempt         int             `json:"attempt"`
	PipelineVersion string          `json:"pipeline_version"`
	CreatedAt       time.Time       `json:"created_at"`
	UpdatedAt       time.Time       `json:"updated_at"`
	Progress        json.RawMessage `json:"progress"`
	Result          any             `json:"result"`
	Error           any             `json:"error"`
}

func id(prefix string) string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return prefix + hex.EncodeToString(b)
}
func reply(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
func fail(w http.ResponseWriter, r *http.Request, status int, code, msg string) {
	reply(w, status, map[string]any{"error": map[string]any{"code": code, "message": msg, "retryable": status >= 500 || status == 429}, "request_id": r.Header.Get("X-Request-ID")})
}
func decode(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, 64*1024)
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return errors.New("expected one JSON object")
	}
	return nil
}
