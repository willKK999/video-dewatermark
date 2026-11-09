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
