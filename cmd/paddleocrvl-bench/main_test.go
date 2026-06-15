package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestPostCountsGeneratedTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"tokens":[1,2,3,4],"prompt_tokens":2}`))
	}))
	defer srv.Close()
	got, err := post(srv.Client(), srv.URL, []byte(`{}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 2 {
		t.Fatalf("tokens=%d want 2", got)
	}
}

func TestGeneratedTokenCountPrefersServerField(t *testing.T) {
	got := generatedTokenCount(generateResponse{
		Tokens:          []int{1, 2, 3, 4},
		PromptTokens:    4,
		GeneratedTokens: 9,
	})
	if got != 9 {
		t.Fatalf("tokens=%d want 9", got)
	}
	got = generatedTokenCount(generateResponse{Tokens: []int{1, 2, 3, 4}, PromptTokens: 2})
	if got != 2 {
		t.Fatalf("fallback tokens=%d want 2", got)
	}
}

func TestPostStatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad", http.StatusBadRequest)
	}))
	defer srv.Close()
	_, err := post(srv.Client(), srv.URL, []byte(`{}`), 1)
	if err == nil || !strings.Contains(err.Error(), "status 400") {
		t.Fatalf("err=%v", err)
	}
}

func TestPostCountsBatchGeneratedTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"responses":[{"tokens":[1,2,3],"prompt_tokens":1},{"tokens":[1,2,3,4],"prompt_tokens":3}]}`))
	}))
	defer srv.Close()
	got, err := post(srv.Client(), srv.URL, []byte(`{}`), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != 3 {
		t.Fatalf("tokens=%d want 3", got)
	}
}

func TestPostCountsBatchGeneratedTokenSummary(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":2,"generated_tokens":11,"responses":[{"tokens":[1],"prompt_tokens":1}]}`))
	}))
	defer srv.Close()
	got, err := post(srv.Client(), srv.URL, []byte(`{}`), 2)
	if err != nil {
		t.Fatal(err)
	}
	if got != 11 {
		t.Fatalf("tokens=%d want 11", got)
	}
}

func TestBenchPayloadBatch(t *testing.T) {
	payload, err := benchPayload(requestBody{Prompt: "x", MaxNewTokens: 1}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(payload), `"requests"`) || strings.Count(string(payload), `"prompt"`) != 3 {
		t.Fatalf("payload=%s", payload)
	}
}

func TestNormalizeBenchURL(t *testing.T) {
	if got := normalizeBenchURL("http://x/v1/generate", 2); got != "http://x/v1/batch" {
		t.Fatalf("got %q", got)
	}
	if got := normalizeBenchURL("http://x/custom", 2); got != "http://x/custom" {
		t.Fatalf("got %q", got)
	}
	if got := normalizeBenchURL("http://x/v1/generate", 1); got != "http://x/v1/generate" {
		t.Fatalf("got %q", got)
	}
}

func TestLoadLastError(t *testing.T) {
	var v atomic.Value
	if got := loadLastError(&v); got != "" {
		t.Fatalf("empty error=%q", got)
	}
	v.Store("status 500")
	if got := loadLastError(&v); got != "status 500" {
		t.Fatalf("error=%q", got)
	}
}

func TestBuildReport(t *testing.T) {
	r := buildReport([]time.Duration{
		10 * time.Millisecond,
		20 * time.Millisecond,
		30 * time.Millisecond,
	}, time.Second, 1, 7, 4, "status 500", benchContext{Mode: "local", Backend: "cpu", Quantization: "q4", WeightPath: "model-q4.gguf", WeightSource: "existing_gguf"})
	if r.Requests != 3 || r.OK != 2 || r.Errors != 1 {
		t.Fatalf("counts=%+v", r)
	}
	if r.Items != 8 || r.BatchSize != 4 || r.Tokens != 7 {
		t.Fatalf("throughput fields=%+v", r)
	}
	if r.LastError != "status 500" {
		t.Fatalf("last_error=%q", r.LastError)
	}
	if r.Mode != "local" || r.Backend != "cpu" || r.Quantization != "q4" || r.WeightPath != "model-q4.gguf" || r.WeightSource != "existing_gguf" {
		t.Fatalf("context fields=%+v", r)
	}
	if r.LatencyMinMS != 10 || r.LatencyP50MS != 20 || r.LatencyMaxMS != 30 || r.LatencyAvgMS != 20 {
		t.Fatalf("latency fields=%+v", r)
	}
	if r.CPU.GOOS == "" || r.Memory.SysBytes == 0 {
		t.Fatalf("runtime fields=%+v", r)
	}
}
