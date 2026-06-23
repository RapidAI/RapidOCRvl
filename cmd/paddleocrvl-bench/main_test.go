package main

import (
	"encoding/base64"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
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

func TestPostRejectsHugeResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(strings.Repeat(" ", maxBenchResponseBytes+1)))
	}))
	defer srv.Close()
	_, err := post(srv.Client(), srv.URL, []byte(`{}`), 1)
	if err == nil || !strings.Contains(err.Error(), "response too large") {
		t.Fatalf("err=%v want response too large", err)
	}
}

func TestDecodeLimitedJSON(t *testing.T) {
	var out generateResponse
	if err := decodeLimitedJSON(strings.NewReader(`{"tokens":[1,2],"prompt_tokens":1}`), 64, &out); err != nil {
		t.Fatal(err)
	}
	if got := generatedTokenCount(out); got != 1 {
		t.Fatalf("tokens=%d want 1", got)
	}
	if err := decodeLimitedJSON(strings.NewReader(`{"tokens":[1]}`), 4, &out); err == nil {
		t.Fatal("expected size error")
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

func TestBodyImageBytesDecodesDataURL(t *testing.T) {
	want := []byte("image")
	got, err := bodyImageBytes(requestBody{ImageBase64: "data:image/png;base64," + base64.StdEncoding.EncodeToString(want)})
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestDecodedBase64LenAccountsForPaddingAndCRLF(t *testing.T) {
	for _, raw := range [][]byte{
		{},
		{1},
		{1, 2},
		{1, 2, 3},
		{1, 2, 3, 4},
	} {
		encoded := base64.StdEncoding.EncodeToString(raw)
		if got := decodedBase64Len(encoded); got != len(raw) {
			t.Fatalf("decodedBase64Len(%q)=%d want %d", encoded, got, len(raw))
		}
		wrapped := encoded
		if len(wrapped) > 2 {
			wrapped = wrapped[:2] + "\r\n" + wrapped[2:]
		}
		if got := decodedBase64LenIgnoringCRLF(wrapped); got != len(raw) {
			t.Fatalf("decodedBase64LenIgnoringCRLF(%q)=%d want %d", wrapped, got, len(raw))
		}
	}
}

func TestBodyImageBytesRejectsOversize(t *testing.T) {
	rawLen := maxBenchImageBytes + 1
	encodedLen := ((rawLen + 2) / 3) * 4
	_, err := bodyImageBytes(requestBody{ImageBase64: strings.Repeat("A", encodedLen)})
	if err == nil || !strings.Contains(err.Error(), "image_base64 exceeds") {
		t.Fatalf("err=%v want size error", err)
	}
}

func BenchmarkBodyImageBytesBase64(b *testing.B) {
	raw := make([]byte, 1<<20)
	for i := range raw {
		raw[i] = byte(i)
	}
	body := requestBody{ImageBase64: "data:image/png;base64," + base64.StdEncoding.EncodeToString(raw)}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := bodyImageBytes(body)
		if err != nil {
			b.Fatal(err)
		}
		if len(got) != len(raw) {
			b.Fatalf("got len %d want %d", len(got), len(raw))
		}
	}
}

func BenchmarkBodyImageBytesBase64Raw(b *testing.B) {
	raw := make([]byte, 1<<20)
	for i := range raw {
		raw[i] = byte(i)
	}
	body := requestBody{ImageBase64: base64.StdEncoding.EncodeToString(raw)}
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := bodyImageBytes(body)
		if err != nil {
			b.Fatal(err)
		}
		if len(got) != len(raw) {
			b.Fatalf("got len %d want %d", len(got), len(raw))
		}
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

func TestValidateBenchURL(t *testing.T) {
	for _, raw := range []string{"http://x/v1/generate", "https://x/v1/batch"} {
		if err := validateBenchURL(raw); err != nil {
			t.Fatalf("%q err=%v", raw, err)
		}
	}
	for _, raw := range []string{"", "x/v1/generate", "ftp://x/v1/generate"} {
		if err := validateBenchURL(raw); err == nil {
			t.Fatalf("expected %q to be rejected", raw)
		}
	}
}

func TestValidateBenchArgs(t *testing.T) {
	valid := requestBody{MaxNewTokens: 1}
	if err := validateBenchArgs(1, 1, 1, time.Second, valid); err != nil {
		t.Fatalf("valid args err=%v", err)
	}
	cases := []struct {
		name        string
		requests    int
		concurrency int
		batchSize   int
		timeout     time.Duration
		body        requestBody
		want        string
	}{
		{"requests", 0, 1, 1, 0, valid, "-n"},
		{"concurrency", 1, 0, 1, 0, valid, "-c"},
		{"batch", 1, 1, 0, 0, valid, "-batch-size"},
		{"timeout", 1, 1, 1, -time.Second, valid, "-timeout"},
		{"max-new", 1, 1, 1, 0, requestBody{MaxNewTokens: -1}, "max_new_tokens"},
		{"temperature", 1, 1, 1, 0, requestBody{Temperature: -1}, "temperature"},
		{"nan-temperature", 1, 1, 1, 0, requestBody{Temperature: math.NaN()}, "temperature"},
		{"top-k", 1, 1, 1, 0, requestBody{TopK: -1}, "top_k"},
	}
	for _, tc := range cases {
		err := validateBenchArgs(tc.requests, tc.concurrency, tc.batchSize, tc.timeout, tc.body)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("%s err=%v want %q", tc.name, err, tc.want)
		}
	}
}

func TestInputIDsCopiesExplicitTokens(t *testing.T) {
	body := requestBody{Tokens: []int{1, 2, 3}}
	got, err := inputIDs(nil, body)
	if err != nil {
		t.Fatal(err)
	}
	body.Tokens[0] = 9
	if !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Fatalf("tokens=%v", got)
	}
}

func TestInputIDsRejectsNegativeTokens(t *testing.T) {
	_, err := inputIDs(nil, requestBody{Tokens: []int{1, -2}})
	if err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("err=%v want negative token error", err)
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
