package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestNormalizeAPIURL(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "host only", in: "127.0.0.1:8080", want: "http://127.0.0.1:8080/v1/ocr"},
		{name: "root path", in: "http://localhost:8080/", want: "http://localhost:8080/v1/ocr"},
		{name: "custom path", in: "https://example.test/custom", want: "https://example.test/custom"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeAPIURL(tt.in)
			if err != nil {
				t.Fatalf("normalizeAPIURL err=%v", err)
			}
			if got != tt.want {
				t.Fatalf("normalizeAPIURL=%q want %q", got, tt.want)
			}
		})
	}
}

func TestNormalizeAPIURLRejectsEmpty(t *testing.T) {
	if _, err := normalizeAPIURL(" "); err == nil {
		t.Fatal("expected empty URL error")
	}
}

func TestDerivedServiceURLs(t *testing.T) {
	ready, err := readyURL("http://127.0.0.1:8080/v1/ocr?x=1#frag")
	if err != nil {
		t.Fatalf("readyURL err=%v", err)
	}
	if ready != "http://127.0.0.1:8080/ready" {
		t.Fatalf("readyURL=%q", ready)
	}
	docs, err := docsURL("127.0.0.1:8080")
	if err != nil {
		t.Fatalf("docsURL err=%v", err)
	}
	if docs != "http://127.0.0.1:8080/doc" {
		t.Fatalf("docsURL=%q", docs)
	}
}

func TestAddAPIKey(t *testing.T) {
	req, err := http.NewRequest(http.MethodGet, "http://example.test", nil)
	if err != nil {
		t.Fatal(err)
	}
	addAPIKey(req, " secret ")
	if got := req.Header.Get("Authorization"); got != "Bearer secret" {
		t.Fatalf("Authorization=%q", got)
	}
	if got := req.Header.Get("X-API-Key"); got != "secret" {
		t.Fatalf("X-API-Key=%q", got)
	}
}

func TestResponseErrorPrefersJSONError(t *testing.T) {
	err := responseError(http.StatusUnauthorized, []byte(`{"error":"bad key"}`))
	if err == nil || !strings.Contains(err.Error(), "bad key") {
		t.Fatalf("responseError=%v", err)
	}
}

func TestRequestContextDefaultsAndExpires(t *testing.T) {
	ctx, cancel := requestContext(0)
	defer cancel()
	if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > 601*time.Second {
		t.Fatalf("deadline=%v ok=%v", deadline, ok)
	}

	ctx, cancel = requestContext(1)
	defer cancel()
	select {
	case <-ctx.Done():
		t.Fatal("context expired too early")
	default:
	}
}

func TestRequestErrorClassifiesCancel(t *testing.T) {
	if got := requestError(context.Canceled).Error(); got != "request canceled" {
		t.Fatalf("cancel err=%q", got)
	}
	if got := requestError(context.DeadlineExceeded).Error(); got != "request timed out" {
		t.Fatalf("timeout err=%q", got)
	}
}

func TestCheckFileSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.txt")
	if err := os.WriteFile(path, []byte("abcd"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := checkFileSize(path, 4, "test"); err != nil {
		t.Fatalf("checkFileSize exact err=%v", err)
	}
	if err := checkFileSize(path, 3, "test"); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("checkFileSize err=%v want too large", err)
	}
}

func TestReadLimited(t *testing.T) {
	got, err := readLimited(strings.NewReader("abcd"), 4, "test")
	if err != nil {
		t.Fatalf("readLimited exact err=%v", err)
	}
	if string(got) != "abcd" {
		t.Fatalf("readLimited=%q", got)
	}
	if _, err := readLimited(strings.NewReader("abcde"), 4, "test"); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Fatalf("readLimited err=%v want too large", err)
	}
	if _, err := readLimited(errReader{}, 4, "test"); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("readLimited err=%v want boom", err)
	}
}

type errReader struct{}

func (errReader) Read(_ []byte) (int, error) {
	return 0, errors.New("boom")
}
