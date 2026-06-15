package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestUniqueShards(t *testing.T) {
	got := uniqueShards(map[string]string{
		"a": "b.safetensors",
		"b": "a.safetensors",
		"c": "b.safetensors",
		"d": "",
	})
	want := []string{"a.safetensors", "b.safetensors"}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d got=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("shard[%d]=%q want %q", i, got[i], want[i])
		}
	}
}

func TestJoinURL(t *testing.T) {
	if got := joinURL("http://x/base", "a.bin"); got != "http://x/base/a.bin" {
		t.Fatalf("got %q", got)
	}
	if got := joinURL("http://x/base/", "a.bin"); got != "http://x/base/a.bin" {
		t.Fatalf("got %q", got)
	}
	if got := joinURL("", "a.bin"); got != baseURL+"a.bin" {
		t.Fatalf("got %q", got)
	}
}

func TestDownloadSkipsExisting(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.bin")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	item, err := download(http.DefaultClient, path, "http://127.0.0.1/unused")
	if err != nil {
		t.Fatal(err)
	}
	if item.Name != "x.bin" || item.Bytes != 3 || len(item.SHA256) != 64 || item.Status != "skipped" {
		t.Fatalf("item=%+v", item)
	}
}

func TestDownloadWritesFile(t *testing.T) {
	var sawUA bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawUA = r.Header.Get("User-Agent") == "paddleocrvl-go"
		_, _ = w.Write([]byte("abc"))
	}))
	defer srv.Close()
	dir := t.TempDir()
	path := filepath.Join(dir, "x.bin")
	item, err := download(srv.Client(), path, srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !sawUA {
		t.Fatal("missing user-agent")
	}
	if item.Name != "x.bin" || item.Bytes != 3 || len(item.SHA256) != 64 || item.Status != "downloaded" {
		t.Fatalf("item=%+v", item)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "abc" {
		t.Fatalf("file=%q", raw)
	}
}

func TestDownloadSummaryAdd(t *testing.T) {
	var s downloadSummary
	s.add(downloadFile{Name: "a", Bytes: 2})
	s.add(downloadFile{Name: "b", Bytes: 3})
	if s.Bytes != 5 || len(s.Files) != 2 {
		t.Fatalf("summary=%+v", s)
	}
}

func TestReplaceDownloadFile(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "x.bin")
	tmp := dst + ".part"
	if err := os.WriteFile(dst, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tmp, []byte("new"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := replaceDownloadFile(tmp, dst); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "new" {
		t.Fatalf("dst=%q", raw)
	}
}
