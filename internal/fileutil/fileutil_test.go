package fileutil

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAbs(t *testing.T) {
	got := Abs(filepath.Join(".", "model.gguf"))
	if !filepath.IsAbs(got) || filepath.Base(got) != "model.gguf" {
		t.Fatalf("path=%q", got)
	}
}

func TestSHA256(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.bin")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	if got := SHA256(path); got != want {
		t.Fatalf("sha=%q want %q", got, want)
	}
	if got := SHA256(filepath.Join(dir, "missing")); got != "" {
		t.Fatalf("missing sha=%q", got)
	}
}
