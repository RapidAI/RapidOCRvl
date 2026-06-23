package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultOutputPath(t *testing.T) {
	dir := filepath.Join("x", "model")
	cases := map[string]string{
		"":     "model.gguf",
		"f32":  "model.gguf",
		" Q4 ": "model-q4.gguf",
		"q8":   "model-q8.gguf",
		"q6":   "model-q6.gguf",
		"q4":   "model-q4.gguf",
	}
	for quant, wantFile := range cases {
		got := defaultOutputPath(dir, quant)
		want := filepath.Join(dir, wantFile)
		if got != want {
			t.Fatalf("quant %q path=%q want %q", quant, got, want)
		}
	}
}

func TestNormalizedQuant(t *testing.T) {
	if got := normalizedQuant(""); got != "f32" {
		t.Fatalf("empty quant=%q want f32", got)
	}
	if got := normalizedQuant(" Q4 "); got != "q4" {
		t.Fatalf("quant=%q want q4", got)
	}
}

func TestValidateQuant(t *testing.T) {
	for _, q := range []string{"", "f32", " Q8 ", "q6", "q4"} {
		if err := validateQuant(q); err != nil {
			t.Fatalf("quant %q err=%v", q, err)
		}
	}
	if err := validateQuant("auto"); err == nil {
		t.Fatal("expected unsupported quantization error")
	}
}

func TestMetadataUint64(t *testing.T) {
	meta := map[string]any{"x": uint64(7), "y": "bad"}
	if got := metadataUint64(meta, "x"); got != 7 {
		t.Fatalf("got %d want 7", got)
	}
	if got := metadataUint64(meta, "y"); got != 0 {
		t.Fatalf("got %d want 0", got)
	}
	if got := metadataUint64(meta, "z"); got != 0 {
		t.Fatalf("got %d want 0", got)
	}
}

func TestEnsureOutputDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "out")
	if err := ensureOutputDir(filepath.Join(dir, "model.gguf")); err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		t.Fatalf("dir stat=%v err=%v", st, err)
	}
	if err := ensureOutputDir("model.gguf"); err != nil {
		t.Fatal(err)
	}
}
