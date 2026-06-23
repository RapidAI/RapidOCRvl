package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadAppliesDefaults(t *testing.T) {
	dir := t.TempDir()
	writeConfigForTest(t, dir, `{
		"vocab_size": 10,
		"hidden_size": 8,
		"intermediate_size": 16,
		"num_hidden_layers": 1,
		"num_attention_heads": 2
	}`)

	cfg, err := Load(dir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.HeadDim != 4 {
		t.Fatalf("HeadDim=%d want 4", cfg.HeadDim)
	}
	if cfg.NumKeyValueHeads != 2 {
		t.Fatalf("NumKeyValueHeads=%d want 2", cfg.NumKeyValueHeads)
	}
	if cfg.RopeTheta != 10000 {
		t.Fatalf("RopeTheta=%f want 10000", cfg.RopeTheta)
	}
	if cfg.RMSNormEps != 1e-6 {
		t.Fatalf("RMSNormEps=%f want 1e-6", cfg.RMSNormEps)
	}
}

func TestLoadRejectsHugeConfig(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxConfigBytes + 1); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	_, err = Load(dir)
	if err == nil || !strings.Contains(err.Error(), "config.json too large") {
		t.Fatalf("err=%v want too large error", err)
	}
}

func TestLoadRejectsDuplicateJSONKeys(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "top-level",
			body: `{"hidden_size":8,"hidden_size":16,"num_attention_heads":2}`,
		},
		{
			name: "nested-vision-config",
			body: `{"hidden_size":8,"num_attention_heads":2,"vision_config":{"hidden_size":8,"hidden_size":16}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeConfigForTest(t, dir, tc.body)
			_, err := Load(dir)
			if err == nil || !strings.Contains(err.Error(), "duplicate JSON key") {
				t.Fatalf("err=%v want duplicate JSON key", err)
			}
		})
	}
}

func writeConfigForTest(t *testing.T, dir, data string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
}
