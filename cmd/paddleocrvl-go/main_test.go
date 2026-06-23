package main

import (
	"encoding/json"
	"testing"
)

func TestParseTokens(t *testing.T) {
	got, err := parseTokens(" 1,2, 3 ,,")
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]=%d want %d", i, got[i], want[i])
		}
	}
	if _, err := parseTokens("bad"); err == nil {
		t.Fatal("expected bad token error")
	}
	if _, err := parseTokens("-1"); err == nil {
		t.Fatal("expected negative token error")
	}
	if _, err := parseTokens(" , "); err == nil {
		t.Fatal("expected empty token error")
	}
}

func TestTaskPrompt(t *testing.T) {
	got, err := taskPrompt(" OCR ")
	if err != nil {
		t.Fatal(err)
	}
	if got != "OCR:" {
		t.Fatalf("prompt=%q", got)
	}
	if _, err := taskPrompt("bad"); err == nil {
		t.Fatal("expected bad task error")
	}
}

func TestNeedsSamplingSeed(t *testing.T) {
	if needsSamplingSeed(1, 0, 0) {
		t.Fatal("greedy temperature=0 should not need seed")
	}
	if needsSamplingSeed(1, 1, 1) {
		t.Fatal("top_k=1 should not need seed")
	}
	if needsSamplingSeed(0, 1, 0) {
		t.Fatal("max_new_tokens=0 should not need seed")
	}
	if !needsSamplingSeed(1, 1, 0) {
		t.Fatal("sampling should need seed")
	}
}

func TestCLIGenerateJSONShape(t *testing.T) {
	raw, err := json.Marshal(cliGenerate{
		Tokens:          []int{1, 2, 3},
		PromptTokens:    2,
		GeneratedTokens: 1,
		Text:            "x",
		Seed:            7,
		Quantization:    "q4",
		WeightPath:      "model-q4.gguf",
		WeightSource:    "existing_gguf",
		Backend:         "cpu",
	})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got["quantization"] != "q4" || got["weight_path"] != "model-q4.gguf" || got["weight_source"] != "existing_gguf" || got["backend"] != "cpu" || got["text"] != "x" || got["generated_tokens"] != float64(1) {
		t.Fatalf("json=%s", raw)
	}
}
