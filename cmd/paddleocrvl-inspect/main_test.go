package main

import (
	"os"
	"path/filepath"
	"testing"

	"paddleocrvl-go/internal/fileutil"
)

func TestFilteredMetadata(t *testing.T) {
	in := map[string]any{
		"paddleocrvl.quantization": "q6",
		"general.file_type":        uint32(6),
		"general.name":             "PaddleOCR-VL-0.9B",
		"other":                    "hidden",
	}
	got := filteredMetadata(in)
	if got["paddleocrvl.quantization"] != "q6" || got["general.file_type"] != uint32(6) || got["other"] != nil {
		t.Fatalf("metadata=%v", got)
	}
}

func TestFirstTensorMap(t *testing.T) {
	keys := []string{"a", "b", "c"}
	tensors := map[string]tensorMeta{
		"a": {DType: "F32", Shape: []int64{1}, Bytes: 4},
		"b": {DType: "Q6_ROW", Shape: []int64{2, 3}, Bytes: 14},
		"c": {DType: "Q4_ROW", Shape: []int64{2, 3}, Bytes: 12},
	}
	got := firstTensorMap(keys, tensors, 2)
	if len(got) != 2 || got["a"].DType != "F32" || got["b"].DType != "Q6_ROW" || got["c"].DType != "" {
		t.Fatalf("first=%v", got)
	}
}

func TestLargestTensorMap(t *testing.T) {
	tensors := map[string]tensorMeta{
		"a": {DType: "F32", Bytes: 4},
		"b": {DType: "Q6_ROW", Bytes: 14},
		"c": {DType: "Q4_ROW", Bytes: 12},
	}
	got := largestTensorMap(tensors, 2)
	if len(got) != 2 || got["b"].Bytes != 14 || got["c"].Bytes != 12 || got["a"].Bytes != 0 {
		t.Fatalf("largest=%v", got)
	}
}

func TestQuantFromGGUFFile(t *testing.T) {
	cases := map[string]string{
		"model-q8.gguf": "q8",
		"model-q6.gguf": "q6",
		"model-q4.gguf": "q4",
		"model.gguf":    "f32",
	}
	for file, want := range cases {
		if got := quantFromGGUFFile(file); got != want {
			t.Fatalf("%s -> %s want %s", file, got, want)
		}
	}
}

func TestMetadataString(t *testing.T) {
	if got := metadataString(map[string]any{"x": "q6"}, "x", "f32"); got != "q6" {
		t.Fatalf("got %q", got)
	}
	if got := metadataString(map[string]any{"x": 6}, "x", "f32"); got != "f32" {
		t.Fatalf("got %q", got)
	}
}

func TestSafetensorsPathAndFileBytes(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(model, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := safetensorsPath(dir); got != fileutil.Abs(model) {
		t.Fatalf("path=%q want %q", got, model)
	}
	if got := safetensorsFileBytes(dir); got != 3 {
		t.Fatalf("bytes=%d want 3", got)
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(index, []byte("idx"), 0o644); err != nil {
		t.Fatal(err)
	}
	shard := filepath.Join(dir, "shard.safetensors")
	if err := os.WriteFile(shard, []byte("shard"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := safetensorsPath(dir); got != fileutil.Abs(index) {
		t.Fatalf("path=%q want %q", got, index)
	}
	if got := safetensorsFileBytes(dir); got != 11 {
		t.Fatalf("bytes=%d want 11", got)
	}
	files := safetensorsFiles(dir)
	if len(files) != 3 || filepath.Base(files[0]) != "model.safetensors" || filepath.Base(files[1]) != "model.safetensors.index.json" || filepath.Base(files[2]) != "shard.safetensors" {
		t.Fatalf("files=%v", files)
	}
	items := weightFiles(files)
	if len(items) != 3 {
		t.Fatalf("items=%v", items)
	}
	for _, item := range items {
		if item.Path == "" || item.Bytes == 0 || len(item.SHA256) != 64 {
			t.Fatalf("bad item=%+v", item)
		}
	}
}

func TestSafetensorsFilesSkipsDirectoriesAndCapsList(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "dir.safetensors"), 0o755); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(index, []byte("idx"), 0o644); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < maxInspectWeightFiles+1; i++ {
		name := filepath.Join(dir, "shard-"+stringID(i)+".safetensors")
		if err := os.WriteFile(name, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	files := safetensorsFiles(dir)
	if len(files) != maxInspectWeightFiles {
		t.Fatalf("files=%d want %d", len(files), maxInspectWeightFiles)
	}
	for _, path := range files {
		if filepath.Base(path) == "dir.safetensors" {
			t.Fatalf("directory included: %v", files)
		}
	}
	if !containsPath(files, index) {
		t.Fatalf("index file not retained in capped list")
	}
	items := weightFiles(append(files, filepath.Join(dir, "dir.safetensors")))
	if len(items) != maxInspectWeightFiles {
		t.Fatalf("items=%d want %d", len(items), maxInspectWeightFiles)
	}
}

func containsPath(paths []string, want string) bool {
	for _, path := range paths {
		if path == want {
			return true
		}
	}
	return false
}

func stringID(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
