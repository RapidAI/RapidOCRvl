package gguf

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func BenchmarkFloat32RowsF32(b *testing.B) {
	dir := b.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vision_config":{}}`), 0o644); err != nil {
		b.Fatal(err)
	}
	values := make([]float32, 512*1024)
	for i := range values {
		values[i] = float32(i%17-8) / 17
	}
	src := filepath.Join(dir, "model.safetensors")
	dst := filepath.Join(dir, "model.gguf")
	writeTestSafetensors(b, src, "x.weight", []int64{512, 1024}, values)
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{}); err != nil {
		b.Fatal(err)
	}
	gf, err := Open(dst)
	if err != nil {
		b.Fatal(err)
	}
	defer gf.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := gf.Float32Rows("x.weight", func(row int, values []float32) error {
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkOpenManyTensorMetadata(b *testing.B) {
	dir := b.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vision_config":{}}`), 0o644); err != nil {
		b.Fatal(err)
	}
	values := make([]float32, 4)
	tensors := make([]benchTensor, 512)
	for i := range tensors {
		tensors[i] = benchTensor{name: "tensor." + strconv.Itoa(i) + ".weight", shape: []int64{2, 2}, values: values}
	}
	src := filepath.Join(dir, "model.safetensors")
	dst := filepath.Join(dir, "model.gguf")
	writeTestSafetensorsMulti(b, src, tensors)
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{}); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gf, err := Open(dst)
		if err != nil {
			b.Fatal(err)
		}
		if len(gf.Tensors) != len(tensors) {
			b.Fatalf("tensors=%d want %d", len(gf.Tensors), len(tensors))
		}
		_ = gf.Close()
	}
}

func BenchmarkQ8Row(b *testing.B) {
	benchmarkQRow(b, "q8", func(gf *File, name string) error {
		_, _, _, err := gf.Q8Row(name)
		return err
	})
}

func BenchmarkQ6Row(b *testing.B) {
	benchmarkQRow(b, "q6", func(gf *File, name string) error {
		_, _, _, err := gf.Q6Row(name)
		return err
	})
}

func BenchmarkQ4Row(b *testing.B) {
	benchmarkQRow(b, "q4", func(gf *File, name string) error {
		_, _, _, err := gf.Q4Row(name)
		return err
	})
}

func benchmarkQRow(b *testing.B, quant string, read func(*File, string) error) {
	dir := b.TempDir()
	configJSON := `{
		"vocab_size": 1024,
		"hidden_size": 1024,
		"intermediate_size": 4096,
		"num_hidden_layers": 0,
		"num_attention_heads": 8,
		"num_key_value_heads": 8,
		"head_dim": 128,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		b.Fatal(err)
	}
	values := make([]float32, 1024*1024)
	for i := range values {
		values[i] = float32(i%17-8) / 17
	}
	src := filepath.Join(dir, "model.safetensors")
	dst := filepath.Join(dir, "model-"+quant+".gguf")
	writeTestSafetensors(b, src, "lm_head.weight", []int64{1024, 1024}, values)
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: quant}); err != nil {
		b.Fatal(err)
	}
	gf, err := Open(dst)
	if err != nil {
		b.Fatal(err)
	}
	defer gf.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := read(gf, "lm_head.weight"); err != nil {
			b.Fatal(err)
		}
	}
}
