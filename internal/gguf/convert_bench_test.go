package gguf

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkConvertSafetensorsQ8(b *testing.B) {
	benchmarkConvertSafetensorsQuant(b, "q8")
}

func BenchmarkConvertSafetensorsF32(b *testing.B) {
	benchmarkConvertSafetensorsQuant(b, "f32")
}

func BenchmarkConvertSafetensorsQ8BF16(b *testing.B) {
	benchmarkConvertSafetensorsQuantDType(b, "q8", "BF16")
}

func BenchmarkConvertSafetensorsQ6(b *testing.B) {
	benchmarkConvertSafetensorsQuant(b, "q6")
}

func BenchmarkConvertSafetensorsQ4(b *testing.B) {
	benchmarkConvertSafetensorsQuant(b, "q4")
}

func BenchmarkConvertSafetensorsQ8MultiTensor(b *testing.B) {
	dir := b.TempDir()
	configJSON := `{
		"vocab_size": 1024,
		"hidden_size": 1024,
		"intermediate_size": 4096,
		"num_hidden_layers": 1,
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
	writeTestSafetensorsMulti(b, src, []benchTensor{
		{name: "model.layers.0.self_attn.q_proj.weight", shape: []int64{1024, 1024}, values: values},
		{name: "model.layers.0.self_attn.o_proj.weight", shape: []int64{1024, 1024}, values: values},
		{name: "lm_head.weight", shape: []int64{1024, 1024}, values: values},
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst := filepath.Join(dir, "model-q8-multi.gguf")
		if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: "q8"}); err != nil {
			b.Fatal(err)
		}
	}
}

type benchTensor struct {
	name   string
	shape  []int64
	values []float32
}

func writeTestSafetensorsMulti(t testing.TB, path string, tensors []benchTensor) {
	t.Helper()
	raw := make(map[string]any, len(tensors))
	offset := 0
	for _, tensor := range tensors {
		size := len(tensor.values) * 4
		raw[tensor.name] = map[string]any{
			"dtype":        "F32",
			"shape":        tensor.shape,
			"data_offsets": []int{offset, offset + size},
		}
		offset += size
	}
	header, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8+len(header)+offset)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	pos := 8 + len(header)
	for _, tensor := range tensors {
		for _, v := range tensor.values {
			binary.LittleEndian.PutUint32(data[pos:], math.Float32bits(v))
			pos += 4
		}
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func benchmarkConvertSafetensorsQuant(b *testing.B, quant string) {
	benchmarkConvertSafetensorsQuantDType(b, quant, "F32")
}

func benchmarkConvertSafetensorsQuantDType(b *testing.B, quant, dtype string) {
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
	if dtype == "BF16" {
		writeTestSafetensorsBF16(b, src, "lm_head.weight", []int64{1024, 1024}, values)
	} else {
		writeTestSafetensors(b, src, "lm_head.weight", []int64{1024, 1024}, values)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst := filepath.Join(dir, "model-"+quant+".gguf")
		if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: quant}); err != nil {
			b.Fatal(err)
		}
	}
}

func writeTestSafetensorsBF16(t testing.TB, path, name string, shape []int64, values []float32) {
	t.Helper()
	raw := map[string]any{
		name: map[string]any{
			"dtype":        "BF16",
			"shape":        shape,
			"data_offsets": []int{0, len(values) * 2},
		},
	}
	header, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8+len(header)+len(values)*2)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	pos := 8 + len(header)
	for i, v := range values {
		binary.LittleEndian.PutUint16(data[pos+i*2:], uint16(math.Float32bits(v)>>16))
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
