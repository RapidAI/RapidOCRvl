package gguf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"paddleocrvl-go/internal/config"
)

func TestConvertSafetensorsQ8RoundTrip(t *testing.T) {
	dir := t.TempDir()
	configJSON := `{
		"vocab_size": 2,
		"hidden_size": 3,
		"intermediate_size": 4,
		"num_hidden_layers": 0,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 3,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	values := []float32{1, -2, 3, 0.5, 0.25, -0.75}
	src := filepath.Join(dir, "model.safetensors")
	writeTestSafetensors(t, src, "lm_head.weight", []int64{2, 3}, values)
	dst := filepath.Join(dir, "model-q8.gguf")
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: "q8"}); err != nil {
		t.Fatal(err)
	}

	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	data, scales, shape, err := gf.Q8Row("lm_head.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 6 || len(scales) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("bad q8 tensor: data=%d scales=%d shape=%v", len(data), len(scales), shape)
	}
	if got := gf.Metadata["paddleocrvl.quantized_tensors"]; got != uint64(1) {
		t.Fatalf("metadata quantized_tensors=%v want 1", got)
	}
	if got := gf.Metadata["paddleocrvl.f32_tensors"]; got != uint64(0) {
		t.Fatalf("metadata f32_tensors=%v want 0", got)
	}
	for i, q := range data {
		row := i / 3
		got := float32(q) * scales[row]
		if math.Abs(float64(got-values[i])) > 0.03 {
			t.Fatalf("value %d got %.4f want %.4f", i, got, values[i])
		}
	}
}

func TestConvertSafetensorsF32RoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vision_config":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	values := []float32{1, -2, 3, 0.5}
	src := filepath.Join(dir, "model.safetensors")
	writeTestSafetensors(t, src, "x.weight", []int64{2, 2}, values)
	dst := filepath.Join(dir, "model.gguf")
	var progressDone, progressTotal int
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Progress: func(done, total int, name string, typ string) {
		progressDone, progressTotal = done, total
	}}); err != nil {
		t.Fatal(err)
	}
	if progressDone != 1 || progressTotal != 1 {
		t.Fatalf("progress done=%d total=%d want 1/1", progressDone, progressTotal)
	}
	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	if got := gf.Metadata["paddleocrvl.quantization"]; got != "f32" {
		t.Fatalf("metadata quantization=%v want f32", got)
	}
	if got := gf.Metadata["paddleocrvl.source_tensors"]; got != uint64(1) {
		t.Fatalf("metadata source_tensors=%v want 1", got)
	}
	if got := gf.Metadata["paddleocrvl.f32_tensors"]; got != uint64(1) {
		t.Fatalf("metadata f32_tensors=%v want 1", got)
	}
	if got := gf.Metadata["paddleocrvl.quantized_tensors"]; got != uint64(0) {
		t.Fatalf("metadata quantized_tensors=%v want 0", got)
	}
	got, shape, err := gf.Float32("x.weight")
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != 2 || shape[1] != 2 {
		t.Fatalf("shape=%v", shape)
	}
	for i := range got {
		if got[i] != values[i] {
			t.Fatalf("value %d got %f want %f", i, got[i], values[i])
		}
	}
	var rows [][]float32
	shape, err = gf.Float32Rows("x.weight", func(row int, values []float32) error {
		rows = append(rows, append([]float32(nil), values...))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != 2 || shape[1] != 2 || len(rows) != 2 || rows[0][0] != 1 || rows[1][1] != 0.5 {
		t.Fatalf("rows shape=%v rows=%v", shape, rows)
	}
}

func TestConvertSafetensorsBF16F32RoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vision_config":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, 256*1024+5)
	for i := range values {
		values[i] = float32(i%251-125) / 8
	}
	src := filepath.Join(dir, "model.safetensors")
	writeTestSafetensorsBF16(t, src, "x.weight", []int64{1, int64(len(values))}, values)
	dst := filepath.Join(dir, "model.gguf")
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{}); err != nil {
		t.Fatal(err)
	}
	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	got, shape, err := gf.Float32("x.weight")
	if err != nil {
		t.Fatal(err)
	}
	if shape[0] != 1 || shape[1] != int64(len(values)) {
		t.Fatalf("shape=%v", shape)
	}
	for _, idx := range []int{0, 1024, len(values) - 1} {
		want := math.Float32frombits(math.Float32bits(values[idx]) & 0xffff0000)
		if got[idx] != want {
			t.Fatalf("value[%d]=%f want %f", idx, got[idx], want)
		}
	}
}

func TestNormalizeQuantization(t *testing.T) {
	cases := map[string]string{
		"":      "f32",
		" F32 ": "f32",
		" Q8 ":  "q8",
	}
	for in, want := range cases {
		if got := normalizeQuantization(in); got != want {
			t.Fatalf("%q -> %q want %q", in, got, want)
		}
	}
}

func TestIsQuantizedTextTensorLayerParser(t *testing.T) {
	cfg := &config.Config{
		VocabSize:         16,
		HiddenSize:        8,
		IntermediateSize:  32,
		NumHiddenLayers:   2,
		NumAttentionHeads: 2,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}
	cases := []struct {
		name  string
		shape []int64
		want  bool
	}{
		{"lm_head.weight", []int64{16, 8}, true},
		{"model.layers.0.self_attn.q_proj.weight", []int64{8, 8}, true},
		{"model.layers.1.self_attn.k_proj.weight", []int64{4, 8}, true},
		{"model.layers.1.self_attn.v_proj.weight", []int64{4, 8}, true},
		{"model.layers.1.self_attn.o_proj.weight", []int64{8, 8}, true},
		{"model.layers.1.mlp.gate_proj.weight", []int64{32, 8}, true},
		{"model.layers.1.mlp.up_proj.weight", []int64{32, 8}, true},
		{"model.layers.1.mlp.down_proj.weight", []int64{8, 32}, true},
		{"model.layers.2.self_attn.q_proj.weight", []int64{8, 8}, false},
		{"model.layers.x.self_attn.q_proj.weight", []int64{8, 8}, false},
		{"model.layers.0.self_attn.q_proj.bias", []int64{8}, false},
		{"model.layers.0.self_attn.q_proj.weight", []int64{4, 8}, false},
	}
	for _, tc := range cases {
		if got := isQuantizedTextTensor(tc.name, tc.shape, cfg); got != tc.want {
			t.Fatalf("%s shape=%v got %v want %v", tc.name, tc.shape, got, tc.want)
		}
	}
}

func TestReplaceFileOverwritesExisting(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "model.gguf")
	tmp := dst + ".tmp"
	if err := os.WriteFile(dst, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(tmp, []byte("new"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := replaceFile(tmp, dst); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "new" {
		t.Fatalf("dst=%q", raw)
	}
	if _, err := os.Stat(tmp); !os.IsNotExist(err) {
		t.Fatalf("tmp still exists err=%v", err)
	}
}

func TestWriteFloat32sChunksLargeInput(t *testing.T) {
	values := make([]float32, 70*1024)
	for i := range values {
		values[i] = float32(i%97) / 97
	}
	var buf bytes.Buffer
	if err := writeFloat32s(&buf, values); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	if len(raw) != len(values)*4 {
		t.Fatalf("bytes=%d want %d", len(raw), len(values)*4)
	}
	for _, idx := range []int{0, 65535, len(values) - 1} {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[idx*4:]))
		if got != values[idx] {
			t.Fatalf("value[%d]=%f want %f", idx, got, values[idx])
		}
	}
}

func TestBytesToInt8(t *testing.T) {
	src := []byte{0, 1, 127, 128, 129, 255, 42, 200, 7}
	dst := make([]int8, len(src))
	bytesToInt8(dst, src)
	want := []int8{0, 1, 127, -128, -127, -1, 42, -56, 7}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%d want %d", i, dst[i], want[i])
		}
	}
}

func TestConvertShardedSafetensorsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vision_config":{}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeTestSafetensors(t, filepath.Join(dir, "a.safetensors"), "a.weight", []int64{1}, []float32{7})
	writeTestSafetensors(t, filepath.Join(dir, "b.safetensors"), "b.weight", []int64{1}, []float32{9})
	index := map[string]any{
		"weight_map": map[string]string{
			"a.weight": "a.safetensors",
			"b.weight": "b.safetensors",
		},
	}
	raw, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "model.gguf")
	if err := ConvertSafetensors(dir, dst, dir); err != nil {
		t.Fatal(err)
	}
	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	got, _, err := gf.Float32("b.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 9 {
		t.Fatalf("got %v", got)
	}
}

func TestConvertSafetensorsQ4RoundTrip(t *testing.T) {
	dir := t.TempDir()
	configJSON := `{
		"vocab_size": 2,
		"hidden_size": 3,
		"intermediate_size": 4,
		"num_hidden_layers": 0,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 3,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	values := []float32{1, -2, 3, 0.5, 0.25, -0.75}
	src := filepath.Join(dir, "model.safetensors")
	writeTestSafetensors(t, src, "lm_head.weight", []int64{2, 3}, values)
	dst := filepath.Join(dir, "model-q4.gguf")
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: "q4"}); err != nil {
		t.Fatal(err)
	}

	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	data, scales, shape, err := gf.Q4Row("lm_head.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 4 || len(scales) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("bad q4 tensor: data=%d scales=%d shape=%v", len(data), len(scales), shape)
	}
	for i, want := range values {
		col := i % 3
		p := data[(i/3)*2+col/2]
		nib := p & 0x0F
		if col&1 == 1 {
			nib = p >> 4
		}
		got := float32(int(nib)-8) * scales[i/3]
		if math.Abs(float64(got-want)) > 0.25 {
			t.Fatalf("value %d got %.4f want %.4f", i, got, want)
		}
	}
}

func TestConvertSafetensorsQ6RoundTrip(t *testing.T) {
	dir := t.TempDir()
	configJSON := `{
		"vocab_size": 2,
		"hidden_size": 3,
		"intermediate_size": 4,
		"num_hidden_layers": 0,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 3,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	values := []float32{1, -2, 3, 0.5, 0.25, -0.75}
	src := filepath.Join(dir, "model.safetensors")
	writeTestSafetensors(t, src, "lm_head.weight", []int64{2, 3}, values)
	dst := filepath.Join(dir, "model-q6.gguf")
	if err := ConvertSafetensorsWithOptions(src, dst, dir, ConvertOptions{Quantization: "q6"}); err != nil {
		t.Fatal(err)
	}

	gf, err := Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	data, scales, shape, err := gf.Q6Row("lm_head.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 6 || len(scales) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("bad q6 tensor: data=%d scales=%d shape=%v", len(data), len(scales), shape)
	}
	for i, want := range values {
		col := i % 3
		bit := col * 6
		idx := (i/3)*3 + bit/8
		shift := uint(bit % 8)
		x := uint16(data[idx])
		if idx+1 < (i/3+1)*3 {
			x |= uint16(data[idx+1]) << 8
		}
		got := float32(int(byte((x>>shift)&0x3F))-32) * scales[i/3]
		if math.Abs(float64(got-want)) > 0.08 {
			t.Fatalf("value %d got %.4f want %.4f", i, got, want)
		}
	}
}

func writeTestSafetensors(t testing.TB, path, name string, shape []int64, values []float32) {
	t.Helper()
	raw := map[string]any{
		name: map[string]any{
			"dtype":        "F32",
			"shape":        shape,
			"data_offsets": []int{0, len(values) * 4},
		},
	}
	header, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8+len(header)+len(values)*4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	pos := 8 + len(header)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[pos+i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
