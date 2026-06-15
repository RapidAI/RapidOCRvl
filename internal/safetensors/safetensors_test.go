package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenModelSharded(t *testing.T) {
	dir := t.TempDir()
	writeSafetensorForTest(t, filepath.Join(dir, "a.safetensors"), "a.weight", []int64{2}, []float32{1, 2})
	writeSafetensorForTest(t, filepath.Join(dir, "b.safetensors"), "b.weight", []int64{2}, []float32{3, 4})
	index := map[string]any{
		"weight_map": map[string]string{
			"a.weight": "a.safetensors",
			"b.weight": "b.safetensors",
		},
	}
	b, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	got, shape, err := sf.Float32("b.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 1 || shape[0] != 2 || len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Fatalf("got values=%v shape=%v", got, shape)
	}
	raw, meta, err := sf.Raw("a.weight")
	if err != nil {
		t.Fatal(err)
	}
	if meta.DType != "F32" || len(raw) != 8 {
		t.Fatalf("raw dtype=%s len=%d", meta.DType, len(raw))
	}
	if _, err := sf.Float32Rows("b.weight", func(row int, values []float32) error { return nil }); err == nil {
		t.Fatal("expected 1D row streaming error")
	}
}

func TestOpenModelRejectsBadSingleBeforeIndex(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSafetensorForTest(t, filepath.Join(dir, "shard.safetensors"), "x.weight", []int64{1}, []float32{1})
	raw, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"x.weight": "shard.safetensors"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenModel(dir); err == nil {
		t.Fatal("expected bad model.safetensors error")
	}
}

func TestFloat32Rows(t *testing.T) {
	dir := t.TempDir()
	writeSafetensorForTest(t, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	var rows [][]float32
	shape, err := sf.Float32Rows("x.weight", func(row int, values []float32) error {
		rows = append(rows, append([]float32(nil), values...))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Fatalf("shape=%v", shape)
	}
	if len(rows) != 2 || rows[0][0] != 1 || rows[0][2] != 3 || rows[1][0] != 4 || rows[1][2] != 6 {
		t.Fatalf("rows=%v", rows)
	}
}

func TestWriteRawTo(t *testing.T) {
	dir := t.TempDir()
	writeSafetensorForTest(t, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{2}, []float32{1, -2})
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	var out bytes.Buffer
	meta, err := sf.WriteRawTo("x.weight", &out, make([]byte, 3))
	if err != nil {
		t.Fatal(err)
	}
	if meta.DType != "F32" || out.Len() != 8 {
		t.Fatalf("meta=%+v len=%d", meta, out.Len())
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(out.Bytes())); got != 1 {
		t.Fatalf("first=%f", got)
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(out.Bytes()[4:])); got != -2 {
		t.Fatalf("second=%f", got)
	}
}

func TestWriteFloat32ToBF16(t *testing.T) {
	dir := t.TempDir()
	values := make([]float32, 256*1024+3)
	for i := range values {
		values[i] = float32(i%257-128) / 16
	}
	writeSafetensorBF16ForTest(t, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{int64(len(values))}, values)
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	var out bytes.Buffer
	meta, err := sf.WriteFloat32To("x.weight", &out, make([]byte, 17), make([]float32, 19))
	if err != nil {
		t.Fatal(err)
	}
	if meta.DType != "BF16" || out.Len() != len(values)*4 {
		t.Fatalf("meta=%+v len=%d", meta, out.Len())
	}
	raw := out.Bytes()
	for _, idx := range []int{0, 1024, len(values) - 1} {
		got := math.Float32frombits(binary.LittleEndian.Uint32(raw[idx*4:]))
		want := math.Float32frombits(math.Float32bits(values[idx]) & 0xffff0000)
		if got != want {
			t.Fatalf("value[%d]=%f want %f", idx, got, want)
		}
	}
}

func TestFloat32BF16UsesBlockDecode(t *testing.T) {
	dir := t.TempDir()
	values := make([]float32, 512*1024+7)
	for i := range values {
		values[i] = float32(i%509-254) / 32
	}
	writeSafetensorBF16ForTest(t, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{int64(len(values))}, values)
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	got, shape, err := sf.Float32("x.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 1 || shape[0] != int64(len(values)) {
		t.Fatalf("shape=%v", shape)
	}
	for _, idx := range []int{0, 512*1024 - 1, len(values) - 1} {
		want := math.Float32frombits(math.Float32bits(values[idx]) & 0xffff0000)
		if got[idx] != want {
			t.Fatalf("value[%d]=%f want %f", idx, got[idx], want)
		}
	}
}

func TestFloat32RowsF32BlockValuesAreStableDuringCallback(t *testing.T) {
	dir := t.TempDir()
	values := make([]float32, 2048)
	for i := range values {
		values[i] = float32(i)
	}
	writeSafetensorForTest(t, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{2, 1024}, values)
	sf, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	var got [2]float32
	shape, err := sf.Float32Rows("x.weight", func(row int, values []float32) error {
		got[row] = values[len(values)-1]
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 1024 {
		t.Fatalf("shape=%v", shape)
	}
	if got[0] != 1023 || got[1] != 2047 {
		t.Fatalf("got=%v", got)
	}
}

func TestOpenRejectsHugeHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.safetensors")
	var raw [8]byte
	binary.LittleEndian.PutUint64(raw[:], uint64(maxHeaderBytes+1))
	if err := os.WriteFile(path, raw[:], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Fatal("expected huge header error")
	}
}

func TestDecodeF16KnownValues(t *testing.T) {
	raw := []byte{
		0x00, 0x3c, // 1
		0x00, 0xc0, // -2
		0x00, 0x00, // 0
		0x00, 0x7c, // +Inf
	}
	out := make([]float32, 4)
	decodeF16(out, raw)
	if out[0] != 1 || out[1] != -2 || out[2] != 0 || !math.IsInf(float64(out[3]), 1) {
		t.Fatalf("decoded=%v", out)
	}
}

func writeSafetensorForTest(t testing.TB, path, name string, shape []int64, values []float32) {
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

func writeSafetensorBF16ForTest(t testing.TB, path, name string, shape []int64, values []float32) {
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
