package safetensors

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"
)

func BenchmarkFloat32RowsF32(b *testing.B) {
	dir := b.TempDir()
	values := make([]float32, 512*1024)
	for i := range values {
		values[i] = float32(i%17-8) / 17
	}
	writeSafetensorForTest(b, filepath.Join(dir, "model.safetensors"), "x.weight", []int64{512, 1024}, values)
	sf, err := OpenModel(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer sf.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sf.Float32Rows("x.weight", func(row int, values []float32) error {
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFloat32RowsBF16(b *testing.B) {
	benchmarkFloat32RowsEncoded(b, "BF16")
}

func BenchmarkSafeShardPath(b *testing.B) {
	dir := filepath.Join("C:", "models", "paddleocrvl")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := safeShardPath(dir, "model-00001-of-00002.safetensors"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFloat32RowsF16(b *testing.B) {
	benchmarkFloat32RowsEncoded(b, "F16")
}

func benchmarkFloat32RowsEncoded(b *testing.B, dtype string) {
	dir := b.TempDir()
	const rows, cols = 512, 1024
	data := make([]byte, rows*cols*2)
	for i := 0; i < len(data)/2; i++ {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(i))
	}
	writeSafetensorMetaForTest(b, filepath.Join(dir, "model.safetensors"), map[string]any{
		"dtype":        dtype,
		"shape":        []int64{rows, cols},
		"data_offsets": []int{0, len(data)},
	}, data)
	sf, err := OpenModel(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer sf.Close()
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := sf.Float32Rows("x.weight", func(row int, values []float32) error {
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeBF16(b *testing.B) {
	raw := make([]byte, 2*512*1024)
	out := make([]float32, len(raw)/2)
	for i := 0; i < len(out); i++ {
		raw[i*2] = byte(i)
		raw[i*2+1] = byte(i >> 8)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeBF16(out, raw)
	}
}

func BenchmarkDecodeF16(b *testing.B) {
	raw := make([]byte, 2*512*1024)
	out := make([]float32, len(raw)/2)
	for i := 0; i < len(out); i++ {
		raw[i*2] = byte(i)
		raw[i*2+1] = byte(i >> 8)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeF16(out, raw)
	}
}

func BenchmarkDecodeBF16Shift(b *testing.B) {
	raw := make([]byte, 2*512*1024)
	out := make([]float32, len(raw)/2)
	for i := 0; i < len(out); i++ {
		raw[i*2] = byte(i)
		raw[i*2+1] = byte(i >> 8)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeBF16ShiftBench(out, raw)
	}
}

func BenchmarkDecodeF16Slow(b *testing.B) {
	raw := make([]byte, 2*512*1024)
	out := make([]float32, len(raw)/2)
	for i := 0; i < len(out); i++ {
		raw[i*2] = byte(i)
		raw[i*2+1] = byte(i >> 8)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeF16SlowBench(out, raw)
	}
}

func decodeBF16ShiftBench(out []float32, raw []byte) {
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
	}
}

func decodeF16SlowBench(out []float32, raw []byte) {
	for i := range out {
		out[i] = f16Slow(binary.LittleEndian.Uint16(raw[i*2:]))
	}
}
