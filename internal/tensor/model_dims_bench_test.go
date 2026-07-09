package tensor

import "testing"

func BenchmarkMatVecQ8ModelKV(b *testing.B) {
	// K/V projection: 512 rows, 2048 cols (4 KV heads * 128 dim)
	rows, cols := 512, 2048
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ8(out, x, q)
	}
}

func BenchmarkMatVecQ8ModelDown(b *testing.B) {
	// Down projection: 2048 rows, 8192 cols
	rows, cols := 2048, 8192
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ8(out, x, q)
	}
}

func BenchmarkMatVecQ8ModelGateUp(b *testing.B) {
	// Gate/Up projection: 8192 rows, 2048 cols
	rows, cols := 8192, 2048
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ8(out, x, q)
	}
}
