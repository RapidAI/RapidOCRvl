package tensor

import "testing"

func BenchmarkMatVecQ4ModelDown(b *testing.B) {
	rows, cols := 2048, 8192
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w { w[i] = float32(i%17-8) / 17 }
	for i := range x { x[i] = float32(i%13-6) / 13 }
	q := QuantizeQ4Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { MatVecQ4(out, x, q) }
}
