package tensor

import "testing"

func BenchmarkMatVecQ8AddRMSNormModel(b *testing.B) {
	rows, cols := 2048, 8192
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	residual := make([]float32, rows)
	out := make([]float32, rows)
	normW := make([]float32, rows)
	for i := range w { w[i] = float32(i%17-8) / 17 }
	for i := range x { x[i] = float32(i%13-6) / 13 }
	for i := range residual { residual[i] = float32(i%11-5) / 11 }
	for i := range normW { normW[i] = 1 + float32(i%7)/32 }
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { MatVecQ8AddRMSNorm(out, residual, x, q, normW, 1e-6) }
}

func BenchmarkMatVecQ4AddRMSNormModel(b *testing.B) {
	rows, cols := 2048, 8192
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	residual := make([]float32, rows)
	out := make([]float32, rows)
	normW := make([]float32, rows)
	for i := range w { w[i] = float32(i%17-8) / 17 }
	for i := range x { x[i] = float32(i%13-6) / 13 }
	for i := range residual { residual[i] = float32(i%11-5) / 11 }
	for i := range normW { normW[i] = 1 + float32(i%7)/32 }
	q := QuantizeQ4Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { MatVecQ4AddRMSNorm(out, residual, x, q, normW, 1e-6) }
}
