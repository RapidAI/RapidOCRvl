package tensor

import (
	"math/rand"
	"testing"
)

func BenchmarkMatVecArgmaxLarge(b *testing.B) {
	rows := 32000
	cols := 1024
	x := make([]float32, cols)
	w := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(42))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatVecArgmax(x, w, rows, cols)
	}
}

func BenchmarkMatVecArgmaxQ8Large(b *testing.B) {
	rows := 32000
	cols := 1024
	x := make([]float32, cols)
	rng := rand.New(rand.NewSource(42))
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	q := &Q8Matrix{
		Rows:  rows,
		Cols:  cols,
		Data:  make([]int8, rows*cols),
		Scale: make([]float32, rows),
	}
	for i := range q.Data {
		q.Data[i] = int8(rng.Intn(256) - 128)
	}
	for i := range q.Scale {
		q.Scale[i] = rng.Float32()*0.01 + 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = MatVecArgmaxQ8(x, q)
	}
}