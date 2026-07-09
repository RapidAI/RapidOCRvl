package tensor

import (
	"runtime"
	"testing"
)

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
func BenchmarkMatVecQ8PairModelGateUp(b *testing.B) {
	// Gate/Up pair: 8192 rows, 2048 cols, two matrices
	rows, cols := 8192, 2048
	gateW := make([]float32, rows*cols)
	upW := make([]float32, rows*cols)
	x := make([]float32, cols)
	outA := make([]float32, rows)
	outB := make([]float32, rows)
	for i := range gateW {
		gateW[i] = float32(i%17-8) / 17
		upW[i] = float32(i%13-6) / 13
	}
	for i := range x {
		x[i] = float32(i%19-9) / 19
	}
	gate := QuantizeQ8Row(gateW, rows, cols)
	up := QuantizeQ8Row(upW, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matVecQ8Pair(outA, outB, x, gate, up)
	}
}
func BenchmarkMatVecQ8PairModelGateUpSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatVecQ8PairModelGateUp(b)
}
