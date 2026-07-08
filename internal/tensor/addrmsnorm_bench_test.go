package tensor

import (
	"testing"
)

func BenchmarkAddRMSNormOutOnly(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	add := make([]float32, n)
	w := make([]float32, n)
	out := make([]float32, n)
	fillBench(dst)
	fillBench(add)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddRMSNormOutOnly(out, dst, add, w, 1e-6)
	}
}