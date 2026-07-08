package tensor

import (
	"math/rand"
	"testing"
)

func BenchmarkDotQuad128(b *testing.B) {
	n := 128
	a := make([]float32, n)
	bb := make([]float32, n)
	c := make([]float32, n)
	d := make([]float32, n)
	x := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		a[i] = rng.Float32()*2 - 1
		bb[i] = rng.Float32()*2 - 1
		c[i] = rng.Float32()*2 - 1
		d[i] = rng.Float32()*2 - 1
		x[i] = rng.Float32()*2 - 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _ = dotF32Quad(a, bb, c, d, x)
	}
}

func BenchmarkDotPair128x2(b *testing.B) {
	n := 128
	a := make([]float32, n)
	bb := make([]float32, n)
	c := make([]float32, n)
	d := make([]float32, n)
	x := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		a[i] = rng.Float32()*2 - 1
		bb[i] = rng.Float32()*2 - 1
		c[i] = rng.Float32()*2 - 1
		d[i] = rng.Float32()*2 - 1
		x[i] = rng.Float32()*2 - 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dotF32Pair(a, bb, x)
		_, _ = dotF32Pair(c, d, x)
	}
}