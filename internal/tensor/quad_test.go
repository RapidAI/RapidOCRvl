package tensor

import (
	"math/rand"
	"testing"
)

func TestDotQuadFMA(t *testing.T) {
	n := 128
	a := make([]float32, n)
	b := make([]float32, n)
	c := make([]float32, n)
	d := make([]float32, n)
	x := make([]float32, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		a[i] = rng.Float32()*2 - 1
		b[i] = rng.Float32()*2 - 1
		c[i] = rng.Float32()*2 - 1
		d[i] = rng.Float32()*2 - 1
		x[i] = rng.Float32()*2 - 1
	}
	s0, s1, s2, s3 := dotF32Quad(a, b, c, d, x)
	exp0 := dotF32(a, x)
	exp1 := dotF32(b, x)
	exp2 := dotF32(c, x)
	exp3 := dotF32(d, x)
	const tol = 1e-4
	if absfQuad(s0-exp0) > tol*absfQuad(exp0) || absfQuad(s0-exp0) > tol {
		t.Errorf("s0: got %v, want %v", s0, exp0)
	}
	if absfQuad(s1-exp1) > tol*absfQuad(exp1) || absfQuad(s1-exp1) > tol {
		t.Errorf("s1: got %v, want %v", s1, exp1)
	}
	if absfQuad(s2-exp2) > tol*absfQuad(exp2) || absfQuad(s2-exp2) > tol {
		t.Errorf("s2: got %v, want %v", s2, exp2)
	}
	if absfQuad(s3-exp3) > tol*absfQuad(exp3) || absfQuad(s3-exp3) > tol {
		t.Errorf("s3: got %v, want %v", s3, exp3)
	}
}

func absfQuad(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}