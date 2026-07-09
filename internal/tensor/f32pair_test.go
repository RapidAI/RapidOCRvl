package tensor

import (
	"math"
	"testing"
)

func TestDotF32PairFMA(t *testing.T) {
	a := make([]float32, 128)
	b := make([]float32, 128)
	x := make([]float32, 128)
	for i := range a {
		a[i] = float32(i%17 - 8) / 17
		b[i] = float32(i%13 - 6) / 13
		x[i] = float32(i%7) / 7
	}
	r0, r1 := dotF32PairFMA(a, b, x)
	s0 := dotF32Scalar(a, x)
	s1 := dotF32Scalar(b, x)
	if math.Abs(float64(r0-s0)) > 1e-4 || math.Abs(float64(r1-s1)) > 1e-4 {
		t.Errorf("got (%f,%f) want (%f,%f)", r0, r1, s0, s1)
	}
}

func TestDotF32TripletFMA(t *testing.T) {
	a := make([]float32, 128)
	b := make([]float32, 128)
	c := make([]float32, 128)
	x := make([]float32, 128)
	for i := range a {
		a[i] = float32(i%17 - 8) / 17
		b[i] = float32(i%13 - 6) / 13
		c[i] = float32(i%11 - 5) / 11
		x[i] = float32(i%7) / 7
	}
	r0, r1, r2 := dotF32TripletFMA(a, b, c, x)
	s0 := dotF32Scalar(a, x)
	s1 := dotF32Scalar(b, x)
	s2 := dotF32Scalar(c, x)
	if math.Abs(float64(r0-s0)) > 1e-4 || math.Abs(float64(r1-s1)) > 1e-4 || math.Abs(float64(r2-s2)) > 1e-4 {
		t.Errorf("got (%f,%f,%f) want (%f,%f,%f)", r0, r1, r2, s0, s1, s2)
	}
}
