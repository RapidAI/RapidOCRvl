package tensor

import (
	"math"
	"testing"
)

func TestDotQ8PairFMACorrectness(t *testing.T) {
	a := make([]int8, 128)
	b := make([]int8, 128)
	x := make([]float32, 128)
	for i := range a {
		a[i] = int8(i%17 - 8)
		b[i] = int8(i%13 - 6)
		x[i] = float32(i%7) / 7
	}
	r0, r1 := dotQ8PairFMA(a, b, x)
	s0, s1 := dotQ8PairScalar(a, b, x)
	if math.Abs(float64(r0-s0)) > 1e-3 || math.Abs(float64(r1-s1)) > 1e-3 {
		t.Errorf("got (%f,%f) want (%f,%f)", r0, r1, s0, s1)
	}
}
