package tensor

import (
	"math"
	"testing"
)

func TestDotQ8TripletFMACorrectness(t *testing.T) {
	a := make([]int8, 128)
	b := make([]int8, 128)
	c := make([]int8, 128)
	x := make([]float32, 128)
	for i := range a {
		a[i] = int8(i%17 - 8)
		b[i] = int8(i%13 - 6)
		c[i] = int8(i%11 - 5)
		x[i] = float32(i%7) / 7
	}
	r0, r1, r2 := dotQ8TripletFMA(a, b, c, x)
	s0, s1, s2 := dotQ8TripletScalar(a, b, c, x)
	if math.Abs(float64(r0-s0)) > 1e-3 || math.Abs(float64(r1-s1)) > 1e-3 || math.Abs(float64(r2-s2)) > 1e-3 {
		t.Errorf("got (%f,%f,%f) want (%f,%f,%f)", r0, r1, r2, s0, s1, s2)
	}
}
