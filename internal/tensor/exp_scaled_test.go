package tensor

import (
	"math"
	"testing"
)

func TestExpScaledVec(t *testing.T) {
	x := make([]float32, 256)
	for i := range x {
		x[i] = float32(i-128) * 0.01
	}
	scale := float32(0.5)
	bias := float32(-1.5)

	expected := make([]float32, len(x))
	var expectedSum float32
	for i, v := range x {
		expected[i] = float32(math.Exp(float64(v*scale + bias)))
		expectedSum += expected[i]
	}

	out := make([]float32, len(x))
	sum := ExpScaledVec(out, x, scale, bias)

	sumDiff := math.Abs(float64(sum - expectedSum))
	if sumDiff > 1e-3 {
		t.Errorf("ExpScaledVec sum: got %f, want %f (diff %e)", sum, expectedSum, sumDiff)
	}

	for i := range x {
		diff := math.Abs(float64(out[i] - expected[i]))
		if diff > 1e-5 {
			t.Errorf("ExpScaledVec[%d]: got %f, want %f (diff %e)", i, out[i], expected[i], diff)
		}
	}
}
