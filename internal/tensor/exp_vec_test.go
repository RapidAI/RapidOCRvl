package tensor

import (
	"math"
	"testing"
)

func TestExpVec(t *testing.T) {
	// Test with known values
	x := []float32{-2, 0, 3, 1, -1, 2, 4, 0.5, -0.5}
	xcopy := make([]float32, len(x))
	copy(xcopy, x)
	
	m := float32(4.0) // max
	
	sum := ExpVec(x, m)
	
	// Compare with math.Exp
	var expectedSum float32
	for i, v := range xcopy {
		expected := float32(math.Exp(float64(v - m)))
		expectedSum += expected
		diff := math.Abs(float64(x[i] - expected))
		if diff > 1e-5 {
			t.Errorf("ExpVec[%d]: got %f, want %f (diff %e)", i, x[i], expected, diff)
		}
	}
	
	sumDiff := math.Abs(float64(sum - expectedSum))
	if sumDiff > 1e-4 {
		t.Errorf("ExpVec sum: got %f, want %f (diff %e)", sum, expectedSum, sumDiff)
	}
}