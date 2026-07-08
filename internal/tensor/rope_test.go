package tensor

import (
	"math"
	"testing"
)

func TestApplyMRoPETable(t *testing.T) {
	heads, dim := 4, 128
	half := dim / 2
	total := heads * dim
	x := make([]float32, total)
	for i := range x {
		x[i] = float32(i%17 - 8) / 4.0
	}
	xwant := make([]float32, total)
	copy(xwant, x)

	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		ang := float64(i+1) * 0.1
		cosTable[i] = float32(math.Cos(ang))
		sinTable[i] = float32(math.Sin(ang))
	}

	// Reference implementation
	for h := 0; h < heads; h++ {
		base := h * dim
		for i := 0; i < half; i++ {
			cs, sn := cosTable[i], sinTable[i]
			a, b := xwant[base+i], xwant[base+half+i]
			xwant[base+i] = a*cs - b*sn
			xwant[base+half+i] = b*cs + a*sn
		}
	}

	// AVX implementation
	ApplyMRoPETable(x, cosTable, sinTable, heads, dim)

	for i := range x {
		if math.Abs(float64(x[i]-xwant[i])) > 1e-5 {
			t.Fatalf("ApplyMRoPETable[%d]: got %f, want %f", i, x[i], xwant[i])
		}
	}
}