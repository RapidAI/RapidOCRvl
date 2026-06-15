package model

import (
	"math"
	"testing"

	"paddleocrvl-go/internal/config"
)

func TestRuntimeMRoPEMatchesScalarReference(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		HeadDim:   128,
		RopeTheta: 10000,
		RopeScaling: &config.RopeScaling{
			MropeSection: []int{16, 24, 24},
		},
	}}
	rt.initTextRoPE()

	in := make([]float32, 2*128)
	for i := range in {
		in[i] = float32(math.Sin(float64(i) * 0.17))
	}
	want := append([]float32(nil), in...)
	got := append([]float32(nil), in...)

	pos := ropePos{T: 7, H: 3, W: 11}
	applyMRoPE(want, 2, 128, pos, 10000)
	rt.applyMRoPE(got, 2, 128, pos)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("index %d got %.8f want %.8f", i, got[i], want[i])
		}
	}
}
