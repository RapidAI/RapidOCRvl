package backend

import (
	"math"
	"os"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanChainedMatVecAddRMSNormMatVecF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU chained matvec+addrmsnorm+matvec test")
	}

	// matvec1: 5x4 (rows1=5, cols1=4), AddRMSNorm: 5, matvec2: 3x5 (rows2=3, cols2=5)
	rows1, cols1 := 5, 4
	rows2, cols2 := 3, 5

	x := []float32{1, 2, 3, 4}
	w1 := make([]float32, rows1*cols1)
	for i := range w1 {
		w1[i] = float32(math.Sin(float64(i) * 0.2))
	}
	residual := []float32{0.5, 0.3, 0.1, 0.2, 0.4}
	normWeight := []float32{1, 1, 1, 1, 1}
	w2 := make([]float32, rows2*rows1)
	for i := range w2 {
		w2[i] = float32(math.Cos(float64(i) * 0.15))
	}

	// Reference: matvec1 -> AddRMSNorm -> matvec2
	out1 := make([]float32, rows1)
	tensor.MatVec(out1, x, w1, rows1, cols1)

	// AddRMSNorm: residual2 = residual + out1; out2 = rmsnorm(residual2, normWeight)
	added := make([]float32, rows1)
	for i := range added {
		added[i] = residual[i] + out1[i]
	}
	normOut := make([]float32, rows1)
	tensor.RMSNorm(normOut, added, normWeight, 1e-6)

	want := make([]float32, rows2)
	tensor.MatVec(want, normOut, w2, rows2, rows1)

	out := make([]float32, rows2)
	err := VulkanChainedMatVecAddRMSNormMatVecF32(out, x, residual, w1, normWeight, w2, rows1, cols1, rows2, cols2)
	if err != nil {
		t.Fatalf("VulkanChainedMatVecAddRMSNormMatVecF32 failed: %v", err)
	}

	tol := float32(0.01)
	for i := range want {
		if diff := float32(math.Abs(float64(out[i] - want[i]))); diff > tol {
			t.Errorf("out[%d]: got %f want %f diff %f", i, out[i], want[i], diff)
		}
	}
}
