package backend

import (
	"os"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanChainedRMSNormMatVecF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU chained rmsnorm+matvec test")
	}

	// Input: 5-element hidden state
	x := []float32{2, -1, 0.5, 4, -3}
	n := len(x)

	// RMSNorm weight (all ones for simplicity)
	normWeight := []float32{1, 1, 1, 1, 1}

	// MatVec: out = w @ rmsnorm(x, weight), rows=4, cols=5
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	rows, cols := 4, 5

	out := make([]float32, rows)
	if err := VulkanChainedRMSNormMatVecF32(out, x, normWeight, w, n, rows, cols); err != nil {
		t.Fatal(err)
	}

	// Compute reference: normOut = rmsnorm(x, weight), then out = w @ normOut
	normOut := make([]float32, n)
	tensor.RMSNorm(normOut, x, normWeight, 1e-6)
	want := make([]float32, rows)
	tensor.MatVec(want, normOut, w, rows, cols)

	for i := range want {
		if d := out[i] - want[i]; d < -1e-3 || d > 1e-3 {
			t.Fatalf("chained out[%d] = %.6f, want %.6f (normOut=%v)", i, out[i], want[i], normOut)
		}
	}
}