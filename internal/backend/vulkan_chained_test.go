package backend

import (
	"os"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanChainedMatVec2F32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU chained matvec2 test")
	}

	// Op1: out1 = w1 @ x,  rows1=4, cols1=5
	x := []float32{2, -1, 0.5, 4, -3}
	w1 := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	// Op2: out2 = w2 @ out1, rows2=3, cols2=4 (rows1)
	w2 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	out2 := make([]float32, 3)
	if err := VulkanChainedMatVec2F32(out2, x, w1, w2, 4, 5, 3); err != nil {
		t.Fatal(err)
	}

	// Compute reference: out1 = w1 @ x, then out2 = w2 @ out1
	out1Ref := make([]float32, 4)
	tensor.MatVec(out1Ref, x, w1, 4, 5)
	want := make([]float32, 3)
	tensor.MatVec(want, out1Ref, w2, 3, 4)

	for i := range want {
		if d := out2[i] - want[i]; d < -1e-4 || d > 1e-4 {
			t.Fatalf("chained out2[%d] = %.6f, want %.6f (out1Ref=%v)", i, out2[i], want[i], out1Ref)
		}
	}
}