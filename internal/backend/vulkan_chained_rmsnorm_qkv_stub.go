//go:build !windows

package backend

import (
	"fmt"
	"paddleocrvl-go/internal/tensor"
)

// VulkanChainedRMSNormFusedQKVMRoPEF32 is a stub for non-Windows platforms.
func VulkanChainedRMSNormFusedQKVMRoPEF32(outA, outB, outC, x, normWeight, wa, wb, wc, cosTable, sinTable []float32, n, rowsA, rowsB, rowsC, cols, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan chained rmsnorm+qkvmrope not available on this platform")
}

// VulkanChainedRMSNormFusedQKVMRoPEQ8 is a stub for non-Windows platforms.
func VulkanChainedRMSNormFusedQKVMRoPEQ8(outA, outB, outC, x, normWeight, cosTable, sinTable []float32, a, b, c *tensor.Q8Matrix, n, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan chained rmsnorm+qkvmrope q8 not available on this platform")
}
