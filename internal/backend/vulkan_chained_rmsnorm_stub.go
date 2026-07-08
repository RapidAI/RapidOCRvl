//go:build !windows

package backend

import "fmt"

func VulkanChainedRMSNormMatVecF32(out, x, normWeight, w []float32, n, rows, cols int) error {
	return fmt.Errorf("vulkan chained rmsnorm+matvec requires Windows GPU dispatch support")
}