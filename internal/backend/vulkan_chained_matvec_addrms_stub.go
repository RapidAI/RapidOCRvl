//go:build !windows

package backend

import "fmt"

func VulkanChainedMatVecAddRMSNormMatVecF32(out, x, residual, w1, normWeight, w2 []float32, rows1, cols1, rows2, cols2 int) error {
	return fmt.Errorf("VulkanChainedMatVecAddRMSNormMatVecF32 not available on this platform")
}
