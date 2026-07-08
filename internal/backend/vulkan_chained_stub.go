//go:build !windows

package backend

import "fmt"

func VulkanChainedMatVec2F32(out2, x, w1, w2 []float32, rows1, cols1, rows2 int) error {
	return fmt.Errorf("vulkan chained matvec2 requires Windows GPU dispatch support")
}