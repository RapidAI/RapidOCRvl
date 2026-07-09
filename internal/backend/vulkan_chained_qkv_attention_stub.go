//go:build !windows

package backend

import "fmt"

func VulkanChainedQKVMRoPEAttentionOutAddRMSNormF32(
	normOut, residual, x []float32,
	wa, wb, wc []float32,
	cosTable, sinTable []float32,
	w, bias, normWeight []float32,
	kCache, vCache []float32,
	cacheEpoch uint64, cacheLen, hidden, numHeads, kvHeads, headDim int,
	outK, outV []float32,
) error {
	return fmt.Errorf("vulkan chained qkv+attention+out+norm not available on this platform")
}