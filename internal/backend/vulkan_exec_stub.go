//go:build (!cgo && !windows) || (!windows && !linux)

package backend

import (
	"fmt"
	"paddleocrvl-go/internal/tensor"
)

func VulkanDispatchSmokeTest() error {
	return fmt.Errorf("vulkan GPU dispatch requires cgo on Windows or Linux")
}

func VulkanMatVecF32(out, x, w []float32, rows, cols int) error {
	return fmt.Errorf("vulkan matvec requires GPU dispatch support")
}

func VulkanMatVecArgmaxF32(x, w []float32, rows, cols int) (int, float32, error) {
	return 0, 0, fmt.Errorf("vulkan matvec argmax requires GPU dispatch support")
}

func VulkanMatVecTopKF32(x, w []float32, rows, cols, k int) ([]VulkanTokenScore, error) {
	return nil, fmt.Errorf("vulkan matvec top-k requires GPU dispatch support")
}

func VulkanRMSNormF32(out, x, weight []float32) error {
	return fmt.Errorf("vulkan rmsnorm requires GPU dispatch support")
}

func VulkanAddRMSNormF32(out, dst, add, weight []float32) error {
	return fmt.Errorf("vulkan add rmsnorm requires GPU dispatch support")
}

func VulkanAddRMSNormF32OutOnly(out, dst, add, weight []float32) error {
	return fmt.Errorf("vulkan add rmsnorm requires GPU dispatch support")
}

func VulkanMRoPEF32(x, cosTable, sinTable []float32, heads, dim int) error {
	return fmt.Errorf("vulkan mrope requires GPU dispatch support")
}

func VulkanMRoPEPairF32(q, k, cosTable, sinTable []float32, qHeads, kvHeads, dim int) error {
	return fmt.Errorf("vulkan mrope pair requires GPU dispatch support")
}

func VulkanFusedMatVec3F32(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) error {
	return fmt.Errorf("vulkan fused matvec3 requires GPU dispatch support")
}

func VulkanFusedMatVec2F32(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) error {
	return fmt.Errorf("vulkan fused matvec2 requires GPU dispatch support")
}

func VulkanFusedMatVec2MRoPEF32(outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsB, rowsC, cols, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan fused matvec2+mrope requires GPU dispatch support")
}

func VulkanFusedMatVec3MRoPEF32(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan fused matvec3+mrope requires GPU dispatch support")
}

func VulkanSwiGLUGateUpF32(out, x, gate, up []float32, rows, cols int) error {
	return fmt.Errorf("vulkan swiglu gate/up requires GPU dispatch support")
}

func VulkanSwiGLUDownF32(out, x, gate, up, down []float32, rows, cols, outRows int) error {
	return fmt.Errorf("vulkan swiglu/down requires GPU dispatch support")
}

func VulkanMatVecAddRMSNormF32(normOut, residual, x, w, normWeight []float32, rows, cols int) error {
	return fmt.Errorf("vulkan matvec+add+rmsnorm requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return fmt.Errorf("vulkan swiglu/down+add+rmsnorm requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormF32OutOnly(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return fmt.Errorf("vulkan swiglu/down+add+rmsnorm out-only requires GPU dispatch support")
}

func VulkanMatRowsBiasF32(out, xs [][]float32, w, bias []float32, rows, cols int) error {
	return fmt.Errorf("vulkan matrows+bias requires GPU dispatch support")
}

func VulkanMatRowsBiasAddRowsF32(out, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) error {
	return fmt.Errorf("vulkan matrows+bias+addrows requires GPU dispatch support")
}

func VulkanMatRowsBias3F32(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) error {
	return fmt.Errorf("vulkan matrows+bias3 requires GPU dispatch support")
}

func VulkanVisionAttentionF32(out, q, k, v [][]float32, tokens, heads, headDim int) error {
	return fmt.Errorf("vulkan vision attention requires GPU dispatch support")
}

func VulkanVisionAttentionOutF32(out, q, k, v [][]float32, w, bias []float32, tokens, heads, headDim int) error {
	return fmt.Errorf("vulkan vision attention+out requires GPU dispatch support")
}

func VulkanVisionRoPEAttentionOutF32(out, q, k, v [][]float32, w, bias, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	return fmt.Errorf("vulkan vision rope+attention+out requires GPU dispatch support")
}

func VulkanVisionQKVRoPEAttentionOutF32(out, x [][]float32, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim, hidden int) error {
	return fmt.Errorf("vulkan vision qkv+rope+attention+out requires GPU dispatch support")
}

func VulkanVisionRoPEPairF32(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	return fmt.Errorf("vulkan vision rope pair requires GPU dispatch support")
}

func VulkanMatRowsGELU2F32(out, xs [][]float32, w1, b1, w2, b2 []float32, hiddenRows, cols, outRows int) error {
	return fmt.Errorf("vulkan matrows gelu2 requires GPU dispatch support")
}

func VulkanMatRowsGELU2AddLayerNormF32(out, x, residual [][]float32, w1, b1, w2, b2, normW, normB []float32, hiddenRows, cols, outRows int, eps float32) error {
	return fmt.Errorf("vulkan matrows gelu2+add layernorm requires GPU dispatch support")
}

func VulkanProjectImageF32(out, x [][]float32, normW, normB, w1, b1, w2, b2 []float32, gridT, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	return fmt.Errorf("vulkan project image requires GPU dispatch support")
}

func VulkanLayerNormRowsF32(out, x [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return fmt.Errorf("vulkan layernorm rows requires GPU dispatch support")
}

func VulkanAddThenLayerNormRowsF32(out, x, add [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return fmt.Errorf("vulkan add+layernorm rows requires GPU dispatch support")
}

func VulkanTextAttentionF32(out, q, kCache, vCache []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan text attention requires GPU dispatch support")
}

func VulkanTextAttentionOutF32(out, q, kCache, vCache, w, bias []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan text attention+out requires GPU dispatch support")
}

func VulkanTextAttentionOutAddRMSNormF32(normOut, residual, q, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan text attention+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan first-token value+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q8 first-token value+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q6 first-token value+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q4 first-token value+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextAttentionOutQ8(out, q, kCache, vCache []float32, w *tensor.Q8Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q8 text attention+out requires GPU dispatch support")
}

func VulkanTextAttentionOutAddRMSNormQ8(normOut, residual, q, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q8 text attention+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextAttentionOutQ6(out, q, kCache, vCache []float32, w *tensor.Q6Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q6 text attention+out requires GPU dispatch support")
}

func VulkanTextAttentionOutAddRMSNormQ6(normOut, residual, q, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q6 text attention+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanTextAttentionOutQ4(out, q, kCache, vCache []float32, w *tensor.Q4Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q4 text attention+out requires GPU dispatch support")
}

func VulkanTextAttentionOutAddRMSNormQ4(normOut, residual, q, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q4 text attention+out+add+rmsnorm requires GPU dispatch support")
}

func VulkanMatVecQ8(out, x []float32, q *tensor.Q8Matrix) error {
	return fmt.Errorf("vulkan q8 matvec requires GPU dispatch support")
}

func VulkanMatVecArgmaxQ8(x []float32, q *tensor.Q8Matrix) (int, float32, error) {
	return 0, 0, fmt.Errorf("vulkan q8 matvec argmax requires GPU dispatch support")
}

func VulkanMatVecTopKQ8(x []float32, q *tensor.Q8Matrix, k int) ([]VulkanTokenScore, error) {
	return nil, fmt.Errorf("vulkan q8 matvec top-k requires GPU dispatch support")
}

func VulkanMatVecAddRMSNormQ8(normOut, residual, x []float32, q *tensor.Q8Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q8 matvec+add+rmsnorm requires GPU dispatch support")
}

func VulkanFusedMatVec3Q8(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	return fmt.Errorf("vulkan q8 fused matvec3 requires GPU dispatch support")
}

func VulkanFusedMatVec2Q8(outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	return fmt.Errorf("vulkan q8 fused matvec2 requires GPU dispatch support")
}

func VulkanFusedMatVec2MRoPEQ8(outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q8 fused matvec2+mrope requires GPU dispatch support")
}

func VulkanFusedMatVec3MRoPEQ8(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q8 fused matvec3+mrope requires GPU dispatch support")
}

func VulkanSwiGLUDownQ8(out, x []float32, gate, up, down *tensor.Q8Matrix) error {
	return fmt.Errorf("vulkan q8 swiglu/down requires GPU dispatch support")
}

func VulkanSwiGLUGateUpQ8(out, x []float32, gate, up *tensor.Q8Matrix) error {
	return fmt.Errorf("vulkan q8 swiglu gate/up requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q8 swiglu/down+add+rmsnorm requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ8OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q8 swiglu/down+add+rmsnorm out-only requires GPU dispatch support")
}

func VulkanMatVecQ4(out, x []float32, q *tensor.Q4Matrix) error {
	return fmt.Errorf("vulkan q4 matvec requires GPU dispatch support")
}

func VulkanMatVecArgmaxQ4(x []float32, q *tensor.Q4Matrix) (int, float32, error) {
	return 0, 0, fmt.Errorf("vulkan q4 matvec argmax requires GPU dispatch support")
}

func VulkanMatVecTopKQ4(x []float32, q *tensor.Q4Matrix, k int) ([]VulkanTokenScore, error) {
	return nil, fmt.Errorf("vulkan q4 matvec top-k requires GPU dispatch support")
}

func VulkanMatVecAddRMSNormQ4(normOut, residual, x []float32, q *tensor.Q4Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q4 matvec+add+rmsnorm requires GPU dispatch support")
}

func VulkanFusedMatVec3Q4(outA, outB, outC, x []float32, a, b, c *tensor.Q4Matrix) error {
	return fmt.Errorf("vulkan q4 fused matvec3 requires GPU dispatch support")
}

func VulkanFusedMatVec2Q4(outB, outC, x []float32, a, b, c *tensor.Q4Matrix) error {
	return fmt.Errorf("vulkan q4 fused matvec2 requires GPU dispatch support")
}

func VulkanFusedMatVec2MRoPEQ4(outB, outC, x []float32, a, b, c *tensor.Q4Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q4 fused matvec2+mrope requires GPU dispatch support")
}

func VulkanFusedMatVec3MRoPEQ4(outA, outB, outC, x []float32, a, b, c *tensor.Q4Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q4 fused matvec3+mrope requires GPU dispatch support")
}

func VulkanSwiGLUDownQ4(out, x []float32, gate, up, down *tensor.Q4Matrix) error {
	return fmt.Errorf("vulkan q4 swiglu/down requires GPU dispatch support")
}

func VulkanSwiGLUGateUpQ4(out, x []float32, gate, up *tensor.Q4Matrix) error {
	return fmt.Errorf("vulkan q4 swiglu gate/up requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x []float32, gate, up, down *tensor.Q4Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q4 swiglu/down+add+rmsnorm requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ4OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q4Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q4 swiglu/down+add+rmsnorm out-only requires GPU dispatch support")
}

func VulkanMatVecQ6(out, x []float32, q *tensor.Q6Matrix) error {
	return fmt.Errorf("vulkan q6 matvec requires GPU dispatch support")
}

func VulkanMatVecArgmaxQ6(x []float32, q *tensor.Q6Matrix) (int, float32, error) {
	return 0, 0, fmt.Errorf("vulkan q6 matvec argmax requires GPU dispatch support")
}

func VulkanMatVecTopKQ6(x []float32, q *tensor.Q6Matrix, k int) ([]VulkanTokenScore, error) {
	return nil, fmt.Errorf("vulkan q6 matvec top-k requires GPU dispatch support")
}

func VulkanMatVecAddRMSNormQ6(normOut, residual, x []float32, q *tensor.Q6Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q6 matvec+add+rmsnorm requires GPU dispatch support")
}

func VulkanFusedMatVec3Q6(outA, outB, outC, x []float32, a, b, c *tensor.Q6Matrix) error {
	return fmt.Errorf("vulkan q6 fused matvec3 requires GPU dispatch support")
}

func VulkanFusedMatVec2Q6(outB, outC, x []float32, a, b, c *tensor.Q6Matrix) error {
	return fmt.Errorf("vulkan q6 fused matvec2 requires GPU dispatch support")
}

func VulkanFusedMatVec2MRoPEQ6(outB, outC, x []float32, a, b, c *tensor.Q6Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q6 fused matvec2+mrope requires GPU dispatch support")
}

func VulkanFusedMatVec3MRoPEQ6(outA, outB, outC, x []float32, a, b, c *tensor.Q6Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	return fmt.Errorf("vulkan q6 fused matvec3+mrope requires GPU dispatch support")
}

func VulkanSwiGLUDownQ6(out, x []float32, gate, up, down *tensor.Q6Matrix) error {
	return fmt.Errorf("vulkan q6 swiglu/down requires GPU dispatch support")
}

func VulkanSwiGLUGateUpQ6(out, x []float32, gate, up *tensor.Q6Matrix) error {
	return fmt.Errorf("vulkan q6 swiglu gate/up requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x []float32, gate, up, down *tensor.Q6Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q6 swiglu/down+add+rmsnorm requires GPU dispatch support")
}

func VulkanSwiGLUDownAddRMSNormQ6OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q6Matrix, normWeight []float32) error {
	return fmt.Errorf("vulkan q6 swiglu/down+add+rmsnorm out-only requires GPU dispatch support")
}
