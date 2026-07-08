package backend

import (
	"math"
	"os"
	"testing"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanChainedRMSNormFusedQKVMRoPEF32(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU chained rmsnorm+qkvmrope test")
	}

	// Use realistic dimensions: hidden=64, qHeads=4, kvHeads=2, headDim=16
	n := 64
	cols := n
	rowsA := 64
	rowsB := 32
	rowsC := 32
	kvHeads := 2
	headDim := 16
	half := headDim / 2

	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.3))
	}

	normWeight := make([]float32, n)
	for i := range normWeight {
		normWeight[i] = 1.0
	}

	wa := make([]float32, rowsA*cols)
	wb := make([]float32, rowsB*cols)
	wc := make([]float32, rowsC*cols)
	for i := range wa {
		wa[i] = float32(math.Cos(float64(i)*0.1)) * 0.01
	}
	for i := range wb {
		wb[i] = float32(math.Sin(float64(i)*0.1)) * 0.01
	}
	for i := range wc {
		wc[i] = float32(math.Cos(float64(i)*0.15)) * 0.01
	}

	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		angle := float64(i) / float64(half) * 0.5
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}

	// Reference: RMSNorm then matvec3+mrope
	normed := make([]float32, n)
	tensor.RMSNorm(normed, x, normWeight, 1e-6)

	outARef := make([]float32, rowsA)
	outBRef := make([]float32, rowsB)
	outCRef := make([]float32, rowsC)
	for r := 0; r < rowsA; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += wa[r*cols+c] * normed[c]
		}
		outARef[r] = sum
	}
	for r := 0; r < rowsB; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += wb[r*cols+c] * normed[c]
		}
		outBRef[r] = sum
	}
	for r := 0; r < rowsC; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += wc[r*cols+c] * normed[c]
		}
		outCRef[r] = sum
	}
	// Apply RoPE to q and k
	for h := 0; h < rowsA/headDim; h++ {
		for d := 0; d < half; d++ {
			idx := h*headDim + d
			idx2 := idx + half
			c := cosTable[d]
			s := sinTable[d]
			outARef[idx], outARef[idx2] = outARef[idx]*c-outARef[idx2]*s, outARef[idx]*s+outARef[idx2]*c
		}
	}
	for h := 0; h < rowsB/headDim; h++ {
		for d := 0; d < half; d++ {
			idx := h*headDim + d
			idx2 := idx + half
			c := cosTable[d]
			s := sinTable[d]
			outBRef[idx], outBRef[idx2] = outBRef[idx]*c-outBRef[idx2]*s, outBRef[idx]*s+outBRef[idx2]*c
		}
	}

	outA := make([]float32, rowsA)
	outB := make([]float32, rowsB)
	outC := make([]float32, rowsC)
	err := VulkanChainedRMSNormFusedQKVMRoPEF32(outA, outB, outC, x, normWeight, wa, wb, wc, cosTable, sinTable, n, rowsA, rowsB, rowsC, cols, kvHeads, headDim)
	if err != nil {
		t.Fatalf("VulkanChainedRMSNormFusedQKVMRoPEF32 failed: %v", err)
	}

	tol := float32(0.01)
	for i := range outA {
		if diff := float32(math.Abs(float64(outA[i] - outARef[i]))); diff > tol {
			t.Errorf("outA[%d]: got %f want %f diff %f", i, outA[i], outARef[i], diff)
			break
		}
	}
	for i := range outB {
		if diff := float32(math.Abs(float64(outB[i] - outBRef[i]))); diff > tol {
			t.Errorf("outB[%d]: got %f want %f diff %f", i, outB[i], outBRef[i], diff)
			break
		}
	}
	for i := range outC {
		if diff := float32(math.Abs(float64(outC[i] - outCRef[i]))); diff > tol {
			t.Errorf("outC[%d]: got %f want %f diff %f", i, outC[i], outCRef[i], diff)
			break
		}
	}
}
func TestVulkanChainedRMSNormFusedQKVMRoPEQ8(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run the Vulkan GPU chained rmsnorm+qkvmrope q8 test")
	}

	// hidden=64, qHeads=4 (rowsA=64), kvHeads=2 (rowsB=rowsC=32), headDim=16
	n := 64
	cols := n
	rowsA := 64
	rowsB := 32
	rowsC := 32
	kvHeads := 2
	headDim := 16
	half := headDim / 2

	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.3))
	}

	normWeight := make([]float32, n)
	for i := range normWeight {
		normWeight[i] = 1.0
	}

	wa := make([]float32, rowsA*cols)
	wb := make([]float32, rowsB*cols)
	wc := make([]float32, rowsC*cols)
	for i := range wa {
		wa[i] = float32(math.Cos(float64(i)*0.1)) * 0.01
	}
	for i := range wb {
		wb[i] = float32(math.Sin(float64(i)*0.1)) * 0.01
	}
	for i := range wc {
		wc[i] = float32(math.Cos(float64(i)*0.15)) * 0.01
	}

	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)

	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		angle := float64(i) / float64(half) * 0.5
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}

	// Reference: RMSNorm then Q8 matvec (matches GPU quantisation) then RoPE
	normed := make([]float32, n)
	tensor.RMSNorm(normed, x, normWeight, 1e-6)

	outARef := make([]float32, rowsA)
	outBRef := make([]float32, rowsB)
	outCRef := make([]float32, rowsC)
	tensor.MatVecQ8(outARef, normed, qa)
	tensor.MatVecQ8(outBRef, normed, qb)
	tensor.MatVecQ8(outCRef, normed, qc)

	// Apply RoPE to q and k (v is not rotated)
	for h := 0; h < rowsA/headDim; h++ {
		for d := 0; d < half; d++ {
			idx := h*headDim + d
			idx2 := idx + half
			c := cosTable[d]
			s := sinTable[d]
			outARef[idx], outARef[idx2] = outARef[idx]*c-outARef[idx2]*s, outARef[idx]*s+outARef[idx2]*c
		}
	}
	for h := 0; h < rowsB/headDim; h++ {
		for d := 0; d < half; d++ {
			idx := h*headDim + d
			idx2 := idx + half
			c := cosTable[d]
			s := sinTable[d]
			outBRef[idx], outBRef[idx2] = outBRef[idx]*c-outBRef[idx2]*s, outBRef[idx]*s+outBRef[idx2]*c
		}
	}

	outA := make([]float32, rowsA)
	outB := make([]float32, rowsB)
	outC := make([]float32, rowsC)
	err := VulkanChainedRMSNormFusedQKVMRoPEQ8(outA, outB, outC, x, normWeight, cosTable, sinTable, qa, qb, qc, n, kvHeads, headDim)
	if err != nil {
		t.Fatalf("VulkanChainedRMSNormFusedQKVMRoPEQ8 failed: %v", err)
	}

	tol := float32(0.01)
	for i := range outA {
		if diff := float32(math.Abs(float64(outA[i] - outARef[i]))); diff > tol {
			t.Errorf("outA[%d]: got %f want %f diff %f", i, outA[i], outARef[i], diff)
			break
		}
	}
	for i := range outB {
		if diff := float32(math.Abs(float64(outB[i] - outBRef[i]))); diff > tol {
			t.Errorf("outB[%d]: got %f want %f diff %f", i, outB[i], outBRef[i], diff)
			break
		}
	}
	for i := range outC {
		if diff := float32(math.Abs(float64(outC[i] - outCRef[i]))); diff > tol {
			t.Errorf("outC[%d]: got %f want %f diff %f", i, outC[i], outCRef[i], diff)
			break
		}
	}
}
