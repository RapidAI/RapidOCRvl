package tensor

import "math"

// AddRMSNormOutOnly computes out = RMSNorm(dst + add, weight) without
// modifying dst in place. This is the "out-only" variant of AddRMSNorm
// used when the residual must be preserved.
func AddRMSNormOutOnly(out, dst, add, weight []float32, eps float32) {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(dst)
	for ; i+7 < n; i += 8 {
		v0 := dst[i] + add[i]
		v1 := dst[i+1] + add[i+1]
		v2 := dst[i+2] + add[i+2]
		v3 := dst[i+3] + add[i+3]
		v4 := dst[i+4] + add[i+4]
		v5 := dst[i+5] + add[i+5]
		v6 := dst[i+6] + add[i+6]
		v7 := dst[i+7] + add[i+7]
		s0 += v0 * v0
		s1 += v1 * v1
		s2 += v2 * v2
		s3 += v3 * v3
		s4 += v4 * v4
		s5 += v5 * v5
		s6 += v6 * v6
		s7 += v7 * v7
	}
	ss := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		v := dst[i] + add[i]
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss)/float64(n)+float64(eps)))
	i = 0
	for ; i+7 < n; i += 8 {
		out[i] = (dst[i] + add[i]) * scale * weight[i]
		out[i+1] = (dst[i+1] + add[i+1]) * scale * weight[i+1]
		out[i+2] = (dst[i+2] + add[i+2]) * scale * weight[i+2]
		out[i+3] = (dst[i+3] + add[i+3]) * scale * weight[i+3]
		out[i+4] = (dst[i+4] + add[i+4]) * scale * weight[i+4]
		out[i+5] = (dst[i+5] + add[i+5]) * scale * weight[i+5]
		out[i+6] = (dst[i+6] + add[i+6]) * scale * weight[i+6]
		out[i+7] = (dst[i+7] + add[i+7]) * scale * weight[i+7]
	}
	for ; i < n; i++ {
		out[i] = (dst[i] + add[i]) * scale * weight[i]
	}
}

// FusedMatVec2 computes outB = x . wB^T and outC = x . wC^T in a single
// pass over x, reducing memory bandwidth for the K and V projections.
func FusedMatVec2(outB, outC, x, wB, wC []float32, rowsB, rowsC, cols int) {
	totalRows := rowsB + rowsC
	if shouldParallel(totalRows*cols, totalRows) {
		parallelFor(totalRows, func(start, end int) {
			fusedMatVec2Serial(outB, outC, x, wB, wC, rowsB, cols, start, end)
		})
		return
	}
	fusedMatVec2Serial(outB, outC, x, wB, wC, rowsB, cols, 0, totalRows)
}

func fusedMatVec2Serial(outB, outC, x, wB, wC []float32, rowsB, cols, start, end int) {
	bEnd := min(end, rowsB)
	for r := start; r < bEnd; r++ {
		outB[r] = dotF32(wB[r*cols:(r+1)*cols], x)
	}
	cStart := max(start, rowsB)
	for r := cStart; r < end; r++ {
		cr := r - rowsB
		outC[cr] = dotF32(wC[cr*cols:(cr+1)*cols], x)
	}
}
