package tensor

import "math"

// WeightedSum2 computes dst[i] = a0*x0[i] + a1*x1[i].
func WeightedSum2(dst, x0, x1 []float32, a0, a1 float32) {
	if useDotFMA && len(dst) >= 8 {
		weightedSum2FMA(dst, x0, x1, a0, a1)
		return
	}
	if useDotF32AVX && len(dst) >= 8 {
		weightedSum2AVX(dst, x0, x1, a0, a1)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7]
	}
	for ; i < n; i++ {
		dst[i] = a0*x0[i] + a1*x1[i]
	}
}

// WeightedSum3 computes dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i].
func WeightedSum3(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	if useDotFMA && len(dst) >= 8 {
		weightedSum3FMA(dst, x0, x1, x2, a0, a1, a2)
		return
	}
	if useDotF32AVX && len(dst) >= 8 {
		weightedSum3AVX(dst, x0, x1, x2, a0, a1, a2)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7]
	}
	for ; i < n; i++ {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
	}
}

// WeightedSum4 computes dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i].
func WeightedSum4(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	if useDotFMA && len(dst) >= 8 {
		weightedSum4FMA(dst, x0, x1, x2, x3, a0, a1, a2, a3)
		return
	}
	if useDotF32AVX && len(dst) >= 8 {
		weightedSum4AVX(dst, x0, x1, x2, x3, a0, a1, a2, a3)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
	}
	for ; i < n; i++ {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
	}
}

// WeightedSumAdd2 computes dst[i] += a0*x0[i] + a1*x1[i].
func WeightedSumAdd2(dst, x0, x1 []float32, a0, a1 float32) {
	if useDotFMA && len(dst) >= 8 {
		weightedSumAdd2FMA(dst, x0, x1, a0, a1)
		return
	}
	if useDotF32AVX && len(dst) >= 8 {
		weightedSumAdd2AVX(dst, x0, x1, a0, a1)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] += a0*x0[i] + a1*x1[i]
		dst[i+1] += a0*x0[i+1] + a1*x1[i+1]
		dst[i+2] += a0*x0[i+2] + a1*x1[i+2]
		dst[i+3] += a0*x0[i+3] + a1*x1[i+3]
		dst[i+4] += a0*x0[i+4] + a1*x1[i+4]
		dst[i+5] += a0*x0[i+5] + a1*x1[i+5]
		dst[i+6] += a0*x0[i+6] + a1*x1[i+6]
		dst[i+7] += a0*x0[i+7] + a1*x1[i+7]
	}
	for ; i < n; i++ {
		dst[i] += a0*x0[i] + a1*x1[i]
	}
}

// WeightedSumAdd4 computes dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i].
func WeightedSumAdd4(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	if useDotFMA && len(dst) >= 8 {
		weightedSumAdd4FMA(dst, x0, x1, x2, x3, a0, a1, a2, a3)
		return
	}
	if useDotF32AVX && len(dst) >= 8 {
		weightedSumAdd4AVX(dst, x0, x1, x2, x3, a0, a1, a2, a3)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		dst[i+1] += a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
		dst[i+2] += a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
		dst[i+3] += a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
		dst[i+4] += a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
		dst[i+5] += a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
		dst[i+6] += a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
		dst[i+7] += a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
	}
	for ; i < n; i++ {
		dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
	}
}

// ScaleCopy computes dst[i] = a * x[i].
func ScaleCopy(dst, x []float32, a float32) {
	if useDotF32AVX && len(dst) >= 8 {
		scaleCopyAVX(dst, x, a)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] = a * x[i]
		dst[i+1] = a * x[i+1]
		dst[i+2] = a * x[i+2]
		dst[i+3] = a * x[i+3]
		dst[i+4] = a * x[i+4]
		dst[i+5] = a * x[i+5]
		dst[i+6] = a * x[i+6]
		dst[i+7] = a * x[i+7]
	}
	for ; i < n; i++ {
		dst[i] = a * x[i]
	}
}

// ScaleAdd computes dst[i] += a * x[i].
func ScaleAdd(dst, x []float32, a float32) {
	if useDotF32AVX && len(dst) >= 8 {
		scaleAddAVX(dst, x, a)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] += a * x[i]
		dst[i+1] += a * x[i+1]
		dst[i+2] += a * x[i+2]
		dst[i+3] += a * x[i+3]
		dst[i+4] += a * x[i+4]
		dst[i+5] += a * x[i+5]
		dst[i+6] += a * x[i+6]
		dst[i+7] += a * x[i+7]
	}
	for ; i < n; i++ {
		dst[i] += a * x[i]
	}
}

// ScaleInPlace computes dst[i] *= scale.
func ScaleInPlace(dst []float32, scale float32) {
	if useDotF32AVX && len(dst) >= 8 {
		scaleInPlaceAVX(dst, scale)
		return
	}
	n := len(dst)
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] *= scale
		dst[i+1] *= scale
		dst[i+2] *= scale
		dst[i+3] *= scale
		dst[i+4] *= scale
		dst[i+5] *= scale
		dst[i+6] *= scale
		dst[i+7] *= scale
	}
	for ; i < n; i++ {
		dst[i] *= scale
	}
}

// ExpVec replaces x[i] with exp(x[i] - m) and returns the sum of all exp values.
// Uses AVX2 vectorized exp for the generic path (n >= 8), scalar math.Exp fallback otherwise.
func ExpVec(x []float32, m float32) float32 {
	if useDotFMA && len(x) >= 8 {
		return expF32VecFMA(x, m)
	}
	if useDotF32AVX && len(x) >= 8 {
		return expF32VecAVX(x, m)
	}
	var sum float32
	for i := range x {
		e := float32(math.Exp(float64(x[i] - m)))
		x[i] = e
		sum += e
	}
	return sum
}

// ApplyMRoPETable applies multi-dimensional rotary position embedding using precomputed cos/sin tables.
// For each head, it performs complex rotation: x[i] = x[i]*cos - x[half+i]*sin, x[half+i] = x[half+i]*cos + x[i]*sin
func ApplyMRoPETable(x []float32, cosTable, sinTable []float32, heads, dim int) {
	if useDotFMA && useDotF32AVX && dim >= 16 {
		ropeFMA(x, cosTable, sinTable, heads, dim)
		return
	}
	if useDotF32AVX && dim >= 16 {
		ropeAVX(x, cosTable, sinTable, heads, dim)
		return
	}
	half := dim / 2
	for h := 0; h < heads; h++ {
		base := h * dim
		for i := 0; i < half; i++ {
			cs, sn := cosTable[i], sinTable[i]
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
		}
	}
}
