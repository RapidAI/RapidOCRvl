package tensor

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
	if useDotFMA && len(dst) >= 8 {
		scaleAddFMA(dst, x, a)
		return
	}
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
// Uses AVX2 vectorized exp for the generic path (n >= 8), scalar FastExpF32 fallback otherwise.
func ExpVec(x []float32, m float32) float32 {
	if useDotFMA && len(x) >= 8 {
		return expF32VecFMA(x, m)
	}
	if useDotF32AVX && len(x) >= 8 {
		return expF32VecAVX(x, m)
	}
	var sum float32
	for i := range x {
		e := FastExpF32(x[i] - m)
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


// RoPEPairAxis rotates q and k in-place for a single axis segment.
// It rotates q[start:start+axisLen] and k[start:start+axisLen] using
// cosTable[0:axisLen/2] and sinTable[0:axisLen/2].
// The rotation is: for i in [0, half):
//   a = q[start+i], b = q[start+half+i]
//   q[start+i] = a*cos - b*sin, q[start+half+i] = b*cos + a*sin
//   same for k
func RoPEPairAxis(q, k []float32, start, axisLen int, cosTable, sinTable []float32) {
	half := axisLen / 2
	if half == 0 {
		return
	}
	if useDotFMA && useDotF32AVX && half >= 8 {
		ropePairAxisFMA(q, k, start, half, cosTable, sinTable)
		return
	}
	if useDotF32AVX && half >= 8 {
		ropePairAxisAVX(q, k, start, half, cosTable, sinTable)
		return
	}
	q0 := q[start : start+half]
	q1 := q[start+half : start+axisLen]
	k0 := k[start : start+half]
	k1 := k[start+half : start+axisLen]
	for i := 0; i < half; i++ {
		cs, sn := cosTable[i], sinTable[i]
		qa, qb := q0[i], q1[i]
		ka, kb := k0[i], k1[i]
		q0[i] = qa*cs - qb*sn
		q1[i] = qb*cs + qa*sn
		k0[i] = ka*cs - kb*sn
		k1[i] = kb*cs + ka*sn
	}
}


// Max returns the maximum value in x. Uses AVX2 when available.
func Max(x []float32) float32 {
	if useDotF32AVX && len(x) >= 8 {
		return maxF32AVX2(x)
	}
	if len(x) == 0 {
		return float32(0)
	}
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// MaxInPlace finds the maximum value in x. Same as Max but exported for model use.
func MaxInPlace(x []float32) float32 {
	return Max(x)
}


// ExpScaledVec fills out[i] = exp(x[i]*scale + bias) and returns the sum of all out values.
// Uses AVX2+FMA vectorized exp when available.
func ExpScaledVec(out, x []float32, scale, bias float32) float32 {
	n := min(len(out), len(x))
	if n == 0 {
		return 0
	}
	// Compute scaled+offset values into out, then use ExpVec
	for i := 0; i < n; i++ {
		out[i] = x[i]*scale + bias
	}
	// ExpVec computes out[i] = exp(out[i] - 0) = exp(out[i]) and returns sum
	// But we want exp(out[i]) not exp(out[i]-m). So use m=0.
	return ExpVec(out[:n], 0)
}
