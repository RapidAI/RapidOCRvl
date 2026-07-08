//go:build !amd64

package tensor

const useDotF32AVX = false
const useDotQ8AVX2 = false
const useDotQ4AVX2 = false

func addAVX(out, a, b []float32) {
	addScalar(out, a, b)
}

func addInPlaceAVX(dst, x []float32) {
	addInPlaceScalar(dst, x)
}

func addInPlaceSumAVX(dst, x []float32) float32 {
	return addInPlaceSumScalar(dst, x)
}

func addInPlaceSumSquaresAVX(dst, x []float32) float32 {
	return addInPlaceSumSquaresScalar(dst, x)
}

func addSumSquaresAVX(out, dst, add []float32) float32 {
	return addSumSquaresScalar(out, dst, add)
}

func addSumSquaresFMA(out, dst, add []float32) float32 {
	return addSumSquaresScalar(out, dst, add)
}

func addScaledAVX(dst, x []float32, scale float32) {
	addScaledScalar(dst, x, scale)
}

func scaleInPlaceAVX(dst []float32, scale float32) {
	scaleInPlaceScalar(dst, scale)
}

func mulScaleAVX(out, x, weight []float32, scale float32) {
	mulScaleScalar(out, x, weight, scale)
}

func affineNormAVX(out, x, weight, bias []float32, mean, scale float32) {
	affineNormScalar(out, x, weight, bias, mean, scale)
}

func sumF32AVX(x []float32) float32 {
	return sumF32Scalar(x)
}

func sumSquaresCenteredAVX(x []float32, mean float32) float32 {
	return sumSquaresCenteredScalar(x, mean)
}

func maxAbsFloat32AVX(x []float32) float32 {
	return maxAbsFloat32Scalar(x)
}

func sumSquaresF32AVX(x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		v0, v1, v2, v3 := x[i], x[i+1], x[i+2], x[i+3]
		v4, v5, v6, v7 := x[i+4], x[i+5], x[i+6], x[i+7]
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
		ss += x[i] * x[i]
	}
	return ss
}


func maxF32AVX(x []float32) float32 {
	var m0, m1, m2, m3, m4, m5, m6, m7 float32
	m := float32(-1) / 0
	m0, m1, m2, m3, m4, m5, m6, m7 = m, m, m, m, m, m, m, m
	i := 0
	n := len(x)
	for ; i+15 < n; i += 16 {
		m0 = max32(m0, x[i])
		m1 = max32(m1, x[i+1])
		m2 = max32(m2, x[i+2])
		m3 = max32(m3, x[i+3])
		m4 = max32(m4, x[i+4])
		m5 = max32(m5, x[i+5])
		m6 = max32(m6, x[i+6])
		m7 = max32(m7, x[i+7])
		m0 = max32(m0, x[i+8])
		m1 = max32(m1, x[i+9])
		m2 = max32(m2, x[i+10])
		m3 = max32(m3, x[i+11])
		m4 = max32(m4, x[i+12])
		m5 = max32(m5, x[i+13])
		m6 = max32(m6, x[i+14])
		m7 = max32(m7, x[i+15])
	}
	for ; i+7 < n; i += 8 {
		m0 = max32(m0, x[i])
		m1 = max32(m1, x[i+1])
		m2 = max32(m2, x[i+2])
		m3 = max32(m3, x[i+3])
		m4 = max32(m4, x[i+4])
		m5 = max32(m5, x[i+5])
		m6 = max32(m6, x[i+6])
		m7 = max32(m7, x[i+7])
	}
	m = max32(max32(m0, m1), max32(m2, m3))
	m = max32(m, max32(max32(m4, m5), max32(m6, m7)))
	for ; i < n; i++ {
		m = max32(m, x[i])
	}
	return m
}
func dotF32AVX(a, b []float32) float32 {
	return dotF32Scalar(a, b)
}

func dotF32PairAVX(a, b, x []float32) (ret0, ret1 float32) {
	return dotF32PairScalar(a, b, x)
}

func dotF32TripletAVX(a, b, c, x []float32) (ret0, ret1, ret2 float32) {
	return dotF32TripletScalar(a, b, c, x)
}

func dotQ8AVX2(a []int8, b []float32) float32 {
	return dotQ8Scalar(a, b)
}

func dotQ8PairAVX2(a, b []int8, x []float32) (ret0, ret1 float32) {
	return dotQ8PairScalar(a, b, x)
}

func dotQ8TripletAVX2(a, b, c []int8, x []float32) (ret0, ret1, ret2 float32) {
	return dotQ8TripletScalar(a, b, c, x)
}

func dotQ4AVX2(a []byte, b []float32, cols int) float32 {
	return dotQ4Scalar(a, b, cols)
}

func dotQ4PairAVX2(a, b []byte, x []float32, cols int) (ret0, ret1 float32) {
	return dotQ4PairScalar(a, b, x, cols)
}

func dotQ4TripletAVX2(a, b, c []byte, x []float32, cols int) (ret0, ret1, ret2 float32) {
	return dotQ4TripletScalar(a, b, c, x, cols)
}

func dotQ6AVX2(a []byte, b []float32, cols int) float32 {
	return 0
}

func dotQ6AVX2Body(a []byte, b []float32, cols int) float32 {
	return 0
}
func dotF32Scalar(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(a)
	for ; i+15 < n; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func dotQ8Scalar(a []int8, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(a)
	for ; i+15 < n; i += 16 {
		s0 += q8Value(a[i])*b[i] + q8Value(a[i+8])*b[i+8]
		s1 += q8Value(a[i+1])*b[i+1] + q8Value(a[i+9])*b[i+9]
		s2 += q8Value(a[i+2])*b[i+2] + q8Value(a[i+10])*b[i+10]
		s3 += q8Value(a[i+3])*b[i+3] + q8Value(a[i+11])*b[i+11]
		s4 += q8Value(a[i+4])*b[i+4] + q8Value(a[i+12])*b[i+12]
		s5 += q8Value(a[i+5])*b[i+5] + q8Value(a[i+13])*b[i+13]
		s6 += q8Value(a[i+6])*b[i+6] + q8Value(a[i+14])*b[i+14]
		s7 += q8Value(a[i+7])*b[i+7] + q8Value(a[i+15])*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += q8Value(a[i]) * b[i]
	}
	return sum
}

func dotQ4Scalar(a []byte, b []float32, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	packed := (cols + 1) / 2
	for ; i+15 < cols; i += 16 {
		s0 += q4Low(a[i/2])*b[i] + q4High(a[(i+8)/2])*b[i+8]
		s1 += q4Low(a[(i+1)/2])*b[i+1] + q4High(a[(i+9)/2])*b[i+9]
		s2 += q4Low(a[(i+2)/2])*b[i+2] + q4High(a[(i+10)/2])*b[i+10]
		s3 += q4Low(a[(i+3)/2])*b[i+3] + q4High(a[(i+11)/2])*b[i+11]
		s4 += q4Low(a[(i+4)/2])*b[i+4] + q4High(a[(i+12)/2])*b[i+12]
		s5 += q4Low(a[(i+5)/2])*b[i+5] + q4High(a[(i+13)/2])*b[i+13]
		s6 += q4Low(a[(i+6)/2])*b[i+6] + q4High(a[(i+14)/2])*b[i+14]
		s7 += q4Low(a[(i+7)/2])*b[i+7] + q4High(a[(i+15)/2])*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < cols; i++ {
		sum += q4Val(a, i, packed) * b[i]
	}
	return sum
}

func dotQ8PairScalar(a, b []int8, x []float32) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q8Value(a[i])*x0 + q8Value(a[i+4])*x4
		a1 += q8Value(a[i+1])*x1 + q8Value(a[i+5])*x5
		a2 += q8Value(a[i+2])*x2 + q8Value(a[i+6])*x6
		a3 += q8Value(a[i+3])*x3 + q8Value(a[i+7])*x7
		b0 += q8Value(b[i])*x0 + q8Value(b[i+4])*x4
		b1 += q8Value(b[i+1])*x1 + q8Value(b[i+5])*x5
		b2 += q8Value(b[i+2])*x2 + q8Value(b[i+6])*x6
		b3 += q8Value(b[i+3])*x3 + q8Value(b[i+7])*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	for ; i < n; i++ {
		xi := x[i]
		sa += q8Value(a[i]) * xi
		sb += q8Value(b[i]) * xi
	}
	return sa, sb
}

func dotQ8TripletScalar(a, b, c []int8, x []float32) (float32, float32, float32) {
	return dotQ8PairScalar(a, b, x)
}

func dotQ4PairScalar(a, b []byte, x []float32, cols int) (float32, float32) {
	return dotQ4Scalar(a, x, cols), dotQ4Scalar(b, x, cols)
}

func dotQ4TripletScalar(a, b, c []byte, x []float32, cols int) (float32, float32, float32) {
	return dotQ4Scalar(a, x, cols), dotQ4Scalar(b, x, cols), dotQ4Scalar(c, x, cols)
}

func dotF32PairScalar(a, b, x []float32) (float32, float32) {
	return dotF32Scalar(a, x), dotF32Scalar(b, x)
}

func dotF32TripletScalar(a, b, c, x []float32) (float32, float32, float32) {
	return dotF32Scalar(a, x), dotF32Scalar(b, x), dotF32Scalar(c, x)
}


func weightedSum2AVX(dst, x0, x1 []float32, a0, a1 float32) {}
func weightedSum3AVX(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {}
func weightedSum4AVX(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {}
func weightedSumAdd2AVX(dst, x0, x1 []float32, a0, a1 float32) {}
func weightedSumAdd4AVX(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {}
func scaleCopyAVX(dst, x []float32, a float32) {}
func scaleAddAVX(dst, x []float32, a float32) {}

func expF32VecAVX(x []float32, m float32) float32 {
	return 0
}

func ropeAVX(x []float32, cosTable, sinTable []float32, heads, dim int) {}

func geluTanhAVX(x []float32) {
	for i := range x {
		x[i] = GELUTanh(x[i])
	}
}

func siluMulInPlaceAVX(gate, up []float32) {
	n := min(len(gate), len(up))
	for i := 0; i < n; i++ {
		gate[i] = SiLU(gate[i]) * up[i]
	}
}


func dotF32FMA(a, b []float32) float32 { return dotF32Scalar(a, b) }
func dotF32PairFMA(a, b, x []float32) (float32, float32) { return dotF32PairScalar(a, b, x) }
func dotF32TripletFMA(a, b, c, x []float32) (float32, float32, float32) { return dotF32TripletScalar(a, b, c, x) }
func dotF32QuadFMA(a, b, c, d, x []float32) (float32, float32, float32, float32) { return dotF32QuadScalar(a, b, c, d, x) }
func dotQ8FMA(a []int8, b []float32) float32 { return dotQ8Scalar(a, b) }
func dotQ8PairFMA(a, b []int8, x []float32) (float32, float32) { return dotQ8PairScalar(a, b, x) }
func dotQ8TripletFMA(a, b, c []int8, x []float32) (float32, float32, float32) { return dotQ8TripletScalar(a, b, c, x) }


func siluMulInPlaceFMA(gate, up []float32) { siluMulInPlaceAVX(gate, up) }
func geluTanhFMA(x []float32) { geluTanhAVX(x) }


func quantizeQ8RowAVX2(w []float32, data []int8, inv float32) {
	for i, v := range w {
		data[i] = quantInt8(v * inv)
	}
}


func quantizeQ4RowAVX2(w []float32, data []byte, inv float32) {
	for i := 0; i+1 < len(w); i += 2 {
		data[i/2] = quantNibble4(w[i]*inv) | (quantNibble4(w[i+1]*inv) << 4)
	}
	if len(w)%2 == 1 {
		data[len(w)/2] = quantNibble4(w[len(w)-1] * inv)
	}
}


func quantizeQ6RowAVX2(w []float32, data []byte, inv float32) {}


func expF32VecFMA(x []float32, m float32) float32 { return expF32VecAVX(x, m) }


func addInPlaceSumSquaresFMA(dst, x []float32) float32 { return addInPlaceSumSquaresAVX(dst, x) }


func weightedSum2FMA(dst, x0, x1 []float32, a0, a1 float32) { weightedSum2AVX(dst, x0, x1, a0, a1) }
func weightedSum3FMA(dst, x0, x1, x2 []float32, a0, a1, a2 float32) { weightedSum3AVX(dst, x0, x1, x2, a0, a1, a2) }
func weightedSum4FMA(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) { weightedSum4AVX(dst, x0, x1, x2, x3, a0, a1, a2, a3) }
func weightedSumAdd4FMA(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) { weightedSumAdd4AVX(dst, x0, x1, x2, x3, a0, a1, a2, a3) }


func scaleAddFMA(dst, x []float32, a float32) { scaleAddAVX(dst, x, a) }
func weightedSumAdd2FMA(dst, x0, x1 []float32, a0, a1 float32) { weightedSumAdd2AVX(dst, x0, x1, a0, a1) }


func maxF32AVX2(x []float32) float32 { return maxF32AVX(x) }


func sumSquaresF32FMA(x []float32) float32 { return sumSquaresF32AVX(x) }

func maxAbsFloat32AVX2(x []float32) float32 { return maxAbsFloat32AVX(x) }
