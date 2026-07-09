//go:build amd64

package tensor

import "golang.org/x/sys/cpu"

var useDotF32AVX = cpu.X86.HasAVX
var useDotQ8AVX2 = cpu.X86.HasAVX2
var useDotQ4AVX2 = cpu.X86.HasAVX2
var useDotQ6AVX2 = false // Q6 AVX2 is slower than Go scalar+LUT
var useDotFMA = cpu.X86.HasFMA
var useVNNI = cpu.X86.HasAVX512VNNI // VPDPBUSD via EVEX encoding

func addAVX(out, a, b []float32)
func addInPlaceAVX(dst, x []float32)
func addInPlaceSumAVX(dst, x []float32) float32
func addInPlaceSumSquaresAVX(dst, x []float32) float32
func addSumSquaresAVX(out, dst, add []float32) float32
func addSumSquaresFMA(out, dst, add []float32) float32
func addScaledAVX(dst, x []float32, scale float32)
func addScaledFMA(dst, x []float32, scale float32)
func scaleInPlaceAVX(dst []float32, scale float32)
func mulScaleAVX(out, x, weight []float32, scale float32)
func affineNormAVX(out, x, weight, bias []float32, mean, scale float32)
func sumF32AVX(x []float32) float32
func sumSquaresCenteredAVX(x []float32, mean float32) float32
func maxAbsFloat32AVX(x []float32) float32
func sumSquaresF32AVX(x []float32) float32
func maxF32AVX(x []float32) float32
func dotF32AVX(a, b []float32) float32
func dotF32PairAVX(a, b, x []float32) (ret0, ret1 float32)
func dotF32TripletAVX(a, b, c, x []float32) (ret0, ret1, ret2 float32)
func dotQ8AVX2(a []int8, b []float32) float32
func dotQ8PairAVX2(a, b []int8, x []float32) (ret0, ret1 float32)
func dotQ8TripletAVX2(a, b, c []int8, x []float32) (ret0, ret1, ret2 float32)
func dotQ4AVX2(a []byte, b []float32, cols int) float32
func dotQ4PairAVX2(a, b []byte, x []float32, cols int) (ret0, ret1 float32)
func dotQ4TripletAVX2(a, b, c []byte, x []float32, cols int) (ret0, ret1, ret2 float32)
func dotQ6AVX2(a []byte, b []float32, cols int) float32

func weightedSum2AVX(dst, x0, x1 []float32, a0, a1 float32)
func weightedSum3AVX(dst, x0, x1, x2 []float32, a0, a1, a2 float32)
func weightedSum4AVX(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32)
func weightedSumAdd2AVX(dst, x0, x1 []float32, a0, a1 float32)
func weightedSumAdd4AVX(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32)
func scaleCopyAVX(dst, x []float32, a float32)
func scaleAddAVX(dst, x []float32, a float32)
func expF32VecAVX(x []float32, m float32) float32
func ropeAVX(x []float32, cosTable, sinTable []float32, heads, dim int)
func ropeFMA(x []float32, cosTable, sinTable []float32, heads, dim int)
func ropePairAxisFMA(q, k []float32, start, half int, cosTable, sinTable []float32)
func ropePairAxisAVX(q, k []float32, start, half int, cosTable, sinTable []float32)
func geluTanhAVX(x []float32)
func siluMulInPlaceAVX(gate, up []float32)

func dotF32FMA(a, b []float32) float32
func dotF32PairFMA(a, b, x []float32) (ret0, ret1 float32)
func dotF32TripletFMA(a, b, c, x []float32) (ret0, ret1, ret2 float32)
func dotF32QuadFMA(a, b, c, d, x []float32) (ret0, ret1, ret2, ret3 float32)
func dotQ8FMA(a []int8, b []float32) float32
func dotQ8PairFMA(a, b []int8, x []float32) (ret0, ret1 float32)
func dotQ8TripletFMA(a, b, c []int8, x []float32) (ret0, ret1, ret2 float32)
func dotQ4FMA(a []byte, b []float32, cols int) float32
func dotQ4PairFMA(a, b []byte, x []float32, cols int) (ret0, ret1 float32)
func dotQ4TripletFMA(a, b, c []byte, x []float32, cols int) (ret0, ret1, ret2 float32)
func siluMulInPlaceFMA(gate, up []float32)
func geluTanhFMA(x []float32)
func quantizeQ8RowAVX2(w []float32, data []int8, inv float32)

func quantizeQ4RowAVX2(w []float32, data []byte, inv float32)

func quantizeQ6RowAVX2(w []float32, data []byte, inv float32)

func expF32VecFMA(x []float32, m float32) float32

func addInPlaceSumSquaresFMA(dst, x []float32) float32

func weightedSum2FMA(dst, x0, x1 []float32, a0, a1 float32)
func weightedSum3FMA(dst, x0, x1, x2 []float32, a0, a1, a2 float32)
func weightedSum4FMA(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32)
func weightedSumAdd4FMA(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32)

func scaleAddFMA(dst, x []float32, a float32)
func weightedSumAdd2FMA(dst, x0, x1 []float32, a0, a1 float32)

func maxF32AVX2(x []float32) float32

// argmaxF32AVX2 returns (index, value) of the maximum element in x using a single-pass SIMD scan.
func argmaxF32AVX2(x []float32) (int, float32)

// expScaledVecFMA computes out[i] = exp(x[i]*scale + bias) and returns the sum of all out values.
// Fuses the scale*bias multiply-add into the exp polynomial, avoiding a separate pass.
func expScaledVecFMA(out, x []float32, scale, bias float32) float32

func sumSquaresF32FMA(x []float32) float32

func mulScaleFMA(out, x, weight []float32, scale float32)
func affineNormFMA(out, x, weight, bias []float32, mean, scale float32)
func sumSquaresCenteredFMA(x []float32, mean float32) float32
func addInPlaceSumFMA(dst, x []float32) float32
func sumF32FMA(x []float32) float32

func maxAbsFloat32AVX2(x []float32) float32

// Assembly helper declarations for VNNI kernels (provided by dot_vnni_amd64.s).
func dotQ8VNNICore(a *int8, xq *uint8, n int) int32
func dotQ8VNNICoreZMM(a *int8, xq *uint8, n int) int32
func dotQ8PairVNNICore(a *int8, b *int8, xq *uint8, n int) (int32, int32)
func dotQ8TripletVNNICore(a *int8, b *int8, c *int8, xq *uint8, n int) (int32, int32, int32)
func dotQ8VNNICoreMultiRowZMM(a *int8, xq *uint8, out *int32, rows int, cols int)
func dotQ8PairVNNICoreMultiRowZMM(a *int8, b *int8, xq *uint8, outA *int32, outB *int32, rows int, cols int)
func dotQ8TripletVNNICoreMultiRowZMM(a *int8, b *int8, c *int8, xq *uint8, outA *int32, outB *int32, outC *int32, rows int, cols int)
//go:noescape
func finalizeDotQ8VNNI(dots *int32, rowSum *int32, scale *float32, out *float32, n int, scaleX float32)
//go:noescape
func finalizeDotQ8PairVNNI(dotsA *int32, rowSumA *int32, scaleA *float32, outA *float32, dotsB *int32, rowSumB *int32, scaleB *float32, outB *float32, n int, scaleX float32)


//go:noescape
func finalizeDotQ8BiasVNNI(dots *int32, rowSum *int32, scale *float32, out *float32, bias *float32, n int, scaleX float32)

//go:noescape
func finalizeDotQ8TripletVNNI(dotsA *int32, rowSumA *int32, scaleA *float32, outA *float32, dotsB *int32, rowSumB *int32, scaleB *float32, outB *float32, dotsC *int32, rowSumC *int32, scaleC *float32, outC *float32, n int, scaleX float32)


//go:noescape
func addBias3FMA(outA *float32, biasA *float32, outB *float32, biasB *float32, outC *float32, biasC *float32, n int)
//go:noescape
func finalizeAddSumSquaresInPlaceVNNI(dots *int32, rowSum *int32, scale *float32, out *float32, residual *float32, n int, scaleX float32) float32

//go:noescape
func finalizeAddSumSquaresOutOnlyVNNI(dots *int32, rowSum *int32, scale *float32, out *float32, residual *float32, n int, scaleX float32) float32

//go:noescape
func rowSumQ8Asm(a *int8, n int) int32
func quantizeXForVNNIAsm(x_base *float32, xq_base *uint8, n int, inv float32)

// VNNI-accelerated Q8 dot product using VPDPBUSD.
// dotQ8VNNI scalar fallback (VNNI not available in assembly).
func dotQ8VNNI(a []int8, xq []uint8, scaleX, scaleW float32, rowSumW int32) float32 {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumW) * scaleX * scaleW
	}
	var dot int32
	if useVNNI {
		dot = dotQ8VNNICoreZMM(&a[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dot += int32(a[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dot-128*rowSumW) * scaleX * scaleW
}

func dotQ8PairVNNI(a, b []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB int32, scaleWA, scaleWB float32) (float32, float32) {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumWA) * scaleX * scaleWA, float32(-128*rowSumWB) * scaleX * scaleWB
	}
	var dotA, dotB int32
	if useVNNI {
		dotA, dotB = dotQ8PairVNNICore(&a[0], &b[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dotA += int32(a[i]) * (int32(xq[i]) - 128)
			dotB += int32(b[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA, float32(dotB-128*rowSumWB) * scaleX * scaleWB
}

func dotQ8TripletVNNI(a, b, c []int8, xq []uint8, scaleX float32, rowSumWA, rowSumWB, rowSumWC int32, scaleWA, scaleWB, scaleWC float32) (float32, float32, float32) {
	n := len(a)
	if len(xq) < n {
		n = len(xq)
	}
	if n == 0 {
		return float32(-128*rowSumWA) * scaleX * scaleWA,
			float32(-128*rowSumWB) * scaleX * scaleWB,
			float32(-128*rowSumWC) * scaleX * scaleWC
	}
	var dotA, dotB, dotC int32
	if useVNNI {
		dotA, dotB, dotC = dotQ8TripletVNNICore(&a[0], &b[0], &c[0], &xq[0], n)
	} else {
		for i := 0; i < n; i++ {
			dotA += int32(a[i]) * (int32(xq[i]) - 128)
			dotB += int32(b[i]) * (int32(xq[i]) - 128)
			dotC += int32(c[i]) * (int32(xq[i]) - 128)
		}
	}
	return float32(dotA-128*rowSumWA) * scaleX * scaleWA,
		float32(dotB-128*rowSumWB) * scaleX * scaleWB,
		float32(dotC-128*rowSumWC) * scaleX * scaleWC
}

func rowSumQ8AVX2(a []int8) int32 {
	if len(a) == 0 {
		return 0
	}
	if useVNNI {
		return rowSumQ8Asm(&a[0], len(a))
	}
	var s int32
	for _, v := range a {
		s += int32(v)
	}
	return s
}

func quantizeXForVNNIAVX2(x []float32, xq []uint8) float32 {
	maxAbs := maxAbsFloat32(x)
	if maxAbs == 0 {
		for i := range xq[:len(x)] {
			xq[i] = 128
		}
		return 1
	}
	scale := maxAbs / 127
	inv := 1 / scale
	if useDotQ8AVX2 && len(x) >= 8 && len(xq) >= len(x) {
		quantizeXForVNNIAsm(&x[0], &xq[0], len(x), inv)
		return scale
	}
	for i, v := range x {
		q := int(v*inv) + 128
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		xq[i] = uint8(q)
	}
	return scale
}