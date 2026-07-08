//go:build amd64

package tensor

import "golang.org/x/sys/cpu"

var useDotF32AVX = cpu.X86.HasAVX
var useDotQ8AVX2 = cpu.X86.HasAVX2
var useDotQ4AVX2 = cpu.X86.HasAVX2
var useDotQ6AVX2 = false // Q6 AVX2 is slower than Go scalar+LUT
var useDotFMA = cpu.X86.HasFMA

func addAVX(out, a, b []float32)
func addInPlaceAVX(dst, x []float32)
func addInPlaceSumAVX(dst, x []float32) float32
func addInPlaceSumSquaresAVX(dst, x []float32) float32
func addSumSquaresAVX(out, dst, add []float32) float32
func addScaledAVX(dst, x []float32, scale float32)
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
func geluTanhAVX(x []float32)
func siluMulInPlaceAVX(gate, up []float32)

func dotF32FMA(a, b []float32) float32
func dotF32PairFMA(a, b, x []float32) (ret0, ret1 float32)
func dotF32TripletFMA(a, b, c, x []float32) (ret0, ret1, ret2 float32)
func dotQ8FMA(a []int8, b []float32) float32
func dotQ8PairFMA(a, b []int8, x []float32) (ret0, ret1 float32)
func dotQ8TripletFMA(a, b, c []int8, x []float32) (ret0, ret1, ret2 float32)

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
