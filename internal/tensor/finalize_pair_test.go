package tensor

import (
	"testing"
)

func TestFinalizeDotQ8PairVNNI(t *testing.T) {
	n := 16
	dotsA := make([]int32, n)
	rowSumA := make([]int32, n)
	scaleA := make([]float32, n)
	outA := make([]float32, n)
	dotsB := make([]int32, n)
	rowSumB := make([]int32, n)
	scaleB := make([]float32, n)
	outB := make([]float32, n)
	scaleX := float32(0.5)

	for i := 0; i < n; i++ {
		dotsA[i] = int32(i*100 + 128)
		rowSumA[i] = int32(i)
		scaleA[i] = float32(i+1) * 0.1
		dotsB[i] = int32(i*200 + 256)
		rowSumB[i] = int32(i*2)
		scaleB[i] = float32(i+1) * 0.2
	}

	finalizeDotQ8PairVNNI(&dotsA[0], &rowSumA[0], &scaleA[0], &outA[0],
		&dotsB[0], &rowSumB[0], &scaleB[0], &outB[0], n, scaleX)

	for i := 0; i < n; i++ {
		expA := float32(dotsA[i]-128*rowSumA[i]) * scaleX * scaleA[i]
		expB := float32(dotsB[i]-128*rowSumB[i]) * scaleX * scaleB[i]
		if outA[i]-expA > 0.001 || outA[i]-expA < -0.001 {
			t.Errorf("A[%d]: got %f expected %f", i, outA[i], expA)
		}
		if outB[i]-expB > 0.001 || outB[i]-expB < -0.001 {
			t.Errorf("B[%d]: got %f expected %f", i, outB[i], expB)
		}
	}
}

func TestFinalizeDotQ8PairVNNIVarSizes(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 9, 15, 16, 17, 32, 33, 100} {
		dotsA := make([]int32, n)
		rowSumA := make([]int32, n)
		scaleA := make([]float32, n)
		outA := make([]float32, n)
		dotsB := make([]int32, n)
		rowSumB := make([]int32, n)
		scaleB := make([]float32, n)
		outB := make([]float32, n)
		scaleX := float32(0.007)
		for i := 0; i < n; i++ {
			dotsA[i] = int32(i*37 - 128*(i%5))
			rowSumA[i] = int32(i % 5)
			scaleA[i] = float32(i+1) * 0.003
			dotsB[i] = int32(i*73 - 128*(i%7))
			rowSumB[i] = int32(i % 7)
			scaleB[i] = float32(i+1) * 0.005
		}
		finalizeDotQ8PairVNNI(&dotsA[0], &rowSumA[0], &scaleA[0], &outA[0],
			&dotsB[0], &rowSumB[0], &scaleB[0], &outB[0], n, scaleX)
		for i := 0; i < n; i++ {
			expA := float32(dotsA[i]-128*rowSumA[i]) * scaleX * scaleA[i]
			expB := float32(dotsB[i]-128*rowSumB[i]) * scaleX * scaleB[i]
			if outA[i]-expA > 0.001 || outA[i]-expA < -0.001 {
				t.Errorf("n=%d A[%d]: got %f expected %f", n, i, outA[i], expA)
			}
			if outB[i]-expB > 0.001 || outB[i]-expB < -0.001 {
				t.Errorf("n=%d B[%d]: got %f expected %f", n, i, outB[i], expB)
			}
		}
	}
}
