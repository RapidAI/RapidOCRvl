package tensor

import (
	"math"
	"testing"
)

func TestDotQ8VNNICoreDebug(t *testing.T) {
	a := []int8{1, 1, 1, 1, 1, 1, 1, 1}
	xq := []uint8{129, 129, 129, 129, 129, 129, 129, 129}
	result := dotQ8VNNICore(&a[0], &xq[0], 8)
	if result != 1032 {
		t.Errorf("core = %d (expected 1032)", result)
	}
}

func TestDotQ8VNNICoreTail(t *testing.T) {
	a := []int8{1, 1, 1}
	xq := []uint8{129, 129, 129}
	result := dotQ8VNNICore(&a[0], &xq[0], 3)
	if result != 387 {
		t.Errorf("tail n=3: core = %d (expected 387)", result)
	}
}

func TestDotQ8VNNICorrectness(t *testing.T) {
	for _, n := range []int{32, 64, 128, 256, 1024} {
		a := make([]int8, n)
		x := make([]float32, n)
		for i := range a {
			a[i] = int8(i%17 - 8)
			x[i] = float32(i%13-6) / 13
		}
		w := make([]float32, n)
		for i, v := range a {
			w[i] = float32(v)
		}
		q := QuantizeQ8Row(w, 1, n)
		xq := make([]uint8, n)
		scaleX := quantizeXForVNNI(x, xq)
		vnniResult := dotQ8VNNI(q.Data[:n], xq, scaleX, q.Scale[0], q.RowSum[0])
		refResult := dotQ8(q.Data[:n], x)
		diff := math.Abs(float64(vnniResult - refResult))
		relTol := math.Abs(float64(refResult)) * 0.02
		if diff > math.Max(relTol, 0.5) {
			t.Errorf("n=%d: VNNI=%f ref=%f diff=%f", n, vnniResult, refResult, diff)
		}
	}
}
