package tensor

import "testing"

func TestVNNIDebug(t *testing.T) {
	n := 32
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
	t.Logf("q.Data[:8] = %v", q.Data[:8])
	t.Logf("x[:8] = %v", x[:8])
	t.Logf("xq[:8] = %v", xq[:8])
	t.Logf("scaleX = %f", scaleX)
	t.Logf("q.Scale[0] = %f", q.Scale[0])
	t.Logf("q.RowSum[0] = %d", q.RowSum[0])
	vnniResult := dotQ8VNNI(q.Data[:n], xq, scaleX, q.Scale[0], q.RowSum[0])
	refResult := dotQ8(q.Data[:n], x)
	t.Logf("VNNI = %f", vnniResult)
	t.Logf("ref  = %f", refResult)
	t.Logf("ratio = %f", vnniResult/refResult)
}

func TestVNNICore32(t *testing.T) {
	// 32 elements: a=[1]*32, xq=[129]*32
	// Expected: 32 * 1 * 129 = 4128
	a := make([]int8, 32)
	xq := make([]uint8, 32)
	for i := range a {
		a[i] = 1
		xq[i] = 129
	}
	result := dotQ8VNNICore(&a[0], &xq[0], 32)
	t.Logf("core32 = %d (expected 4128)", result)
}

func TestVNNICoreMixed(t *testing.T) {
	// a = [-127, -111, -95, -79, -64, -48, -32, -16] (8 elements, matching q.Data[:8])
	// xq = [2, 23, 44, 65, 86, 107, 128, 149] (matching test xq[:8])
	// Expected raw dot = sum(a[i]*xq[i]) = -127*2 + -111*23 + -95*44 + -79*65 + -64*86 + -48*107 + -32*128 + -16*149
	// = -254 - 2553 - 4180 - 5135 - 5504 - 5136 - 4096 - 2384 = -29242
	a := []int8{-127, -111, -95, -79, -64, -48, -32, -16}
	xq := []uint8{2, 23, 44, 65, 86, 107, 128, 149}
	result := dotQ8VNNICore(&a[0], &xq[0], 8)
	t.Logf("coreMixed = %d (expected -29242)", result)
}

func TestVNNIRowSum(t *testing.T) {
	n := 32
	a := make([]int8, n)
	for i := range a {
		a[i] = int8(i%17 - 8)
	}
	w := make([]float32, n)
	for i, v := range a {
		w[i] = float32(v)
	}
	q := QuantizeQ8Row(w, 1, n)
	// Compute expected rowSum
	var expected int32
	for i := 0; i < n; i++ {
		expected += int32(q.Data[i])
	}
	t.Logf("q.RowSum[0] = %d, expected = %d", q.RowSum[0], expected)
	// Also compute raw dot
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	xq := make([]uint8, n)
	scaleX := quantizeXForVNNI(x, xq)
	rawDot := dotQ8VNNICore(&q.Data[0], &xq[0], n)
	t.Logf("rawDot = %d", rawDot)
	t.Logf("128*rowSum = %d", 128*q.RowSum[0])
	t.Logf("rawDot - 128*rowSum = %d", rawDot-128*q.RowSum[0])
	t.Logf("scaleX = %f", scaleX)
	t.Logf("(rawDot-128*rowSum)*scaleX = %f", float32(rawDot-128*q.RowSum[0])*scaleX)
	// ref
	refDot := dotQ8(q.Data[:n], x)
	t.Logf("refDot = %f", refDot)
	t.Logf("refDot * q.Scale[0] = %f", refDot * q.Scale[0])
}
