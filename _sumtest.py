with open("internal/tensor/vnni_debug_test.go", "r", encoding="utf-8") as f:
    c = f.read()
c += """
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
"""
with open("internal/tensor/vnni_debug_test.go", "w", encoding="utf-8") as f:
    f.write(c)
print("done")
