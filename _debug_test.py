content = """package tensor

import (
	"math"
	"testing"
)

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
"""
with open("internal/tensor/vnni_debug_test.go", "w", encoding="utf-8") as f:
    f.write(content)
print("done")
