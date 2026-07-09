with open("internal/tensor/vnni_debug_test.go", "r", encoding="utf-8") as f:
    c = f.read()
c += """
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
"""
with open("internal/tensor/vnni_debug_test.go", "w", encoding="utf-8") as f:
    f.write(c)
print("done")
