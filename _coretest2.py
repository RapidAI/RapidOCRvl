with open("internal/tensor/vnni_debug_test.go", "r", encoding="utf-8") as f:
    c = f.read()
c += """
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
"""
with open("internal/tensor/vnni_debug_test.go", "w", encoding="utf-8") as f:
    f.write(c)
print("done")
