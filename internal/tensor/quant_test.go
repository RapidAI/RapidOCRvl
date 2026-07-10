package tensor

import (
	"math"
	"runtime"
	"testing"
)

func TestQ8MatVecApprox(t *testing.T) {
	w := []float32{
		1, -2, 3,
		0.5, 0.25, -0.75,
	}
	x := []float32{2, -1, 0.5}
	want := make([]float32, 2)
	MatVec(want, x, w, 2, 3)
	q := QuantizeQ8Row(w, 2, 3)
	got := make([]float32, 2)
	MatVecQ8(got, x, q)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 0.05 {
			t.Fatalf("row %d got %f want %f", i, got[i], want[i])
		}
	}
}

func TestQuantizeQ8RowBytesIntoMatchesInt8(t *testing.T) {
	w := []float32{1, -2, 3, 0.5, 0.25, -0.75, 0}
	asInt := make([]int8, len(w))
	asBytes := make([]byte, len(w))
	s1 := QuantizeQ8RowInto(w, asInt)
	s2 := QuantizeQ8RowBytesInto(w, asBytes)
	if s1 != s2 {
		t.Fatalf("scale %f != %f", s1, s2)
	}
	for i := range asInt {
		if byte(asInt[i]) != asBytes[i] {
			t.Fatalf("q[%d]=%d byte=%d", i, asInt[i], asBytes[i])
		}
	}
}

func TestQ4MatVecApprox(t *testing.T) {
	w := []float32{
		1, -2, 3,
		0.5, 0.25, -0.75,
	}
	x := []float32{2, -1, 0.5}
	want := make([]float32, 2)
	MatVec(want, x, w, 2, 3)
	q := QuantizeQ4Row(w, 2, 3)
	got := make([]float32, 2)
	MatVecQ4(got, x, q)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 0.5 {
			t.Fatalf("row %d got %f want %f", i, got[i], want[i])
		}
	}
}

func TestQuantizeQ4RowIntoOddColsClearsTailNibble(t *testing.T) {
	data := []byte{0xFF, 0xFF}
	_ = QuantizeQ4RowInto([]float32{1, -1, 0.5}, data)
	if data[1]&0xF0 != 0 {
		t.Fatalf("tail high nibble not cleared: %#02x", data[1])
	}
}

func TestQ6MatVecApprox(t *testing.T) {
	w := []float32{
		1, -2, 3,
		0.5, 0.25, -0.75,
	}
	x := []float32{2, -1, 0.5}
	want := make([]float32, 2)
	MatVec(want, x, w, 2, 3)
	q := QuantizeQ6Row(w, 2, 3)
	got := make([]float32, 2)
	MatVecQ6(got, x, q)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 0.12 {
			t.Fatalf("row %d got %f want %f", i, got[i], want[i])
		}
	}
}

func TestQ6PackRoundTripOddCols(t *testing.T) {
	row := make([]byte, PackedQ6Cols(11))
	for i := 0; i < 11; i++ {
		putQ6(row, i, byte((i*7)&0x3F))
	}
	for i := 0; i < 11; i++ {
		want := byte((i * 7) & 0x3F)
		if got := getQ6(row, i); got != want {
			t.Fatalf("col %d got %d want %d", i, got, want)
		}
	}
}

func TestQuantHelpers(t *testing.T) {
	if got := maxAbsFloat32([]float32{-1, 2.5, -3, 0.25, 1, -0.5, 7, -6, 4}); got != 7 {
		t.Fatalf("maxAbs=%f", got)
	}
	cases := map[float32]int{
		1.49:  1,
		1.50:  2,
		-1.49: -1,
		-1.50: -2,
		0:     0,
	}
	for in, want := range cases {
		if got := roundToInt(in); got != want {
			t.Fatalf("roundToInt(%f)=%d want %d", in, got, want)
		}
	}
	if q4Value(0) != -8 || q4Value(8) != 0 || q4Value(15) != 7 {
		t.Fatalf("bad q4Value")
	}
	if q4ByteLo[0xF0] != -8 || q4ByteHi[0xF0] != 7 || q4ByteLo[0x08] != 0 {
		t.Fatalf("bad q4 byte tables")
	}
	if q8Value(-128) != -128 || q8Value(0) != 0 || q8Value(127) != 127 {
		t.Fatalf("bad q8Value")
	}
	if q6Value(0) != -32 || q6Value(32) != 0 || q6Value(63) != 31 {
		t.Fatalf("bad q6Value")
	}
}

func TestParallelForSingleProcUsesSingleRange(t *testing.T) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	calls := 0
	parallelFor(7, func(start, end int) {
		calls++
		if start != 0 || end != 7 {
			t.Fatalf("range=%d:%d", start, end)
		}
	})
	if calls != 1 {
		t.Fatalf("calls=%d want 1", calls)
	}
}

func TestFusedMatVec3MatchesSeparate(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	a := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	b := []float32{
		0.5, 1, -1,
	}
	c := []float32{
		2, -0.25, 0.75,
		-0.5, 1.5, 0.25,
	}
	wantA := make([]float32, 2)
	wantB := make([]float32, 1)
	wantC := make([]float32, 2)
	MatVec(wantA, x, a, 2, 3)
	MatVec(wantB, x, b, 1, 3)
	MatVec(wantC, x, c, 2, 3)

	gotA := make([]float32, 2)
	gotB := make([]float32, 1)
	gotC := make([]float32, 2)
	FusedMatVec3(gotA, gotB, gotC, x, a, b, c, 2, 1, 2, 3)
	assertCloseVec(t, "f32 a", gotA, wantA, 1e-6)
	assertCloseVec(t, "f32 b", gotB, wantB, 1e-6)
	assertCloseVec(t, "f32 c", gotC, wantC, 1e-6)

	gotA = make([]float32, 2)
	gotB = make([]float32, 1)
	gotC = make([]float32, 2)
	qa, qb, qc := QuantizeQ8Row(a, 2, 3), QuantizeQ8Row(b, 1, 3), QuantizeQ8Row(c, 2, 3)
	FusedMatVec3Q8(gotA, gotB, gotC, x, qa, qb, qc)
	assertCloseVec(t, "q8 a", gotA, wantA, 0.05)
	assertCloseVec(t, "q8 b", gotB, wantB, 0.05)
	assertCloseVec(t, "q8 c", gotC, wantC, 0.05)

	gotA = make([]float32, 2)
	gotB = make([]float32, 1)
	gotC = make([]float32, 2)
	q6a, q6b, q6c := QuantizeQ6Row(a, 2, 3), QuantizeQ6Row(b, 1, 3), QuantizeQ6Row(c, 2, 3)
	FusedMatVec3Q6(gotA, gotB, gotC, x, q6a, q6b, q6c)
	assertCloseVec(t, "q6 a", gotA, wantA, 0.12)
	assertCloseVec(t, "q6 b", gotB, wantB, 0.12)
	assertCloseVec(t, "q6 c", gotC, wantC, 0.12)

	gotA = make([]float32, 2)
	gotB = make([]float32, 1)
	gotC = make([]float32, 2)
	q4a, q4b, q4c := QuantizeQ4Row(a, 2, 3), QuantizeQ4Row(b, 1, 3), QuantizeQ4Row(c, 2, 3)
	FusedMatVec3Q4(gotA, gotB, gotC, x, q4a, q4b, q4c)
	assertCloseVec(t, "q4 a", gotA, wantA, 0.5)
	assertCloseVec(t, "q4 b", gotB, wantB, 0.5)
	assertCloseVec(t, "q4 c", gotC, wantC, 0.5)
}

func TestFusedMatVec3EqualRowsMatchesSeparate(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25, 2}
	rows, cols := 3, 4
	a := []float32{
		1, 2, 3, 4,
		-1, 0.5, 2, -0.25,
		0.5, -1.5, 0.75, 1,
	}
	b := []float32{
		0.5, 1, -1, 2,
		2, -0.25, 0.75, 0.5,
		-1, 1.25, 0.5, -0.75,
	}
	c := []float32{
		2, -0.25, 0.75, 1,
		-0.5, 1.5, 0.25, -1,
		1, 0.5, -0.5, 2,
	}
	wantA, wantB, wantC := make([]float32, rows), make([]float32, rows), make([]float32, rows)
	gotA, gotB, gotC := make([]float32, rows), make([]float32, rows), make([]float32, rows)
	MatVec(wantA, x, a, rows, cols)
	MatVec(wantB, x, b, rows, cols)
	MatVec(wantC, x, c, rows, cols)
	FusedMatVec3(gotA, gotB, gotC, x, a, b, c, rows, rows, rows, cols)
	assertCloseVec(t, "equal f32 a", gotA, wantA, 1e-6)
	assertCloseVec(t, "equal f32 b", gotB, wantB, 1e-6)
	assertCloseVec(t, "equal f32 c", gotC, wantC, 1e-6)

	gotA = make([]float32, rows)
	gotB = make([]float32, rows)
	gotC = make([]float32, rows)
	FusedMatVec3Q8(gotA, gotB, gotC, x, QuantizeQ8Row(a, rows, cols), QuantizeQ8Row(b, rows, cols), QuantizeQ8Row(c, rows, cols))
	assertCloseVec(t, "equal q8 a", gotA, wantA, 0.05)
	assertCloseVec(t, "equal q8 b", gotB, wantB, 0.05)
	assertCloseVec(t, "equal q8 c", gotC, wantC, 0.05)

	gotA = make([]float32, rows)
	gotB = make([]float32, rows)
	gotC = make([]float32, rows)
	FusedMatVec3Q6(gotA, gotB, gotC, x, QuantizeQ6Row(a, rows, cols), QuantizeQ6Row(b, rows, cols), QuantizeQ6Row(c, rows, cols))
	assertCloseVec(t, "equal q6 a", gotA, wantA, 0.12)
	assertCloseVec(t, "equal q6 b", gotB, wantB, 0.12)
	assertCloseVec(t, "equal q6 c", gotC, wantC, 0.12)

	gotA = make([]float32, rows)
	gotB = make([]float32, rows)
	gotC = make([]float32, rows)
	FusedMatVec3Q4(gotA, gotB, gotC, x, QuantizeQ4Row(a, rows, cols), QuantizeQ4Row(b, rows, cols), QuantizeQ4Row(c, rows, cols))
	assertCloseVec(t, "equal q4 a", gotA, wantA, 0.5)
	assertCloseVec(t, "equal q4 b", gotB, wantB, 0.5)
	assertCloseVec(t, "equal q4 c", gotC, wantC, 0.5)
}

func TestFusedMatVec3SerialPartialRanges(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	a := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	b := []float32{
		0.5, 1, -1,
	}
	c := []float32{
		2, -0.25, 0.75,
		-0.5, 1.5, 0.25,
	}
	wantA, wantB, wantC := make([]float32, 2), make([]float32, 1), make([]float32, 2)
	FusedMatVec3(wantA, wantB, wantC, x, a, b, c, 2, 1, 2, 3)
	gotA, gotB, gotC := make([]float32, 2), make([]float32, 1), make([]float32, 2)
	fusedMatVec3Serial(gotA, gotB, gotC, x, a, b, c, 2, 1, 3, 1, 4)
	if gotA[0] != 0 {
		t.Fatalf("range touched outA[0]=%f", gotA[0])
	}
	assertCloseVec(t, "partial a", gotA[1:], wantA[1:], 1e-6)
	assertCloseVec(t, "partial b", gotB, wantB, 1e-6)
	assertCloseVec(t, "partial c", gotC[:1], wantC[:1], 1e-6)
	if gotC[1] != 0 {
		t.Fatalf("range touched outC[1]=%f", gotC[1])
	}
}

func TestFusedMatVec3QuantSerialPartialRanges(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	a := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	b := []float32{
		0.5, 1, -1,
	}
	c := []float32{
		2, -0.25, 0.75,
		-0.5, 1.5, 0.25,
	}

	wantA, wantB, wantC := make([]float32, 2), make([]float32, 1), make([]float32, 2)
	qa, qb, qc := QuantizeQ8Row(a, 2, 3), QuantizeQ8Row(b, 1, 3), QuantizeQ8Row(c, 2, 3)
	FusedMatVec3Q8(wantA, wantB, wantC, x, qa, qb, qc)
	gotA, gotB, gotC := make([]float32, 2), make([]float32, 1), make([]float32, 2)
	fusedMatVec3Q8Serial(gotA, gotB, gotC, x, qa, qb, qc, 1, 4)
	if gotA[0] != 0 || gotC[1] != 0 {
		t.Fatalf("q8 range touched outA[0]=%f outC[1]=%f", gotA[0], gotC[1])
	}
	assertCloseVec(t, "q8 partial a", gotA[1:], wantA[1:], 0)
	assertCloseVec(t, "q8 partial b", gotB, wantB, 0)
	assertCloseVec(t, "q8 partial c", gotC[:1], wantC[:1], 0)

	wantA, wantB, wantC = make([]float32, 2), make([]float32, 1), make([]float32, 2)
	q6a, q6b, q6c := QuantizeQ6Row(a, 2, 3), QuantizeQ6Row(b, 1, 3), QuantizeQ6Row(c, 2, 3)
	FusedMatVec3Q6(wantA, wantB, wantC, x, q6a, q6b, q6c)
	gotA, gotB, gotC = make([]float32, 2), make([]float32, 1), make([]float32, 2)
	fusedMatVec3Q6Serial(gotA, gotB, gotC, x, q6a, q6b, q6c, 1, 4)
	if gotA[0] != 0 || gotC[1] != 0 {
		t.Fatalf("q6 range touched outA[0]=%f outC[1]=%f", gotA[0], gotC[1])
	}
	assertCloseVec(t, "q6 partial a", gotA[1:], wantA[1:], 0)
	assertCloseVec(t, "q6 partial b", gotB, wantB, 0)
	assertCloseVec(t, "q6 partial c", gotC[:1], wantC[:1], 0)

	wantA, wantB, wantC = make([]float32, 2), make([]float32, 1), make([]float32, 2)
	q4a, q4b, q4c := QuantizeQ4Row(a, 2, 3), QuantizeQ4Row(b, 1, 3), QuantizeQ4Row(c, 2, 3)
	FusedMatVec3Q4(wantA, wantB, wantC, x, q4a, q4b, q4c)
	gotA, gotB, gotC = make([]float32, 2), make([]float32, 1), make([]float32, 2)
	fusedMatVec3Q4Serial(gotA, gotB, gotC, x, q4a, q4b, q4c, 1, 4)
	if gotA[0] != 0 || gotC[1] != 0 {
		t.Fatalf("q4 range touched outA[0]=%f outC[1]=%f", gotA[0], gotC[1])
	}
	assertCloseVec(t, "q4 partial a", gotA[1:], wantA[1:], 0)
	assertCloseVec(t, "q4 partial b", gotB, wantB, 0)
	assertCloseVec(t, "q4 partial c", gotC[:1], wantC[:1], 0)
}

func TestDotAndAddScaled(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	b := []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}
	if got, want := Dot(a, b), float32(165); got != want {
		t.Fatalf("Dot got %f want %f", got, want)
	}
	dst := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}
	AddScaled(dst, b, 0.5)
	for i := range dst {
		want := 1 + 0.5*b[i]
		if dst[i] != want {
			t.Fatalf("AddScaled[%d] got %f want %f", i, dst[i], want)
		}
	}
	out := make([]float32, len(a))
	Add(out, a, b)
	for i := range out {
		if out[i] != a[i]+b[i] {
			t.Fatalf("Add[%d] got %f want %f", i, out[i], a[i]+b[i])
		}
	}
	inPlace := append([]float32(nil), a...)
	AddInPlace(inPlace, b)
	for i := range inPlace {
		if inPlace[i] != a[i]+b[i] {
			t.Fatalf("AddInPlace[%d] got %f want %f", i, inPlace[i], a[i]+b[i])
		}
	}
	alias := append([]float32(nil), a...)
	Add(alias, alias, b)
	for i := range alias {
		if alias[i] != a[i]+b[i] {
			t.Fatalf("Add alias[%d] got %f want %f", i, alias[i], a[i]+b[i])
		}
	}
}

func assertCloseVec(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] got %f want %f", name, i, got[i], want[i])
		}
	}
}

func TestSoftmaxSingle(t *testing.T) {
	x := []float32{42}
	SoftmaxInPlace(x)
	if x[0] != 1 {
		t.Fatalf("SoftmaxInPlace single got %f want 1", x[0])
	}
}

func TestSoftmaxLen2(t *testing.T) {
	x := []float32{3, 1}
	SoftmaxInPlace(x)
	e := float32(math.Exp(-2))
	want0 := 1 / (1 + e)
	want1 := e / (1 + e)
	if math.Abs(float64(x[0]-want0)) > 1e-6 || math.Abs(float64(x[1]-want1)) > 1e-6 {
		t.Fatalf("SoftmaxInPlace len2 got %v want [%f %f]", x, want0, want1)
	}
}

func TestSoftmaxSmallLengths(t *testing.T) {
	for _, in := range [][]float32{
		{3, 1, -2},
		{1.25, -0.75, 0.5, 2},
		{1.25, -0.75, 0.5, 2, -1},
		{1.25, -0.75, 0.5, 2, -1, 0.25},
		{1.25, -0.75, 0.5, 2, -1, 0.25, 1},
		{1.25, -0.75, 0.5, 2, -1, 0.25, 1, -0.5},
	} {
		x := append([]float32(nil), in...)
		SoftmaxInPlace(x)
		m := in[0]
		for _, v := range in[1:] {
			if v > m {
				m = v
			}
		}
		var sum float64
		for _, v := range in {
			sum += math.Exp(float64(v - m))
		}
		for i, v := range in {
			want := math.Exp(float64(v-m)) / sum
			if math.Abs(float64(x[i])-want) > 1e-6 {
				t.Fatalf("SoftmaxInPlace(%v)[%d]=%f want %f", in, i, x[i], want)
			}
		}
	}
}

func TestSoftmaxInPlace(t *testing.T) {
	x := []float32{-2, 0, 3, 1, -1, 2, 4, 0.5, -0.5}
	SoftmaxInPlace(x)
	var sum float32
	for _, v := range x {
		if v <= 0 {
			t.Fatalf("softmax value not positive: %f", v)
		}
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Fatalf("sum=%f want 1", sum)
	}
	if Argmax(x) != 6 {
		t.Fatalf("argmax=%d want 6", Argmax(x))
	}
}

func TestArgmax(t *testing.T) {
	x := []float32{-1, 0.5, 2, 1.9, 3, 2.5, 3.1, 0}
	if got := Argmax(x); got != 6 {
		t.Fatalf("Argmax got %d want 6", got)
	}
	ties := []float32{-1, 4, 2, 4, 3, 4}
	if got := Argmax(ties); got != 1 {
		t.Fatalf("Argmax tie got %d want 1", got)
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	w := []float32{1, 0.5, 1.5, 2, 0.25, 1.25, 0.75, 1, 1.1}
	got := make([]float32, len(x))
	RMSNorm(got, x, w, 1e-6)
	var ss float64
	for _, v := range x {
		ss += float64(v * v)
	}
	scale := 1 / math.Sqrt(ss/float64(len(x))+1e-6)
	for i := range x {
		want := float64(x[i]) * scale * float64(w[i])
		if math.Abs(float64(got[i])-want) > 1e-5 {
			t.Fatalf("RMSNorm[%d] got %f want %f", i, got[i], want)
		}
	}
}

func TestAddRMSNormMatchesSeparateOps(t *testing.T) {
	dst := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	add := []float32{0.5, 1, -1.5, 2, -2.5, 3, -3.5, 4, -4.5}
	w := []float32{1, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6}
	separateDst := append([]float32(nil), dst...)
	AddInPlace(separateDst, add)
	want := make([]float32, len(dst))
	RMSNorm(want, separateDst, w, 1e-6)
	gotDst := append([]float32(nil), dst...)
	got := make([]float32, len(dst))
	AddRMSNorm(got, gotDst, add, w, 1e-6)
	assertCloseVec(t, "addrmsnorm dst", gotDst, separateDst, 0)
	assertCloseVec(t, "addrmsnorm out", got, want, 1e-6)
}

func TestAddLayerNormMatchesSeparateOps(t *testing.T) {
	dst := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	add := []float32{0.5, 1, -1.5, 2, -2.5, 3, -3.5, 4, -4.5}
	w := []float32{1, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9}
	separateDst := append([]float32(nil), dst...)
	AddInPlace(separateDst, add)
	want := make([]float32, len(dst))
	LayerNorm(want, separateDst, w, bias, 1e-6)
	gotDst := append([]float32(nil), dst...)
	got := make([]float32, len(dst))
	AddLayerNorm(got, gotDst, add, w, bias, 1e-6)
	assertCloseVec(t, "addlayernorm dst", gotDst, separateDst, 0)
	assertCloseVec(t, "addlayernorm out", got, want, 1e-6)
}

func TestLayerNorm(t *testing.T) {
	x := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	w := []float32{1, 0.5, 1.5, 2, 0.25, 1.25, 0.75, 1, 1.1}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9}
	got := make([]float32, len(x))
	LayerNorm(got, x, w, bias, 1e-6)
	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	scale := 1 / math.Sqrt(variance/float64(len(x))+1e-6)
	for i := range x {
		want := (float64(x[i])-mean)*scale*float64(w[i]) + float64(bias[i])
		if math.Abs(float64(got[i])-want) > 1e-5 {
			t.Fatalf("LayerNorm[%d] got %f want %f", i, got[i], want)
		}
	}
}

func TestLayerNormInPlace(t *testing.T) {
	x := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	w := []float32{1, 0.5, 1.5, 2, 0.25, 1.25, 0.75, 1, 1.1}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9}
	want := make([]float32, len(x))
	LayerNorm(want, x, w, bias, 1e-6)
	got := append([]float32(nil), x...)
	LayerNorm(got, got, w, bias, 1e-6)
	assertCloseVec(t, "layernorm-inplace", got, want, 1e-6)
}

func TestGELUTanhInPlace(t *testing.T) {
	got := []float32{-3, -1, 0, 0.5, 1, 2, 3, 4, 5}
	want := append([]float32(nil), got...)
	for i := range want {
		want[i] = GELUTanh(want[i])
	}
	GELUTanhInPlace(got)
	assertCloseVec(t, "gelu-inplace", got, want, 0)
}

func TestGELUTanhRowsInPlace(t *testing.T) {
	got := makeRowsForTest(3, 9)
	for i := range got {
		copy(got[i], []float32{-3, -1, 0, 0.5, 1, 2, 3, 4, 5})
	}
	want := makeRowsForTest(3, 9)
	for i := range got {
		copy(want[i], got[i])
		for j := range want[i] {
			want[i][j] = GELUTanh(want[i][j])
		}
	}
	GELUTanhRowsInPlace(got)
	for i := range got {
		assertCloseVec(t, "gelu-rows", got[i], want[i], 0)
	}
}

func TestAddThenLayerNormMatchesAddLayerNorm(t *testing.T) {
	dstA := []float32{1.25, -0.75, 0.5, 2, -1, 0.25, 1, -0.5}
	dstB := append([]float32(nil), dstA...)
	add := []float32{-0.25, 0.5, 1, -1, 0.75, -0.5, 0.25, 1.5}
	weight := []float32{1, 1.25, 0.75, 1.5, 0.5, 1.1, 0.9, 1.3}
	bias := []float32{0, 0.1, -0.2, 0.3, -0.4, 0.2, -0.1, 0.05}
	outA := make([]float32, len(dstA))
	outB := make([]float32, len(dstB))
	AddLayerNorm(outA, dstA, add, weight, bias, 1e-6)
	AddThenLayerNorm(outB, dstB, add, weight, bias, 1e-6)
	assertCloseVec(t, "dst", dstB, dstA, 1e-6)
	assertCloseVec(t, "norm", outB, outA, 1e-5)
}

func TestLayerNormRowsMatchesPerRow(t *testing.T) {
	rows, cols := 5, 16
	x := makeRowsForTest(rows, cols)
	add := makeRowsForTest(rows, cols)
	for i := range x {
		fillTestValues(x[i])
		fillTestValues(add[i])
	}
	weight := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range weight {
		weight[i] = 1 + float32(i%5)/16
		bias[i] = float32(i%7-3) / 17
	}

	want := makeRowsForTest(rows, cols)
	got := makeRowsForTest(rows, cols)
	for i := range x {
		LayerNorm(want[i], x[i], weight, bias, 1e-6)
	}
	LayerNormRows(got, x, weight, bias, 1e-6)
	for i := range x {
		assertCloseVec(t, "layernorm rows", got[i], want[i], 0)
	}

	x2 := makeRowsForTest(rows, cols)
	x3 := makeRowsForTest(rows, cols)
	for i := range x {
		copy(x2[i], x[i])
		copy(x3[i], x[i])
	}
	for i := range x {
		AddThenLayerNorm(want[i], x2[i], add[i], weight, bias, 1e-6)
	}
	AddThenLayerNormRows(got, x3, add, weight, bias, 1e-6)
	for i := range x {
		assertCloseVec(t, "addthenlayernorm rows", got[i], want[i], 1e-5)
	}
}

func TestSiLUMulInPlace(t *testing.T) {
	gate := []float32{-3, -1, 0, 0.5, 1, 2, 3, 4, 5}
	up := []float32{2, -1, 3, 4, -2, 0.5, 1, -0.25, 0.75}
	want := append([]float32(nil), gate...)
	for i := range want {
		want[i] = SiLU(want[i]) * up[i]
	}
	SiLUMulInPlace(gate, up)
	assertCloseVec(t, "silu-mul", gate, want, 1e-6)
}

func TestFusedSwiGLUF32Scratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}
	g := make([]float32, 2)
	u := make([]float32, 2)
	want := make([]float32, 2)
	MatVec(g, x, gate, 2, 3)
	MatVec(u, x, up, 2, 3)
	for i := range g {
		g[i] = SiLU(g[i]) * u[i]
	}
	MatVec(want, g, down, 2, 2)

	got := make([]float32, 2)
	FusedSwiGLUF32Scratch(got, x, gate, up, down, 2, 3, 2, make([]float32, 2))
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("row %d got %f want %f", i, got[i], want[i])
		}
	}
}

func TestFusedSwiGLUQuantScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}
	want := make([]float32, 2)
	FusedSwiGLUF32Scratch(want, x, gate, up, down, 2, 3, 2, make([]float32, 2))

	got8 := make([]float32, 2)
	FusedSwiGLUQ8Scratch(got8, x, QuantizeQ8Row(gate, 2, 3), QuantizeQ8Row(up, 2, 3), QuantizeQ8Row(down, 2, 2), make([]float32, 2))
	for i := range got8 {
		if math.Abs(float64(got8[i]-want[i])) > 0.1 {
			t.Fatalf("q8 row %d got %f want %f", i, got8[i], want[i])
		}
	}

	got4 := make([]float32, 2)
	FusedSwiGLUQ4Scratch(got4, x, QuantizeQ4Row(gate, 2, 3), QuantizeQ4Row(up, 2, 3), QuantizeQ4Row(down, 2, 2), make([]float32, 2))
	for i := range got4 {
		if math.Abs(float64(got4[i]-want[i])) > 0.8 {
			t.Fatalf("q4 row %d got %f want %f", i, got4[i], want[i])
		}
	}

	got6 := make([]float32, 2)
	FusedSwiGLUQ6Scratch(got6, x, QuantizeQ6Row(gate, 2, 3), QuantizeQ6Row(up, 2, 3), QuantizeQ6Row(down, 2, 2), make([]float32, 2))
	for i := range got6 {
		if math.Abs(float64(got6[i]-want[i])) > 0.25 {
			t.Fatalf("q6 row %d got %f want %f", i, got6[i], want[i])
		}
	}
}

func TestFusedSwiGLUQuantScratchUsesSingleScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}
	q8Gate := QuantizeQ8Row(gate, 2, 3)
	q8Up := QuantizeQ8Row(up, 2, 3)
	q8Down := QuantizeQ8Row(down, 2, 2)
	q4Gate := QuantizeQ4Row(gate, 2, 3)
	q4Up := QuantizeQ4Row(up, 2, 3)
	q4Down := QuantizeQ4Row(down, 2, 2)
	q6Gate := QuantizeQ6Row(gate, 2, 3)
	q6Up := QuantizeQ6Row(up, 2, 3)
	q6Down := QuantizeQ6Row(down, 2, 2)

	want8 := make([]float32, 2)
	got8 := make([]float32, 2)
	FusedSwiGLUQ8(want8, x, q8Gate, q8Up, q8Down)
	FusedSwiGLUQ8Scratch(got8, x, q8Gate, q8Up, q8Down, make([]float32, 2))
	assertCloseVec(t, "q8 single scratch", got8, want8, 0)

	want4 := make([]float32, 2)
	got4 := make([]float32, 2)
	FusedSwiGLUQ4(want4, x, q4Gate, q4Up, q4Down)
	FusedSwiGLUQ4Scratch(got4, x, q4Gate, q4Up, q4Down, make([]float32, 2))
	assertCloseVec(t, "q4 single scratch", got4, want4, 0)

	want6 := make([]float32, 2)
	got6 := make([]float32, 2)
	FusedSwiGLUQ6(want6, x, q6Gate, q6Up, q6Down)
	FusedSwiGLUQ6Scratch(got6, x, q6Gate, q6Up, q6Down, make([]float32, 2))
	assertCloseVec(t, "q6 single scratch", got6, want6, 0)
}

func TestFusedSwiGLUQuantConvenienceAllocatesSingleScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25, 0.75}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1, 0.25,
		2, -0.25, 0.75, -1,
		1.25, -0.5, 0.5, 2,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}

	q8Gate := QuantizeQ8Row(gate, 2, 3)
	q8Up := QuantizeQ8Row(up, 3, 4)
	q8Down := QuantizeQ8Row(down, 2, 2)
	want8, got8 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ8Scratch(want8, x, q8Gate, q8Up, q8Down, make([]float32, 4))
	FusedSwiGLUQ8(got8, x, q8Gate, q8Up, q8Down)
	assertCloseVec(t, "q8 convenience fallback", got8, want8, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ8(got8, x, q8Gate, q8Up, q8Down)
	}); allocs != 1 {
		t.Fatalf("q8 convenience allocs=%f want 1", allocs)
	}

	q4Gate := QuantizeQ4Row(gate, 2, 3)
	q4Up := QuantizeQ4Row(up, 3, 4)
	q4Down := QuantizeQ4Row(down, 2, 2)
	want4, got4 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ4Scratch(want4, x, q4Gate, q4Up, q4Down, make([]float32, 4))
	FusedSwiGLUQ4(got4, x, q4Gate, q4Up, q4Down)
	assertCloseVec(t, "q4 convenience fallback", got4, want4, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ4(got4, x, q4Gate, q4Up, q4Down)
	}); allocs != 1 {
		t.Fatalf("q4 convenience allocs=%f want 1", allocs)
	}

	q6Gate := QuantizeQ6Row(gate, 2, 3)
	q6Up := QuantizeQ6Row(up, 3, 4)
	q6Down := QuantizeQ6Row(down, 2, 2)
	want6, got6 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ6Scratch(want6, x, q6Gate, q6Up, q6Down, make([]float32, 4))
	FusedSwiGLUQ6(got6, x, q6Gate, q6Up, q6Down)
	assertCloseVec(t, "q6 convenience fallback", got6, want6, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ6(got6, x, q6Gate, q6Up, q6Down)
	}); allocs != 1 {
		t.Fatalf("q6 convenience allocs=%f want 1", allocs)
	}
}

func TestFusedSwiGLUQuantConvenienceEqualShapeAllocatesSingleScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}

	q8Gate := QuantizeQ8Row(gate, 2, 3)
	q8Up := QuantizeQ8Row(up, 2, 3)
	q8Down := QuantizeQ8Row(down, 2, 2)
	got8 := make([]float32, 2)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ8(got8, x, q8Gate, q8Up, q8Down)
	}); allocs != 1 {
		t.Fatalf("q8 equal-shape allocs=%f want 1", allocs)
	}

	q4Gate := QuantizeQ4Row(gate, 2, 3)
	q4Up := QuantizeQ4Row(up, 2, 3)
	q4Down := QuantizeQ4Row(down, 2, 2)
	got4 := make([]float32, 2)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ4(got4, x, q4Gate, q4Up, q4Down)
	}); allocs != 1 {
		t.Fatalf("q4 equal-shape allocs=%f want 1", allocs)
	}

	q6Gate := QuantizeQ6Row(gate, 2, 3)
	q6Up := QuantizeQ6Row(up, 2, 3)
	q6Down := QuantizeQ6Row(down, 2, 2)
	got6 := make([]float32, 2)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ6(got6, x, q6Gate, q6Up, q6Down)
	}); allocs != 1 {
		t.Fatalf("q6 equal-shape allocs=%f want 1", allocs)
	}
}

func TestQuantSwiGLUFallbackHandlesShortUpRows(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
		0.75, -1.25, 0.5,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}

	q8Gate := QuantizeQ8Row(gate, 3, 3)
	q8Up := QuantizeQ8Row(up, 2, 3)
	wantGate, wantUp := make([]float32, 3), make([]float32, 2)
	MatVecQ8(wantGate, x, q8Gate)
	MatVecQ8(wantUp, x, q8Up)
	got := []float32{9, 9, 9}
	matVecQ8SwiGLU(got, x, q8Gate, q8Up)
	assertCloseVec(t, "q8 fallback head", got[:2], []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)
	if got[2] != 0 {
		t.Fatalf("q8 fallback tail=%f want 0", got[2])
	}

	q4Gate := QuantizeQ4Row(gate, 3, 3)
	q4Up := QuantizeQ4Row(up, 2, 3)
	MatVecQ4(wantGate, x, q4Gate)
	MatVecQ4(wantUp, x, q4Up)
	got = []float32{9, 9, 9}
	matVecQ4SwiGLU(got, x, q4Gate, q4Up)
	assertCloseVec(t, "q4 fallback head", got[:2], []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)
	if got[2] != 0 {
		t.Fatalf("q4 fallback tail=%f want 0", got[2])
	}

	q6Gate := QuantizeQ6Row(gate, 3, 3)
	q6Up := QuantizeQ6Row(up, 2, 3)
	MatVecQ6(wantGate, x, q6Gate)
	MatVecQ6(wantUp, x, q6Up)
	got = []float32{9, 9, 9}
	matVecQ6SwiGLU(got, x, q6Gate, q6Up)
	assertCloseVec(t, "q6 fallback head", got[:2], []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)
	if got[2] != 0 {
		t.Fatalf("q6 fallback tail=%f want 0", got[2])
	}
}

func TestQuantSwiGLUFallbackHandlesLongUpRowsAndDifferentCols(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25, 0.75}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1, 0.25,
		2, -0.25, 0.75, -1,
		1.25, -0.5, 0.5, 2,
	}

	q8Gate := QuantizeQ8Row(gate, 2, 3)
	q8Up := QuantizeQ8Row(up, 3, 4)
	wantGate, wantUp := make([]float32, 2), make([]float32, 3)
	MatVecQ8(wantGate, x, q8Gate)
	MatVecQ8(wantUp, x, q8Up)
	got := []float32{9, 9}
	matVecQ8SwiGLU(got, x, q8Gate, q8Up)
	assertCloseVec(t, "q8 fallback long up", got, []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)

	q4Gate := QuantizeQ4Row(gate, 2, 3)
	q4Up := QuantizeQ4Row(up, 3, 4)
	MatVecQ4(wantGate, x, q4Gate)
	MatVecQ4(wantUp, x, q4Up)
	got = []float32{9, 9}
	matVecQ4SwiGLU(got, x, q4Gate, q4Up)
	assertCloseVec(t, "q4 fallback long up", got, []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)

	q6Gate := QuantizeQ6Row(gate, 2, 3)
	q6Up := QuantizeQ6Row(up, 3, 4)
	MatVecQ6(wantGate, x, q6Gate)
	MatVecQ6(wantUp, x, q6Up)
	got = []float32{9, 9}
	matVecQ6SwiGLU(got, x, q6Gate, q6Up)
	assertCloseVec(t, "q6 fallback long up", got, []float32{SiLU(wantGate[0]) * wantUp[0], SiLU(wantGate[1]) * wantUp[1]}, 0)
}

func TestFusedSwiGLUQuantScratchFallbackReusesExtraScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25, 0.75}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	up := []float32{
		0.5, 1, -1, 0.25,
		2, -0.25, 0.75, -1,
		1.25, -0.5, 0.5, 2,
	}
	down := []float32{
		1, -2,
		0.5, 0.25,
	}

	q8Gate := QuantizeQ8Row(gate, 2, 3)
	q8Up := QuantizeQ8Row(up, 3, 4)
	q8Down := QuantizeQ8Row(down, 2, 2)
	want8, got8 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ8Scratch(want8, x, q8Gate, q8Up, q8Down, make([]float32, 2))
	FusedSwiGLUQ8Scratch(got8, x, q8Gate, q8Up, q8Down, make([]float32, 4))
	assertCloseVec(t, "q8 fallback extra scratch", got8, want8, 0)
	scratch8 := make([]float32, 4)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ8Scratch(got8, x, q8Gate, q8Up, q8Down, scratch8)
	}); allocs != 0 {
		t.Fatalf("q8 fallback allocs=%f want 0", allocs)
	}

	q4Gate := QuantizeQ4Row(gate, 2, 3)
	q4Up := QuantizeQ4Row(up, 3, 4)
	q4Down := QuantizeQ4Row(down, 2, 2)
	want4, got4 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ4Scratch(want4, x, q4Gate, q4Up, q4Down, make([]float32, 2))
	FusedSwiGLUQ4Scratch(got4, x, q4Gate, q4Up, q4Down, make([]float32, 4))
	assertCloseVec(t, "q4 fallback extra scratch", got4, want4, 0)
	scratch4 := make([]float32, 4)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ4Scratch(got4, x, q4Gate, q4Up, q4Down, scratch4)
	}); allocs != 0 {
		t.Fatalf("q4 fallback allocs=%f want 0", allocs)
	}

	q6Gate := QuantizeQ6Row(gate, 2, 3)
	q6Up := QuantizeQ6Row(up, 3, 4)
	q6Down := QuantizeQ6Row(down, 2, 2)
	want6, got6 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ6Scratch(want6, x, q6Gate, q6Up, q6Down, make([]float32, 2))
	FusedSwiGLUQ6Scratch(got6, x, q6Gate, q6Up, q6Down, make([]float32, 4))
	assertCloseVec(t, "q6 fallback extra scratch", got6, want6, 0)
	scratch6 := make([]float32, 4)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ6Scratch(got6, x, q6Gate, q6Up, q6Down, scratch6)
	}); allocs != 0 {
		t.Fatalf("q6 fallback allocs=%f want 0", allocs)
	}
}

func TestFusedSwiGLUQuantScratchFallbackReusesShortUpScratch(t *testing.T) {
	x := []float32{0.25, -0.5, 1.25}
	gate := []float32{
		1, 2, 3,
		-1, 0.5, 2,
		0.75, -1.25, 0.5,
	}
	up := []float32{
		0.5, 1, -1,
		2, -0.25, 0.75,
	}
	down := []float32{
		1, -2, 0.5,
		0.5, 0.25, -1,
	}

	q8Gate := QuantizeQ8Row(gate, 3, 3)
	q8Up := QuantizeQ8Row(up, 2, 3)
	q8Down := QuantizeQ8Row(down, 2, 3)
	want8, got8 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ8Scratch(want8, x, q8Gate, q8Up, q8Down, make([]float32, 3))
	scratch8 := make([]float32, 5)
	FusedSwiGLUQ8Scratch(got8, x, q8Gate, q8Up, q8Down, scratch8)
	assertCloseVec(t, "q8 fallback short-up scratch", got8, want8, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ8Scratch(got8, x, q8Gate, q8Up, q8Down, scratch8)
	}); allocs != 0 {
		t.Fatalf("q8 short-up fallback allocs=%f want 0", allocs)
	}

	q4Gate := QuantizeQ4Row(gate, 3, 3)
	q4Up := QuantizeQ4Row(up, 2, 3)
	q4Down := QuantizeQ4Row(down, 2, 3)
	want4, got4 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ4Scratch(want4, x, q4Gate, q4Up, q4Down, make([]float32, 3))
	scratch4 := make([]float32, 5)
	FusedSwiGLUQ4Scratch(got4, x, q4Gate, q4Up, q4Down, scratch4)
	assertCloseVec(t, "q4 fallback short-up scratch", got4, want4, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ4Scratch(got4, x, q4Gate, q4Up, q4Down, scratch4)
	}); allocs != 0 {
		t.Fatalf("q4 short-up fallback allocs=%f want 0", allocs)
	}

	q6Gate := QuantizeQ6Row(gate, 3, 3)
	q6Up := QuantizeQ6Row(up, 2, 3)
	q6Down := QuantizeQ6Row(down, 2, 3)
	want6, got6 := make([]float32, 2), make([]float32, 2)
	FusedSwiGLUQ6Scratch(want6, x, q6Gate, q6Up, q6Down, make([]float32, 3))
	scratch6 := make([]float32, 5)
	FusedSwiGLUQ6Scratch(got6, x, q6Gate, q6Up, q6Down, scratch6)
	assertCloseVec(t, "q6 fallback short-up scratch", got6, want6, 0)
	if allocs := testing.AllocsPerRun(100, func() {
		FusedSwiGLUQ6Scratch(got6, x, q6Gate, q6Up, q6Down, scratch6)
	}); allocs != 0 {
		t.Fatalf("q6 short-up fallback allocs=%f want 0", allocs)
	}
}

func TestMatRowsBias(t *testing.T) {
	xs := [][]float32{
		{1, 2, 3},
		{-1, 0.5, 2},
	}
	w := []float32{
		1, 0, -1,
		0.5, 2, 1,
	}
	bias := []float32{0.25, -0.5}
	out := makeRowsForTest(2, 2)
	MatRowsBias(out, xs, w, bias, 2, 3)
	for i := range xs {
		want := make([]float32, 2)
		MatVecBias(want, xs[i], w, bias, 2, 3)
		for j := range want {
			if out[i][j] != want[j] {
				t.Fatalf("out[%d][%d] got %f want %f", i, j, out[i][j], want[j])
			}
		}
	}
}

func TestMatRowsBiasAddRowsMatchesSeparate(t *testing.T) {
	xs := [][]float32{
		{1, 2, 3},
		{-1, 0.5, 2},
	}
	w := []float32{
		1, 0, -1,
		0.5, 2, 1,
	}
	bias := []float32{0.25, -0.5}
	add := [][]float32{
		{0.1, 0.2},
		{-0.3, 0.4},
	}
	want := makeRowsForTest(2, 2)
	got := makeRowsForTest(2, 2)
	MatRowsBias(want, xs, w, bias, 2, 3)
	for i := range want {
		AddInPlace(want[i], add[i])
	}
	MatRowsBiasAddRows(got, xs, w, bias, add, 2, 3)
	for i := range got {
		assertCloseVec(t, "row", got[i], want[i], 1e-6)
	}
}

func TestMatRowsBiasAddRowsEmptyAddMatchesMatRowsBias(t *testing.T) {
	xs := [][]float32{
		{1, 2, 3},
		{-1, 0.5, 2},
	}
	w := []float32{
		1, 0, -1,
		0.5, 2, 1,
	}
	bias := []float32{0.25, -0.5}
	want := makeRowsForTest(2, 2)
	got := makeRowsForTest(2, 2)
	MatRowsBias(want, xs, w, bias, 2, 3)
	MatRowsBiasAddRows(got, xs, w, bias, nil, 2, 3)
	for i := range got {
		assertCloseVec(t, "row", got[i], want[i], 1e-6)
	}
}

func TestMatRowsBiasAddRowsRepeatsAddRows(t *testing.T) {
	rows, cols, batch := 5, 7, 6
	xs := makeRowsForTest(batch, cols)
	out := makeRowsForTest(batch, rows)
	want := makeRowsForTest(batch, rows)
	add := makeRowsForTest(2, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillTestValues(w)
	fillTestValues(bias)
	for i := range xs {
		fillTestValues(xs[i])
	}
	for i := range add {
		fillTestValues(add[i])
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i%len(add)])
	}
	MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	for i := range out {
		assertCloseVec(t, "row", out[i], want[i], 1e-6)
	}
}

func TestMatRowsBiasAddRowsCols588MatchesSeparate(t *testing.T) {
	rows, cols, batch := 3, 588, 2
	xs := makeRowsForTest(batch, cols)
	out := makeRowsForTest(batch, rows)
	want := makeRowsForTest(batch, rows)
	add := makeRowsForTest(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range bias {
		bias[i] = float32(i%5-2) / 5
	}
	for i := range xs {
		for j := range xs[i] {
			xs[i][j] = float32((i+j)%13-6) / 13
		}
		for j := range add[i] {
			add[i][j] = float32((i+j)%7-3) / 7
		}
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i])
	}
	MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	for i := range out {
		assertCloseVec(t, "row", out[i], want[i], 1e-5)
	}
}

func TestMatRowsBiasAddRowsCols16MatchesSeparate(t *testing.T) {
	rows, cols, batch := 5, 16, 3
	xs := makeRowsForTest(batch, cols)
	out := makeRowsForTest(batch, rows)
	want := makeRowsForTest(batch, rows)
	add := makeRowsForTest(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range bias {
		bias[i] = float32(i%5-2) / 5
	}
	for i := range xs {
		for j := range xs[i] {
			xs[i][j] = float32((i+j)%13-6) / 13
		}
		for j := range add[i] {
			add[i][j] = float32((i+j)%7-3) / 7
		}
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i])
	}
	MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	for i := range out {
		assertCloseVec(t, "row", out[i], want[i], 1e-5)
	}
}

func TestMatRowsBiasAddRowsCols16RepeatsAddRows(t *testing.T) {
	rows, cols, batch := 5, 16, 6
	xs := makeRowsForTest(batch, cols)
	out := makeRowsForTest(batch, rows)
	want := makeRowsForTest(batch, rows)
	add := makeRowsForTest(2, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillTestValues(w)
	fillTestValues(bias)
	for i := range xs {
		fillTestValues(xs[i])
	}
	for i := range add {
		fillTestValues(add[i])
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i%len(add)])
	}
	MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	for i := range out {
		assertCloseVec(t, "row", out[i], want[i], 1e-5)
	}
}

func TestMatRowsBiasAddRowsCols588RepeatsAddRows(t *testing.T) {
	rows, cols, batch := 5, 588, 6
	xs := makeRowsForTest(batch, cols)
	out := makeRowsForTest(batch, rows)
	want := makeRowsForTest(batch, rows)
	add := makeRowsForTest(2, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillTestValues(w)
	fillTestValues(bias)
	for i := range xs {
		fillTestValues(xs[i])
	}
	for i := range add {
		fillTestValues(add[i])
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i%len(add)])
	}
	MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	for i := range out {
		assertCloseVec(t, "row", out[i], want[i], 1e-4)
	}
}

func TestMatRowsBias3MatchesSeparate(t *testing.T) {
	xs := [][]float32{
		{1, 2, 3},
		{-1, 0.5, 2},
	}
	wa := []float32{
		1, 0, -1,
		0.5, 2, 1,
	}
	wb := []float32{
		-0.25, 1, 0.75,
	}
	wc := []float32{
		2, -1, 0.5,
		0.25, 0.5, -0.75,
	}
	ba := []float32{0.25, -0.5}
	bb := []float32{0.75}
	bc := []float32{1, -1}
	wantA, wantB, wantC := makeRowsForTest(2, 2), makeRowsForTest(2, 1), makeRowsForTest(2, 2)
	gotA, gotB, gotC := makeRowsForTest(2, 2), makeRowsForTest(2, 1), makeRowsForTest(2, 2)
	for i := range xs {
		MatVecBias(wantA[i], xs[i], wa, ba, 2, 3)
		MatVecBias(wantB[i], xs[i], wb, bb, 1, 3)
		MatVecBias(wantC[i], xs[i], wc, bc, 2, 3)
	}
	MatRowsBias3(gotA, gotB, gotC, xs, wa, ba, wb, bb, wc, bc, 2, 1, 2, 3)
	for i := range xs {
		assertCloseVec(t, "a", gotA[i], wantA[i], 1e-6)
		assertCloseVec(t, "b", gotB[i], wantB[i], 1e-6)
		assertCloseVec(t, "c", gotC[i], wantC[i], 1e-6)
	}
}

func TestMatRowsBias3EqualRowsMatchesSeparate(t *testing.T) {
	xs := [][]float32{
		{1, 2, 3, 4},
		{-1, 0.5, 2, -0.25},
	}
	rows, cols := 3, 4
	wa := []float32{
		1, 0, -1, 0.5,
		0.5, 2, 1, -0.25,
		-1, 0.25, 0.75, 1,
	}
	wb := []float32{
		-0.25, 1, 0.75, 2,
		0.5, -1, 1.5, 0.25,
		2, 0.5, -0.5, 1,
	}
	wc := []float32{
		2, -1, 0.5, 0.25,
		0.25, 0.5, -0.75, 1,
		-0.5, 1.5, 0.25, -1,
	}
	ba := []float32{0.25, -0.5, 0.75}
	bb := []float32{0.75, -0.25, 0.5}
	bc := []float32{1, -1, 0.25}
	wantA, wantB, wantC := makeRowsForTest(2, rows), makeRowsForTest(2, rows), makeRowsForTest(2, rows)
	gotA, gotB, gotC := makeRowsForTest(2, rows), makeRowsForTest(2, rows), makeRowsForTest(2, rows)
	for i := range xs {
		MatVecBias(wantA[i], xs[i], wa, ba, rows, cols)
		MatVecBias(wantB[i], xs[i], wb, bb, rows, cols)
		MatVecBias(wantC[i], xs[i], wc, bc, rows, cols)
	}
	MatRowsBias3(gotA, gotB, gotC, xs, wa, ba, wb, bb, wc, bc, rows, rows, rows, cols)
	for i := range xs {
		assertCloseVec(t, "equal a", gotA[i], wantA[i], 1e-6)
		assertCloseVec(t, "equal b", gotB[i], wantB[i], 1e-6)
		assertCloseVec(t, "equal c", gotC[i], wantC[i], 1e-6)
	}
}

func TestMatVecBiasMatchesSeparateLarge(t *testing.T) {
	rows, cols := 128, 2048
	x := make([]float32, cols)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillTestValues(x)
	fillTestValues(w)
	for i := range bias {
		bias[i] = float32(i%9-4) / 9
	}
	want := make([]float32, rows)
	MatVec(want, x, w, rows, cols)
	for i := range want {
		want[i] += bias[i]
	}
	got := make([]float32, rows)
	MatVecBias(got, x, w, bias, rows, cols)
	assertCloseVec(t, "matvecbias", got, want, 1e-5)
}

func TestMatVecCols256SpecializationMatchesReference(t *testing.T) {
	rows, cols := 7, 256
	x := make([]float32, cols)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillTestValues(x)
	fillTestValues(w)
	for i := range bias {
		bias[i] = float32(i%5-2) / 7
	}

	want := make([]float32, rows)
	for r := 0; r < rows; r++ {
		want[r] = referenceDot(w[r*cols:(r+1)*cols], x)
	}
	got := make([]float32, rows)
	MatVec(got, x, w, rows, cols)
	assertCloseVec(t, "matvec256", got, want, 1e-4)

	for r := range want {
		want[r] += bias[r]
	}
	MatVecBias(got, x, w, bias, rows, cols)
	assertCloseVec(t, "matvecbias256", got, want, 1e-4)

	xs := [][]float32{x, append([]float32(nil), x...)}
	xs[1][0] += 0.25
	out := makeRowsForTest(len(xs), rows)
	MatRowsBias(out, xs, w, bias, rows, cols)
	for i := range xs {
		for r := 0; r < rows; r++ {
			want[r] = referenceDot(w[r*cols:(r+1)*cols], xs[i]) + bias[r]
		}
		assertCloseVec(t, "matrowsbias256", out[i], want, 1e-4)
	}
}

func referenceDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func fillTestValues(x []float32) {
	for i := range x {
		x[i] = float32(i%17-8) / 17
	}
}

func makeRowsForTest(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := range out {
		out[i] = make([]float32, cols)
	}
	return out
}


func TestDotOctet(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
	b0 := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}
	b1 := []float32{0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}
	b2 := []float32{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}
	b3 := []float32{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}
	b4 := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	b5 := []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	b6 := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	b7 := []float32{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}
	s0, s1, s2, s3, s4, s5, s6, s7 := DotOctet(b0, b1, b2, b3, b4, b5, b6, b7, a)
	if s0 != 120 { t.Errorf("s0: got %v want 120", s0) }
	if s1 != 128 { t.Errorf("s1: got %v want 128", s1) }
	if s2 != 136 { t.Errorf("s2: got %v want 136", s2) }
	if s3 != 144 { t.Errorf("s3: got %v want 144", s3) }
	if s4 != 1056 { t.Errorf("s4: got %v want 1056", s4) }
	if s5 != 264 { t.Errorf("s5: got %v want 264", s5) }
	if s6 != 528 { t.Errorf("s6: got %v want 528", s6) }
	if s7 != 1584 { t.Errorf("s7: got %v want 1584", s7) }
}

func TestDotOctetSimple(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	b0 := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	b1 := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	b2 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b3 := []float32{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b4 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b5 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b6 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	b7 := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	s0, s1, s2, s3, s4, s5, s6, s7 := DotOctet(b0, b1, b2, b3, b4, b5, b6, b7, a)
	t.Logf("results: %v %v %v %v %v %v %v %v", s0, s1, s2, s3, s4, s5, s6, s7)
	if s0 != 136 { t.Errorf("s0: got %v want 136", s0) }
	if s1 != 272 { t.Errorf("s1: got %v want 272", s1) }
	if s2 != 0 { t.Errorf("s2: got %v want 0", s2) }
	if s3 != 1 { t.Errorf("s3: got %v want 1", s3) }
	if s4 != 0 { t.Errorf("s4: got %v want 0", s4) }
	if s5 != 0 { t.Errorf("s5: got %v want 0", s5) }
	if s6 != 0 { t.Errorf("s6: got %v want 0", s6) }
	if s7 != 0 { t.Errorf("s7: got %v want 0", s7) }
}
