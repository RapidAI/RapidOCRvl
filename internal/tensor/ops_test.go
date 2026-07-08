package tensor

import (
	"math"
	"testing"
)

func TestMatVecCols96MatchesGenericRange(t *testing.T) {
	rows, cols := 17, 96
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%29)-14) / 19
	}
	for i := range x {
		x[i] = float32((i%17)-8) / 13
	}
	want := make([]float32, rows)
	got := make([]float32, rows)
	matVecSerialRange(want, x, w, cols, 0, rows)
	MatVec(got, x, w, rows, cols)
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("row %d got %f want %f", i, got[i], want[i])
		}
	}
}

func TestMatVecBiasCols96MatchesGenericRange(t *testing.T) {
	rows, cols := 17, 96
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	for i := range w {
		w[i] = float32((i%29)-14) / 19
	}
	for i := range x {
		x[i] = float32((i%17)-8) / 13
	}
	for i := range bias {
		bias[i] = float32((i%11)-5) / 7
	}
	want := make([]float32, rows)
	got := make([]float32, rows)
	matVecBiasSerialRange(want, x, w, bias, cols, 0, rows)
	MatVecBias(got, x, w, bias, rows, cols)
	assertCloseFloats(t, "bias", got, want)
}

func TestMatRowsBiasAddRowsCols96MatchesSeparate(t *testing.T) {
	batch, rows, cols := 5, 17, 96
	xs := makeRowsBench(batch, cols)
	want := makeRowsBench(batch, rows)
	got := makeRowsBench(batch, rows)
	add := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
		fillBench(add[i])
	}
	MatRowsBias(want, xs, w, bias, rows, cols)
	for i := range want {
		AddInPlace(want[i], add[i])
	}
	MatRowsBiasAddRows(got, xs, w, bias, add, rows, cols)
	for i := range got {
		assertCloseFloats(t, "row", got[i], want[i])
	}
}

func TestMatRowsBias3Cols96MatchesSeparate(t *testing.T) {
	batch, rows, cols := 5, 17, 96
	xs := makeRowsBench(batch, cols)
	wantA := makeRowsBench(batch, rows)
	wantB := makeRowsBench(batch, rows)
	wantC := makeRowsBench(batch, rows)
	gotA := makeRowsBench(batch, rows)
	gotB := makeRowsBench(batch, rows)
	gotC := makeRowsBench(batch, rows)
	wa := make([]float32, rows*cols)
	wb := make([]float32, rows*cols)
	wc := make([]float32, rows*cols)
	ba := make([]float32, rows)
	bb := make([]float32, rows)
	bc := make([]float32, rows)
	fillBench(wa)
	fillBench(wb)
	fillBench(wc)
	fillBench(ba)
	fillBench(bb)
	fillBench(bc)
	for i := range xs {
		fillBench(xs[i])
		MatVecBias(wantA[i], xs[i], wa, ba, rows, cols)
		MatVecBias(wantB[i], xs[i], wb, bb, rows, cols)
		MatVecBias(wantC[i], xs[i], wc, bc, rows, cols)
	}
	MatRowsBias3(gotA, gotB, gotC, xs, wa, ba, wb, bb, wc, bc, rows, rows, rows, cols)
	for i := range gotA {
		assertCloseFloats(t, "a", gotA[i], wantA[i])
		assertCloseFloats(t, "b", gotB[i], wantB[i])
		assertCloseFloats(t, "c", gotC[i], wantC[i])
	}
}

func TestFusedMatVec3Cols96MatchesSeparate(t *testing.T) {
	cols, rowsA, rowsB, rowsC := 96, 17, 7, 7
	x := make([]float32, cols)
	wa := make([]float32, rowsA*cols)
	wb := make([]float32, rowsB*cols)
	wc := make([]float32, rowsC*cols)
	for i := range x {
		x[i] = float32((i%19)-9) / 17
	}
	for i := range wa {
		wa[i] = float32((i%31)-15) / 23
	}
	for i := range wb {
		wb[i] = float32((i%37)-18) / 29
	}
	for i := range wc {
		wc[i] = float32((i%41)-20) / 31
	}
	wantA, wantB, wantC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	gotA, gotB, gotC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	MatVec(wantA, x, wa, rowsA, cols)
	MatVec(wantB, x, wb, rowsB, cols)
	MatVec(wantC, x, wc, rowsC, cols)
	FusedMatVec3(gotA, gotB, gotC, x, wa, wb, wc, rowsA, rowsB, rowsC, cols)
	assertCloseFloats(t, "a", gotA, wantA)
	assertCloseFloats(t, "b", gotB, wantB)
	assertCloseFloats(t, "c", gotC, wantC)
}

func assertCloseFloats(t *testing.T, name string, got, want []float32) {
	t.Helper()
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("%s[%d] got %f want %f", name, i, got[i], want[i])
		}
	}
}

func TestMatVecArgmaxMatchesMatVecArgmax(t *testing.T) {
	rows, cols := 17, 9
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range w {
		w[i] = float32(i%11-5) / 7
	}
	for i := range x {
		x[i] = float32(i%7-3) / 5
	}
	copy(w[3*cols:(3+1)*cols], []float32{3, 0, 0, 0, 0, 0, 0, 0, 0})
	copy(w[12*cols:(12+1)*cols], []float32{3, 0, 0, 0, 0, 0, 0, 0, 0})
	x[0] = 2

	out := make([]float32, rows)
	MatVec(out, x, w, rows, cols)
	want := Argmax(out)
	got, score := MatVecArgmax(x, w, rows, cols)
	if got != want || score != out[want] {
		t.Fatalf("MatVecArgmax=(%d,%f) want (%d,%f)", got, score, want, out[want])
	}
}

func TestMatVecTopKMatchesMatVecTopK(t *testing.T) {
	rows, cols, k := 11, 5, 4
	w := make([]float32, rows*cols)
	x := []float32{1, -0.25, 0.5, 0.75, -0.5}
	for r := 0; r < rows; r++ {
		base := r * cols
		w[base] = float32(r - 5)
		w[base+1] = float32((r%3)-1) / 10
		w[base+2] = float32((r%5)-2) / 20
	}
	out := make([]float32, rows)
	MatVec(out, x, w, rows, cols)
	got, maxScore := MatVecTopK(nil, x, w, rows, cols, k)
	wantIDs := topKIDsForTest(out, k)
	if len(got) != k {
		t.Fatalf("len got %d want %d", len(got), k)
	}
	seen := map[int]bool{}
	for _, s := range got {
		seen[s.ID] = true
		if math.Abs(float64(s.Score-out[s.ID])) > 1e-2 {
			t.Fatalf("score id=%d got %f want %f", s.ID, s.Score, out[s.ID])
		}
	}
	for _, id := range wantIDs {
		if !seen[id] {
			t.Fatalf("missing id %d in %v want ids %v", id, got, wantIDs)
		}
	}
	if math.Abs(float64(maxScore-out[wantIDs[0]])) > 1e-2 {
		t.Fatalf("max score got %f want %f", maxScore, out[wantIDs[0]])
	}
}

func TestMatVecTopKParallelMatchesMatVecTopK(t *testing.T) {
	rows, cols, k := 1024, 256, 7
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%17)-8) / 17
	}
	for r := 0; r < rows; r++ {
		base := r * cols
		w[base] = float32(r - rows/2)
		for c := 1; c < cols; c++ {
			w[base+c] = float32(((r+c)%23)-11) / 100
		}
	}
	out := make([]float32, rows)
	MatVec(out, x, w, rows, cols)
	got, maxScore := MatVecTopK(nil, x, w, rows, cols, k)
	wantIDs := topKIDsForTest(out, k)
	if len(got) != k {
		t.Fatalf("len got %d want %d", len(got), k)
	}
	seen := map[int]bool{}
	for _, s := range got {
		seen[s.ID] = true
		if math.Abs(float64(s.Score-out[s.ID])) > 1e-2 {
			t.Fatalf("score id=%d got %f want %f", s.ID, s.Score, out[s.ID])
		}
	}
	for _, id := range wantIDs {
		if !seen[id] {
			t.Fatalf("missing id %d in %v want ids %v", id, got, wantIDs)
		}
	}
	if math.Abs(float64(maxScore-out[wantIDs[0]])) > 1e-2 {
		t.Fatalf("max score got %f want %f", maxScore, out[wantIDs[0]])
	}
}

func topKIDsForTest(scores []float32, k int) []int {
	ids := make([]int, len(scores))
	for i := range ids {
		ids[i] = i
	}
	for i := 1; i < len(ids); i++ {
		id := ids[i]
		j := i - 1
		for ; j >= 0 && scores[ids[j]] < scores[id]; j-- {
			ids[j+1] = ids[j]
		}
		ids[j+1] = id
	}
	return ids[:k]
}
