package tensor

import (
	"math"
	"testing"
)

func TestDotQ8VNNICorrectness(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 100, 128, 256, 1024} {
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

func TestDotQ8PairVNNICorrectness(t *testing.T) {
	n := 256
	a := make([]int8, n)
	b := make([]int8, n)
	x := make([]float32, n)
	for i := range a {
		a[i] = int8(i%17 - 8)
		b[i] = int8(i%13 - 6)
		x[i] = float32(i%7) / 7
	}
	wa := make([]float32, n)
	wb := make([]float32, n)
	for i, v := range a {
		wa[i] = float32(v)
	}
	for i, v := range b {
		wb[i] = float32(v)
	}
	qa := QuantizeQ8Row(wa, 1, n)
	qb := QuantizeQ8Row(wb, 1, n)
	xq := make([]uint8, n)
	scaleX := quantizeXForVNNI(x, xq)

	r0, r1 := dotQ8PairVNNI(qa.Data[:n], qb.Data[:n], xq, scaleX, qa.RowSum[0], qb.RowSum[0], qa.Scale[0], qb.Scale[0])
	s0 := dotQ8(qa.Data[:n], x)
	s1 := dotQ8(qb.Data[:n], x)

	if math.Abs(float64(r0-s0)) > 0.5 || math.Abs(float64(r1-s1)) > 0.5 {
		t.Errorf("pair: VNNI=(%f,%f) ref=(%f,%f)", r0, r1, s0, s1)
	}
}

func TestDotQ8TripletVNNICorrectness(t *testing.T) {
	n := 256
	a := make([]int8, n)
	b := make([]int8, n)
	c := make([]int8, n)
	x := make([]float32, n)
	for i := range a {
		a[i] = int8(i%17 - 8)
		b[i] = int8(i%13 - 6)
		c[i] = int8(i%11 - 5)
		x[i] = float32(i%7) / 7
	}
	wa := make([]float32, n)
	wb := make([]float32, n)
	wc := make([]float32, n)
	for i, v := range a {
		wa[i] = float32(v)
	}
	for i, v := range b {
		wb[i] = float32(v)
	}
	for i, v := range c {
		wc[i] = float32(v)
	}
	qa := QuantizeQ8Row(wa, 1, n)
	qb := QuantizeQ8Row(wb, 1, n)
	qc := QuantizeQ8Row(wc, 1, n)
	xq := make([]uint8, n)
	scaleX := quantizeXForVNNI(x, xq)

	r0, r1, r2 := dotQ8TripletVNNI(qa.Data[:n], qb.Data[:n], qc.Data[:n], xq, scaleX,
		qa.RowSum[0], qb.RowSum[0], qc.RowSum[0],
		qa.Scale[0], qb.Scale[0], qc.Scale[0])
	s0 := dotQ8(qa.Data[:n], x)
	s1 := dotQ8(qb.Data[:n], x)
	s2 := dotQ8(qc.Data[:n], x)

	if math.Abs(float64(r0-s0)) > 0.5 || math.Abs(float64(r1-s1)) > 0.5 || math.Abs(float64(r2-s2)) > 0.5 {
		t.Errorf("triplet: VNNI=(%f,%f,%f) ref=(%f,%f,%f)", r0, r1, r2, s0, s1, s2)
	}
}

func TestMatVecQ8VNNIvsRef(t *testing.T) {
	rows, cols := 64, 256
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ8Row(w, rows, cols)

	out := make([]float32, rows)
	ref := make([]float32, rows)
	MatVecQ8(out, x, q)

	for r := 0; r < rows; r++ {
		base := r * cols
		ref[r] = dotQ8(q.Data[base:base+cols], x) * q.Scale[r]
	}

	for r := range out {
		diff := math.Abs(float64(out[r] - ref[r]))
		if diff > math.Abs(float64(ref[r]))*0.02+0.01 {
			t.Errorf("row %d: VNNI=%f ref=%f diff=%f", r, out[r], ref[r], diff)
		}
	}
}

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

func TestQuantizeQ8RowDebug(t *testing.T) {
	w := []float32{-8, -7, -6, -5, -4, -3, -2, -1}
	data := make([]int8, 8)
	scale := QuantizeQ8RowInto(w, data)
	expected := []int8{-127, -111, -95, -79, -63, -47, -32, -16}
	for i, v := range data {
		if math.Abs(float64(v)-float64(expected[i])) > 2 {
			t.Errorf("data[%d] = %d, expected ~%d (scale=%f)", i, v, expected[i], scale)
		}
	}
}
