package tensor

import (
	"runtime"
	"testing"
)

func BenchmarkMatVecF32(b *testing.B) {
	rows, cols := 1024, 1024
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(out, x, w, rows, cols)
	}
}

func BenchmarkMatVecF32SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatVecF32(b)
}

func BenchmarkMatRowsBias(b *testing.B) {
	batch, rows, cols := 196, 256, 256
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias(out, xs, w, bias, rows, cols)
	}
}

func BenchmarkMatRowsBias3(b *testing.B) {
	batch, rows, cols := 196, 1024, 1024
	xs := makeRowsBench(batch, cols)
	outA := makeRowsBench(batch, rows)
	outB := makeRowsBench(batch, rows)
	outC := makeRowsBench(batch, rows)
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
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias3(outA, outB, outC, xs, wa, ba, wb, bb, wc, bc, rows, rows, rows, cols)
	}
}

func BenchmarkMatVecQ8(b *testing.B) {
	rows, cols := 1024, 1024
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ8(out, x, q)
	}
}

func BenchmarkMatVecQ8SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatVecQ8(b)
}

func BenchmarkMatVecQ4(b *testing.B) {
	rows, cols := 1024, 1024
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ4Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ4(out, x, q)
	}
}

func BenchmarkMatVecQ6(b *testing.B) {
	rows, cols := 1024, 1024
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	q := QuantizeQ6Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ6(out, x, q)
	}
}

func BenchmarkQuantizeQ8Row(b *testing.B) {
	rows, cols := 4096, 1024
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ8Row(w, rows, cols)
	}
}

func BenchmarkQuantizeQ4Row(b *testing.B) {
	rows, cols := 4096, 1024
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ4Row(w, rows, cols)
	}
}

func BenchmarkQuantizeQ6Row(b *testing.B) {
	rows, cols := 4096, 1024
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ6Row(w, rows, cols)
	}
}

func BenchmarkFusedSwiGLUQ8(b *testing.B) {
	hidden, inter := 1024, 4096
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	tmpU := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, inter, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG, tmpU)
	}
}

func BenchmarkFusedSwiGLUF32(b *testing.B) {
	hidden, inter := 1024, 4096
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	tmpU := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUF32Scratch(out, x, gateW, upW, downW, inter, hidden, hidden, tmpG, tmpU)
	}
}

func BenchmarkFusedSwiGLUQ4(b *testing.B) {
	hidden, inter := 1024, 4096
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	tmpU := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ4Row(gateW, inter, hidden)
	up := QuantizeQ4Row(upW, inter, hidden)
	down := QuantizeQ4Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG, tmpU)
	}
}

func BenchmarkFusedSwiGLUQ6(b *testing.B) {
	hidden, inter := 1024, 4096
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	tmpU := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ6Row(gateW, inter, hidden)
	up := QuantizeQ6Row(upW, inter, hidden)
	down := QuantizeQ6Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG, tmpU)
	}
}

func BenchmarkRMSNorm(b *testing.B) {
	n := 4096
	x := make([]float32, n)
	w := make([]float32, n)
	out := make([]float32, n)
	fillBench(x)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RMSNorm(out, x, w, 1e-6)
	}
}

func BenchmarkAddRMSNorm(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	add := make([]float32, n)
	w := make([]float32, n)
	out := make([]float32, n)
	fillBench(dst)
	fillBench(add)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddRMSNorm(out, dst, add, w, 1e-6)
	}
}

func BenchmarkAddThenRMSNorm(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	add := make([]float32, n)
	w := make([]float32, n)
	out := make([]float32, n)
	fillBench(dst)
	fillBench(add)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddInPlace(dst, add)
		RMSNorm(out, dst, w, 1e-6)
	}
}

func BenchmarkLayerNorm(b *testing.B) {
	n := 4096
	x := make([]float32, n)
	w := make([]float32, n)
	bias := make([]float32, n)
	out := make([]float32, n)
	fillBench(x)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LayerNorm(out, x, w, bias, 1e-6)
	}
}

func BenchmarkAddLayerNorm(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	add := make([]float32, n)
	w := make([]float32, n)
	bias := make([]float32, n)
	out := make([]float32, n)
	fillBench(dst)
	fillBench(add)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddLayerNorm(out, dst, add, w, bias, 1e-6)
	}
}

func BenchmarkAddThenLayerNorm(b *testing.B) {
	n := 4096
	dst := make([]float32, n)
	add := make([]float32, n)
	w := make([]float32, n)
	bias := make([]float32, n)
	out := make([]float32, n)
	fillBench(dst)
	fillBench(add)
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddInPlace(dst, add)
		LayerNorm(out, dst, w, bias, 1e-6)
	}
}

func BenchmarkArgmax(b *testing.B) {
	x := make([]float32, 128000)
	fillBench(x)
	x[len(x)-17] = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Argmax(x)
	}
}

func BenchmarkSoftmaxInPlace(b *testing.B) {
	x := make([]float32, 4096)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
	}
}

func BenchmarkSoftmaxInPlaceLen2(b *testing.B) {
	x := []float32{1.25, -0.75}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1] = 1.25, -0.75
	}
}

func BenchmarkSoftmaxInPlaceLen4(b *testing.B) {
	x := []float32{1.25, -0.75, 0.5, 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1], x[2], x[3] = 1.25, -0.75, 0.5, 2
	}
}

func BenchmarkSoftmaxInPlaceLen5(b *testing.B) {
	x := []float32{1.25, -0.75, 0.5, 2, -1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1], x[2], x[3], x[4] = 1.25, -0.75, 0.5, 2, -1
	}
}

func BenchmarkSoftmaxInPlaceLen6(b *testing.B) {
	x := []float32{1.25, -0.75, 0.5, 2, -1, 0.25}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1], x[2], x[3] = 1.25, -0.75, 0.5, 2
		x[4], x[5] = -1, 0.25
	}
}

func BenchmarkSoftmaxInPlaceLen7(b *testing.B) {
	x := []float32{1.25, -0.75, 0.5, 2, -1, 0.25, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1], x[2], x[3] = 1.25, -0.75, 0.5, 2
		x[4], x[5], x[6] = -1, 0.25, 1
	}
}

func BenchmarkSoftmaxInPlaceLen8(b *testing.B) {
	x := []float32{1.25, -0.75, 0.5, 2, -1, 0.25, 1, -0.5}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
		x[0], x[1], x[2], x[3] = 1.25, -0.75, 0.5, 2
		x[4], x[5], x[6], x[7] = -1, 0.25, 1, -0.5
	}
}

func BenchmarkGELUTanhRowsInPlace(b *testing.B) {
	x := makeRowsBench(196, 4096)
	for i := range x {
		fillBench(x[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GELUTanhRowsInPlace(x)
	}
}

func BenchmarkFusedMatVec3Q8(b *testing.B) {
	hidden, qRows, kvRows := 1024, 1024, 256
	x := make([]float32, hidden)
	qw := make([]float32, qRows*hidden)
	kw := make([]float32, kvRows*hidden)
	vw := make([]float32, kvRows*hidden)
	q := make([]float32, qRows)
	k := make([]float32, kvRows)
	v := make([]float32, kvRows)
	fillBench(x)
	fillBench(qw)
	fillBench(kw)
	fillBench(vw)
	qq := QuantizeQ8Row(qw, qRows, hidden)
	qk := QuantizeQ8Row(kw, kvRows, hidden)
	qv := QuantizeQ8Row(vw, kvRows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q8(q, k, v, x, qq, qk, qv)
	}
}

func BenchmarkFusedMatVec3Q8EqualRows(b *testing.B) {
	hidden, rows := 1024, 1024
	x := make([]float32, hidden)
	wa := make([]float32, rows*hidden)
	wb := make([]float32, rows*hidden)
	wc := make([]float32, rows*hidden)
	a := make([]float32, rows)
	bb := make([]float32, rows)
	c := make([]float32, rows)
	fillBench(x)
	fillBench(wa)
	fillBench(wb)
	fillBench(wc)
	qa := QuantizeQ8Row(wa, rows, hidden)
	qb := QuantizeQ8Row(wb, rows, hidden)
	qc := QuantizeQ8Row(wc, rows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q8(a, bb, c, x, qa, qb, qc)
	}
}

func BenchmarkFusedMatVec3F32EqualRows(b *testing.B) {
	hidden, rows := 1024, 1024
	x := make([]float32, hidden)
	wa := make([]float32, rows*hidden)
	wb := make([]float32, rows*hidden)
	wc := make([]float32, rows*hidden)
	a := make([]float32, rows)
	bb := make([]float32, rows)
	c := make([]float32, rows)
	fillBench(x)
	fillBench(wa)
	fillBench(wb)
	fillBench(wc)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3(a, bb, c, x, wa, wb, wc, rows, rows, rows, hidden)
	}
}

func BenchmarkFusedMatVec3Q8SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkFusedMatVec3Q8(b)
}

func BenchmarkFusedMatVec3Q6(b *testing.B) {
	hidden, qRows, kvRows := 1024, 1024, 256
	x := make([]float32, hidden)
	qw := make([]float32, qRows*hidden)
	kw := make([]float32, kvRows*hidden)
	vw := make([]float32, kvRows*hidden)
	q := make([]float32, qRows)
	k := make([]float32, kvRows)
	v := make([]float32, kvRows)
	fillBench(x)
	fillBench(qw)
	fillBench(kw)
	fillBench(vw)
	qq := QuantizeQ6Row(qw, qRows, hidden)
	qk := QuantizeQ6Row(kw, kvRows, hidden)
	qv := QuantizeQ6Row(vw, kvRows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q6(q, k, v, x, qq, qk, qv)
	}
}

func BenchmarkFusedMatVec3Q6EqualRows(b *testing.B) {
	hidden, rows := 1024, 1024
	x := make([]float32, hidden)
	wa := make([]float32, rows*hidden)
	wb := make([]float32, rows*hidden)
	wc := make([]float32, rows*hidden)
	a := make([]float32, rows)
	bb := make([]float32, rows)
	c := make([]float32, rows)
	fillBench(x)
	fillBench(wa)
	fillBench(wb)
	fillBench(wc)
	qa := QuantizeQ6Row(wa, rows, hidden)
	qb := QuantizeQ6Row(wb, rows, hidden)
	qc := QuantizeQ6Row(wc, rows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q6(a, bb, c, x, qa, qb, qc)
	}
}

func BenchmarkFusedMatVec3Q4(b *testing.B) {
	hidden, qRows, kvRows := 1024, 1024, 256
	x := make([]float32, hidden)
	qw := make([]float32, qRows*hidden)
	kw := make([]float32, kvRows*hidden)
	vw := make([]float32, kvRows*hidden)
	q := make([]float32, qRows)
	k := make([]float32, kvRows)
	v := make([]float32, kvRows)
	fillBench(x)
	fillBench(qw)
	fillBench(kw)
	fillBench(vw)
	qq := QuantizeQ4Row(qw, qRows, hidden)
	qk := QuantizeQ4Row(kw, kvRows, hidden)
	qv := QuantizeQ4Row(vw, kvRows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q4(q, k, v, x, qq, qk, qv)
	}
}

func BenchmarkFusedMatVec3Q4EqualRows(b *testing.B) {
	hidden, rows := 1024, 1024
	x := make([]float32, hidden)
	wa := make([]float32, rows*hidden)
	wb := make([]float32, rows*hidden)
	wc := make([]float32, rows*hidden)
	a := make([]float32, rows)
	bb := make([]float32, rows)
	c := make([]float32, rows)
	fillBench(x)
	fillBench(wa)
	fillBench(wb)
	fillBench(wc)
	qa := QuantizeQ4Row(wa, rows, hidden)
	qb := QuantizeQ4Row(wb, rows, hidden)
	qc := QuantizeQ4Row(wc, rows, hidden)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3Q4(a, bb, c, x, qa, qb, qc)
	}
}

func fillBench(x []float32) {
	for i := range x {
		x[i] = float32(i%17-8) / 17
	}
}

func makeRowsBench(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	data := make([]float32, rows*cols)
	for i := range out {
		out[i] = data[i*cols : (i+1)*cols]
	}
	return out
}
