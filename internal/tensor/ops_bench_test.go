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

func BenchmarkMatVecF32Medium(b *testing.B) {
	rows, cols := 512, 256
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

func BenchmarkMatVecF32Cols16(b *testing.B) {
	rows, cols := 1024, 16
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(out, x, w, rows, cols)
	}
}

func BenchmarkMatVecF32Cols588(b *testing.B) {
	rows, cols := 1024, 588
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(out, x, w, rows, cols)
	}
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

func BenchmarkMatRowsBiasPatch14(b *testing.B) {
	batch, rows, cols := 196, 1024, 588
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

func BenchmarkMatRowsBiasAddRowsPatch14Separate(b *testing.B) {
	batch, rows, cols := 196, 1024, 588
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias(out, xs, w, bias, rows, cols)
		for j := range out {
			AddInPlace(out[j], add[j])
		}
	}
}

func BenchmarkMatRowsBiasAddRowsPatch14Fused(b *testing.B) {
	batch, rows, cols := 196, 1024, 588
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	}
}

func BenchmarkMatRowsBiasAddRowsCols16Separate(b *testing.B) {
	batch, rows, cols := 196, 1024, 16
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias(out, xs, w, bias, rows, cols)
		for j := range out {
			AddInPlace(out[j], add[j])
		}
	}
}

func BenchmarkMatRowsBiasAddRowsCols16Fused(b *testing.B) {
	batch, rows, cols := 196, 1024, 16
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(batch, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	}
}

func BenchmarkMatRowsBiasAddRowsCols16RepeatedSeparate(b *testing.B) {
	batch, rows, cols := 392, 1024, 16
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(196, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
	}
	for i := range add {
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias(out, xs, w, bias, rows, cols)
		for j := range out {
			AddInPlace(out[j], add[j%len(add)])
		}
	}
}

func BenchmarkMatRowsBiasAddRowsCols16RepeatedFused(b *testing.B) {
	batch, rows, cols := 392, 1024, 16
	xs := makeRowsBench(batch, cols)
	out := makeRowsBench(batch, rows)
	add := makeRowsBench(196, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	for i := range xs {
		fillBench(xs[i])
	}
	for i := range add {
		fillBench(add[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBiasAddRows(out, xs, w, bias, add, rows, cols)
	}
}

func BenchmarkMatVecBiasMedium(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	fillBench(bias)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecBias(out, x, w, bias, rows, cols)
	}
}

func BenchmarkMatVecBiasVisionMLPUp(b *testing.B) {
	rows, cols := 4096, 1024
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	fillBench(bias)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecBias(out, x, w, bias, rows, cols)
	}
}

func BenchmarkMatVecBiasVisionMLPDown(b *testing.B) {
	rows, cols := 1024, 4096
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	fillBench(bias)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecBias(out, x, w, bias, rows, cols)
	}
}

func BenchmarkMatVecBiasSerialCols16(b *testing.B) {
	rows, cols := 1024, 16
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	fillBench(bias)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecBiasSerial(out, x, w, bias, rows, cols)
	}
}

func BenchmarkMatVecBiasSerialCols256(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	fillBench(w)
	fillBench(x)
	fillBench(bias)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecBiasSerial(out, x, w, bias, rows, cols)
	}
}

func BenchmarkMatRowsBiasSingleBatchMedium(b *testing.B) {
	rows, cols := 512, 256
	xs := makeRowsBench(1, cols)
	out := makeRowsBench(1, rows)
	w := make([]float32, rows*cols)
	bias := make([]float32, rows)
	fillBench(w)
	fillBench(bias)
	fillBench(xs[0])
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatRowsBias(out, xs, w, bias, rows, cols)
	}
}

func BenchmarkMatRowsBiasVisionMLPUp196x1024(b *testing.B) {
	batch, rows, cols := 196, 4096, 1024
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

func BenchmarkMatRowsBiasVisionMLPDown196x4096(b *testing.B) {
	batch, rows, cols := 196, 1024, 4096
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

func BenchmarkMatRowsBiasSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatRowsBias(b)
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

func BenchmarkMatRowsBias3SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatRowsBias3(b)
}

func BenchmarkMatRowsBias3Medium(b *testing.B) {
	batch, rows, cols := 64, 256, 256
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

func BenchmarkMatVecQ8Medium(b *testing.B) {
	rows, cols := 512, 256
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

func BenchmarkMatVecQ8BiasMedium(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	bias := make([]float32, rows)
	out := make([]float32, rows)
	for i := range w {
		w[i] = float32(i%17-8) / 17
	}
	for i := range x {
		x[i] = float32(i%13-6) / 13
	}
	for i := range bias {
		bias[i] = float32(i%11-5) / 11
	}
	q := QuantizeQ8Row(w, rows, cols)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecQ8Bias(out, x, q, bias)
	}
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

func BenchmarkMatVecQ4SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatVecQ4(b)
}

func BenchmarkMatVecQ4Medium(b *testing.B) {
	rows, cols := 512, 256
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

func BenchmarkMatVecQ6SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkMatVecQ6(b)
}

func BenchmarkMatVecQ6Medium(b *testing.B) {
	rows, cols := 512, 256
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

func BenchmarkQuantizeQ8RowMedium(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ8Row(w, rows, cols)
	}
}

func BenchmarkQuantizeQ8RowInto256(b *testing.B) {
	w := make([]float32, 256)
	data := make([]int8, len(w))
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ8RowInto(w, data)
	}
}

func BenchmarkMaxAbsFloat32256(b *testing.B) {
	x := make([]float32, 256)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = maxAbsFloat32(x)
	}
}

func BenchmarkQuantizeQ8RowSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkQuantizeQ8Row(b)
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

func BenchmarkQuantizeQ4RowMedium(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ4Row(w, rows, cols)
	}
}

func BenchmarkQuantizeQ4RowInto256(b *testing.B) {
	w := make([]float32, 256)
	data := make([]byte, (len(w)+1)/2)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ4RowInto(w, data)
	}
}

func BenchmarkQuantizeQ4RowSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkQuantizeQ4Row(b)
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

func BenchmarkQuantizeQ6RowMedium(b *testing.B) {
	rows, cols := 512, 256
	w := make([]float32, rows*cols)
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ6Row(w, rows, cols)
	}
}

func BenchmarkQuantizeQ6RowInto256(b *testing.B) {
	w := make([]float32, 256)
	data := make([]byte, PackedQ6Cols(len(w)))
	fillBench(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = QuantizeQ6RowInto(w, data)
	}
}

func BenchmarkQuantizeQ6RowSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkQuantizeQ6Row(b)
}

func BenchmarkFusedSwiGLUQ8(b *testing.B) {
	hidden, inter := 1024, 4096
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter*2)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, inter, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ8SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkFusedSwiGLUQ8(b)
}

func BenchmarkFusedSwiGLUQ8Medium(b *testing.B) {
	hidden, inter := 256, 512
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, inter, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ8FallbackLongUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 1024
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, upRows, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ8FallbackShortUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 256
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+upRows)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, upRows, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
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
		FusedSwiGLUF32ScratchWithU(out, x, gateW, upW, downW, inter, hidden, hidden, tmpG, tmpU)
	}
}

func BenchmarkFusedSwiGLUF32SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkFusedSwiGLUF32(b)
}

func BenchmarkFusedSwiGLUF32Medium(b *testing.B) {
	hidden, inter := 256, 512
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUF32Scratch(out, x, gateW, upW, downW, inter, hidden, hidden, tmpG)
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
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ4Row(gateW, inter, hidden)
	up := QuantizeQ4Row(upW, inter, hidden)
	down := QuantizeQ4Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ4SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkFusedSwiGLUQ4(b)
}

func BenchmarkFusedSwiGLUQ4Medium(b *testing.B) {
	hidden, inter := 256, 512
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ4Row(gateW, inter, hidden)
	up := QuantizeQ4Row(upW, inter, hidden)
	down := QuantizeQ4Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ4FallbackLongUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 1024
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ4Row(gateW, inter, hidden)
	up := QuantizeQ4Row(upW, upRows, hidden)
	down := QuantizeQ4Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ4FallbackShortUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 256
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+upRows)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ4Row(gateW, inter, hidden)
	up := QuantizeQ4Row(upW, upRows, hidden)
	down := QuantizeQ4Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG)
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
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ6Row(gateW, inter, hidden)
	up := QuantizeQ6Row(upW, inter, hidden)
	down := QuantizeQ6Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ6FallbackLongUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 1024
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ6Row(gateW, inter, hidden)
	up := QuantizeQ6Row(upW, upRows, hidden)
	down := QuantizeQ6Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ6FallbackShortUp(b *testing.B) {
	hidden, inter, upRows := 256, 512, 256
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, upRows*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter+upRows)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ6Row(gateW, inter, hidden)
	up := QuantizeQ6Row(upW, upRows, hidden)
	down := QuantizeQ6Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG)
	}
}

func BenchmarkFusedSwiGLUQ6SingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkFusedSwiGLUQ6(b)
}

func BenchmarkFusedSwiGLUQ6Medium(b *testing.B) {
	hidden, inter := 256, 512
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ6Row(gateW, inter, hidden)
	up := QuantizeQ6Row(upW, inter, hidden)
	down := QuantizeQ6Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG)
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

func BenchmarkLayerNormRowsSerial196x1024(b *testing.B) {
	rows, cols := 196, 1024
	x := makeRowsBench(rows, cols)
	out := makeRowsBench(rows, cols)
	w := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range x {
		fillBench(x[i])
	}
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := range x {
			LayerNorm(out[r], x[r], w, bias, 1e-6)
		}
	}
}

func BenchmarkLayerNormRows196x1024(b *testing.B) {
	rows, cols := 196, 1024
	x := makeRowsBench(rows, cols)
	out := makeRowsBench(rows, cols)
	w := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range x {
		fillBench(x[i])
	}
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		LayerNormRows(out, x, w, bias, 1e-6)
	}
}

func BenchmarkAddThenLayerNormRowsSerial196x1024(b *testing.B) {
	rows, cols := 196, 1024
	dst := makeRowsBench(rows, cols)
	add := makeRowsBench(rows, cols)
	out := makeRowsBench(rows, cols)
	w := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range dst {
		fillBench(dst[i])
		fillBench(add[i])
	}
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := range dst {
			AddThenLayerNorm(out[r], dst[r], add[r], w, bias, 1e-6)
		}
	}
}

func BenchmarkAddThenLayerNormRows196x1024(b *testing.B) {
	rows, cols := 196, 1024
	dst := makeRowsBench(rows, cols)
	add := makeRowsBench(rows, cols)
	out := makeRowsBench(rows, cols)
	w := make([]float32, cols)
	bias := make([]float32, cols)
	for i := range dst {
		fillBench(dst[i])
		fillBench(add[i])
	}
	for i := range w {
		w[i] = 1 + float32(i%11)/32
		bias[i] = float32(i%7-3) / 13
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AddThenLayerNormRows(out, dst, add, w, bias, 1e-6)
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

func BenchmarkSoftmaxInPlaceLen196(b *testing.B) {
	x := make([]float32, 196)
	fillBench(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SoftmaxInPlace(x)
	}
}

func BenchmarkSoftmaxInPlaceLen512(b *testing.B) {
	x := make([]float32, 512)
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

func BenchmarkGELUTanhRowsInPlaceSingleProc(b *testing.B) {
	orig := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(orig)
	BenchmarkGELUTanhRowsInPlace(b)
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

func BenchmarkFusedMatVec3Q8Medium(b *testing.B) {
	hidden, qRows, kvRows := 256, 256, 64
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

func BenchmarkFusedMatVec3Q8EqualRowsMedium(b *testing.B) {
	hidden, rows := 256, 256
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

func BenchmarkFusedMatVec3F32Medium(b *testing.B) {
	hidden, qRows, kvRows := 256, 256, 64
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
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedMatVec3(q, k, v, x, qw, kw, vw, qRows, kvRows, kvRows, hidden)
	}
}

func BenchmarkFusedMatVec3F32EqualRowsMedium(b *testing.B) {
	hidden, rows := 256, 256
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

func BenchmarkFusedMatVec3Q6Medium(b *testing.B) {
	hidden, qRows, kvRows := 256, 256, 64
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

func BenchmarkFusedMatVec3Q6EqualRowsMedium(b *testing.B) {
	hidden, rows := 256, 256
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

func BenchmarkFusedMatVec3Q4Medium(b *testing.B) {
	hidden, qRows, kvRows := 256, 256, 64
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

func BenchmarkFusedMatVec3Q4EqualRowsMedium(b *testing.B) {
	hidden, rows := 256, 256
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
func BenchmarkFusedSwiGLUQ8ModelSize(b *testing.B) {
	// Model dimensions: hidden=2048, inter=8192
	hidden, inter := 2048, 8192
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, inter, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
	}
}
func BenchmarkFusedSwiGLUQ8ModelSizeParts(b *testing.B) {
	hidden, inter := 2048, 8192
	x := make([]float32, hidden)
	gateW := make([]float32, inter*hidden)
	upW := make([]float32, inter*hidden)
	downW := make([]float32, hidden*inter)
	out := make([]float32, hidden)
	tmpG := make([]float32, inter*2)
	fillBench(x)
	fillBench(gateW)
	fillBench(upW)
	fillBench(downW)
	gate := QuantizeQ8Row(gateW, inter, hidden)
	up := QuantizeQ8Row(upW, inter, hidden)
	down := QuantizeQ8Row(downW, hidden, inter)
	tmpU := tmpG[inter:]
	b.Run("GateUp", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			matVecQ8SwiGLUScratch(tmpG, x, gate, up, tmpU)
		}
	})
	b.Run("Down", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MatVecQ8(out, tmpG, down)
		}
	})
}
func BenchmarkSiLUMulInPlace8192(b *testing.B) {
	gate := make([]float32, 8192)
	up := make([]float32, 8192)
	fillBench(gate)
	fillBench(up)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SiLUMulInPlace(gate, up)
	}
}
func BenchmarkSwiGLUSeparateMatVecs(b *testing.B) {
	// SwiGLU using 2 separate MatVecQ8 calls + SiLUMulInPlace
	// vs FusedSwiGLUQ8Scratch which uses dotQ8Pair
	hidden, inter := 2048, 8192
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
		MatVecQ8(tmpG, x, gate)
		MatVecQ8(tmpU, x, up)
		SiLUMulInPlace(tmpG, tmpU)
		MatVecQ8(out, tmpG, down)
	}
}
