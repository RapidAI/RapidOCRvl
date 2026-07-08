package tensor

import (
	"math"
	"runtime"
	"sync"
	"unsafe"
)

const parallelWork = 1 << 18

func MatVec(out, x, w []float32, rows, cols int) {
	if cols == 16 {
		matVecSerial16(out, x, w, rows)
		return
	}
	if cols == 256 {
		if shouldParallelMatVec(rows*cols, rows) {
			parallelForMatVec(rows, func(start, end int) {
				for r := start; r < end; r++ {
					base := r * 256
					out[r] = dotF32_256(w[base:base+256], x)
				}
			})
			return
		}
		matVecSerial256(out, x, w, rows)
		return
	}
	if shouldParallelMatVec(rows*cols, rows) {
		parallelForMatVec(rows, func(start, end int) {
			for r := start; r < end; r++ {
				out[r] = dotF32(w[r*cols:(r+1)*cols], x)
			}
		})
		return
	}
	for r := 0; r < rows; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x)
	}
}

func FusedMatVec3(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) {
	totalRows := rowsA + rowsB + rowsC
	if rowsA == rowsB && rowsB == rowsC {
		if shouldParallelFusedMatVec3EqualRows(totalRows*cols, rowsA) {
			parallelForFusedMatVec3EqualRows(rowsA, func(start, end int) {
				fusedMatVec3EqualRowsSerial(outA, outB, outC, x, wa, wb, wc, cols, start, end)
			})
			return
		}
		if totalRows*cols < parallelWork {
			fusedMatVec3EqualRowsSerial(outA, outB, outC, x, wa, wb, wc, cols, 0, rowsA)
			return
		}
	}
	if shouldParallel(totalRows*cols, totalRows) {
		parallelFor(totalRows, func(start, end int) {
			fusedMatVec3Serial(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, cols, start, end)
		})
		return
	}
	fusedMatVec3Serial(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, cols, 0, totalRows)
}

func shouldParallelFusedMatVec3EqualRows(work, rows int) bool {
	if work >= parallelWork/2 && rows >= 256 && rows <= 512 {
		return shouldParallel(parallelWork, rows)
	}
	return false
}

func parallelForFusedMatVec3EqualRows(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func fusedMatVec3EqualRowsSerial(outA, outB, outC, x, wa, wb, wc []float32, cols, start, end int) {
	for r := start; r < end; r++ {
		base := r * cols
		outA[r], outB[r], outC[r] = dotF32Triplet(wa[base:base+cols], wb[base:base+cols], wc[base:base+cols], x)
	}
}

func fusedMatVec3Serial(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, cols, start, end int) {
	splitB := rowsA + rowsB
	aEnd := min(end, rowsA)
	for r := start; r < aEnd; r++ {
		outA[r] = dotF32(wa[r*cols:(r+1)*cols], x)
	}
	bStart := max(start, rowsA)
	bEnd := min(end, splitB)
	for r := bStart; r < bEnd; r++ {
		br := r - rowsA
		outB[br] = dotF32(wb[br*cols:(br+1)*cols], x)
	}
	cStart := max(start, splitB)
	for r := cStart; r < end; r++ {
		cr := r - splitB
		outC[cr] = dotF32(wc[cr*cols:(cr+1)*cols], x)
	}
}

func MatVecBias(out, x, w, bias []float32, rows, cols int) {
	if cols == 16 {
		matVecBiasSerial16(out, x, w, bias, rows)
		return
	}
	if cols == 256 {
		if shouldParallelMatVec(rows*cols, rows) {
			parallelForMatVec(rows, func(start, end int) {
				for r := start; r < end; r++ {
					base := r * 256
					out[r] = dotF32_256(w[base:base+256], x) + bias[r]
				}
			})
			return
		}
		matVecBiasSerial256(out, x, w, bias, rows)
		return
	}
	if shouldParallelMatVec(rows*cols, rows) {
		parallelForMatVec(rows, func(start, end int) {
			for r := start; r < end; r++ {
				out[r] = dotF32(w[r*cols:(r+1)*cols], x) + bias[r]
			}
		})
		return
	}
	matVecBiasSerial(out, x, w, bias, rows, cols)
}

func MatVecBiasSerial(out, x, w, bias []float32, rows, cols int) {
	if cols == 16 {
		matVecBiasSerial16(out, x, w, bias, rows)
		return
	}
	if cols == 256 {
		matVecBiasSerial256(out, x, w, bias, rows)
		return
	}
	if cols == 588 {
		matVecBiasSerial588(out, x, w, bias, rows)
		return
	}
	matVecBiasSerial(out, x, w, bias, rows, cols)
}

func MatRowsBias(out [][]float32, xs [][]float32, w, bias []float32, rows, cols int) {
	work := len(xs) * rows * cols
	if !shouldParallel(work, len(xs)) || len(xs) == 1 {
		if cols == 16 {
			for i := range xs {
				matVecBiasSerial16(out[i], xs[i], w, bias, rows)
			}
			return
		}
		if cols == 256 && len(xs) > 1 {
			for i := range xs {
				matVecBiasSerial256(out[i], xs[i], w, bias, rows)
			}
			return
		}
		if cols == 588 {
			for i := range xs {
				matVecBiasSerial588(out[i], xs[i], w, bias, rows)
			}
			return
		}
		for i := range xs {
			if len(xs) == 1 {
				MatVecBias(out[i], xs[i], w, bias, rows, cols)
			} else {
				matVecBiasSerial(out[i], xs[i], w, bias, rows, cols)
			}
		}
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if rows*cols < 1<<16 {
		workers = min(workers, 8)
	}
	if cols == 16 {
		parallelForMax(len(xs), workers, func(start, end int) {
			for i := start; i < end; i++ {
				matVecBiasSerial16(out[i], xs[i], w, bias, rows)
			}
		})
		return
	}
	if cols == 256 {
		parallelForMax(len(xs), workers, func(start, end int) {
			for i := start; i < end; i++ {
				matVecBiasSerial256(out[i], xs[i], w, bias, rows)
			}
		})
		return
	}
	if cols == 588 {
		parallelForMax(len(xs), workers, func(start, end int) {
			for i := start; i < end; i++ {
				matVecBiasSerial588(out[i], xs[i], w, bias, rows)
			}
		})
		return
	}
	parallelForMax(len(xs), workers, func(start, end int) {
		for i := start; i < end; i++ {
			matVecBiasSerial(out[i], xs[i], w, bias, rows, cols)
		}
	})
}

func MatRowsBiasAddRows(out [][]float32, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) {
	if len(add) == 0 {
		MatRowsBias(out, xs, w, bias, rows, cols)
		return
	}
	if cols != 16 && cols != 588 {
		MatRowsBias(out, xs, w, bias, rows, cols)
		if len(add) == len(xs) {
			for i := range xs {
				AddInPlace(out[i], add[i])
			}
			return
		}
		addIdx := 0
		for i := range xs {
			AddInPlace(out[i], add[addIdx])
			addIdx++
			if addIdx == len(add) {
				addIdx = 0
			}
		}
		return
	}
	work := len(xs) * rows * cols
	if !shouldParallel(work, len(xs)) || len(xs) == 1 {
		if cols == 16 {
			matRowsBiasAddRowsSerial16(out, xs, w, bias, add, rows, 0, len(xs))
			return
		}
		matRowsBiasAddRowsSerial588(out, xs, w, bias, add, rows, 0, len(xs))
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if cols == 16 {
		parallelForMax(len(xs), min(workers, 8), func(start, end int) {
			matRowsBiasAddRowsSerial16(out, xs, w, bias, add, rows, start, end)
		})
		return
	}
	parallelForMax(len(xs), workers, func(start, end int) {
		matRowsBiasAddRowsSerial588(out, xs, w, bias, add, rows, start, end)
	})
}

func matRowsBiasAddRowsSerial16(out [][]float32, xs [][]float32, w, bias []float32, add [][]float32, rows, start, end int) {
	if len(add) == len(xs) {
		for i := start; i < end; i++ {
			matVecBiasAddSerial16(out[i], xs[i], w, bias, add[i], rows)
		}
		return
	}
	addIdx := start % len(add)
	for i := start; i < end; i++ {
		matVecBiasAddSerial16(out[i], xs[i], w, bias, add[addIdx], rows)
		addIdx++
		if addIdx == len(add) {
			addIdx = 0
		}
	}
}

func matRowsBiasAddRowsSerial588(out [][]float32, xs [][]float32, w, bias []float32, add [][]float32, rows, start, end int) {
	if len(add) == len(xs) {
		for i := start; i < end; i++ {
			matVecBiasAddSerial588(out[i], xs[i], w, bias, add[i], rows)
		}
		return
	}
	addIdx := start % len(add)
	for i := start; i < end; i++ {
		matVecBiasAddSerial588(out[i], xs[i], w, bias, add[addIdx], rows)
		addIdx++
		if addIdx == len(add) {
			addIdx = 0
		}
	}
}

func MatRowsBias3(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) {
	if len(xs) == 0 {
		return
	}
	work := len(xs) * (rowsA + rowsB + rowsC) * cols
	if rowsA == rowsB && rowsB == rowsC {
		if !shouldParallel(work, len(xs)) || len(xs) == 1 {
			for i := range xs {
				matVecBias3EqualRowsSerial(outA[i], outB[i], outC[i], xs[i], wa, ba, wb, bb, wc, bc, rowsA, cols)
			}
			return
		}
		parallelFor(len(xs), func(start, end int) {
			for i := start; i < end; i++ {
				matVecBias3EqualRowsSerial(outA[i], outB[i], outC[i], xs[i], wa, ba, wb, bb, wc, bc, rowsA, cols)
			}
		})
		return
	}
	if !shouldParallel(work, len(xs)) || len(xs) == 1 {
		for i := range xs {
			matVecBias3Serial(outA[i], outB[i], outC[i], xs[i], wa, ba, wb, bb, wc, bc, rowsA, rowsB, rowsC, cols)
		}
		return
	}
	parallelFor(len(xs), func(start, end int) {
		for i := start; i < end; i++ {
			matVecBias3Serial(outA[i], outB[i], outC[i], xs[i], wa, ba, wb, bb, wc, bc, rowsA, rowsB, rowsC, cols)
		}
	})
}

func matVecBias3EqualRowsSerial(outA, outB, outC, x, wa, ba, wb, bb, wc, bc []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		base := r * cols
		a, b, c := dotF32Triplet(wa[base:base+cols], wb[base:base+cols], wc[base:base+cols], x)
		outA[r] = a + ba[r]
		outB[r] = b + bb[r]
		outC[r] = c + bc[r]
	}
}

func matVecBias3Serial(outA, outB, outC, x, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) {
	totalRows := rowsA + rowsB + rowsC
	splitB := rowsA + rowsB
	for r := 0; r < rowsA; r++ {
		outA[r] = dotF32(wa[r*cols:(r+1)*cols], x) + ba[r]
	}
	for r := rowsA; r < splitB; r++ {
		br := r - rowsA
		outB[br] = dotF32(wb[br*cols:(br+1)*cols], x) + bb[br]
	}
	for r := splitB; r < totalRows; r++ {
		cr := r - splitB
		outC[cr] = dotF32(wc[cr*cols:(cr+1)*cols], x) + bc[cr]
	}
}

func FusedSwiGLUF32Scratch(out, x, gate, up, down []float32, rows, cols, outRows int, tmpG []float32) {
	FusedSwiGLUF32ScratchWithU(out, x, gate, up, down, rows, cols, outRows, tmpG, nil)
}

// FusedSwiGLUF32ScratchWithU computes SiLU(gate*x)*up*x then down*x.
// If tmpU is non-nil and large enough, it batches SiLU via the AVX2 kernel.
func FusedSwiGLUF32ScratchWithU(out, x, gate, up, down []float32, rows, cols, outRows int, tmpG, tmpU []float32) {
	tmpG = tmpG[:rows]
	work := rows * cols * 2
	if shouldParallel(work, rows) {
		parallelForGateUp(rows, func(start, end int) {
			fusedSwiGLUGateUpSerialBatched(tmpG, tmpU, x, gate, up, start, end, cols)
		})
	} else {
		fusedSwiGLUGateUpSerialBatched(tmpG, tmpU, x, gate, up, 0, rows, cols)
	}
	MatVec(out, tmpG, down, outRows, rows)
}

func fusedSwiGLUGateUpSerialBatched(tmpG, tmpU, x, gate, up []float32, start, end, cols int) {
	batchSize := end - start
	if useDotF32AVX && batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		for r := start; r < end; r++ {
			base := r * cols
			g, u := dotF32Pair(gate[base:base+cols], up[base:base+cols], x)
			tmpG[r] = g
			tmpU[r] = u
		}
		SiLUMulInPlace(tmpG[start:end], tmpU[start:end])
		return
	}
	for r := start; r < end; r++ {
		base := r * cols
		g, u := dotF32Pair(gate[base:base+cols], up[base:base+cols], x)
		tmpG[r] = SiLU(g) * u
	}
}

func parallelForGateUp(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func matVecParallel(out, x, w []float32, rows, cols int) {
	parallelFor(rows, func(start, end int) {
		for r := start; r < end; r++ {
			out[r] = dotF32(w[r*cols:(r+1)*cols], x)
		}
	})
}

func matVecSerial(out, x, w []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x)
	}
}

func matVecSerial16(out, x, w []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 16
		out[r] = dotF32_16(w[base:base+16], x)
	}
}

func matVecSerial256(out, x, w []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 256
		out[r] = dotF32_256(w[base:base+256], x)
	}
}

func matVecBiasParallel(out, x, w, bias []float32, rows, cols int) {
	parallelFor(rows, func(start, end int) {
		for r := start; r < end; r++ {
			out[r] = dotF32(w[r*cols:(r+1)*cols], x) + bias[r]
		}
	})
}

func matVecBiasSerial(out, x, w, bias []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x) + bias[r]
	}
}

func matVecBiasSerial16(out, x, w, bias []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 16
		out[r] = dotF32_16(w[base:base+16], x) + bias[r]
	}
}

func matVecBiasAddSerial16(out, x, w, bias, add []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 16
		out[r] = dotF32_16(w[base:base+16], x) + bias[r] + add[r]
	}
}

func matVecBiasSerial256(out, x, w, bias []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 256
		out[r] = dotF32_256(w[base:base+256], x) + bias[r]
	}
}

func matVecBiasSerial588(out, x, w, bias []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 588
		out[r] = dotF32_588(w[base:base+588], x) + bias[r]
	}
}

func matVecBiasAddSerial588(out, x, w, bias, add []float32, rows int) {
	for r := 0; r < rows; r++ {
		base := r * 588
		out[r] = dotF32_588(w[base:base+588], x) + bias[r] + add[r]
	}
}

func parallelFor(n int, fn func(start, end int)) {
	parallelForMax(n, runtime.GOMAXPROCS(0), fn)
}

func shouldParallel(work, units int) bool {
	if work < parallelWork || units < 4 {
		return false
	}
	return runtime.GOMAXPROCS(0) > 1
}

func shouldParallelMatVec(work, rows int) bool {
	if work >= parallelWork/2 && rows >= 512 {
		return shouldParallel(parallelWork, rows)
	}
	return shouldParallel(work, rows)
}

func parallelForMatVec(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func parallelForMax(n, maxWorkers int, fn func(start, end int)) {
	workers := min(maxWorkers, n)
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * n / workers
		end := (worker + 1) * n / workers
		wg.Add(1)
		go func() {
			fn(start, end)
			wg.Done()
		}()
	}
	fn(0, n/workers)
	wg.Wait()
}

func dotF32(a, b []float32) float32 {
	if useDotFMA && len(a) >= 8 {
		return dotF32FMA(a, b)
	}
	if useDotF32AVX && len(a) >= 8 {
		return dotF32AVX(a, b)
	}
	return dotF32Scalar(a, b)
}

func dotF32Scalar(a, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(a)
	for ; i+15 < n; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func dotF32_16(a, b []float32) float32 {
	if useDotFMA {
		return dotF32FMA(a, b)
	}
	if useDotF32AVX {
		return dotF32AVX(a, b)
	}
	s0 := a[0]*b[0] + a[8]*b[8]
	s1 := a[1]*b[1] + a[9]*b[9]
	s2 := a[2]*b[2] + a[10]*b[10]
	s3 := a[3]*b[3] + a[11]*b[11]
	s4 := a[4]*b[4] + a[12]*b[12]
	s5 := a[5]*b[5] + a[13]*b[13]
	s6 := a[6]*b[6] + a[14]*b[14]
	s7 := a[7]*b[7] + a[15]*b[15]
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

func dotF32_256(a, b []float32) float32 {
	if useDotFMA {
		return dotF32FMA(a, b)
	}
	if useDotF32AVX {
		return dotF32AVX(a, b)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for i := 0; i < 256; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

func dotF32_588(a, b []float32) float32 {
	if useDotFMA {
		return dotF32FMA(a, b)
	}
	if useDotF32AVX {
		return dotF32AVX(a, b)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for i := 0; i < 576; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	sum += a[576]*b[576] + a[577]*b[577] + a[578]*b[578] + a[579]*b[579]
	sum += a[580]*b[580] + a[581]*b[581] + a[582]*b[582] + a[583]*b[583]
	sum += a[584]*b[584] + a[585]*b[585] + a[586]*b[586] + a[587]*b[587]
	return sum
}

func Dot(a, b []float32) float32 {
	return dotF32(a, b)
}

func dotF32Pair(a, b, x []float32) (float32, float32) {
	if useDotFMA && len(x) >= 8 {
		return dotF32PairFMA(a, b, x)
	}
	if useDotF32AVX && len(x) >= 8 {
		return dotF32PairAVX(a, b, x)
	}
	return dotF32PairScalar(a, b, x)
}

func dotF32PairScalar(a, b, x []float32) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += a[i]*x0 + a[i+4]*x4
		a1 += a[i+1]*x1 + a[i+5]*x5
		a2 += a[i+2]*x2 + a[i+6]*x6
		a3 += a[i+3]*x3 + a[i+7]*x7
		b0 += b[i]*x0 + b[i+4]*x4
		b1 += b[i+1]*x1 + b[i+5]*x5
		b2 += b[i+2]*x2 + b[i+6]*x6
		b3 += b[i+3]*x3 + b[i+7]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	for ; i < n; i++ {
		xi := x[i]
		sa += a[i] * xi
		sb += b[i] * xi
	}
	return sa, sb
}

func dotF32Triplet(a, b, c, x []float32) (float32, float32, float32) {
	if useDotFMA && len(x) >= 8 {
		return dotF32TripletFMA(a, b, c, x)
	}
	if useDotF32AVX && len(x) >= 8 {
		return dotF32TripletAVX(a, b, c, x)
	}
	return dotF32TripletScalar(a, b, c, x)
}

func dotF32TripletScalar(a, b, c, x []float32) (float32, float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += a[i]*x0 + a[i+4]*x4
		a1 += a[i+1]*x1 + a[i+5]*x5
		a2 += a[i+2]*x2 + a[i+6]*x6
		a3 += a[i+3]*x3 + a[i+7]*x7
		b0 += b[i]*x0 + b[i+4]*x4
		b1 += b[i+1]*x1 + b[i+5]*x5
		b2 += b[i+2]*x2 + b[i+6]*x6
		b3 += b[i+3]*x3 + b[i+7]*x7
		c0 += c[i]*x0 + c[i+4]*x4
		c1 += c[i+1]*x1 + c[i+5]*x5
		c2 += c[i+2]*x2 + c[i+6]*x6
		c3 += c[i+3]*x3 + c[i+7]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	sc := (c0 + c1) + (c2 + c3)
	for ; i < n; i++ {
		xi := x[i]
		sa += a[i] * xi
		sb += b[i] * xi
		sc += c[i] * xi
	}
	return sa, sb, sc
}

func dotF32Quad(a, b, c, d, x []float32) (float32, float32, float32, float32) {
	if useDotFMA && len(a) >= 8 {
		return dotF32QuadFMA(a, b, c, d, x)
	}
	if useDotF32AVX && len(a) >= 8 {
		s0, s1 := dotF32PairAVX(a, b, x)
		s2, s3 := dotF32PairAVX(c, d, x)
		return s0, s1, s2, s3
	}
	return dotF32QuadScalar(a, b, c, d, x)
}

func dotF32QuadScalar(a, b, c, d, x []float32) (float32, float32, float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += a[i]*x0 + a[i+4]*x4
		a1 += a[i+1]*x1 + a[i+5]*x5
		a2 += a[i+2]*x2 + a[i+6]*x6
		a3 += a[i+3]*x3 + a[i+7]*x7
		b0 += b[i]*x0 + b[i+4]*x4
		b1 += b[i+1]*x1 + b[i+5]*x5
		b2 += b[i+2]*x2 + b[i+6]*x6
		b3 += b[i+3]*x3 + b[i+7]*x7
		c0 += c[i]*x0 + c[i+4]*x4
		c1 += c[i+1]*x1 + c[i+5]*x5
		c2 += c[i+2]*x2 + c[i+6]*x6
		c3 += c[i+3]*x3 + c[i+7]*x7
		d0 += d[i]*x0 + d[i+4]*x4
		d1 += d[i+1]*x1 + d[i+5]*x5
		d2 += d[i+2]*x2 + d[i+6]*x6
		d3 += d[i+3]*x3 + d[i+7]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	sc := (c0 + c1) + (c2 + c3)
	sd := (d0 + d1) + (d2 + d3)
	for ; i < n; i++ {
		xi := x[i]
		sa += a[i] * xi
		sb += b[i] * xi
		sc += c[i] * xi
		sd += d[i] * xi
	}
	return sa, sb, sc, sd
}

func AddScaled(dst, x []float32, scale float32) {
	if useDotF32AVX && len(dst) >= 8 {
		addScaledAVX(dst, x, scale)
		return
	}
	var i int
	n := len(dst)
	for ; i+7 < n; i += 8 {
		dst[i] += scale * x[i]
		dst[i+1] += scale * x[i+1]
		dst[i+2] += scale * x[i+2]
		dst[i+3] += scale * x[i+3]
		dst[i+4] += scale * x[i+4]
		dst[i+5] += scale * x[i+5]
		dst[i+6] += scale * x[i+6]
		dst[i+7] += scale * x[i+7]
	}
	for ; i < n; i++ {
		dst[i] += scale * x[i]
	}
}

func Add(out, a, b []float32) {
	if useDotF32AVX && len(out) >= 8 {
		addAVX(out, a, b)
		return
	}
	var i int
	n := len(out)
	for ; i+7 < n; i += 8 {
		out[i] = a[i] + b[i]
		out[i+1] = a[i+1] + b[i+1]
		out[i+2] = a[i+2] + b[i+2]
		out[i+3] = a[i+3] + b[i+3]
		out[i+4] = a[i+4] + b[i+4]
		out[i+5] = a[i+5] + b[i+5]
		out[i+6] = a[i+6] + b[i+6]
		out[i+7] = a[i+7] + b[i+7]
	}
	for ; i < n; i++ {
		out[i] = a[i] + b[i]
	}
}

func AddInPlace(dst, x []float32) {
	if useDotF32AVX && len(dst) >= 8 {
		addInPlaceAVX(dst, x)
		return
	}
	var i int
	n := len(dst)
	for ; i+7 < n; i += 8 {
		dst[i] += x[i]
		dst[i+1] += x[i+1]
		dst[i+2] += x[i+2]
		dst[i+3] += x[i+3]
		dst[i+4] += x[i+4]
		dst[i+5] += x[i+5]
		dst[i+6] += x[i+6]
		dst[i+7] += x[i+7]
	}
	for ; i < n; i++ {
		dst[i] += x[i]
	}
}

func AddRMSNorm(out, dst, add, weight []float32, eps float32) {
	n := len(dst)
	var ss float32
	if useDotFMA && n >= 8 {
		ss = addInPlaceSumSquaresFMA(dst, add)
	} else if useDotF32AVX && n >= 8 {
		ss = addInPlaceSumSquaresAVX(dst, add)
	} else {
		ss = addInPlaceSumSquaresScalar(dst, add)
	}
	scale := float32(1 / math.Sqrt(float64(ss)/float64(n)+float64(eps)))
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(out, dst, weight, scale)
		return
	}
	for i := 0; i < n; i++ {
		out[i] = dst[i] * scale * weight[i]
	}
}

func AddLayerNorm(out, dst, add, weight, bias []float32, eps float32) {
	n := len(dst)
	var sum float32
	if useDotF32AVX && n >= 8 {
		sum = addInPlaceSumAVX(dst, add)
	} else {
		sum = addInPlaceSumScalar(dst, add)
	}
	mean := sum / float32(n)
	var variance float32
	if useDotF32AVX && n >= 8 {
		variance = sumSquaresCenteredAVX(dst, mean)
	} else {
		variance = sumSquaresCenteredScalar(dst, mean)
	}
	scale := float32(1 / math.Sqrt(float64(variance)/float64(n)+float64(eps)))
	if useDotF32AVX && n >= 8 {
		affineNormAVX(out, dst, weight, bias, mean, scale)
		return
	}
	for i := 0; i < n; i++ {
		out[i] = (dst[i]-mean)*scale*weight[i] + bias[i]
	}
}
func AddThenLayerNorm(out, dst, add, weight, bias []float32, eps float32) {
	AddLayerNorm(out, dst, add, weight, bias, eps)
}

func LayerNormRows(out, x [][]float32, weight, bias []float32, eps float32) {
	if len(x) == 0 {
		return
	}
	if shouldParallelNormRows(len(x)*len(x[0]), len(x)) {
		parallelForNormRows(len(x), func(start, end int) {
			for i := start; i < end; i++ {
				LayerNorm(out[i], x[i], weight, bias, eps)
			}
		})
		return
	}
	for i := range x {
		LayerNorm(out[i], x[i], weight, bias, eps)
	}
}

func AddThenLayerNormRows(out, dst, add [][]float32, weight, bias []float32, eps float32) {
	if len(dst) == 0 {
		return
	}
	if shouldParallelNormRows(len(dst)*len(dst[0]), len(dst)) {
		parallelForNormRows(len(dst), func(start, end int) {
			for i := start; i < end; i++ {
				AddThenLayerNorm(out[i], dst[i], add[i], weight, bias, eps)
			}
		})
		return
	}
	for i := range dst {
		AddThenLayerNorm(out[i], dst[i], add[i], weight, bias, eps)
	}
}

func shouldParallelNormRows(work, rows int) bool {
	return work >= 1<<16 && rows >= 16 && runtime.GOMAXPROCS(0) > 1
}

func parallelForNormRows(rows int, fn func(start, end int)) {
	parallelForMax(rows, min(runtime.GOMAXPROCS(0), 8), fn)
}

func RMSNorm(out, x, weight []float32, eps float32) {
	n := len(x)
	var ss float32
	if useDotFMA && n >= 8 {
		ss = sumSquaresF32FMA(x)
	} else if useDotF32AVX && n >= 8 {
		ss = sumSquaresF32AVX(x)
	} else {
		ss = sumSquaresScalar(x)
	}
	scale := float32(1 / math.Sqrt(float64(ss)/float64(n)+float64(eps)))
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(out, x, weight, scale)
		return
	}
	for i := 0; i < n; i++ {
		out[i] = x[i] * scale * weight[i]
	}
}

func sumSquaresScalar(x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		s0 += x[i] * x[i]
		s1 += x[i+1] * x[i+1]
		s2 += x[i+2] * x[i+2]
		s3 += x[i+3] * x[i+3]
		s4 += x[i+4] * x[i+4]
		s5 += x[i+5] * x[i+5]
		s6 += x[i+6] * x[i+6]
		s7 += x[i+7] * x[i+7]
	}
	ss := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		ss += x[i] * x[i]
	}
	return ss
}

func LayerNorm(out, x, weight, bias []float32, eps float32) {
	n := len(x)
	var sum float32
	if useDotF32AVX && n >= 8 {
		sum = sumF32AVX(x)
	} else {
		sum = sumF32Scalar(x)
	}
	mean := sum / float32(n)
	var variance float32
	if useDotF32AVX && n >= 8 {
		variance = sumSquaresCenteredAVX(x, mean)
	} else {
		variance = sumSquaresCenteredScalar(x, mean)
	}
	scale := float32(1 / math.Sqrt(float64(variance)/float64(n)+float64(eps)))
	if useDotF32AVX && n >= 8 {
		affineNormAVX(out, x, weight, bias, mean, scale)
		return
	}
	for i := 0; i < n; i++ {
		out[i] = (x[i]-mean)*scale*weight[i] + bias[i]
	}
}
func SiLU(x float32) float32 {
	return x / (1 + fastExpF32(-x))
}

// fastExpF32 computes exp(x) using range reduction to 2^f polynomial.
// Uses the same 5th-degree polynomial as the AVX2 expF32VecAVX kernel,
// ensuring bit-exact consistency between scalar and vectorized paths.
func fastExpF32(x float32) float32 {
	if x > 88.0 {
		return float32(math.MaxFloat32)
	}
	if x < -88.0 {
		return 0
	}
	// exp(x) = 2^(x * log2(e))
	const log2e = float32(1.4426950408889634)
	t := x * log2e
	n := int32(math.RoundToEven(float64(t)))
	f := t - float32(n)
	// 2^f polynomial (Estrin scheme, degree 5)
	// poly = c1 + c2*f + c3*f^2 + c4*f^3 + c5*f^4
	//      = A + f^2 * (B + f^2 * c5)
	// where A = c1 + c2*f, B = c3 + c4*f
	const c1 = float32(0.6931471824645996)
	const c2 = float32(0.24022650718688965)
	const c3 = float32(0.05550410971045494)
	const c4 = float32(0.009618128649890423)
	const c5 = float32(0.0013352063251659274)
	f2 := f * f
	a := float32(1.0) + c1*f
	b := c2 + c3*f
	c := c4 + c5*f
	poly := a + f2*(b+c*f2)
	// Scale by 2^n
	bits := math.Float32bits(poly) + uint32(n)<<23
	return *(*float32)(unsafe.Pointer(&bits))
}

// fastSiLU approximates x*sigmoid(x) using fastExpF32.
// For x >= 0: silu = x / (1 + exp(-x))
// For x < 0: silu = x * exp(x) / (1 + exp(x))
func fastSiLU(x float32) float32 {
	if x >= 0 {
		return x / (1 + fastExpF32(-x))
	}
	ex := fastExpF32(x)
	return x * ex / (1 + ex)
}

func SiLUMulInPlace(gate, up []float32) {
	if useDotF32AVX && len(gate) >= 8 {
		siluMulInPlaceAVX(gate, up)
		return
	}
	n := min(len(gate), len(up))
	for i := 0; i < n; i++ {
		gate[i] = SiLU(gate[i]) * up[i]
	}
}

func GELUTanh(x float32) float32 {
	// Fused exp_input: t = 2*c*log2e * x + 2*c*k*log2e * x^3
	// = gelu2cLog2e * x + gelu2ckLog2e * x^3
	const gelu2cLog2e = float32(2.0 * 0.7978845608028654 * 1.4426950408889634)
	const gelu2ckLog2e = float32(2.0 * 0.7978845608028654 * 0.044715 * 1.4426950408889634)
	x3 := x * (x * x)
	t := gelu2cLog2e*x + gelu2ckLog2e*x3
	e2x := fastExpF32T(t)
	tanh := (e2x - 1) / (e2x + 1)
	return 0.5 * x * (1 + tanh)
}

// fastExpF32T computes exp(x) where x is already scaled by log2(e).
// i.e., x = original_input * log2e. This skips the log2e multiply.
func fastExpF32T(x float32) float32 {
	if x > 88.0*1.4426950 {
		return float32(math.MaxFloat32)
	}
	if x < -88.0*1.4426950 {
		return 0
	}
	n := int32(math.RoundToEven(float64(x)))
	f := x - float32(n)
	const c1 = float32(0.6931471824645996)
	const c2 = float32(0.24022650718688965)
	const c3 = float32(0.05550410971045494)
	const c4 = float32(0.009618128649890423)
	const c5 = float32(0.0013352063251659274)
	f2 := f * f
	a := float32(1.0) + c1*f
	b := c2 + c3*f
	c := c4 + c5*f
	poly := a + f2*(b+c*f2)
	bits := math.Float32bits(poly) + uint32(n)<<23
	return *(*float32)(unsafe.Pointer(&bits))
}

func GELUTanhInPlace(x []float32) {
	if useDotF32AVX && len(x) >= 8 {
		geluTanhAVX(x)
		return
	}
	n := len(x)
	for i := 0; i < n; i++ {
		x[i] = GELUTanh(x[i])
	}
}

func GELUTanhRowsInPlace(x [][]float32) {
	if len(x) > 1 && shouldParallel(len(x)*len(x[0]), len(x)) {
		workers := runtime.GOMAXPROCS(0)
		if len(x[0]) < 1024 {
			workers = min(workers, 8)
		}
		parallelForMax(len(x), workers, func(start, end int) {
			for i := start; i < end; i++ {
				GELUTanhInPlace(x[i])
			}
		})
		return
	}
	for i := range x {
		GELUTanhInPlace(x[i])
	}
}

func SoftmaxInPlace(x []float32) {
	switch n := len(x); n {
	case 1:
		x[0] = 1
		return
	case 2:
		a, b := x[0], x[1]
		if a >= b {
			e := float32(math.Exp(float64(b - a)))
			inv := 1 / (1 + e)
			x[0] = inv
			x[1] = e * inv
		} else {
			e := float32(math.Exp(float64(a - b)))
			inv := 1 / (1 + e)
			x[0] = e * inv
			x[1] = inv
		}
		return
	case 3:
		a, b, c := x[0], x[1], x[2]
		m := max32(max32(a, b), c)
		e0 := float32(math.Exp(float64(a - m)))
		e1 := float32(math.Exp(float64(b - m)))
		e2 := float32(math.Exp(float64(c - m)))
		inv := 1 / (e0 + e1 + e2)
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		return
	case 4:
		a, b, c, d := x[0], x[1], x[2], x[3]
		m := max32(max32(a, b), max32(c, d))
		e0 := float32(math.Exp(float64(a - m)))
		e1 := float32(math.Exp(float64(b - m)))
		e2 := float32(math.Exp(float64(c - m)))
		e3 := float32(math.Exp(float64(d - m)))
		inv := 1 / ((e0 + e1) + (e2 + e3))
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		x[3] = e3 * inv
		return
	case 5:
		a0, a1, a2, a3, a4 := x[0], x[1], x[2], x[3], x[4]
		m := max32(max32(max32(a0, a1), max32(a2, a3)), a4)
		e0 := float32(math.Exp(float64(a0 - m)))
		e1 := float32(math.Exp(float64(a1 - m)))
		e2 := float32(math.Exp(float64(a2 - m)))
		e3 := float32(math.Exp(float64(a3 - m)))
		e4 := float32(math.Exp(float64(a4 - m)))
		inv := 1 / ((e0 + e1) + (e2 + e3) + e4)
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		x[3] = e3 * inv
		x[4] = e4 * inv
		return
	case 6:
		a0, a1, a2, a3 := x[0], x[1], x[2], x[3]
		a4, a5 := x[4], x[5]
		m := max32(max32(max32(a0, a1), max32(a2, a3)), max32(a4, a5))
		e0 := float32(math.Exp(float64(a0 - m)))
		e1 := float32(math.Exp(float64(a1 - m)))
		e2 := float32(math.Exp(float64(a2 - m)))
		e3 := float32(math.Exp(float64(a3 - m)))
		e4 := float32(math.Exp(float64(a4 - m)))
		e5 := float32(math.Exp(float64(a5 - m)))
		inv := 1 / ((e0 + e1) + (e2 + e3) + (e4 + e5))
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		x[3] = e3 * inv
		x[4] = e4 * inv
		x[5] = e5 * inv
		return
	case 7:
		a0, a1, a2, a3 := x[0], x[1], x[2], x[3]
		a4, a5, a6 := x[4], x[5], x[6]
		m := max32(max32(max32(a0, a1), max32(a2, a3)), max32(max32(a4, a5), a6))
		e0 := float32(math.Exp(float64(a0 - m)))
		e1 := float32(math.Exp(float64(a1 - m)))
		e2 := float32(math.Exp(float64(a2 - m)))
		e3 := float32(math.Exp(float64(a3 - m)))
		e4 := float32(math.Exp(float64(a4 - m)))
		e5 := float32(math.Exp(float64(a5 - m)))
		e6 := float32(math.Exp(float64(a6 - m)))
		inv := 1 / ((e0 + e1) + (e2 + e3) + (e4 + e5) + e6)
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		x[3] = e3 * inv
		x[4] = e4 * inv
		x[5] = e5 * inv
		x[6] = e6 * inv
		return
	case 8:
		a0, a1, a2, a3 := x[0], x[1], x[2], x[3]
		a4, a5, a6, a7 := x[4], x[5], x[6], x[7]
		m := max32(max32(max32(a0, a1), max32(a2, a3)), max32(max32(a4, a5), max32(a6, a7)))
		e0 := float32(math.Exp(float64(a0 - m)))
		e1 := float32(math.Exp(float64(a1 - m)))
		e2 := float32(math.Exp(float64(a2 - m)))
		e3 := float32(math.Exp(float64(a3 - m)))
		e4 := float32(math.Exp(float64(a4 - m)))
		e5 := float32(math.Exp(float64(a5 - m)))
		e6 := float32(math.Exp(float64(a6 - m)))
		e7 := float32(math.Exp(float64(a7 - m)))
		inv := 1 / ((e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7))
		x[0] = e0 * inv
		x[1] = e1 * inv
		x[2] = e2 * inv
		x[3] = e3 * inv
		x[4] = e4 * inv
		x[5] = e5 * inv
		x[6] = e6 * inv
		x[7] = e7 * inv
		return
	}
	n := len(x)
	m := float32(math.Inf(-1))
	if useDotF32AVX && n >= 8 {
		m = maxF32AVX2(x)
	} else {
		var m0, m1, m2, m3, m4, m5, m6, m7 = m, m, m, m, m, m, m, m
		i := 0
		for ; i+15 < n; i += 16 {
			m0 = max32(m0, x[i])
			m1 = max32(m1, x[i+1])
			m2 = max32(m2, x[i+2])
			m3 = max32(m3, x[i+3])
			m4 = max32(m4, x[i+4])
			m5 = max32(m5, x[i+5])
			m6 = max32(m6, x[i+6])
			m7 = max32(m7, x[i+7])
			m0 = max32(m0, x[i+8])
			m1 = max32(m1, x[i+9])
			m2 = max32(m2, x[i+10])
			m3 = max32(m3, x[i+11])
			m4 = max32(m4, x[i+12])
			m5 = max32(m5, x[i+13])
			m6 = max32(m6, x[i+14])
			m7 = max32(m7, x[i+15])
		}
		for ; i+7 < n; i += 8 {
			m0 = max32(m0, x[i])
			m1 = max32(m1, x[i+1])
			m2 = max32(m2, x[i+2])
			m3 = max32(m3, x[i+3])
			m4 = max32(m4, x[i+4])
			m5 = max32(m5, x[i+5])
			m6 = max32(m6, x[i+6])
			m7 = max32(m7, x[i+7])
		}
		m = max32(max32(max32(m0, m1), max32(m2, m3)), max32(max32(m4, m5), max32(m6, m7)))
		for ; i < n; i++ {
			m = max32(m, x[i])
		}
	}
	sum := ExpVec(x, m)
	inv := 1 / sum
	ScaleInPlace(x, inv)
}

func max32(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func Argmax(x []float32) int {
	n := len(x)
	if n == 0 {
		return 0
	}
	if useDotF32AVX && n >= 8 {
		// Use AVX to find the max value, then scalar scan for first index
		bestVal := maxF32AVX2(x)
		for i := 0; i < n; i++ {
			if x[i] == bestVal {
				return i
			}
		}
		return 0
	}
	best := 0
	bestVal := x[0]
	for i := 1; i < n; i++ {
		if x[i] > bestVal {
			best = i
			bestVal = x[i]
		}
	}
	return best
}


func addInPlaceSumSquaresScalar(dst, add []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(dst)
	for ; i+7 < n; i += 8 {
		v0 := dst[i] + add[i]
		v1 := dst[i+1] + add[i+1]
		v2 := dst[i+2] + add[i+2]
		v3 := dst[i+3] + add[i+3]
		v4 := dst[i+4] + add[i+4]
		v5 := dst[i+5] + add[i+5]
		v6 := dst[i+6] + add[i+6]
		v7 := dst[i+7] + add[i+7]
		dst[i] = v0
		dst[i+1] = v1
		dst[i+2] = v2
		dst[i+3] = v3
		dst[i+4] = v4
		dst[i+5] = v5
		dst[i+6] = v6
		dst[i+7] = v7
		s0 += v0 * v0
		s1 += v1 * v1
		s2 += v2 * v2
		s3 += v3 * v3
		s4 += v4 * v4
		s5 += v5 * v5
		s6 += v6 * v6
		s7 += v7 * v7
	}
	ss := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		v := dst[i] + add[i]
		dst[i] = v
		ss += v * v
	}
	return ss
}

func sumF32Scalar(x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		s0 += x[i]
		s1 += x[i+1]
		s2 += x[i+2]
		s3 += x[i+3]
		s4 += x[i+4]
		s5 += x[i+5]
		s6 += x[i+6]
		s7 += x[i+7]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += x[i]
	}
	return sum
}

func sumSquaresCenteredScalar(x []float32, mean float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		d0 := x[i] - mean
		d1 := x[i+1] - mean
		d2 := x[i+2] - mean
		d3 := x[i+3] - mean
		d4 := x[i+4] - mean
		d5 := x[i+5] - mean
		d6 := x[i+6] - mean
		d7 := x[i+7] - mean
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	v := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		d := x[i] - mean
		v += d * d
	}
	return v
}


func addInPlaceSumScalar(dst, add []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(dst)
	for ; i+7 < n; i += 8 {
		v0 := dst[i] + add[i]
		v1 := dst[i+1] + add[i+1]
		v2 := dst[i+2] + add[i+2]
		v3 := dst[i+3] + add[i+3]
		v4 := dst[i+4] + add[i+4]
		v5 := dst[i+5] + add[i+5]
		v6 := dst[i+6] + add[i+6]
		v7 := dst[i+7] + add[i+7]
		dst[i] = v0
		dst[i+1] = v1
		dst[i+2] = v2
		dst[i+3] = v3
		dst[i+4] = v4
		dst[i+5] = v5
		dst[i+6] = v6
		dst[i+7] = v7
		s0 += v0
		s1 += v1
		s2 += v2
		s3 += v3
		s4 += v4
		s5 += v5
		s6 += v6
		s7 += v7
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		v := dst[i] + add[i]
		dst[i] = v
		sum += v
	}
	return sum
}
