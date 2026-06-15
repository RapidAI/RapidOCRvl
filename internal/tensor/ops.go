package tensor

import (
	"math"
	"runtime"
	"sync"
)

const parallelWork = 1 << 18

func MatVec(out, x, w []float32, rows, cols int) {
	if rows*cols >= parallelWork && rows >= 4 {
		matVecParallel(out, x, w, rows, cols)
		return
	}
	for r := 0; r < rows; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x)
	}
}

func FusedMatVec3(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) {
	totalRows := rowsA + rowsB + rowsC
	if rowsA == rowsB && rowsB == rowsC && totalRows*cols < parallelWork {
		fusedMatVec3EqualRowsSerial(outA, outB, outC, x, wa, wb, wc, cols, 0, rowsA)
		return
	}
	if totalRows*cols >= parallelWork && totalRows >= 4 {
		parallelFor(totalRows, func(start, end int) {
			fusedMatVec3Serial(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, cols, start, end)
		})
		return
	}
	fusedMatVec3Serial(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, cols, 0, totalRows)
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
	if rows*cols >= parallelWork && rows >= 4 {
		matVecBiasParallel(out, x, w, bias, rows, cols)
		return
	}
	matVecBiasSerial(out, x, w, bias, rows, cols)
}

func MatVecBiasSerial(out, x, w, bias []float32, rows, cols int) {
	matVecBiasSerial(out, x, w, bias, rows, cols)
}

func MatRowsBias(out [][]float32, xs [][]float32, w, bias []float32, rows, cols int) {
	work := len(xs) * rows * cols
	if work < parallelWork || len(xs) == 1 {
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
	parallelForMax(len(xs), workers, func(start, end int) {
		for i := start; i < end; i++ {
			matVecBiasSerial(out[i], xs[i], w, bias, rows, cols)
		}
	})
}

func MatRowsBias3(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) {
	if len(xs) == 0 {
		return
	}
	work := len(xs) * (rowsA + rowsB + rowsC) * cols
	if rowsA == rowsB && rowsB == rowsC {
		if work < parallelWork || len(xs) == 1 {
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
	if work < parallelWork || len(xs) == 1 {
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

func FusedSwiGLUF32Scratch(out, x, gate, up, down []float32, rows, cols, outRows int, tmpG, tmpU []float32) {
	tmpG = tmpG[:rows]
	tmpU = tmpU[:rows]
	work := rows * cols * 2
	if work >= parallelWork && rows >= 4 {
		fusedGateUpParallel(tmpG, tmpU, x, gate, up, rows, cols)
	} else {
		fusedGateUpSerial(tmpG, tmpU, x, gate, up, 0, rows, cols)
	}
	SiLUMulInPlace(tmpG, tmpU)
	MatVec(out, tmpG, down, outRows, rows)
}

func fusedGateUpParallel(tmpG, tmpU, x, gate, up []float32, rows, cols int) {
	parallelFor(rows, func(start, end int) {
		fusedGateUpSerial(tmpG, tmpU, x, gate, up, start, end, cols)
	})
}

func fusedGateUpSerial(tmpG, tmpU, x, gate, up []float32, start, end, cols int) {
	for r := start; r < end; r++ {
		base := r * cols
		tmpG[r], tmpU[r] = dotF32Pair(gate[base:base+cols], up[base:base+cols], x)
	}
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

func parallelFor(n int, fn func(start, end int)) {
	parallelForMax(n, runtime.GOMAXPROCS(0), fn)
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

func Dot(a, b []float32) float32 {
	return dotF32(a, b)
}

func dotF32Pair(a, b, x []float32) (float32, float32) {
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

func AddScaled(dst, x []float32, scale float32) {
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
	scale := float32(1 / math.Sqrt(float64(ss)/float64(n)+float64(eps)))
	i = 0
	for ; i+7 < n; i += 8 {
		out[i] = dst[i] * scale * weight[i]
		out[i+1] = dst[i+1] * scale * weight[i+1]
		out[i+2] = dst[i+2] * scale * weight[i+2]
		out[i+3] = dst[i+3] * scale * weight[i+3]
		out[i+4] = dst[i+4] * scale * weight[i+4]
		out[i+5] = dst[i+5] * scale * weight[i+5]
		out[i+6] = dst[i+6] * scale * weight[i+6]
		out[i+7] = dst[i+7] * scale * weight[i+7]
	}
	for ; i < n; i++ {
		out[i] = dst[i] * scale * weight[i]
	}
}

func AddLayerNorm(out, dst, add, weight, bias []float32, eps float32) {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	var q0, q1, q2, q3, q4, q5, q6, q7 float32
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
		q0 += v0 * v0
		q1 += v1 * v1
		q2 += v2 * v2
		q3 += v3 * v3
		q4 += v4 * v4
		q5 += v5 * v5
		q6 += v6 * v6
		q7 += v7 * v7
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	sumsq := (q0 + q1) + (q2 + q3) + (q4 + q5) + (q6 + q7)
	for ; i < n; i++ {
		v := dst[i] + add[i]
		dst[i] = v
		sum += v
		sumsq += v * v
	}
	mean := sum / float32(n)
	variance := sumsq/float32(n) - mean*mean
	if variance < 0 {
		variance = 0
	}
	scale := float32(1 / math.Sqrt(float64(variance)+float64(eps)))
	i = 0
	for ; i+7 < n; i += 8 {
		out[i] = (dst[i]-mean)*scale*weight[i] + bias[i]
		out[i+1] = (dst[i+1]-mean)*scale*weight[i+1] + bias[i+1]
		out[i+2] = (dst[i+2]-mean)*scale*weight[i+2] + bias[i+2]
		out[i+3] = (dst[i+3]-mean)*scale*weight[i+3] + bias[i+3]
		out[i+4] = (dst[i+4]-mean)*scale*weight[i+4] + bias[i+4]
		out[i+5] = (dst[i+5]-mean)*scale*weight[i+5] + bias[i+5]
		out[i+6] = (dst[i+6]-mean)*scale*weight[i+6] + bias[i+6]
		out[i+7] = (dst[i+7]-mean)*scale*weight[i+7] + bias[i+7]
	}
	for ; i < n; i++ {
		out[i] = (dst[i]-mean)*scale*weight[i] + bias[i]
	}
}

func AddThenLayerNorm(out, dst, add, weight, bias []float32, eps float32) {
	AddInPlace(dst, add)
	LayerNorm(out, dst, weight, bias, eps)
}

func RMSNorm(out, x, weight []float32, eps float32) {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		s0 += x0 * x0
		s1 += x1 * x1
		s2 += x2 * x2
		s3 += x3 * x3
		s4 += x4 * x4
		s5 += x5 * x5
		s6 += x6 * x6
		s7 += x7 * x7
	}
	ss := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		v := x[i]
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss)/float64(n)+float64(eps)))
	i = 0
	for ; i+7 < n; i += 8 {
		out[i] = x[i] * scale * weight[i]
		out[i+1] = x[i+1] * scale * weight[i+1]
		out[i+2] = x[i+2] * scale * weight[i+2]
		out[i+3] = x[i+3] * scale * weight[i+3]
		out[i+4] = x[i+4] * scale * weight[i+4]
		out[i+5] = x[i+5] * scale * weight[i+5]
		out[i+6] = x[i+6] * scale * weight[i+6]
		out[i+7] = x[i+7] * scale * weight[i+7]
	}
	for ; i < n; i++ {
		out[i] = x[i] * scale * weight[i]
	}
}

func LayerNorm(out, x, weight, bias []float32, eps float32) {
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
	mean := sum / float32(n)
	s0, s1, s2, s3, s4, s5, s6, s7 = 0, 0, 0, 0, 0, 0, 0, 0
	i = 0
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
	variance := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		d := x[i] - mean
		variance += d * d
	}
	scale := float32(1 / math.Sqrt(float64(variance)/float64(n)+float64(eps)))
	i = 0
	for ; i+7 < n; i += 8 {
		out[i] = (x[i]-mean)*scale*weight[i] + bias[i]
		out[i+1] = (x[i+1]-mean)*scale*weight[i+1] + bias[i+1]
		out[i+2] = (x[i+2]-mean)*scale*weight[i+2] + bias[i+2]
		out[i+3] = (x[i+3]-mean)*scale*weight[i+3] + bias[i+3]
		out[i+4] = (x[i+4]-mean)*scale*weight[i+4] + bias[i+4]
		out[i+5] = (x[i+5]-mean)*scale*weight[i+5] + bias[i+5]
		out[i+6] = (x[i+6]-mean)*scale*weight[i+6] + bias[i+6]
		out[i+7] = (x[i+7]-mean)*scale*weight[i+7] + bias[i+7]
	}
	for ; i < n; i++ {
		out[i] = (x[i]-mean)*scale*weight[i] + bias[i]
	}
}

func SiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func SiLUMulInPlace(gate, up []float32) {
	i := 0
	n := len(gate)
	for ; i+7 < n; i += 8 {
		gate[i] = SiLU(gate[i]) * up[i]
		gate[i+1] = SiLU(gate[i+1]) * up[i+1]
		gate[i+2] = SiLU(gate[i+2]) * up[i+2]
		gate[i+3] = SiLU(gate[i+3]) * up[i+3]
		gate[i+4] = SiLU(gate[i+4]) * up[i+4]
		gate[i+5] = SiLU(gate[i+5]) * up[i+5]
		gate[i+6] = SiLU(gate[i+6]) * up[i+6]
		gate[i+7] = SiLU(gate[i+7]) * up[i+7]
	}
	for ; i < n; i++ {
		gate[i] = SiLU(gate[i]) * up[i]
	}
}

func GELUTanh(x float32) float32 {
	const c = 0.7978845608028654
	return 0.5 * x * (1 + float32(math.Tanh(c*float64(x)*(1+0.044715*float64(x*x)))))
}

func GELUTanhInPlace(x []float32) {
	i := 0
	n := len(x)
	for ; i+7 < n; i += 8 {
		x[i] = GELUTanh(x[i])
		x[i+1] = GELUTanh(x[i+1])
		x[i+2] = GELUTanh(x[i+2])
		x[i+3] = GELUTanh(x[i+3])
		x[i+4] = GELUTanh(x[i+4])
		x[i+5] = GELUTanh(x[i+5])
		x[i+6] = GELUTanh(x[i+6])
		x[i+7] = GELUTanh(x[i+7])
	}
	for ; i < n; i++ {
		x[i] = GELUTanh(x[i])
	}
}

func GELUTanhRowsInPlace(x [][]float32) {
	if len(x) > 1 && len(x)*len(x[0]) >= parallelWork {
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
	m := float32(math.Inf(-1))
	i := 0
	n := len(x)
	var m0, m1, m2, m3, m4, m5, m6, m7 = m, m, m, m, m, m, m, m
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
	var sum float32
	i = 0
	for ; i+15 < n; i += 16 {
		e0 := float32(math.Exp(float64(x[i] - m)))
		e1 := float32(math.Exp(float64(x[i+1] - m)))
		e2 := float32(math.Exp(float64(x[i+2] - m)))
		e3 := float32(math.Exp(float64(x[i+3] - m)))
		e4 := float32(math.Exp(float64(x[i+4] - m)))
		e5 := float32(math.Exp(float64(x[i+5] - m)))
		e6 := float32(math.Exp(float64(x[i+6] - m)))
		e7 := float32(math.Exp(float64(x[i+7] - m)))
		e8 := float32(math.Exp(float64(x[i+8] - m)))
		e9 := float32(math.Exp(float64(x[i+9] - m)))
		e10 := float32(math.Exp(float64(x[i+10] - m)))
		e11 := float32(math.Exp(float64(x[i+11] - m)))
		e12 := float32(math.Exp(float64(x[i+12] - m)))
		e13 := float32(math.Exp(float64(x[i+13] - m)))
		e14 := float32(math.Exp(float64(x[i+14] - m)))
		e15 := float32(math.Exp(float64(x[i+15] - m)))
		x[i] = e0
		x[i+1] = e1
		x[i+2] = e2
		x[i+3] = e3
		x[i+4] = e4
		x[i+5] = e5
		x[i+6] = e6
		x[i+7] = e7
		x[i+8] = e8
		x[i+9] = e9
		x[i+10] = e10
		x[i+11] = e11
		x[i+12] = e12
		x[i+13] = e13
		x[i+14] = e14
		x[i+15] = e15
		sum += (e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7) +
			(e8 + e9) + (e10 + e11) + (e12 + e13) + (e14 + e15)
	}
	for ; i+7 < n; i += 8 {
		e0 := float32(math.Exp(float64(x[i] - m)))
		e1 := float32(math.Exp(float64(x[i+1] - m)))
		e2 := float32(math.Exp(float64(x[i+2] - m)))
		e3 := float32(math.Exp(float64(x[i+3] - m)))
		e4 := float32(math.Exp(float64(x[i+4] - m)))
		e5 := float32(math.Exp(float64(x[i+5] - m)))
		e6 := float32(math.Exp(float64(x[i+6] - m)))
		e7 := float32(math.Exp(float64(x[i+7] - m)))
		x[i] = e0
		x[i+1] = e1
		x[i+2] = e2
		x[i+3] = e3
		x[i+4] = e4
		x[i+5] = e5
		x[i+6] = e6
		x[i+7] = e7
		sum += (e0 + e1) + (e2 + e3) + (e4 + e5) + (e6 + e7)
	}
	for ; i < n; i++ {
		v := x[i]
		e := float32(math.Exp(float64(v - m)))
		x[i] = e
		sum += e
	}
	inv := 1 / sum
	i = 0
	for ; i+15 < n; i += 16 {
		x[i] *= inv
		x[i+1] *= inv
		x[i+2] *= inv
		x[i+3] *= inv
		x[i+4] *= inv
		x[i+5] *= inv
		x[i+6] *= inv
		x[i+7] *= inv
		x[i+8] *= inv
		x[i+9] *= inv
		x[i+10] *= inv
		x[i+11] *= inv
		x[i+12] *= inv
		x[i+13] *= inv
		x[i+14] *= inv
		x[i+15] *= inv
	}
	for ; i+7 < n; i += 8 {
		x[i] *= inv
		x[i+1] *= inv
		x[i+2] *= inv
		x[i+3] *= inv
		x[i+4] *= inv
		x[i+5] *= inv
		x[i+6] *= inv
		x[i+7] *= inv
	}
	for ; i < n; i++ {
		x[i] *= inv
	}
}

func max32(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func Argmax(x []float32) int {
	n := len(x)
	best := 0
	bestVal := x[0]
	i := 1
	var i0, i1, i2, i3, i4, i5, i6, i7 int
	var v0, v1, v2, v3, v4, v5, v6, v7 = bestVal, bestVal, bestVal, bestVal, bestVal, bestVal, bestVal, bestVal
	for ; i+7 < n; i += 8 {
		if x[i] > v0 {
			i0 = i
			v0 = x[i]
		}
		if x[i+1] > v1 {
			i1 = i + 1
			v1 = x[i+1]
		}
		if x[i+2] > v2 {
			i2 = i + 2
			v2 = x[i+2]
		}
		if x[i+3] > v3 {
			i3 = i + 3
			v3 = x[i+3]
		}
		if x[i+4] > v4 {
			i4 = i + 4
			v4 = x[i+4]
		}
		if x[i+5] > v5 {
			i5 = i + 5
			v5 = x[i+5]
		}
		if x[i+6] > v6 {
			i6 = i + 6
			v6 = x[i+6]
		}
		if x[i+7] > v7 {
			i7 = i + 7
			v7 = x[i+7]
		}
	}
	if v0 > bestVal {
		best, bestVal = i0, v0
	}
	if v1 > bestVal {
		best, bestVal = i1, v1
	}
	if v2 > bestVal {
		best, bestVal = i2, v2
	}
	if v3 > bestVal {
		best, bestVal = i3, v3
	}
	if v4 > bestVal {
		best, bestVal = i4, v4
	}
	if v5 > bestVal {
		best, bestVal = i5, v5
	}
	if v6 > bestVal {
		best, bestVal = i6, v6
	}
	if v7 > bestVal {
		best, bestVal = i7, v7
	}
	for ; i < n; i++ {
		if x[i] > bestVal {
			best = i
			bestVal = x[i]
		}
	}
	return best
}
