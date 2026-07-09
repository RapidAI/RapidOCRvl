package tensor

import (
	"runtime"
	"sync"
)

// AddRMSNormOutOnly computes out = RMSNorm(dst + add, weight) without
// modifying dst in place. This is the "out-only" variant of AddRMSNorm
// used when the residual must be preserved.
func AddRMSNormOutOnly(out, dst, add, weight []float32, eps float32) {
	n := len(dst)
	var ss float32
	if useDotFMA && n >= 8 {
		ss = addSumSquaresFMA(out, dst, add)
	} else if useDotF32AVX && n >= 8 {
		ss = addSumSquaresAVX(out, dst, add)
	} else {
		ss = addSumSquaresOutScalar(out, dst, add)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(out, out, weight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(out, out, weight, scale)
		return
	}
	for i := 0; i < n; i++ {
		out[i] = out[i] * scale * weight[i]
	}
}

func addSumSquaresOutScalar(out, dst, add []float32) float32 {
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
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
		out[i] = v
		ss += v * v
	}
	return ss
}
// FusedMatVec2 computes outB = x . wB^T and outC = x . wC^T in a single
// pass over x, reducing memory bandwidth for the K and V projections.
func FusedMatVec2(outB, outC, x, wB, wC []float32, rowsB, rowsC, cols int) {
	if rowsB == rowsC {
		if shouldParallel(rowsB*cols*2, rowsB) {
			parallelFor(rowsB, func(start, end int) {
				fusedMatVec2EqualRowsSerial(outB, outC, x, wB, wC, cols, start, end)
			})
			return
		}
		fusedMatVec2EqualRowsSerial(outB, outC, x, wB, wC, cols, 0, rowsB)
		return
	}
	totalRows := rowsB + rowsC
	if shouldParallel(totalRows*cols, totalRows) {
		parallelFor(totalRows, func(start, end int) {
			fusedMatVec2Serial(outB, outC, x, wB, wC, rowsB, cols, start, end)
		})
		return
	}
	fusedMatVec2Serial(outB, outC, x, wB, wC, rowsB, cols, 0, totalRows)
}

func fusedMatVec2EqualRowsSerial(outB, outC, x, wB, wC []float32, cols, start, end int) {
	for r := start; r < end; r++ {
		base := r * cols
		outB[r], outC[r] = dotF32Pair(wB[base:base+cols], wC[base:base+cols], x)
	}
}

func fusedMatVec2Serial(outB, outC, x, wB, wC []float32, rowsB, cols, start, end int) {
	bEnd := min(end, rowsB)
	for r := start; r < bEnd; r++ {
		outB[r] = dotF32(wB[r*cols:(r+1)*cols], x)
	}
	cStart := max(start, rowsB)
	for r := cStart; r < end; r++ {
		cr := r - rowsB
		outC[cr] = dotF32(wC[cr*cols:(cr+1)*cols], x)
	}
}

// MatVecAddRMSNormOutOnly fuses the down-projection matvec with AddRMSNormOutOnly.
// Instead of: out = matvec(x, w); normOut = RMSNorm(residual + out, weight)
// It computes: normOut[i] = (residual[i] + dot(w[row_i], x)) * rsqrt(ss/n+eps) * weight[i]
// where ss = sum((residual[i] + dot)^2). This saves one full pass over the output.
func MatVecAddRMSNormOutOnly(normOut, residual, x, w []float32, rows, cols int, normWeight []float32, eps float32) {
	n := rows
	if n == 0 {
		return
	}
	// Phase 1: compute out[i] = residual[i] + dot(w[row_i], x), accumulate sum of squares
	var ss float32
	if shouldParallelMatVec(rows*cols, rows) {
		ss = matVecAddSumSquaresParallel(normOut, residual, x, w, rows, cols)
	} else {
		ss = matVecAddSumSquaresSerial(normOut, residual, x, w, 0, rows, cols)
	}
	// Phase 2: apply RMSNorm scale
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecAddSumSquaresSerial(out, residual, x, w []float32, start, end, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		v0 := residual[i] + dotF32(w[i*cols:(i+1)*cols], x)
		v1 := residual[i+1] + dotF32(w[(i+1)*cols:(i+2)*cols], x)
		v2 := residual[i+2] + dotF32(w[(i+2)*cols:(i+3)*cols], x)
		v3 := residual[i+3] + dotF32(w[(i+3)*cols:(i+4)*cols], x)
		v4 := residual[i+4] + dotF32(w[(i+4)*cols:(i+5)*cols], x)
		v5 := residual[i+5] + dotF32(w[(i+5)*cols:(i+6)*cols], x)
		v6 := residual[i+6] + dotF32(w[(i+6)*cols:(i+7)*cols], x)
		v7 := residual[i+7] + dotF32(w[(i+7)*cols:(i+8)*cols], x)
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + dotF32(w[i*cols:(i+1)*cols], x)
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecAddSumSquaresParallel(out, residual, x, w []float32, rows, cols int) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 {
		return matVecAddSumSquaresSerial(out, residual, x, w, 0, rows, cols)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > rows {
			end = rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecAddSumSquaresSerial(out, residual, x, w, start, end, cols)
		}(worker, start, end)
	}
	// Run first chunk on caller
	firstEnd := chunk
	if firstEnd > rows {
		firstEnd = rows
	}
	partials[0] = matVecAddSumSquaresSerial(out, residual, x, w, 0, firstEnd, cols)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecAddRMSNorm fuses the down-projection matvec with AddRMSNorm (in-place).
// It computes: residual[i] += dot(w[row_i], x); then normOut[i] = residual[i] * rsqrt(ss/n+eps) * weight[i]
func MatVecAddRMSNorm(normOut, residual, x, w []float32, rows, cols int, normWeight []float32, eps float32) {
	n := rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelMatVec(rows*cols, rows) {
		ss = matVecInPlaceAddSumSquaresParallel(normOut, residual, x, w, rows, cols)
	} else {
		ss = matVecInPlaceAddSumSquaresSerial(normOut, residual, x, w, 0, rows, cols)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecInPlaceAddSumSquaresSerial(out, residual, x, w []float32, start, end, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		v0 := residual[i] + dotF32(w[i*cols:(i+1)*cols], x)
		v1 := residual[i+1] + dotF32(w[(i+1)*cols:(i+2)*cols], x)
		v2 := residual[i+2] + dotF32(w[(i+2)*cols:(i+3)*cols], x)
		v3 := residual[i+3] + dotF32(w[(i+3)*cols:(i+4)*cols], x)
		v4 := residual[i+4] + dotF32(w[(i+4)*cols:(i+5)*cols], x)
		v5 := residual[i+5] + dotF32(w[(i+5)*cols:(i+6)*cols], x)
		v6 := residual[i+6] + dotF32(w[(i+6)*cols:(i+7)*cols], x)
		v7 := residual[i+7] + dotF32(w[(i+7)*cols:(i+8)*cols], x)
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + dotF32(w[i*cols:(i+1)*cols], x)
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecInPlaceAddSumSquaresParallel(out, residual, x, w []float32, rows, cols int) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > rows {
		workers = rows
	}
	if workers <= 1 {
		return matVecInPlaceAddSumSquaresSerial(out, residual, x, w, 0, rows, cols)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > rows {
			end = rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecInPlaceAddSumSquaresSerial(out, residual, x, w, start, end, cols)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > rows {
		firstEnd = rows
	}
	partials[0] = matVecInPlaceAddSumSquaresSerial(out, residual, x, w, 0, firstEnd, cols)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ8AddRMSNorm fuses Q8 down-projection matvec with AddRMSNorm (in-place).
// residual[i] += dot(down[row_i], x); normOut[i] = residual[i] * rsqrt(ss/n+eps) * weight[i]
func MatVecQ8AddRMSNorm(normOut, residual, x []float32, down *Q8Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ8InPlaceAddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ8InPlaceAddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ8InPlaceAddSumSquaresSerial(out, residual, x []float32, q *Q8Matrix, start, end int) float32 {
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		return matVecQ8InPlaceAddSumSquaresVNNISerial(out, residual, x, q, start, end)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		v0 := residual[i] + dotQ8(q.Data[i*q.Cols:(i+1)*q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ8(q.Data[(i+1)*q.Cols:(i+2)*q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ8(q.Data[(i+2)*q.Cols:(i+3)*q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ8(q.Data[(i+3)*q.Cols:(i+4)*q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ8(q.Data[(i+4)*q.Cols:(i+5)*q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ8(q.Data[(i+5)*q.Cols:(i+6)*q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ8(q.Data[(i+6)*q.Cols:(i+7)*q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ8(q.Data[(i+7)*q.Cols:(i+8)*q.Cols], x)*q.Scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + dotQ8(q.Data[i*q.Cols:(i+1)*q.Cols], x)*q.Scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ8InPlaceAddSumSquaresVNNISerial(out, residual, x []float32, q *Q8Matrix, start, end int) float32 {
	xq := getVNNIScratch(q.Cols)
	defer putVNNIScratch(xq)
	scaleX := quantizeXForVNNI(x, xq)
	data := q.Data
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * cols
		b1 := (i + 1) * cols
		b2 := (i + 2) * cols
		b3 := (i + 3) * cols
		b4 := (i + 4) * cols
		b5 := (i + 5) * cols
		b6 := (i + 6) * cols
		b7 := (i + 7) * cols
		v0 := residual[i] + float32(dotQ8VNNICoreZMM(&data[b0], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		v1 := residual[i+1] + float32(dotQ8VNNICoreZMM(&data[b1], &xq[0], cols)-128*rowSum[i+1])*scaleX*scale[i+1]
		v2 := residual[i+2] + float32(dotQ8VNNICoreZMM(&data[b2], &xq[0], cols)-128*rowSum[i+2])*scaleX*scale[i+2]
		v3 := residual[i+3] + float32(dotQ8VNNICoreZMM(&data[b3], &xq[0], cols)-128*rowSum[i+3])*scaleX*scale[i+3]
		v4 := residual[i+4] + float32(dotQ8VNNICoreZMM(&data[b4], &xq[0], cols)-128*rowSum[i+4])*scaleX*scale[i+4]
		v5 := residual[i+5] + float32(dotQ8VNNICoreZMM(&data[b5], &xq[0], cols)-128*rowSum[i+5])*scaleX*scale[i+5]
		v6 := residual[i+6] + float32(dotQ8VNNICoreZMM(&data[b6], &xq[0], cols)-128*rowSum[i+6])*scaleX*scale[i+6]
		v7 := residual[i+7] + float32(dotQ8VNNICoreZMM(&data[b7], &xq[0], cols)-128*rowSum[i+7])*scaleX*scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + float32(dotQ8VNNICoreZMM(&data[i*cols], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ8InPlaceAddSumSquaresParallel(out, residual, x []float32, q *Q8Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ8InPlaceAddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ8InPlaceAddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ8InPlaceAddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ8AddRMSNormOutOnly fuses Q8 down-projection matvec with AddRMSNormOutOnly.
func MatVecQ8AddRMSNormOutOnly(normOut, residual, x []float32, down *Q8Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ8AddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ8AddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ8AddSumSquaresSerial(out, residual, x []float32, q *Q8Matrix, start, end int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		v0 := residual[i] + dotQ8(q.Data[i*q.Cols:(i+1)*q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ8(q.Data[(i+1)*q.Cols:(i+2)*q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ8(q.Data[(i+2)*q.Cols:(i+3)*q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ8(q.Data[(i+3)*q.Cols:(i+4)*q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ8(q.Data[(i+4)*q.Cols:(i+5)*q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ8(q.Data[(i+5)*q.Cols:(i+6)*q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ8(q.Data[(i+6)*q.Cols:(i+7)*q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ8(q.Data[(i+7)*q.Cols:(i+8)*q.Cols], x)*q.Scale[i+7]
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + dotQ8(q.Data[i*q.Cols:(i+1)*q.Cols], x)*q.Scale[i]
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ8AddSumSquaresParallel(out, residual, x []float32, q *Q8Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ8AddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ8AddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ8AddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ4AddRMSNorm fuses Q4 down-projection matvec with AddRMSNorm (in-place).
// residual[i] += dot(down[row_i], x); normOut[i] = residual[i] * rsqrt(ss/n+eps) * weight[i]
func MatVecQ4AddRMSNorm(normOut, residual, x []float32, down *Q4Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ4InPlaceAddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ4InPlaceAddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ4InPlaceAddSumSquaresSerial(out, residual, x []float32, q *Q4Matrix, start, end int) float32 {
	if useVNNI && q.Unpacked != nil && q.Cols >= 32 && q.RowSum != nil {
		return matVecQ4InPlaceAddSumSquaresVNNISerial(out, residual, x, q, start, end)
	}
	if q.Unpacked != nil {
		return matVecQ4InPlaceAddSumSquaresUnpacked(out, residual, x, q, start, end)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	packedCols := (q.Cols + 1) / 2
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * packedCols
		b1 := (i + 1) * packedCols
		b2 := (i + 2) * packedCols
		b3 := (i + 3) * packedCols
		b4 := (i + 4) * packedCols
		b5 := (i + 5) * packedCols
		b6 := (i + 6) * packedCols
		b7 := (i + 7) * packedCols
		v0 := residual[i] + dotQ4(q.Data[b0:b0+packedCols], x, q.Cols)*q.Scale[i]
		v1 := residual[i+1] + dotQ4(q.Data[b1:b1+packedCols], x, q.Cols)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ4(q.Data[b2:b2+packedCols], x, q.Cols)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ4(q.Data[b3:b3+packedCols], x, q.Cols)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ4(q.Data[b4:b4+packedCols], x, q.Cols)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ4(q.Data[b5:b5+packedCols], x, q.Cols)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ4(q.Data[b6:b6+packedCols], x, q.Cols)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ4(q.Data[b7:b7+packedCols], x, q.Cols)*q.Scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * packedCols
		v := residual[i] + dotQ4(q.Data[base:base+packedCols], x, q.Cols)*q.Scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ4InPlaceAddSumSquaresUnpacked(out, residual, x []float32, q *Q4Matrix, start, end int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * q.Cols
		b1 := (i + 1) * q.Cols
		b2 := (i + 2) * q.Cols
		b3 := (i + 3) * q.Cols
		b4 := (i + 4) * q.Cols
		b5 := (i + 5) * q.Cols
		b6 := (i + 6) * q.Cols
		b7 := (i + 7) * q.Cols
		v0 := residual[i] + dotQ4Unpacked(q.Unpacked[b0:b0+q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ4Unpacked(q.Unpacked[b1:b1+q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ4Unpacked(q.Unpacked[b2:b2+q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ4Unpacked(q.Unpacked[b3:b3+q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ4Unpacked(q.Unpacked[b4:b4+q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ4Unpacked(q.Unpacked[b5:b5+q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ4Unpacked(q.Unpacked[b6:b6+q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ4Unpacked(q.Unpacked[b7:b7+q.Cols], x)*q.Scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * q.Cols
		v := residual[i] + dotQ4Unpacked(q.Unpacked[base:base+q.Cols], x)*q.Scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ4InPlaceAddSumSquaresVNNISerial(out, residual, x []float32, q *Q4Matrix, start, end int) float32 {
	xq := getVNNIScratch(q.Cols)
	defer putVNNIScratch(xq)
	scaleX := quantizeXForVNNI(x, xq)
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * cols
		b1 := (i + 1) * cols
		b2 := (i + 2) * cols
		b3 := (i + 3) * cols
		b4 := (i + 4) * cols
		b5 := (i + 5) * cols
		b6 := (i + 6) * cols
		b7 := (i + 7) * cols
		v0 := residual[i] + float32(dotQ8VNNICoreZMM(&data[b0], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		v1 := residual[i+1] + float32(dotQ8VNNICoreZMM(&data[b1], &xq[0], cols)-128*rowSum[i+1])*scaleX*scale[i+1]
		v2 := residual[i+2] + float32(dotQ8VNNICoreZMM(&data[b2], &xq[0], cols)-128*rowSum[i+2])*scaleX*scale[i+2]
		v3 := residual[i+3] + float32(dotQ8VNNICoreZMM(&data[b3], &xq[0], cols)-128*rowSum[i+3])*scaleX*scale[i+3]
		v4 := residual[i+4] + float32(dotQ8VNNICoreZMM(&data[b4], &xq[0], cols)-128*rowSum[i+4])*scaleX*scale[i+4]
		v5 := residual[i+5] + float32(dotQ8VNNICoreZMM(&data[b5], &xq[0], cols)-128*rowSum[i+5])*scaleX*scale[i+5]
		v6 := residual[i+6] + float32(dotQ8VNNICoreZMM(&data[b6], &xq[0], cols)-128*rowSum[i+6])*scaleX*scale[i+6]
		v7 := residual[i+7] + float32(dotQ8VNNICoreZMM(&data[b7], &xq[0], cols)-128*rowSum[i+7])*scaleX*scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + float32(dotQ8VNNICoreZMM(&data[i*cols], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ4InPlaceAddSumSquaresParallel(out, residual, x []float32, q *Q4Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ4InPlaceAddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ4InPlaceAddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ4InPlaceAddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ4AddRMSNormOutOnly fuses Q4 down-projection matvec with AddRMSNormOutOnly.
func MatVecQ4AddRMSNormOutOnly(normOut, residual, x []float32, down *Q4Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ4AddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ4AddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ4AddSumSquaresSerial(out, residual, x []float32, q *Q4Matrix, start, end int) float32 {
	if q.Unpacked != nil {
		return matVecQ4AddSumSquaresUnpacked(out, residual, x, q, start, end)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	packedCols := (q.Cols + 1) / 2
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * packedCols
		b1 := (i + 1) * packedCols
		b2 := (i + 2) * packedCols
		b3 := (i + 3) * packedCols
		b4 := (i + 4) * packedCols
		b5 := (i + 5) * packedCols
		b6 := (i + 6) * packedCols
		b7 := (i + 7) * packedCols
		v0 := residual[i] + dotQ4(q.Data[b0:b0+packedCols], x, q.Cols)*q.Scale[i]
		v1 := residual[i+1] + dotQ4(q.Data[b1:b1+packedCols], x, q.Cols)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ4(q.Data[b2:b2+packedCols], x, q.Cols)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ4(q.Data[b3:b3+packedCols], x, q.Cols)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ4(q.Data[b4:b4+packedCols], x, q.Cols)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ4(q.Data[b5:b5+packedCols], x, q.Cols)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ4(q.Data[b6:b6+packedCols], x, q.Cols)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ4(q.Data[b7:b7+packedCols], x, q.Cols)*q.Scale[i+7]
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * packedCols
		v := residual[i] + dotQ4(q.Data[base:base+packedCols], x, q.Cols)*q.Scale[i]
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ4AddSumSquaresUnpacked(out, residual, x []float32, q *Q4Matrix, start, end int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * q.Cols
		b1 := (i + 1) * q.Cols
		b2 := (i + 2) * q.Cols
		b3 := (i + 3) * q.Cols
		b4 := (i + 4) * q.Cols
		b5 := (i + 5) * q.Cols
		b6 := (i + 6) * q.Cols
		b7 := (i + 7) * q.Cols
		v0 := residual[i] + dotQ4Unpacked(q.Unpacked[b0:b0+q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ4Unpacked(q.Unpacked[b1:b1+q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ4Unpacked(q.Unpacked[b2:b2+q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ4Unpacked(q.Unpacked[b3:b3+q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ4Unpacked(q.Unpacked[b4:b4+q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ4Unpacked(q.Unpacked[b5:b5+q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ4Unpacked(q.Unpacked[b6:b6+q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ4Unpacked(q.Unpacked[b7:b7+q.Cols], x)*q.Scale[i+7]
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * q.Cols
		v := residual[i] + dotQ4Unpacked(q.Unpacked[base:base+q.Cols], x)*q.Scale[i]
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ4AddSumSquaresParallel(out, residual, x []float32, q *Q4Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ4AddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ4AddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ4AddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ6AddRMSNorm fuses Q6 down-projection matvec with AddRMSNorm (in-place).
// residual[i] += dot(down[row_i], x); normOut[i] = residual[i] * rsqrt(ss/n+eps) * weight[i]
func MatVecQ6AddRMSNorm(normOut, residual, x []float32, down *Q6Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ6InPlaceAddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ6InPlaceAddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ6InPlaceAddSumSquaresSerial(out, residual, x []float32, q *Q6Matrix, start, end int) float32 {
	if useVNNI && q.Unpacked != nil && q.Cols >= 32 && q.RowSum != nil {
		return matVecQ6InPlaceAddSumSquaresVNNISerial(out, residual, x, q, start, end)
	}
	if q.Unpacked != nil {
		return matVecQ6InPlaceAddSumSquaresUnpacked(out, residual, x, q, start, end)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	packedCols := PackedQ6Cols(q.Cols)
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * packedCols
		b1 := (i + 1) * packedCols
		b2 := (i + 2) * packedCols
		b3 := (i + 3) * packedCols
		b4 := (i + 4) * packedCols
		b5 := (i + 5) * packedCols
		b6 := (i + 6) * packedCols
		b7 := (i + 7) * packedCols
		v0 := residual[i] + dotQ6(q.Data[b0:b0+packedCols], x, q.Cols)*q.Scale[i]
		v1 := residual[i+1] + dotQ6(q.Data[b1:b1+packedCols], x, q.Cols)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ6(q.Data[b2:b2+packedCols], x, q.Cols)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ6(q.Data[b3:b3+packedCols], x, q.Cols)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ6(q.Data[b4:b4+packedCols], x, q.Cols)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ6(q.Data[b5:b5+packedCols], x, q.Cols)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ6(q.Data[b6:b6+packedCols], x, q.Cols)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ6(q.Data[b7:b7+packedCols], x, q.Cols)*q.Scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * packedCols
		v := residual[i] + dotQ6(q.Data[base:base+packedCols], x, q.Cols)*q.Scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ6InPlaceAddSumSquaresUnpacked(out, residual, x []float32, q *Q6Matrix, start, end int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * q.Cols
		b1 := (i + 1) * q.Cols
		b2 := (i + 2) * q.Cols
		b3 := (i + 3) * q.Cols
		b4 := (i + 4) * q.Cols
		b5 := (i + 5) * q.Cols
		b6 := (i + 6) * q.Cols
		b7 := (i + 7) * q.Cols
		v0 := residual[i] + dotQ6Unpacked(q.Unpacked[b0:b0+q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ6Unpacked(q.Unpacked[b1:b1+q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ6Unpacked(q.Unpacked[b2:b2+q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ6Unpacked(q.Unpacked[b3:b3+q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ6Unpacked(q.Unpacked[b4:b4+q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ6Unpacked(q.Unpacked[b5:b5+q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ6Unpacked(q.Unpacked[b6:b6+q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ6Unpacked(q.Unpacked[b7:b7+q.Cols], x)*q.Scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * q.Cols
		v := residual[i] + dotQ6Unpacked(q.Unpacked[base:base+q.Cols], x)*q.Scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ6InPlaceAddSumSquaresVNNISerial(out, residual, x []float32, q *Q6Matrix, start, end int) float32 {
	xq := getVNNIScratch(q.Cols)
	defer putVNNIScratch(xq)
	scaleX := quantizeXForVNNI(x, xq)
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * cols
		b1 := (i + 1) * cols
		b2 := (i + 2) * cols
		b3 := (i + 3) * cols
		b4 := (i + 4) * cols
		b5 := (i + 5) * cols
		b6 := (i + 6) * cols
		b7 := (i + 7) * cols
		v0 := residual[i] + float32(dotQ8VNNICoreZMM(&data[b0], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		v1 := residual[i+1] + float32(dotQ8VNNICoreZMM(&data[b1], &xq[0], cols)-128*rowSum[i+1])*scaleX*scale[i+1]
		v2 := residual[i+2] + float32(dotQ8VNNICoreZMM(&data[b2], &xq[0], cols)-128*rowSum[i+2])*scaleX*scale[i+2]
		v3 := residual[i+3] + float32(dotQ8VNNICoreZMM(&data[b3], &xq[0], cols)-128*rowSum[i+3])*scaleX*scale[i+3]
		v4 := residual[i+4] + float32(dotQ8VNNICoreZMM(&data[b4], &xq[0], cols)-128*rowSum[i+4])*scaleX*scale[i+4]
		v5 := residual[i+5] + float32(dotQ8VNNICoreZMM(&data[b5], &xq[0], cols)-128*rowSum[i+5])*scaleX*scale[i+5]
		v6 := residual[i+6] + float32(dotQ8VNNICoreZMM(&data[b6], &xq[0], cols)-128*rowSum[i+6])*scaleX*scale[i+6]
		v7 := residual[i+7] + float32(dotQ8VNNICoreZMM(&data[b7], &xq[0], cols)-128*rowSum[i+7])*scaleX*scale[i+7]
		residual[i] = v0
		residual[i+1] = v1
		residual[i+2] = v2
		residual[i+3] = v3
		residual[i+4] = v4
		residual[i+5] = v5
		residual[i+6] = v6
		residual[i+7] = v7
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		v := residual[i] + float32(dotQ8VNNICoreZMM(&data[i*cols], &xq[0], cols)-128*rowSum[i])*scaleX*scale[i]
		residual[i] = v
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ6InPlaceAddSumSquaresParallel(out, residual, x []float32, q *Q6Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ6InPlaceAddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ6InPlaceAddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ6InPlaceAddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}

// MatVecQ6AddRMSNormOutOnly fuses Q6 down-projection matvec with AddRMSNormOutOnly.
func MatVecQ6AddRMSNormOutOnly(normOut, residual, x []float32, down *Q6Matrix, normWeight []float32, eps float32) {
	n := down.Rows
	if n == 0 {
		return
	}
	var ss float32
	if shouldParallelQuantMatVec(n*down.Cols, n) {
		ss = matVecQ6AddSumSquaresParallel(normOut, residual, x, down)
	} else {
		ss = matVecQ6AddSumSquaresSerial(normOut, residual, x, down, 0, n)
	}
	scale := fastRsqrtF32(ss/float32(n)+eps)
	if useDotFMA && n >= 8 {
		mulScaleFMA(normOut, normOut, normWeight, scale)
		return
	}
	if useDotF32AVX && n >= 8 {
		mulScaleAVX(normOut, normOut, normWeight, scale)
		return
	}
	for i := 0; i < n; i++ {
		normOut[i] = normOut[i] * scale * normWeight[i]
	}
}

func matVecQ6AddSumSquaresSerial(out, residual, x []float32, q *Q6Matrix, start, end int) float32 {
	if q.Unpacked != nil {
		return matVecQ6AddSumSquaresUnpacked(out, residual, x, q, start, end)
	}
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	packedCols := PackedQ6Cols(q.Cols)
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * packedCols
		b1 := (i + 1) * packedCols
		b2 := (i + 2) * packedCols
		b3 := (i + 3) * packedCols
		b4 := (i + 4) * packedCols
		b5 := (i + 5) * packedCols
		b6 := (i + 6) * packedCols
		b7 := (i + 7) * packedCols
		v0 := residual[i] + dotQ6(q.Data[b0:b0+packedCols], x, q.Cols)*q.Scale[i]
		v1 := residual[i+1] + dotQ6(q.Data[b1:b1+packedCols], x, q.Cols)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ6(q.Data[b2:b2+packedCols], x, q.Cols)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ6(q.Data[b3:b3+packedCols], x, q.Cols)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ6(q.Data[b4:b4+packedCols], x, q.Cols)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ6(q.Data[b5:b5+packedCols], x, q.Cols)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ6(q.Data[b6:b6+packedCols], x, q.Cols)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ6(q.Data[b7:b7+packedCols], x, q.Cols)*q.Scale[i+7]
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * packedCols
		v := residual[i] + dotQ6(q.Data[base:base+packedCols], x, q.Cols)*q.Scale[i]
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ6AddSumSquaresUnpacked(out, residual, x []float32, q *Q6Matrix, start, end int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := start
	for ; i+7 < end; i += 8 {
		b0 := i * q.Cols
		b1 := (i + 1) * q.Cols
		b2 := (i + 2) * q.Cols
		b3 := (i + 3) * q.Cols
		b4 := (i + 4) * q.Cols
		b5 := (i + 5) * q.Cols
		b6 := (i + 6) * q.Cols
		b7 := (i + 7) * q.Cols
		v0 := residual[i] + dotQ6Unpacked(q.Unpacked[b0:b0+q.Cols], x)*q.Scale[i]
		v1 := residual[i+1] + dotQ6Unpacked(q.Unpacked[b1:b1+q.Cols], x)*q.Scale[i+1]
		v2 := residual[i+2] + dotQ6Unpacked(q.Unpacked[b2:b2+q.Cols], x)*q.Scale[i+2]
		v3 := residual[i+3] + dotQ6Unpacked(q.Unpacked[b3:b3+q.Cols], x)*q.Scale[i+3]
		v4 := residual[i+4] + dotQ6Unpacked(q.Unpacked[b4:b4+q.Cols], x)*q.Scale[i+4]
		v5 := residual[i+5] + dotQ6Unpacked(q.Unpacked[b5:b5+q.Cols], x)*q.Scale[i+5]
		v6 := residual[i+6] + dotQ6Unpacked(q.Unpacked[b6:b6+q.Cols], x)*q.Scale[i+6]
		v7 := residual[i+7] + dotQ6Unpacked(q.Unpacked[b7:b7+q.Cols], x)*q.Scale[i+7]
		out[i] = v0
		out[i+1] = v1
		out[i+2] = v2
		out[i+3] = v3
		out[i+4] = v4
		out[i+5] = v5
		out[i+6] = v6
		out[i+7] = v7
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
	for ; i < end; i++ {
		base := i * q.Cols
		v := residual[i] + dotQ6Unpacked(q.Unpacked[base:base+q.Cols], x)*q.Scale[i]
		out[i] = v
		ss += v * v
	}
	return ss
}

func matVecQ6AddSumSquaresParallel(out, residual, x []float32, q *Q6Matrix) float32 {
	workers := runtime.GOMAXPROCS(0)
	if workers > q.Rows {
		workers = q.Rows
	}
	if workers <= 1 {
		return matVecQ6AddSumSquaresSerial(out, residual, x, q, 0, q.Rows)
	}
	var partialsArr [16]float32; partials := partialsArr[:workers]
	chunk := (q.Rows + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 1; worker < workers; worker++ {
		start := worker * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			partials[slot] = matVecQ6AddSumSquaresSerial(out, residual, x, q, start, end)
		}(worker, start, end)
	}
	firstEnd := chunk
	if firstEnd > q.Rows {
		firstEnd = q.Rows
	}
	partials[0] = matVecQ6AddSumSquaresSerial(out, residual, x, q, 0, firstEnd)
	wg.Wait()
	var ss float32
	for _, p := range partials {
		ss += p
	}
	return ss
}
