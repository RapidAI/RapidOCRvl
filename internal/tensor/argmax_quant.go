package tensor

import (
	"runtime"
	"sync"
)

// MatVecArgmaxQ8 computes the Q8 matvec of x with q and returns the token id
// and score of the row with the maximum score.  Used as a CPU reference for
// the Vulkan argmax probe.
func MatVecArgmaxQ8(x []float32, q *Q8Matrix) (int, float32) {
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		return matVecArgmaxQ8Parallel(x, q)
	}
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		data := q.Data
		scale := q.Scale
		rowSum := q.RowSum
		cols := q.Cols
		bestToken := 0
		bestScore := float32(0)
		if useVNNI {
			scratch := getInt32Scratch(q.Rows)
			defer putInt32Scratch(scratch)
			dotQ8VNNICoreMultiRowZMM(&data[0], &xq[0], &scratch[0], q.Rows, cols)
			for r := 0; r < q.Rows; r++ {
				score := float32(scratch[r]-128*rowSum[r]) * scaleX * scale[r]
				if r == 0 || score > bestScore {
					bestToken = r
					bestScore = score
				}
			}
			return bestToken, bestScore
		}
		for r := 0; r < q.Rows; r++ {
			base := r * q.Cols
			score := dotQ8VNNI(data[base:base+q.Cols], xq, scaleX, scale[r], rowSum[r])
			if r == 0 || score > bestScore {
				bestToken = r
				bestScore = score
			}
		}
		return bestToken, bestScore
	}
	bestToken := 0
	bestScore := float32(0)
	for r := 0; r < q.Rows; r++ {
		base := r * q.Cols
		score := dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]
		if r == 0 || score > bestScore {
			bestToken = r
			bestScore = score
		}
	}
	return bestToken, bestScore
}

func matVecArgmaxQ8Parallel(x []float32, q *Q8Matrix) (int, float32) {
	type partial struct {
		idx int
		val float32
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > 16 {
		workers = 16
	}
	if workers > q.Rows {
		workers = q.Rows
	}
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		if workers <= 1 {
			data := q.Data
			scale := q.Scale
			rowSum := q.RowSum
			cols := q.Cols
			bestToken := 0
			bestScore := float32(0)
			if useVNNI {
				scratch := getInt32Scratch(q.Rows)
				defer putInt32Scratch(scratch)
				dotQ8VNNICoreMultiRowZMM(&data[0], &xq[0], &scratch[0], q.Rows, cols)
				for r := 0; r < q.Rows; r++ {
					score := float32(scratch[r]-128*rowSum[r]) * scaleX * scale[r]
					if r == 0 || score > bestScore {
						bestToken = r
						bestScore = score
					}
				}
				return bestToken, bestScore
			}
			for r := 0; r < q.Rows; r++ {
				base := r * q.Cols
				score := dotQ8VNNI(data[base:base+q.Cols], xq, scaleX, scale[r], rowSum[r])
				if r == 0 || score > bestScore {
					bestToken = r
					bestScore = score
				}
			}
			return bestToken, bestScore
		}
		chunk := (q.Rows + workers - 1) / workers
		var resultsArr [16]partial; results := resultsArr[:workers]
		var wg sync.WaitGroup
		for wi := 0; wi < workers; wi++ {
			start := wi * chunk
			end := start + chunk
			if end > q.Rows {
				end = q.Rows
			}
			if start >= end {
				results[wi] = partial{0, 0}
				continue
			}
			wg.Add(1)
			go func(slot, start, end int) {
				defer wg.Done()
				data := q.Data
				scale := q.Scale
				rowSum := q.RowSum
				cols := q.Cols
				bestToken := start
				bestScore := float32(0)
				if useVNNI {
					nRows := end - start
					scratch := getInt32Scratch(nRows)
					defer putInt32Scratch(scratch)
					dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
					for r := 0; r < nRows; r++ {
						score := float32(scratch[r]-128*rowSum[start+r]) * scaleX * scale[start+r]
						if r == 0 || score > bestScore {
							bestToken = start + r
							bestScore = score
						}
					}
				} else {
					for r := start; r < end; r++ {
						base := r * q.Cols
						score := dotQ8VNNI(data[base:base+q.Cols], xq, scaleX, scale[r], rowSum[r])
						if r == start || score > bestScore {
							bestToken = r
							bestScore = score
						}
					}
				}
				results[slot] = partial{bestToken, bestScore}
			}(wi, start, end)
		}
		wg.Wait()
		best := results[0]
		for i := 1; i < workers; i++ {
			if results[i].val > best.val {
				best = results[i]
			}
		}
		return best.idx, best.val
	}
	if workers <= 1 {
		bestToken := 0
		bestScore := float32(0)
		for r := 0; r < q.Rows; r++ {
			base := r * q.Cols
			score := dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]
			if r == 0 || score > bestScore {
				bestToken = r
				bestScore = score
			}
		}
		return bestToken, bestScore
	}
	chunk := (q.Rows + workers - 1) / workers
	var resultsArr [16]partial; results := resultsArr[:workers]
	var wg sync.WaitGroup
	for wi := 0; wi < workers; wi++ {
		start := wi * chunk
		end := start + chunk
		if end > q.Rows {
			end = q.Rows
		}
		if start >= end {
			results[wi] = partial{0, 0}
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			bestToken := start
			bestScore := float32(0)
			for r := start; r < end; r++ {
				base := r * q.Cols
				score := dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]
				if r == start || score > bestScore {
					bestToken = r
					bestScore = score
				}
			}
			results[slot] = partial{bestToken, bestScore}
		}(wi, start, end)
	}
	wg.Wait()
	bestToken := results[0].idx
	bestScore := results[0].val
	for i := 1; i < workers; i++ {
		if results[i].val > bestScore {
			bestScore = results[i].val
			bestToken = results[i].idx
		}
	}
	return bestToken, bestScore
}

// MatVecArgmaxQ6 computes the Q6 matvec of x with q and returns the token id
// and score of the row with the maximum score.
func MatVecArgmaxQ6(x []float32, q *Q6Matrix) (int, float32) {
	if q.Unpacked != nil {
		return matVecArgmaxQ6Unpacked(x, q)
	}
	packedCols := PackedQ6Cols(q.Cols)
	bestToken := 0
	bestScore := float32(0)
	for r := 0; r < q.Rows; r++ {
		base := r * packedCols
		score := dotQ6(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]
		if r == 0 || score > bestScore {
			bestToken = r
			bestScore = score
		}
	}
	return bestToken, bestScore
}

func matVecArgmaxQ6Unpacked(x []float32, q *Q6Matrix) (int, float32) {
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		return matVecArgmaxQ6UnpackedParallel(x, q)
	}
	bestToken := 0
	bestScore := float32(0)
	for r := 0; r < q.Rows; r++ {
		base := r * q.Cols
		score := dotQ6Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]
		if r == 0 || score > bestScore {
			bestToken = r
			bestScore = score
		}
	}
	return bestToken, bestScore
}

func matVecArgmaxQ6UnpackedParallel(x []float32, q *Q6Matrix) (int, float32) {
	return matVecArgmaxUnpackedParallel(q.Rows, q.Cols, q.Unpacked, q.Scale, x, dotQ6Unpacked)
}

// MatVecArgmaxQ4 computes the Q4 matvec of x with q and returns the token id
// and score of the row with the maximum score.
func MatVecArgmaxQ4(x []float32, q *Q4Matrix) (int, float32) {
	if q.Unpacked != nil {
		return matVecArgmaxQ4Unpacked(x, q)
	}
	packedCols := (q.Cols + 1) / 2
	bestToken := 0
	bestScore := float32(0)
	for r := 0; r < q.Rows; r++ {
		base := r * packedCols
		score := dotQ4(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]
		if r == 0 || score > bestScore {
			bestToken = r
			bestScore = score
		}
	}
	return bestToken, bestScore
}

func matVecArgmaxQ4Unpacked(x []float32, q *Q4Matrix) (int, float32) {
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		return matVecArgmaxQ4UnpackedParallel(x, q)
	}
	bestToken := 0
	bestScore := float32(0)
	for r := 0; r < q.Rows; r++ {
		base := r * q.Cols
		score := dotQ4Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]
		if r == 0 || score > bestScore {
			bestToken = r
			bestScore = score
		}
	}
	return bestToken, bestScore
}

func matVecArgmaxQ4UnpackedParallel(x []float32, q *Q4Matrix) (int, float32) {
	return matVecArgmaxUnpackedParallel(q.Rows, q.Cols, q.Unpacked, q.Scale, x, dotQ4Unpacked)
}

type argmaxDotFn func(a []int8, b []float32) float32

func matVecArgmaxUnpackedParallel(rows, cols int, data []int8, scale []float32, x []float32, dotFn argmaxDotFn) (int, float32) {
	type partial struct {
		idx int
		val float32
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > 16 {
		workers = 16
	}
	if workers > rows {
		workers = rows
	}
	if workers <= 1 {
		bestToken := 0
		bestScore := float32(0)
		for r := 0; r < rows; r++ {
			base := r * cols
			score := dotFn(data[base:base+cols], x) * scale[r]
			if r == 0 || score > bestScore {
				bestToken = r
				bestScore = score
			}
		}
		return bestToken, bestScore
	}
	chunk := (rows + workers - 1) / workers
	var resultsArr [16]partial; results := resultsArr[:workers]
	var wg sync.WaitGroup
	for wi := 0; wi < workers; wi++ {
		start := wi * chunk
		end := start + chunk
		if end > rows {
			end = rows
		}
		if start >= end {
			results[wi] = partial{0, 0}
			continue
		}
		wg.Add(1)
		go func(slot, start, end int) {
			defer wg.Done()
			bestToken := start
			bestScore := float32(0)
			for r := start; r < end; r++ {
				base := r * cols
				score := dotFn(data[base:base+cols], x) * scale[r]
				if r == start || score > bestScore {
					bestToken = r
					bestScore = score
				}
			}
			results[slot] = partial{bestToken, bestScore}
		}(wi, start, end)
	}
	wg.Wait()
	bestToken := results[0].idx
	bestScore := results[0].val
	for i := 1; i < workers; i++ {
		if results[i].val > bestScore {
			bestScore = results[i].val
			bestToken = results[i].idx
		}
	}
	return bestToken, bestScore
}

// FusedMatVec2Q8 computes outB = qb @ x and outC = qc @ x for Q8 matrices.
// When both matrices share the same shape, the pair dot kernel reads x once for
// both rows instead of twice.
func FusedMatVec2Q8(outB, outC, x []float32, b, c *Q8Matrix) {
	matVecQ8Pair(outB, outC, x, b, c)
}

// FusedMatVec2Q6 computes outB = qb @ x and outC = qc @ x for Q6 matrices.
func FusedMatVec2Q6(outB, outC, x []float32, b, c *Q6Matrix) {
	matVecQ6Pair(outB, outC, x, b, c)
}

// FusedMatVec2Q4 computes outB = qb @ x and outC = qc @ x for Q4 matrices.
func FusedMatVec2Q4(outB, outC, x []float32, b, c *Q4Matrix) {
	matVecQ4Pair(outB, outC, x, b, c)
}