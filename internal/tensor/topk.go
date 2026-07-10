package tensor

import (
	"runtime"
	"sync"
)

// TopKScore holds a token id and its matvec score, used for top-k sampling.
type TopKScore struct {
	ID    int
	Score float32
}

// DotPair computes two dot products simultaneously: a.b0 and a.b1.
// This shares the a read across both dot products, reducing memory traffic.
func DotPair(b0, b1, a []float32) (float32, float32) {
	return dotF32Pair(b0, b1, a)
}

// DotQuad computes four dot products simultaneously: a.x, b.x, c.x, d.x.
// This shares the x read across all four dot products, reducing memory traffic.
func DotQuad(b0, b1, b2, b3, a []float32) (float32, float32, float32, float32) {
	return dotF32Quad(b0, b1, b2, b3, a)
}

// DotOctet computes eight dot products simultaneously, sharing the x read
// across all eight outputs. This reduces memory traffic and function-call
// overhead compared to two DotQuad calls.
func DotOctet(b0, b1, b2, b3, b4, b5, b6, b7, a []float32) (float32, float32, float32, float32, float32, float32, float32, float32) {
	if useDotFMA && len(a) >= 16 {
		return dotF32OctetFMA(b0, b1, b2, b3, b4, b5, b6, b7, a)
	}
	s0 := dotF32(b0, a)
	s1 := dotF32(b1, a)
	s2 := dotF32(b2, a)
	s3 := dotF32(b3, a)
	s4 := dotF32(b4, a)
	s5 := dotF32(b5, a)
	s6 := dotF32(b6, a)
	s7 := dotF32(b7, a)
	return s0, s1, s2, s3, s4, s5, s6, s7
}

// MatVecArgmax computes the F32 matvec x . w^T and returns the row index
// with the maximum score along with that score.
func MatVecArgmax(x, w []float32, rows, cols int) (int, float32) {
	if shouldParallel(rows*cols, rows) {
		return matVecArgmaxParallel(x, w, rows, cols)
	}
	bestIdx := 0
	bestVal := float32(0)
	for r := 0; r < rows; r++ {
		base := r * cols
		v := dotF32(w[base:base+cols], x)
		if r == 0 || v > bestVal {
			bestVal = v
			bestIdx = r
		}
	}
	return bestIdx, bestVal
}


// finalizeTopKScores converts VNNI dot results to TopKScore entries.
func finalizeTopKScores(scores []TopKScore, dots []int32, rowSum []int32, scale []float32, n int, scaleX float32) {
	for r := 0; r < n; r++ {
		scores[r] = TopKScore{ID: r, Score: float32(dots[r]-128*rowSum[r]) * scaleX * scale[r]}
	}
}

func matVecArgmaxParallel(x, w []float32, rows, cols int) (int, float32) {
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
		bestIdx := 0
		bestVal := float32(0)
		for r := 0; r < rows; r++ {
			base := r * cols
			v := dotF32(w[base:base+cols], x)
			if r == 0 || v > bestVal {
				bestVal = v
				bestIdx = r
			}
		}
		return bestIdx, bestVal
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
			bestIdx := start
			bestVal := float32(0)
			for r := start; r < end; r++ {
				base := r * cols
				v := dotF32(w[base:base+cols], x)
				if r == start || v > bestVal {
					bestVal = v
					bestIdx = r
				}
			}
			results[slot] = partial{bestIdx, bestVal}
		}(wi, start, end)
	}
	wg.Wait()
	bestIdx := results[0].idx
	bestVal := results[0].val
	for i := 1; i < workers; i++ {
		if results[i].val > bestVal {
			bestVal = results[i].val
			bestIdx = results[i].idx
		}
	}
	return bestIdx, bestVal
}

// MatVecTopKQ8WithWork computes the Q8 matvec x . q^T, then selects the
// top-k scores. The work slice is reused for partial-sort scratch space.
// Returns the top-k scores (sorted descending), the work buffer, and the
// number of valid entries.
func MatVecTopKQ8WithWork(scores, work []TopKScore, x []float32, q *Q8Matrix, k int) ([]TopKScore, []TopKScore, int) {
	scores = ensureTopKScoreCap(scores, q.Rows)
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		scratch := getInt32Scratch(q.Rows)
		defer putInt32Scratch(scratch)
		dotQ8VNNICoreMultiRowZMM(&q.Data[0], &xq[0], &scratch[0], q.Rows, q.Cols)
		finalizeTopKScores(scores, scratch, q.RowSum, q.Scale, q.Rows, scaleX)
		return topKSelectWithWork(scores, work, q.Rows, k), work, k
	}
	for r := 0; r < q.Rows; r++ {
		base := r * q.Cols
		scores[r] = TopKScore{ID: r, Score: dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]}
	}
	return topKSelectWithWork(scores, work, q.Rows, k), work, k
}

// MatVecTopKQ6WithWork is the Q6 variant of MatVecTopKQ8WithWork.
func MatVecTopKQ6WithWork(scores, work []TopKScore, x []float32, q *Q6Matrix, k int) ([]TopKScore, []TopKScore, int) {
	scores = ensureTopKScoreCap(scores, q.Rows)
	if q.Unpacked != nil {
		if useVNNI && q.Cols >= 32 && q.RowSum != nil {
			xq := getVNNIScratch(q.Cols)
			defer putVNNIScratch(xq)
			scaleX := quantizeXForVNNI(x, xq)
			scratch := getInt32Scratch(q.Rows)
			defer putInt32Scratch(scratch)
			dotQ8VNNICoreMultiRowZMM(&q.Unpacked[0], &xq[0], &scratch[0], q.Rows, q.Cols)
			finalizeTopKScores(scores, scratch, q.RowSum, q.Scale, q.Rows, scaleX)
			return topKSelectWithWork(scores, work, q.Rows, k), work, k
		}
		for r := 0; r < q.Rows; r++ {
			base := r * q.Cols
			scores[r] = TopKScore{ID: r, Score: dotQ6Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]}
		}
	} else {
		packedCols := PackedQ6Cols(q.Cols)
		for r := 0; r < q.Rows; r++ {
			base := r * packedCols
			scores[r] = TopKScore{ID: r, Score: dotQ6(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]}
		}
	}
	return topKSelectWithWork(scores, work, q.Rows, k), work, k
}

// MatVecTopKQ4WithWork is the Q4 variant of MatVecTopKQ8WithWork.
func MatVecTopKQ4WithWork(scores, work []TopKScore, x []float32, q *Q4Matrix, k int) ([]TopKScore, []TopKScore, int) {
	scores = ensureTopKScoreCap(scores, q.Rows)
	if q.Unpacked != nil {
		if useVNNI && q.Cols >= 32 && q.RowSum != nil {
			xq := getVNNIScratch(q.Cols)
			defer putVNNIScratch(xq)
			scaleX := quantizeXForVNNI(x, xq)
			scratch := getInt32Scratch(q.Rows)
			defer putInt32Scratch(scratch)
			dotQ8VNNICoreMultiRowZMM(&q.Unpacked[0], &xq[0], &scratch[0], q.Rows, q.Cols)
			finalizeTopKScores(scores, scratch, q.RowSum, q.Scale, q.Rows, scaleX)
			return topKSelectWithWork(scores, work, q.Rows, k), work, k
		}
		for r := 0; r < q.Rows; r++ {
			base := r * q.Cols
			scores[r] = TopKScore{ID: r, Score: dotQ4Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]}
		}
	} else {
		packedCols := (q.Cols + 1) / 2
		for r := 0; r < q.Rows; r++ {
			base := r * packedCols
			scores[r] = TopKScore{ID: r, Score: dotQ4(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]}
		}
	}
	return topKSelectWithWork(scores, work, q.Rows, k), work, k
}

// MatVecTopKWithWork is the F32 variant of MatVecTopKQ8WithWork.
func MatVecTopKWithWork(scores, work []TopKScore, x, w []float32, rows, cols, k int) ([]TopKScore, []TopKScore, int) {
	scores = ensureTopKScoreCap(scores, rows)
	for r := 0; r < rows; r++ {
		base := r * cols
		scores[r] = TopKScore{ID: r, Score: dotF32(w[base:base+cols], x)}
	}
	return topKSelectWithWork(scores, work, rows, k), work, k
}

func ensureTopKScoreCap(s []TopKScore, n int) []TopKScore {
	if cap(s) >= n {
		return s[:n]
	}
	return make([]TopKScore, n)
}

// topKSelectWithWork performs a partial selection sort to find the top-k
// entries (by score, ties broken by ID ascending) and returns them sorted
// descending. The work slice is available for future optimizations.
func topKSelectWithWork(scores, work []TopKScore, n, k int) []TopKScore {
	if k <= 0 {
		return scores[:0]
	}
	if k > n {
		k = n
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < n; j++ {
			if scores[j].Score > scores[best].Score ||
				(scores[j].Score == scores[best].Score && scores[j].ID < scores[best].ID) {
				best = j
			}
		}
		if best != i {
			scores[i], scores[best] = scores[best], scores[i]
		}
	}
	return scores[:k]
}