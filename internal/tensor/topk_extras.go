package tensor

// matVecSerialRange computes out[r] = w[r*cols:(r+1)*cols] . x for r in [start, end).
func matVecSerialRange(out, x, w []float32, cols, start, end int) {
	for r := start; r < end; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x)
	}
}

// matVecBiasSerialRange computes out[r] = w[r*cols:(r+1)*cols] . x + bias[r]
// for r in [start, end).
func matVecBiasSerialRange(out, x, w, bias []float32, cols, start, end int) {
	for r := start; r < end; r++ {
		out[r] = dotF32(w[r*cols:(r+1)*cols], x) + bias[r]
	}
}

// MatVecTopK computes the F32 matvec x . w^T, then selects the top-k scores
// (by score descending, ties broken by ID ascending). Returns the top-k
// scores and the maximum score. If dst has sufficient capacity it is reused.
func MatVecTopK(dst []TopKScore, x, w []float32, rows, cols, k int) ([]TopKScore, float32) {
	scores := ensureTopKScoreCap(dst, rows)
	for r := 0; r < rows; r++ {
		base := r * cols
		scores[r] = TopKScore{ID: r, Score: dotF32(w[base:base+cols], x)}
	}
	if k <= 0 {
		return scores[:0], 0
	}
	if k > rows {
		k = rows
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < rows; j++ {
			if scores[j].Score > scores[best].Score ||
				(scores[j].Score == scores[best].Score && scores[j].ID < scores[best].ID) {
				best = j
			}
		}
		if best != i {
			scores[i], scores[best] = scores[best], scores[i]
		}
	}
	return scores[:k], scores[0].Score
}
