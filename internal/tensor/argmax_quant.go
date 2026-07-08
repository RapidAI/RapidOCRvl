package tensor

// MatVecArgmaxQ8 computes the Q8 matvec of x with q and returns the token id
// and score of the row with the maximum score.  Used as a CPU reference for
// the Vulkan argmax probe.
func MatVecArgmaxQ8(x []float32, q *Q8Matrix) (int, float32) {
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

// MatVecArgmaxQ6 computes the Q6 matvec of x with q and returns the token id
// and score of the row with the maximum score.
func MatVecArgmaxQ6(x []float32, q *Q6Matrix) (int, float32) {
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

// MatVecArgmaxQ4 computes the Q4 matvec of x with q and returns the token id
// and score of the row with the maximum score.
func MatVecArgmaxQ4(x []float32, q *Q4Matrix) (int, float32) {
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

// FusedMatVec2Q8 computes outB = qb @ x and outC = qc @ x for Q8 matrices.
func FusedMatVec2Q8(outB, outC, x []float32, b, c *Q8Matrix) {
	MatVecQ8(outB, x, b)
	MatVecQ8(outC, x, c)
}

// FusedMatVec2Q6 computes outB = qb @ x and outC = qc @ x for Q6 matrices.
func FusedMatVec2Q6(outB, outC, x []float32, b, c *Q6Matrix) {
	MatVecQ6(outB, x, b)
	MatVecQ6(outC, x, c)
}

// FusedMatVec2Q4 computes outB = qb @ x and outC = qc @ x for Q4 matrices.
func FusedMatVec2Q4(outB, outC, x []float32, b, c *Q4Matrix) {
	MatVecQ4(outB, x, b)
	MatVecQ4(outC, x, c)
}
