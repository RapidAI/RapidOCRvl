package tensor

// prequant.go — Pre-quantized dispatch variants.
// These functions receive an already-quantized xq and scaleX, avoiding
// redundant quantization when the same x is reused across parallel workers.

// matVecQ4SerialPreQuant is the VNNI path of matVecQ4Serial with pre-quantized x.
func matVecQ4SerialPreQuant(out []float32, q *Q4Matrix, xq []uint8, scaleX float32, start, end int) {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	finalizeDotQ8VNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], nRows, scaleX)
}

// matVecQ6SerialPreQuant is the VNNI path of matVecQ6Serial with pre-quantized x.
func matVecQ6SerialPreQuant(out []float32, q *Q6Matrix, xq []uint8, scaleX float32, start, end int) {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	finalizeDotQ8VNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], nRows, scaleX)
}

// matVecQ8SerialPreQuant is the VNNI path of matVecQ8Serial with pre-quantized x.
func matVecQ8SerialPreQuant(out []float32, q *Q8Matrix, xq []uint8, scaleX float32, start, end int) {
	data := q.Data
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	finalizeDotQ8VNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], nRows, scaleX)
}

// matVecQ8BiasSerialPreQuant is the VNNI path of matVecQ8BiasSerial with pre-quantized x.
func matVecQ8BiasSerialPreQuant(out []float32, q *Q8Matrix, bias []float32, xq []uint8, scaleX float32, start, end int) {
	data := q.Data
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	finalizeDotQ8BiasVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &bias[start], nRows, scaleX)
}

// matVecQ4PairSerialPreQuant is the VNNI path of matVecQ4PairSerial with pre-quantized x.
func matVecQ4PairSerialPreQuant(outA, outB []float32, a, b *Q4Matrix, xq []uint8, scaleX float32, start, end int) {
	aData, bData := a.Unpacked, b.Unpacked
	aScale, bScale := a.Scale, b.Scale
	aRS, bRS := a.RowSum, b.RowSum
	cols := a.Cols
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	dotQ8PairVNNICoreMultiRowZMM(&aData[start*cols], &bData[start*cols], &xq[0], &sA[0], &sB[0], nRows, cols)
	finalizeDotQ8PairVNNI(&sA[0], &aRS[start], &aScale[start], &outA[start],
		&sB[0], &bRS[start], &bScale[start], &outB[start], nRows, scaleX)
}

// matVecQ6PairSerialPreQuant is the VNNI path of matVecQ6PairSerial with pre-quantized x.
func matVecQ6PairSerialPreQuant(outA, outB []float32, a, b *Q6Matrix, xq []uint8, scaleX float32, start, end int) {
	aData, bData := a.Unpacked, b.Unpacked
	aScale, bScale := a.Scale, b.Scale
	aRS, bRS := a.RowSum, b.RowSum
	cols := a.Cols
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	dotQ8PairVNNICoreMultiRowZMM(&aData[start*cols], &bData[start*cols], &xq[0], &sA[0], &sB[0], nRows, cols)
	finalizeDotQ8PairVNNI(&sA[0], &aRS[start], &aScale[start], &outA[start],
		&sB[0], &bRS[start], &bScale[start], &outB[start], nRows, scaleX)
}

// fusedMatVec3Q4EqualRowsSerialPreQuant is the VNNI path with pre-quantized x.
func fusedMatVec3Q4EqualRowsSerialPreQuant(outA, outB, outC []float32, a, b, c *Q4Matrix, xq []uint8, scaleX float32, start, end int) {
	aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
	aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
	aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
	cols := a.Cols
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	sC := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	defer putInt32Scratch(sC)
	dotQ8TripletVNNICoreMultiRowZMM(&aData[start*cols], &bData[start*cols], &cData[start*cols], &xq[0], &sA[0], &sB[0], &sC[0], nRows, cols)
	finalizeDotQ8TripletVNNI(&sA[0], &aRS[start], &aScale[start], &outA[start],
		&sB[0], &bRS[start], &bScale[start], &outB[start],
		&sC[0], &cRS[start], &cScale[start], &outC[start], nRows, scaleX)
}

// fusedMatVec3Q6EqualRowsSerialPreQuant is the VNNI path with pre-quantized x.
func fusedMatVec3Q6EqualRowsSerialPreQuant(outA, outB, outC []float32, a, b, c *Q6Matrix, xq []uint8, scaleX float32, start, end int) {
	aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
	aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
	aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
	cols := a.Cols
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	sC := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	defer putInt32Scratch(sC)
	dotQ8TripletVNNICoreMultiRowZMM(&aData[start*cols], &bData[start*cols], &cData[start*cols], &xq[0], &sA[0], &sB[0], &sC[0], nRows, cols)
	finalizeDotQ8TripletVNNI(&sA[0], &aRS[start], &aScale[start], &outA[start],
		&sB[0], &bRS[start], &bScale[start], &outB[start],
		&sC[0], &cRS[start], &cScale[start], &outC[start], nRows, scaleX)
}

// fusedMatVec3Q8EqualRowsSerialPreQuant is the VNNI path with pre-quantized x.
func fusedMatVec3Q8EqualRowsSerialPreQuant(outA, outB, outC []float32, a, b, c *Q8Matrix, xq []uint8, scaleX float32, start, end int) {
	aData, bData, cData := a.Data, b.Data, c.Data
	aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
	aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
	cols := a.Cols
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	sC := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	defer putInt32Scratch(sC)
	dotQ8TripletVNNICoreMultiRowZMM(&aData[start*cols], &bData[start*cols], &cData[start*cols], &xq[0], &sA[0], &sB[0], &sC[0], nRows, cols)
	finalizeDotQ8TripletVNNI(&sA[0], &aRS[start], &aScale[start], &outA[start],
		&sB[0], &bRS[start], &bScale[start], &outB[start],
		&sC[0], &cRS[start], &cScale[start], &outC[start], nRows, scaleX)
}

// matVecQ8InPlaceAddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ8InPlaceAddSumSquaresVNNIPreQuant(out, residual []float32, q *Q8Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Data
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresInPlaceVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}

// matVecQ8AddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ8AddSumSquaresVNNIPreQuant(out, residual []float32, q *Q8Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Data
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresOutOnlyVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}

// matVecQ4InPlaceAddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ4InPlaceAddSumSquaresVNNIPreQuant(out, residual []float32, q *Q4Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresInPlaceVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}

// matVecQ4AddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ4AddSumSquaresVNNIPreQuant(out, residual []float32, q *Q4Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresOutOnlyVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}

// matVecQ6InPlaceAddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ6InPlaceAddSumSquaresVNNIPreQuant(out, residual []float32, q *Q6Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresInPlaceVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}

// matVecQ6AddSumSquaresVNNIPreQuant — VNNI path with pre-quantized x.
func matVecQ6AddSumSquaresVNNIPreQuant(out, residual []float32, q *Q6Matrix, xq []uint8, scaleX float32, start, end int) float32 {
	data := q.Unpacked
	scale := q.Scale
	rowSum := q.RowSum
	cols := q.Cols
	nRows := end - start
	scratch := getInt32Scratch(nRows)
	defer putInt32Scratch(scratch)
	dotQ8VNNICoreMultiRowZMM(&data[start*cols], &xq[0], &scratch[0], nRows, cols)
	return finalizeAddSumSquaresOutOnlyVNNI(&scratch[0], &rowSum[start], &scale[start], &out[start], &residual[start], nRows, scaleX)
}


// matVecQ4SwiGLUSerialBatchedPreQuant is the VNNI path with pre-quantized x.
func matVecQ4SwiGLUSerialBatchedPreQuant(out, tmpU []float32, gate, up *Q4Matrix, xq []uint8, scaleX float32, start, end int) {
	batchSize := end - start
	gData := gate.Unpacked
	uData := up.Unpacked
	gScale := gate.Scale
	uScale := up.Scale
	gRS := gate.RowSum
	uRS := up.RowSum
	cols := gate.Cols
	if batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		scratchA := getInt32Scratch(batchSize)
		scratchB := getInt32Scratch(batchSize)
		defer putInt32Scratch(scratchA)
		defer putInt32Scratch(scratchB)
		dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &scratchA[0], &scratchB[0], batchSize, cols)
		finalizeDotQ8PairVNNI(&scratchA[0], &gRS[start], &gScale[start], &out[start],
			&scratchB[0], &uRS[start], &uScale[start], &tmpU[start], batchSize, scaleX)
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &sA[0], &sB[0], nRows, cols)
	tmpG := getFloat32Scratch(nRows)
	defer putFloat32Scratch(tmpG)
	finalizeDotQ8PairVNNI(&sA[0], &gRS[start], &gScale[start], &out[start],
		&sB[0], &uRS[start], &uScale[start], &tmpG[0], nRows, scaleX)
	SiLUMulInPlace(out[start:end], tmpG[:nRows])
}

// matVecQ6SwiGLUSerialBatchedPreQuant is the VNNI path with pre-quantized x.
func matVecQ6SwiGLUSerialBatchedPreQuant(out, tmpU []float32, gate, up *Q6Matrix, xq []uint8, scaleX float32, start, end int) {
	batchSize := end - start
	gData := gate.Unpacked
	uData := up.Unpacked
	gScale := gate.Scale
	uScale := up.Scale
	gRS := gate.RowSum
	uRS := up.RowSum
	cols := gate.Cols
	if batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		scratchA := getInt32Scratch(batchSize)
		scratchB := getInt32Scratch(batchSize)
		defer putInt32Scratch(scratchA)
		defer putInt32Scratch(scratchB)
		dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &scratchA[0], &scratchB[0], batchSize, cols)
		finalizeDotQ8PairVNNI(&scratchA[0], &gRS[start], &gScale[start], &out[start],
			&scratchB[0], &uRS[start], &uScale[start], &tmpU[start], batchSize, scaleX)
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &sA[0], &sB[0], nRows, cols)
	tmpG := getFloat32Scratch(nRows)
	defer putFloat32Scratch(tmpG)
	finalizeDotQ8PairVNNI(&sA[0], &gRS[start], &gScale[start], &out[start],
		&sB[0], &uRS[start], &uScale[start], &tmpG[0], nRows, scaleX)
	SiLUMulInPlace(out[start:end], tmpG[:nRows])
}

// matVecQ8SwiGLUSerialBatchedPreQuant is the VNNI path with pre-quantized x.
func matVecQ8SwiGLUSerialBatchedPreQuant(out, tmpU []float32, gate, up *Q8Matrix, xq []uint8, scaleX float32, start, end int) {
	batchSize := end - start
	gData := gate.Data
	uData := up.Data
	gScale := gate.Scale
	uScale := up.Scale
	gRowSum := gate.RowSum
	uRowSum := up.RowSum
	cols := gate.Cols
	if batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		scratchA := getInt32Scratch(batchSize)
		scratchB := getInt32Scratch(batchSize)
		defer putInt32Scratch(scratchA)
		defer putInt32Scratch(scratchB)
		dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &scratchA[0], &scratchB[0], batchSize, cols)
		finalizeDotQ8PairVNNI(&scratchA[0], &gRowSum[start], &gScale[start], &out[start],
			&scratchB[0], &uRowSum[start], &uScale[start], &tmpU[start], batchSize, scaleX)
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	nRows := end - start
	sA := getInt32Scratch(nRows)
	sB := getInt32Scratch(nRows)
	defer putInt32Scratch(sA)
	defer putInt32Scratch(sB)
	dotQ8PairVNNICoreMultiRowZMM(&gData[start*cols], &uData[start*cols], &xq[0], &sA[0], &sB[0], nRows, cols)
	tmpG := getFloat32Scratch(nRows)
	defer putFloat32Scratch(tmpG)
	finalizeDotQ8PairVNNI(&sA[0], &gRowSum[start], &gScale[start], &out[start],
		&sB[0], &uRowSum[start], &uScale[start], &tmpG[0], nRows, scaleX)
	SiLUMulInPlace(out[start:end], tmpG[:nRows])
}
