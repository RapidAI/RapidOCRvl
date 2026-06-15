package tensor

type Q8Matrix struct {
	Rows  int
	Cols  int
	Data  []int8
	Scale []float32
}

type Q4Matrix struct {
	Rows  int
	Cols  int
	Data  []byte
	Scale []float32
}

type Q6Matrix struct {
	Rows  int
	Cols  int
	Data  []byte
	Scale []float32
}

func QuantizeQ8Row(w []float32, rows, cols int) *Q8Matrix {
	q := &Q8Matrix{
		Rows:  rows,
		Cols:  cols,
		Data:  make([]int8, rows*cols),
		Scale: make([]float32, rows),
	}
	if rows*cols >= parallelWork && rows >= 4 {
		parallelRows(rows, func(start, end int) {
			quantizeQ8Rows(w, q, start, end)
		})
		return q
	}
	quantizeQ8Rows(w, q, 0, rows)
	return q
}

func QuantizeQ8RowInto(w []float32, data []int8) float32 {
	maxAbs := maxAbsFloat32(w)
	if maxAbs == 0 {
		clear(data[:len(w)])
		return 1
	}
	scale := maxAbs / 127
	inv := 1 / scale
	i := 0
	for ; i+15 < len(w); i += 16 {
		data[i] = quantInt8(w[i] * inv)
		data[i+1] = quantInt8(w[i+1] * inv)
		data[i+2] = quantInt8(w[i+2] * inv)
		data[i+3] = quantInt8(w[i+3] * inv)
		data[i+4] = quantInt8(w[i+4] * inv)
		data[i+5] = quantInt8(w[i+5] * inv)
		data[i+6] = quantInt8(w[i+6] * inv)
		data[i+7] = quantInt8(w[i+7] * inv)
		data[i+8] = quantInt8(w[i+8] * inv)
		data[i+9] = quantInt8(w[i+9] * inv)
		data[i+10] = quantInt8(w[i+10] * inv)
		data[i+11] = quantInt8(w[i+11] * inv)
		data[i+12] = quantInt8(w[i+12] * inv)
		data[i+13] = quantInt8(w[i+13] * inv)
		data[i+14] = quantInt8(w[i+14] * inv)
		data[i+15] = quantInt8(w[i+15] * inv)
	}
	for ; i+7 < len(w); i += 8 {
		data[i] = quantInt8(w[i] * inv)
		data[i+1] = quantInt8(w[i+1] * inv)
		data[i+2] = quantInt8(w[i+2] * inv)
		data[i+3] = quantInt8(w[i+3] * inv)
		data[i+4] = quantInt8(w[i+4] * inv)
		data[i+5] = quantInt8(w[i+5] * inv)
		data[i+6] = quantInt8(w[i+6] * inv)
		data[i+7] = quantInt8(w[i+7] * inv)
	}
	for ; i < len(w); i++ {
		data[i] = quantInt8(w[i] * inv)
	}
	return scale
}

func QuantizeQ8RowBytesInto(w []float32, data []byte) float32 {
	maxAbs := maxAbsFloat32(w)
	if maxAbs == 0 {
		clear(data[:len(w)])
		return 1
	}
	scale := maxAbs / 127
	inv := 1 / scale
	i := 0
	for ; i+15 < len(w); i += 16 {
		data[i] = byte(quantInt8(w[i] * inv))
		data[i+1] = byte(quantInt8(w[i+1] * inv))
		data[i+2] = byte(quantInt8(w[i+2] * inv))
		data[i+3] = byte(quantInt8(w[i+3] * inv))
		data[i+4] = byte(quantInt8(w[i+4] * inv))
		data[i+5] = byte(quantInt8(w[i+5] * inv))
		data[i+6] = byte(quantInt8(w[i+6] * inv))
		data[i+7] = byte(quantInt8(w[i+7] * inv))
		data[i+8] = byte(quantInt8(w[i+8] * inv))
		data[i+9] = byte(quantInt8(w[i+9] * inv))
		data[i+10] = byte(quantInt8(w[i+10] * inv))
		data[i+11] = byte(quantInt8(w[i+11] * inv))
		data[i+12] = byte(quantInt8(w[i+12] * inv))
		data[i+13] = byte(quantInt8(w[i+13] * inv))
		data[i+14] = byte(quantInt8(w[i+14] * inv))
		data[i+15] = byte(quantInt8(w[i+15] * inv))
	}
	for ; i+7 < len(w); i += 8 {
		data[i] = byte(quantInt8(w[i] * inv))
		data[i+1] = byte(quantInt8(w[i+1] * inv))
		data[i+2] = byte(quantInt8(w[i+2] * inv))
		data[i+3] = byte(quantInt8(w[i+3] * inv))
		data[i+4] = byte(quantInt8(w[i+4] * inv))
		data[i+5] = byte(quantInt8(w[i+5] * inv))
		data[i+6] = byte(quantInt8(w[i+6] * inv))
		data[i+7] = byte(quantInt8(w[i+7] * inv))
	}
	for ; i < len(w); i++ {
		data[i] = byte(quantInt8(w[i] * inv))
	}
	return scale
}

func quantInt8(x float32) int8 {
	v := roundToInt(x)
	if v > 127 {
		v = 127
	} else if v < -127 {
		v = -127
	}
	return int8(v)
}

func quantizeQ8Rows(w []float32, q *Q8Matrix, start, end int) {
	for r := start; r < end; r++ {
		base := r * q.Cols
		q.Scale[r] = QuantizeQ8RowInto(w[base:base+q.Cols], q.Data[base:base+q.Cols])
	}
}

func QuantizeQ4Row(w []float32, rows, cols int) *Q4Matrix {
	q := &Q4Matrix{
		Rows:  rows,
		Cols:  cols,
		Data:  make([]byte, rows*((cols+1)/2)),
		Scale: make([]float32, rows),
	}
	if rows*cols >= parallelWork && rows >= 4 {
		parallelRows(rows, func(start, end int) {
			quantizeQ4Rows(w, q, start, end)
		})
		return q
	}
	quantizeQ4Rows(w, q, 0, rows)
	return q
}

func QuantizeQ6Row(w []float32, rows, cols int) *Q6Matrix {
	q := &Q6Matrix{
		Rows:  rows,
		Cols:  cols,
		Data:  make([]byte, rows*PackedQ6Cols(cols)),
		Scale: make([]float32, rows),
	}
	if rows*cols >= parallelWork && rows >= 4 {
		parallelRows(rows, func(start, end int) {
			quantizeQ6Rows(w, q, start, end)
		})
		return q
	}
	quantizeQ6Rows(w, q, 0, rows)
	return q
}

func QuantizeQ6RowInto(w []float32, data []byte) float32 {
	maxAbs := maxAbsFloat32(w)
	if maxAbs == 0 {
		clear(data)
		return 1
	}
	scale := maxAbs / 31
	inv := 1 / scale
	i := 0
	j := 0
	for ; i+7 < len(w); i, j = i+8, j+6 {
		v0 := quantByte6(w[i] * inv)
		v1 := quantByte6(w[i+1] * inv)
		v2 := quantByte6(w[i+2] * inv)
		v3 := quantByte6(w[i+3] * inv)
		v4 := quantByte6(w[i+4] * inv)
		v5 := quantByte6(w[i+5] * inv)
		v6 := quantByte6(w[i+6] * inv)
		v7 := quantByte6(w[i+7] * inv)
		data[j] = v0 | (v1 << 6)
		data[j+1] = (v1 >> 2) | (v2 << 4)
		data[j+2] = (v2 >> 4) | (v3 << 2)
		data[j+3] = v4 | (v5 << 6)
		data[j+4] = (v5 >> 2) | (v6 << 4)
		data[j+5] = (v6 >> 4) | (v7 << 2)
	}
	for ; i+3 < len(w); i, j = i+4, j+3 {
		v0 := quantByte6(w[i] * inv)
		v1 := quantByte6(w[i+1] * inv)
		v2 := quantByte6(w[i+2] * inv)
		v3 := quantByte6(w[i+3] * inv)
		data[j] = v0 | (v1 << 6)
		data[j+1] = (v1 >> 2) | (v2 << 4)
		data[j+2] = (v2 >> 4) | (v3 << 2)
	}
	if i < len(w) {
		clear(data[j:])
		for ; i < len(w); i++ {
			putQ6(data, i, quantByte6(w[i]*inv))
		}
	}
	return scale
}

func quantByte6(x float32) byte {
	v := roundToInt(x)
	if v > 31 {
		v = 31
	} else if v < -31 {
		v = -31
	}
	return byte(v + 32)
}

func quantizeQ6Rows(w []float32, q *Q6Matrix, start, end int) {
	packedCols := PackedQ6Cols(q.Cols)
	for r := start; r < end; r++ {
		base := r * q.Cols
		row := q.Data[r*packedCols : (r+1)*packedCols]
		q.Scale[r] = QuantizeQ6RowInto(w[base:base+q.Cols], row)
	}
}

func QuantizeQ4RowInto(w []float32, data []byte) float32 {
	maxAbs := maxAbsFloat32(w)
	if maxAbs == 0 {
		clear(data)
		return 1
	}
	scale := maxAbs / 7
	inv := 1 / scale
	i := 0
	j := 0
	for ; i+15 < len(w); i, j = i+16, j+8 {
		data[j] = quantNibble4(w[i]*inv) | (quantNibble4(w[i+1]*inv) << 4)
		data[j+1] = quantNibble4(w[i+2]*inv) | (quantNibble4(w[i+3]*inv) << 4)
		data[j+2] = quantNibble4(w[i+4]*inv) | (quantNibble4(w[i+5]*inv) << 4)
		data[j+3] = quantNibble4(w[i+6]*inv) | (quantNibble4(w[i+7]*inv) << 4)
		data[j+4] = quantNibble4(w[i+8]*inv) | (quantNibble4(w[i+9]*inv) << 4)
		data[j+5] = quantNibble4(w[i+10]*inv) | (quantNibble4(w[i+11]*inv) << 4)
		data[j+6] = quantNibble4(w[i+12]*inv) | (quantNibble4(w[i+13]*inv) << 4)
		data[j+7] = quantNibble4(w[i+14]*inv) | (quantNibble4(w[i+15]*inv) << 4)
	}
	for ; i+7 < len(w); i, j = i+8, j+4 {
		data[j] = quantNibble4(w[i]*inv) | (quantNibble4(w[i+1]*inv) << 4)
		data[j+1] = quantNibble4(w[i+2]*inv) | (quantNibble4(w[i+3]*inv) << 4)
		data[j+2] = quantNibble4(w[i+4]*inv) | (quantNibble4(w[i+5]*inv) << 4)
		data[j+3] = quantNibble4(w[i+6]*inv) | (quantNibble4(w[i+7]*inv) << 4)
	}
	for ; i+1 < len(w); i, j = i+2, j+1 {
		lo := quantNibble4(w[i] * inv)
		hi := quantNibble4(w[i+1] * inv)
		data[j] = lo | (hi << 4)
	}
	if i < len(w) {
		data[j] = quantNibble4(w[i] * inv)
	}
	return scale
}

func quantNibble4(x float32) byte {
	v := roundToInt(x)
	if v > 7 {
		v = 7
	} else if v < -7 {
		v = -7
	}
	return byte(v + 8)
}

func quantizeQ4Rows(w []float32, q *Q4Matrix, start, end int) {
	packedCols := (q.Cols + 1) / 2
	for r := start; r < end; r++ {
		base := r * q.Cols
		row := q.Data[r*packedCols : (r+1)*packedCols]
		q.Scale[r] = QuantizeQ4RowInto(w[base:base+q.Cols], row)
	}
}

func parallelRows(rows int, fn func(start, end int)) {
	parallelFor(rows, fn)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func maxAbsFloat32(x []float32) float32 {
	var m0, m1, m2, m3, m4, m5, m6, m7 float32
	i := 0
	n := len(x)
	for ; i+15 < n; i += 16 {
		m0 = max32(m0, max32(abs32(x[i]), abs32(x[i+8])))
		m1 = max32(m1, max32(abs32(x[i+1]), abs32(x[i+9])))
		m2 = max32(m2, max32(abs32(x[i+2]), abs32(x[i+10])))
		m3 = max32(m3, max32(abs32(x[i+3]), abs32(x[i+11])))
		m4 = max32(m4, max32(abs32(x[i+4]), abs32(x[i+12])))
		m5 = max32(m5, max32(abs32(x[i+5]), abs32(x[i+13])))
		m6 = max32(m6, max32(abs32(x[i+6]), abs32(x[i+14])))
		m7 = max32(m7, max32(abs32(x[i+7]), abs32(x[i+15])))
	}
	for ; i+7 < n; i += 8 {
		m0 = max32(m0, abs32(x[i]))
		m1 = max32(m1, abs32(x[i+1]))
		m2 = max32(m2, abs32(x[i+2]))
		m3 = max32(m3, abs32(x[i+3]))
		m4 = max32(m4, abs32(x[i+4]))
		m5 = max32(m5, abs32(x[i+5]))
		m6 = max32(m6, abs32(x[i+6]))
		m7 = max32(m7, abs32(x[i+7]))
	}
	m := max32(max32(m0, m1), max32(m2, m3))
	m = max32(m, max32(max32(m4, m5), max32(m6, m7)))
	for ; i < n; i++ {
		m = max32(m, abs32(x[i]))
	}
	return m
}

func roundToInt(x float32) int {
	if x >= 0 {
		return int(x + 0.5)
	}
	return int(x - 0.5)
}

func MatVecQ8(out, x []float32, q *Q8Matrix) {
	if q.Rows*q.Cols >= parallelWork && q.Rows >= 4 {
		matVecQ8Parallel(out, x, q)
		return
	}
	matVecQ8Serial(out, x, q, 0, q.Rows)
}

func MatVecQ8Bias(out, x []float32, q *Q8Matrix, bias []float32) {
	if q.Rows*q.Cols >= parallelWork && q.Rows >= 4 {
		parallelFor(q.Rows, func(start, end int) {
			matVecQ8BiasSerial(out, x, q, bias, start, end)
		})
		return
	}
	matVecQ8BiasSerial(out, x, q, bias, 0, q.Rows)
}

func MatVecQ4(out, x []float32, q *Q4Matrix) {
	if q.Rows*q.Cols >= parallelWork && q.Rows >= 4 {
		matVecQ4Parallel(out, x, q)
		return
	}
	matVecQ4Serial(out, x, q, 0, q.Rows)
}

func MatVecQ6(out, x []float32, q *Q6Matrix) {
	if q.Rows*q.Cols >= parallelWork && q.Rows >= 4 {
		matVecQ6Parallel(out, x, q)
		return
	}
	matVecQ6Serial(out, x, q, 0, q.Rows)
}

func FusedMatVec3Q8(outA, outB, outC, x []float32, a, b, c *Q8Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if totalRows*a.Cols >= parallelWork && a.Rows >= 4 {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if totalRows*a.Cols >= parallelWork && totalRows >= 4 {
		fusedMatVec3Q8Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q8Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedMatVec3Q6(outA, outB, outC, x []float32, a, b, c *Q6Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if totalRows*a.Cols >= parallelWork && a.Rows >= 4 {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if totalRows*a.Cols >= parallelWork && totalRows >= 4 {
		fusedMatVec3Q6Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q6Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedMatVec3Q4(outA, outB, outC, x []float32, a, b, c *Q4Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if totalRows*a.Cols >= parallelWork && a.Rows >= 4 {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if totalRows*a.Cols >= parallelWork && totalRows >= 4 {
		fusedMatVec3Q4Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q4Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedSwiGLUQ8(out, x []float32, gate, up, down *Q8Matrix) {
	tmpG := make([]float32, gate.Rows)
	tmpU := make([]float32, up.Rows)
	FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG, tmpU)
}

func FusedSwiGLUQ8Scratch(out, x []float32, gate, up, down *Q8Matrix, tmpG, tmpU []float32) {
	tmpG = tmpG[:gate.Rows]
	tmpU = tmpU[:up.Rows]
	matVecQ8Pair(tmpG, tmpU, x, gate, up)
	SiLUMulInPlace(tmpG, tmpU)
	MatVecQ8(out, tmpG, down)
}

func FusedSwiGLUQ4Scratch(out, x []float32, gate, up, down *Q4Matrix, tmpG, tmpU []float32) {
	tmpG = tmpG[:gate.Rows]
	tmpU = tmpU[:up.Rows]
	matVecQ4Pair(tmpG, tmpU, x, gate, up)
	SiLUMulInPlace(tmpG, tmpU)
	MatVecQ4(out, tmpG, down)
}

func FusedSwiGLUQ6Scratch(out, x []float32, gate, up, down *Q6Matrix, tmpG, tmpU []float32) {
	tmpG = tmpG[:gate.Rows]
	tmpU = tmpU[:up.Rows]
	matVecQ6Pair(tmpG, tmpU, x, gate, up)
	SiLUMulInPlace(tmpG, tmpU)
	MatVecQ6(out, tmpG, down)
}

func matVecQ8Pair(outA, outB, x []float32, a, b *Q8Matrix) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		MatVecQ8(outA, x, a)
		MatVecQ8(outB, x, b)
		return
	}
	if a.Rows*a.Cols >= parallelWork && a.Rows >= 4 {
		parallelFor(a.Rows, func(start, end int) {
			matVecQ8PairSerial(outA, outB, x, a, b, start, end)
		})
		return
	}
	matVecQ8PairSerial(outA, outB, x, a, b, 0, a.Rows)
}

func matVecQ8PairSerial(outA, outB, x []float32, a, b *Q8Matrix, start, end int) {
	for r := start; r < end; r++ {
		base := r * a.Cols
		av, bv := dotQ8Pair(a.Data[base:base+a.Cols], b.Data[base:base+b.Cols], x)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
	}
}

func matVecQ8Parallel(out, x []float32, q *Q8Matrix) {
	parallelFor(q.Rows, func(start, end int) {
		matVecQ8Serial(out, x, q, start, end)
	})
}

func fusedMatVec3Q8Parallel(outA, outB, outC, x []float32, a, b, c *Q8Matrix, totalRows int) {
	parallelFor(totalRows, func(start, end int) {
		fusedMatVec3Q8Serial(outA, outB, outC, x, a, b, c, start, end)
	})
}

func fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x []float32, a, b, c *Q8Matrix, start, end int) {
	for r := start; r < end; r++ {
		base := r * a.Cols
		av, bv, cv := dotQ8Triplet(a.Data[base:base+a.Cols], b.Data[base:base+b.Cols], c.Data[base:base+c.Cols], x)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
		outC[r] = cv * c.Scale[r]
	}
}

func fusedMatVec3Q8Serial(outA, outB, outC, x []float32, a, b, c *Q8Matrix, start, end int) {
	splitB := a.Rows + b.Rows
	aEnd := min(end, a.Rows)
	for r := start; r < aEnd; r++ {
		base := r * a.Cols
		outA[r] = dotQ8(a.Data[base:base+a.Cols], x) * a.Scale[r]
	}
	bStart := max(start, a.Rows)
	bEnd := min(end, splitB)
	for r := bStart; r < bEnd; r++ {
		br := r - a.Rows
		base := br * b.Cols
		outB[br] = dotQ8(b.Data[base:base+b.Cols], x) * b.Scale[br]
	}
	cStart := max(start, splitB)
	for r := cStart; r < end; r++ {
		cr := r - splitB
		base := cr * c.Cols
		outC[cr] = dotQ8(c.Data[base:base+c.Cols], x) * c.Scale[cr]
	}
}

func matVecQ8Serial(out, x []float32, q *Q8Matrix, start, end int) {
	for r := start; r < end; r++ {
		base := r * q.Cols
		out[r] = dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]
	}
}

func matVecQ8BiasSerial(out, x []float32, q *Q8Matrix, bias []float32, start, end int) {
	for r := start; r < end; r++ {
		base := r * q.Cols
		out[r] = dotQ8(q.Data[base:base+q.Cols], x)*q.Scale[r] + bias[r]
	}
}

func matVecQ4Pair(outA, outB, x []float32, a, b *Q4Matrix) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		MatVecQ4(outA, x, a)
		MatVecQ4(outB, x, b)
		return
	}
	if a.Rows*a.Cols >= parallelWork && a.Rows >= 4 {
		parallelFor(a.Rows, func(start, end int) {
			matVecQ4PairSerial(outA, outB, x, a, b, start, end)
		})
		return
	}
	matVecQ4PairSerial(outA, outB, x, a, b, 0, a.Rows)
}

func matVecQ4PairSerial(outA, outB, x []float32, a, b *Q4Matrix, start, end int) {
	packedCols := (a.Cols + 1) / 2
	for r := start; r < end; r++ {
		base := r * packedCols
		av, bv := dotQ4Pair(a.Data[base:base+packedCols], b.Data[base:base+packedCols], x, a.Cols)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
	}
}

func matVecQ6Pair(outA, outB, x []float32, a, b *Q6Matrix) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		MatVecQ6(outA, x, a)
		MatVecQ6(outB, x, b)
		return
	}
	if a.Rows*a.Cols >= parallelWork && a.Rows >= 4 {
		parallelFor(a.Rows, func(start, end int) {
			matVecQ6PairSerial(outA, outB, x, a, b, start, end)
		})
		return
	}
	matVecQ6PairSerial(outA, outB, x, a, b, 0, a.Rows)
}

func matVecQ6PairSerial(outA, outB, x []float32, a, b *Q6Matrix, start, end int) {
	packedCols := PackedQ6Cols(a.Cols)
	for r := start; r < end; r++ {
		base := r * packedCols
		av, bv := dotQ6Pair(a.Data[base:base+packedCols], b.Data[base:base+packedCols], x, a.Cols)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
	}
}

func matVecQ4Parallel(out, x []float32, q *Q4Matrix) {
	parallelFor(q.Rows, func(start, end int) {
		matVecQ4Serial(out, x, q, start, end)
	})
}

func matVecQ6Parallel(out, x []float32, q *Q6Matrix) {
	parallelFor(q.Rows, func(start, end int) {
		matVecQ6Serial(out, x, q, start, end)
	})
}

func fusedMatVec3Q6Parallel(outA, outB, outC, x []float32, a, b, c *Q6Matrix, totalRows int) {
	parallelFor(totalRows, func(start, end int) {
		fusedMatVec3Q6Serial(outA, outB, outC, x, a, b, c, start, end)
	})
}

func fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x []float32, a, b, c *Q6Matrix, start, end int) {
	packedCols := PackedQ6Cols(a.Cols)
	for r := start; r < end; r++ {
		base := r * packedCols
		av, bv, cv := dotQ6Triplet(a.Data[base:base+packedCols], b.Data[base:base+packedCols], c.Data[base:base+packedCols], x, a.Cols)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
		outC[r] = cv * c.Scale[r]
	}
}

func fusedMatVec3Q6Serial(outA, outB, outC, x []float32, a, b, c *Q6Matrix, start, end int) {
	aPacked := PackedQ6Cols(a.Cols)
	bPacked := PackedQ6Cols(b.Cols)
	cPacked := PackedQ6Cols(c.Cols)
	splitB := a.Rows + b.Rows
	aEnd := min(end, a.Rows)
	for r := start; r < aEnd; r++ {
		base := r * aPacked
		outA[r] = dotQ6(a.Data[base:base+aPacked], x, a.Cols) * a.Scale[r]
	}
	bStart := max(start, a.Rows)
	bEnd := min(end, splitB)
	for r := bStart; r < bEnd; r++ {
		br := r - a.Rows
		base := br * bPacked
		outB[br] = dotQ6(b.Data[base:base+bPacked], x, b.Cols) * b.Scale[br]
	}
	cStart := max(start, splitB)
	for r := cStart; r < end; r++ {
		cr := r - splitB
		base := cr * cPacked
		outC[cr] = dotQ6(c.Data[base:base+cPacked], x, c.Cols) * c.Scale[cr]
	}
}

func matVecQ6Serial(out, x []float32, q *Q6Matrix, start, end int) {
	packedCols := PackedQ6Cols(q.Cols)
	for r := start; r < end; r++ {
		base := r * packedCols
		out[r] = dotQ6(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]
	}
}

func matVecQ4Serial(out, x []float32, q *Q4Matrix, start, end int) {
	packedCols := (q.Cols + 1) / 2
	for r := start; r < end; r++ {
		base := r * packedCols
		out[r] = dotQ4(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]
	}
}

func fusedMatVec3Q4Parallel(outA, outB, outC, x []float32, a, b, c *Q4Matrix, totalRows int) {
	parallelFor(totalRows, func(start, end int) {
		fusedMatVec3Q4Serial(outA, outB, outC, x, a, b, c, start, end)
	})
}

func fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x []float32, a, b, c *Q4Matrix, start, end int) {
	packedCols := (a.Cols + 1) / 2
	for r := start; r < end; r++ {
		base := r * packedCols
		av, bv, cv := dotQ4Triplet(a.Data[base:base+packedCols], b.Data[base:base+packedCols], c.Data[base:base+packedCols], x, a.Cols)
		outA[r] = av * a.Scale[r]
		outB[r] = bv * b.Scale[r]
		outC[r] = cv * c.Scale[r]
	}
}

func fusedMatVec3Q4Serial(outA, outB, outC, x []float32, a, b, c *Q4Matrix, start, end int) {
	aPacked := (a.Cols + 1) / 2
	bPacked := (b.Cols + 1) / 2
	cPacked := (c.Cols + 1) / 2
	splitB := a.Rows + b.Rows
	aEnd := min(end, a.Rows)
	for r := start; r < aEnd; r++ {
		base := r * aPacked
		outA[r] = dotQ4(a.Data[base:base+aPacked], x, a.Cols) * a.Scale[r]
	}
	bStart := max(start, a.Rows)
	bEnd := min(end, splitB)
	for r := bStart; r < bEnd; r++ {
		br := r - a.Rows
		base := br * bPacked
		outB[br] = dotQ4(b.Data[base:base+bPacked], x, b.Cols) * b.Scale[br]
	}
	cStart := max(start, splitB)
	for r := cStart; r < end; r++ {
		cr := r - splitB
		base := cr * cPacked
		outC[cr] = dotQ4(c.Data[base:base+cPacked], x, c.Cols) * c.Scale[cr]
	}
}

func dotQ8(a []int8, b []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	n := len(a)
	for ; i+15 < n; i += 16 {
		s0 += q8Value(a[i])*b[i] + q8Value(a[i+8])*b[i+8]
		s1 += q8Value(a[i+1])*b[i+1] + q8Value(a[i+9])*b[i+9]
		s2 += q8Value(a[i+2])*b[i+2] + q8Value(a[i+10])*b[i+10]
		s3 += q8Value(a[i+3])*b[i+3] + q8Value(a[i+11])*b[i+11]
		s4 += q8Value(a[i+4])*b[i+4] + q8Value(a[i+12])*b[i+12]
		s5 += q8Value(a[i+5])*b[i+5] + q8Value(a[i+13])*b[i+13]
		s6 += q8Value(a[i+6])*b[i+6] + q8Value(a[i+14])*b[i+14]
		s7 += q8Value(a[i+7])*b[i+7] + q8Value(a[i+15])*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += q8Value(a[i]) * b[i]
	}
	return sum
}

func dotQ8Pair(a, b []int8, x []float32) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	n := len(x)
	for ; i+15 < n; i += 16 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		x8, x9, x10, x11 := x[i+8], x[i+9], x[i+10], x[i+11]
		x12, x13, x14, x15 := x[i+12], x[i+13], x[i+14], x[i+15]
		a0 += q8Value(a[i])*x0 + q8Value(a[i+4])*x4 + q8Value(a[i+8])*x8 + q8Value(a[i+12])*x12
		a1 += q8Value(a[i+1])*x1 + q8Value(a[i+5])*x5 + q8Value(a[i+9])*x9 + q8Value(a[i+13])*x13
		a2 += q8Value(a[i+2])*x2 + q8Value(a[i+6])*x6 + q8Value(a[i+10])*x10 + q8Value(a[i+14])*x14
		a3 += q8Value(a[i+3])*x3 + q8Value(a[i+7])*x7 + q8Value(a[i+11])*x11 + q8Value(a[i+15])*x15
		b0 += q8Value(b[i])*x0 + q8Value(b[i+4])*x4 + q8Value(b[i+8])*x8 + q8Value(b[i+12])*x12
		b1 += q8Value(b[i+1])*x1 + q8Value(b[i+5])*x5 + q8Value(b[i+9])*x9 + q8Value(b[i+13])*x13
		b2 += q8Value(b[i+2])*x2 + q8Value(b[i+6])*x6 + q8Value(b[i+10])*x10 + q8Value(b[i+14])*x14
		b3 += q8Value(b[i+3])*x3 + q8Value(b[i+7])*x7 + q8Value(b[i+11])*x11 + q8Value(b[i+15])*x15
	}
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q8Value(a[i])*x0 + q8Value(a[i+4])*x4
		a1 += q8Value(a[i+1])*x1 + q8Value(a[i+5])*x5
		a2 += q8Value(a[i+2])*x2 + q8Value(a[i+6])*x6
		a3 += q8Value(a[i+3])*x3 + q8Value(a[i+7])*x7
		b0 += q8Value(b[i])*x0 + q8Value(b[i+4])*x4
		b1 += q8Value(b[i+1])*x1 + q8Value(b[i+5])*x5
		b2 += q8Value(b[i+2])*x2 + q8Value(b[i+6])*x6
		b3 += q8Value(b[i+3])*x3 + q8Value(b[i+7])*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	for ; i < n; i++ {
		xi := x[i]
		sa += q8Value(a[i]) * xi
		sb += q8Value(b[i]) * xi
	}
	return sa, sb
}

func dotQ8Triplet(a, b, c []int8, x []float32) (float32, float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 float32
	i := 0
	n := len(x)
	for ; i+15 < n; i += 16 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		x8, x9, x10, x11 := x[i+8], x[i+9], x[i+10], x[i+11]
		x12, x13, x14, x15 := x[i+12], x[i+13], x[i+14], x[i+15]
		a0 += q8Value(a[i])*x0 + q8Value(a[i+4])*x4 + q8Value(a[i+8])*x8 + q8Value(a[i+12])*x12
		a1 += q8Value(a[i+1])*x1 + q8Value(a[i+5])*x5 + q8Value(a[i+9])*x9 + q8Value(a[i+13])*x13
		a2 += q8Value(a[i+2])*x2 + q8Value(a[i+6])*x6 + q8Value(a[i+10])*x10 + q8Value(a[i+14])*x14
		a3 += q8Value(a[i+3])*x3 + q8Value(a[i+7])*x7 + q8Value(a[i+11])*x11 + q8Value(a[i+15])*x15
		b0 += q8Value(b[i])*x0 + q8Value(b[i+4])*x4 + q8Value(b[i+8])*x8 + q8Value(b[i+12])*x12
		b1 += q8Value(b[i+1])*x1 + q8Value(b[i+5])*x5 + q8Value(b[i+9])*x9 + q8Value(b[i+13])*x13
		b2 += q8Value(b[i+2])*x2 + q8Value(b[i+6])*x6 + q8Value(b[i+10])*x10 + q8Value(b[i+14])*x14
		b3 += q8Value(b[i+3])*x3 + q8Value(b[i+7])*x7 + q8Value(b[i+11])*x11 + q8Value(b[i+15])*x15
		c0 += q8Value(c[i])*x0 + q8Value(c[i+4])*x4 + q8Value(c[i+8])*x8 + q8Value(c[i+12])*x12
		c1 += q8Value(c[i+1])*x1 + q8Value(c[i+5])*x5 + q8Value(c[i+9])*x9 + q8Value(c[i+13])*x13
		c2 += q8Value(c[i+2])*x2 + q8Value(c[i+6])*x6 + q8Value(c[i+10])*x10 + q8Value(c[i+14])*x14
		c3 += q8Value(c[i+3])*x3 + q8Value(c[i+7])*x7 + q8Value(c[i+11])*x11 + q8Value(c[i+15])*x15
	}
	for ; i+7 < n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q8Value(a[i])*x0 + q8Value(a[i+4])*x4
		a1 += q8Value(a[i+1])*x1 + q8Value(a[i+5])*x5
		a2 += q8Value(a[i+2])*x2 + q8Value(a[i+6])*x6
		a3 += q8Value(a[i+3])*x3 + q8Value(a[i+7])*x7
		b0 += q8Value(b[i])*x0 + q8Value(b[i+4])*x4
		b1 += q8Value(b[i+1])*x1 + q8Value(b[i+5])*x5
		b2 += q8Value(b[i+2])*x2 + q8Value(b[i+6])*x6
		b3 += q8Value(b[i+3])*x3 + q8Value(b[i+7])*x7
		c0 += q8Value(c[i])*x0 + q8Value(c[i+4])*x4
		c1 += q8Value(c[i+1])*x1 + q8Value(c[i+5])*x5
		c2 += q8Value(c[i+2])*x2 + q8Value(c[i+6])*x6
		c3 += q8Value(c[i+3])*x3 + q8Value(c[i+7])*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	sc := (c0 + c1) + (c2 + c3)
	for ; i < n; i++ {
		xi := x[i]
		sa += q8Value(a[i]) * xi
		sb += q8Value(b[i]) * xi
		sc += q8Value(c[i]) * xi
	}
	return sa, sb, sc
}

func q8Value(v int8) float32 {
	return q8ValueTable[byte(v)]
}

var q8ValueTable = func() [256]float32 {
	var t [256]float32
	for i := range t {
		t[i] = float32(int8(byte(i)))
	}
	return t
}()

func dotQ4(a []byte, b []float32, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+15 < cols; i += 16 {
		base := i / 2
		p0, p1, p2, p3 := a[base], a[base+1], a[base+2], a[base+3]
		p4, p5, p6, p7 := a[base+4], a[base+5], a[base+6], a[base+7]
		s0 += q4ByteLo[p0]*b[i] + q4ByteLo[p4]*b[i+8]
		s1 += q4ByteHi[p0]*b[i+1] + q4ByteHi[p4]*b[i+9]
		s2 += q4ByteLo[p1]*b[i+2] + q4ByteLo[p5]*b[i+10]
		s3 += q4ByteHi[p1]*b[i+3] + q4ByteHi[p5]*b[i+11]
		s4 += q4ByteLo[p2]*b[i+4] + q4ByteLo[p6]*b[i+12]
		s5 += q4ByteHi[p2]*b[i+5] + q4ByteHi[p6]*b[i+13]
		s6 += q4ByteLo[p3]*b[i+6] + q4ByteLo[p7]*b[i+14]
		s7 += q4ByteHi[p3]*b[i+7] + q4ByteHi[p7]*b[i+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < cols; i++ {
		p := a[i/2]
		v := q4ByteLo[p]
		if i&1 == 1 {
			v = q4ByteHi[p]
		}
		sum += v * b[i]
	}
	return sum
}

func dotQ4Pair(a, b []byte, x []float32, cols int) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	fullBytes := cols / 2
	for ; i+7 < fullBytes; i += 8 {
		ax0, ax1, ax2, ax3 := a[i], a[i+1], a[i+2], a[i+3]
		ax4, ax5, ax6, ax7 := a[i+4], a[i+5], a[i+6], a[i+7]
		bx0, bx1, bx2, bx3 := b[i], b[i+1], b[i+2], b[i+3]
		bx4, bx5, bx6, bx7 := b[i+4], b[i+5], b[i+6], b[i+7]
		j := i * 2
		x0, x1, x2, x3 := x[j], x[j+1], x[j+2], x[j+3]
		x4, x5, x6, x7 := x[j+4], x[j+5], x[j+6], x[j+7]
		x8, x9, x10, x11 := x[j+8], x[j+9], x[j+10], x[j+11]
		x12, x13, x14, x15 := x[j+12], x[j+13], x[j+14], x[j+15]
		a0 += q4ByteLo[ax0]*x0 + q4ByteLo[ax2]*x4 + q4ByteLo[ax4]*x8 + q4ByteLo[ax6]*x12
		a1 += q4ByteHi[ax0]*x1 + q4ByteHi[ax2]*x5 + q4ByteHi[ax4]*x9 + q4ByteHi[ax6]*x13
		a2 += q4ByteLo[ax1]*x2 + q4ByteLo[ax3]*x6 + q4ByteLo[ax5]*x10 + q4ByteLo[ax7]*x14
		a3 += q4ByteHi[ax1]*x3 + q4ByteHi[ax3]*x7 + q4ByteHi[ax5]*x11 + q4ByteHi[ax7]*x15
		b0 += q4ByteLo[bx0]*x0 + q4ByteLo[bx2]*x4 + q4ByteLo[bx4]*x8 + q4ByteLo[bx6]*x12
		b1 += q4ByteHi[bx0]*x1 + q4ByteHi[bx2]*x5 + q4ByteHi[bx4]*x9 + q4ByteHi[bx6]*x13
		b2 += q4ByteLo[bx1]*x2 + q4ByteLo[bx3]*x6 + q4ByteLo[bx5]*x10 + q4ByteLo[bx7]*x14
		b3 += q4ByteHi[bx1]*x3 + q4ByteHi[bx3]*x7 + q4ByteHi[bx5]*x11 + q4ByteHi[bx7]*x15
	}
	for ; i+3 < fullBytes; i += 4 {
		ax0, ax1, ax2, ax3 := a[i], a[i+1], a[i+2], a[i+3]
		bx0, bx1, bx2, bx3 := b[i], b[i+1], b[i+2], b[i+3]
		x0, x1, x2, x3 := x[i*2], x[i*2+1], x[i*2+2], x[i*2+3]
		x4, x5, x6, x7 := x[i*2+4], x[i*2+5], x[i*2+6], x[i*2+7]
		a0 += q4ByteLo[ax0]*x0 + q4ByteLo[ax2]*x4
		a1 += q4ByteHi[ax0]*x1 + q4ByteHi[ax2]*x5
		a2 += q4ByteLo[ax1]*x2 + q4ByteLo[ax3]*x6
		a3 += q4ByteHi[ax1]*x3 + q4ByteHi[ax3]*x7
		b0 += q4ByteLo[bx0]*x0 + q4ByteLo[bx2]*x4
		b1 += q4ByteHi[bx0]*x1 + q4ByteHi[bx2]*x5
		b2 += q4ByteLo[bx1]*x2 + q4ByteLo[bx3]*x6
		b3 += q4ByteHi[bx1]*x3 + q4ByteHi[bx3]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	for ; i < fullBytes; i++ {
		ai, bi := a[i], b[i]
		x0, x1 := x[i*2], x[i*2+1]
		sa += q4ByteLo[ai]*x0 + q4ByteHi[ai]*x1
		sb += q4ByteLo[bi]*x0 + q4ByteHi[bi]*x1
	}
	if cols&1 == 1 {
		idx := cols / 2
		xi := x[cols-1]
		sa += q4ByteLo[a[idx]] * xi
		sb += q4ByteLo[b[idx]] * xi
	}
	return sa, sb
}

func dotQ4Triplet(a, b, c []byte, x []float32, cols int) (float32, float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 float32
	i := 0
	fullBytes := cols / 2
	for ; i+7 < fullBytes; i += 8 {
		ax0, ax1, ax2, ax3 := a[i], a[i+1], a[i+2], a[i+3]
		ax4, ax5, ax6, ax7 := a[i+4], a[i+5], a[i+6], a[i+7]
		bx0, bx1, bx2, bx3 := b[i], b[i+1], b[i+2], b[i+3]
		bx4, bx5, bx6, bx7 := b[i+4], b[i+5], b[i+6], b[i+7]
		cx0, cx1, cx2, cx3 := c[i], c[i+1], c[i+2], c[i+3]
		cx4, cx5, cx6, cx7 := c[i+4], c[i+5], c[i+6], c[i+7]
		j := i * 2
		x0, x1, x2, x3 := x[j], x[j+1], x[j+2], x[j+3]
		x4, x5, x6, x7 := x[j+4], x[j+5], x[j+6], x[j+7]
		x8, x9, x10, x11 := x[j+8], x[j+9], x[j+10], x[j+11]
		x12, x13, x14, x15 := x[j+12], x[j+13], x[j+14], x[j+15]
		a0 += q4ByteLo[ax0]*x0 + q4ByteLo[ax2]*x4 + q4ByteLo[ax4]*x8 + q4ByteLo[ax6]*x12
		a1 += q4ByteHi[ax0]*x1 + q4ByteHi[ax2]*x5 + q4ByteHi[ax4]*x9 + q4ByteHi[ax6]*x13
		a2 += q4ByteLo[ax1]*x2 + q4ByteLo[ax3]*x6 + q4ByteLo[ax5]*x10 + q4ByteLo[ax7]*x14
		a3 += q4ByteHi[ax1]*x3 + q4ByteHi[ax3]*x7 + q4ByteHi[ax5]*x11 + q4ByteHi[ax7]*x15
		b0 += q4ByteLo[bx0]*x0 + q4ByteLo[bx2]*x4 + q4ByteLo[bx4]*x8 + q4ByteLo[bx6]*x12
		b1 += q4ByteHi[bx0]*x1 + q4ByteHi[bx2]*x5 + q4ByteHi[bx4]*x9 + q4ByteHi[bx6]*x13
		b2 += q4ByteLo[bx1]*x2 + q4ByteLo[bx3]*x6 + q4ByteLo[bx5]*x10 + q4ByteLo[bx7]*x14
		b3 += q4ByteHi[bx1]*x3 + q4ByteHi[bx3]*x7 + q4ByteHi[bx5]*x11 + q4ByteHi[bx7]*x15
		c0 += q4ByteLo[cx0]*x0 + q4ByteLo[cx2]*x4 + q4ByteLo[cx4]*x8 + q4ByteLo[cx6]*x12
		c1 += q4ByteHi[cx0]*x1 + q4ByteHi[cx2]*x5 + q4ByteHi[cx4]*x9 + q4ByteHi[cx6]*x13
		c2 += q4ByteLo[cx1]*x2 + q4ByteLo[cx3]*x6 + q4ByteLo[cx5]*x10 + q4ByteLo[cx7]*x14
		c3 += q4ByteHi[cx1]*x3 + q4ByteHi[cx3]*x7 + q4ByteHi[cx5]*x11 + q4ByteHi[cx7]*x15
	}
	for ; i+3 < fullBytes; i += 4 {
		ax0, ax1, ax2, ax3 := a[i], a[i+1], a[i+2], a[i+3]
		bx0, bx1, bx2, bx3 := b[i], b[i+1], b[i+2], b[i+3]
		cx0, cx1, cx2, cx3 := c[i], c[i+1], c[i+2], c[i+3]
		j := i * 2
		x0, x1, x2, x3 := x[j], x[j+1], x[j+2], x[j+3]
		x4, x5, x6, x7 := x[j+4], x[j+5], x[j+6], x[j+7]
		a0 += q4ByteLo[ax0]*x0 + q4ByteLo[ax2]*x4
		a1 += q4ByteHi[ax0]*x1 + q4ByteHi[ax2]*x5
		a2 += q4ByteLo[ax1]*x2 + q4ByteLo[ax3]*x6
		a3 += q4ByteHi[ax1]*x3 + q4ByteHi[ax3]*x7
		b0 += q4ByteLo[bx0]*x0 + q4ByteLo[bx2]*x4
		b1 += q4ByteHi[bx0]*x1 + q4ByteHi[bx2]*x5
		b2 += q4ByteLo[bx1]*x2 + q4ByteLo[bx3]*x6
		b3 += q4ByteHi[bx1]*x3 + q4ByteHi[bx3]*x7
		c0 += q4ByteLo[cx0]*x0 + q4ByteLo[cx2]*x4
		c1 += q4ByteHi[cx0]*x1 + q4ByteHi[cx2]*x5
		c2 += q4ByteLo[cx1]*x2 + q4ByteLo[cx3]*x6
		c3 += q4ByteHi[cx1]*x3 + q4ByteHi[cx3]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	sc := (c0 + c1) + (c2 + c3)
	for ; i < fullBytes; i++ {
		ai, bi, ci := a[i], b[i], c[i]
		x0, x1 := x[i*2], x[i*2+1]
		sa += q4ByteLo[ai]*x0 + q4ByteHi[ai]*x1
		sb += q4ByteLo[bi]*x0 + q4ByteHi[bi]*x1
		sc += q4ByteLo[ci]*x0 + q4ByteHi[ci]*x1
	}
	if cols&1 == 1 {
		idx := cols / 2
		xi := x[cols-1]
		sa += q4ByteLo[a[idx]] * xi
		sb += q4ByteLo[b[idx]] * xi
		sc += q4ByteLo[c[idx]] * xi
	}
	return sa, sb, sc
}

func q4Value(nib byte) float32 {
	return q4ValueTable[nib&0x0F]
}

var q4ValueTable = [16]float32{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}

var q4ByteLo, q4ByteHi = func() ([256]float32, [256]float32) {
	var lo, hi [256]float32
	for i := range lo {
		b := byte(i)
		lo[i] = q4Value(b & 0x0F)
		hi[i] = q4Value(b >> 4)
	}
	return lo, hi
}()

func dotQ6(a []byte, b []float32, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+15 < cols; i += 16 {
		base := (i * 6) / 8
		b0 := uint32(a[base])
		b1 := uint32(a[base+1])
		b2 := uint32(a[base+2])
		b3 := uint32(a[base+3])
		b4 := uint32(a[base+4])
		b5 := uint32(a[base+5])
		b6 := uint32(a[base+6])
		b7 := uint32(a[base+7])
		b8 := uint32(a[base+8])
		b9 := uint32(a[base+9])
		b10 := uint32(a[base+10])
		b11 := uint32(a[base+11])
		s0 += q6ValueTable[byte(b0&0x3F)]*b[i] + q6ValueTable[byte(b6&0x3F)]*b[i+8]
		s1 += q6ValueTable[byte(((b0>>6)|((b1&0x0F)<<2))&0x3F)]*b[i+1] + q6ValueTable[byte(((b6>>6)|((b7&0x0F)<<2))&0x3F)]*b[i+9]
		s2 += q6ValueTable[byte(((b1>>4)|((b2&0x03)<<4))&0x3F)]*b[i+2] + q6ValueTable[byte(((b7>>4)|((b8&0x03)<<4))&0x3F)]*b[i+10]
		s3 += q6ValueTable[byte((b2>>2)&0x3F)]*b[i+3] + q6ValueTable[byte((b8>>2)&0x3F)]*b[i+11]
		s4 += q6ValueTable[byte(b3&0x3F)]*b[i+4] + q6ValueTable[byte(b9&0x3F)]*b[i+12]
		s5 += q6ValueTable[byte(((b3>>6)|((b4&0x0F)<<2))&0x3F)]*b[i+5] + q6ValueTable[byte(((b9>>6)|((b10&0x0F)<<2))&0x3F)]*b[i+13]
		s6 += q6ValueTable[byte(((b4>>4)|((b5&0x03)<<4))&0x3F)]*b[i+6] + q6ValueTable[byte(((b10>>4)|((b11&0x03)<<4))&0x3F)]*b[i+14]
		s7 += q6ValueTable[byte((b5>>2)&0x3F)]*b[i+7] + q6ValueTable[byte((b11>>2)&0x3F)]*b[i+15]
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		b0 := uint32(a[base])
		b1 := uint32(a[base+1])
		b2 := uint32(a[base+2])
		b3 := uint32(a[base+3])
		b4 := uint32(a[base+4])
		b5 := uint32(a[base+5])
		s0 += q6ValueTable[byte(b0&0x3F)] * b[i]
		s1 += q6ValueTable[byte(((b0>>6)|((b1&0x0F)<<2))&0x3F)] * b[i+1]
		s2 += q6ValueTable[byte(((b1>>4)|((b2&0x03)<<4))&0x3F)] * b[i+2]
		s3 += q6ValueTable[byte((b2>>2)&0x3F)] * b[i+3]
		s4 += q6ValueTable[byte(b3&0x3F)] * b[i+4]
		s5 += q6ValueTable[byte(((b3>>6)|((b4&0x0F)<<2))&0x3F)] * b[i+5]
		s6 += q6ValueTable[byte(((b4>>4)|((b5&0x03)<<4))&0x3F)] * b[i+6]
		s7 += q6ValueTable[byte((b5>>2)&0x3F)] * b[i+7]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < cols; i++ {
		sum += q6ValueTable[getQ6(a, i)] * b[i]
	}
	return sum
}

func dotQ6Pair(a, b []byte, x []float32, cols int) (float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3 float32
	i := 0
	for ; i+15 < cols; i += 16 {
		base := (i * 6) / 8
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		x8, x9, x10, x11 := x[i+8], x[i+9], x[i+10], x[i+11]
		x12, x13, x14, x15 := x[i+12], x[i+13], x[i+14], x[i+15]
		av0, av1, av2, av3 := uint32(a[base]), uint32(a[base+1]), uint32(a[base+2]), uint32(a[base+3])
		av4, av5, av6, av7 := uint32(a[base+4]), uint32(a[base+5]), uint32(a[base+6]), uint32(a[base+7])
		av8, av9, av10, av11 := uint32(a[base+8]), uint32(a[base+9]), uint32(a[base+10]), uint32(a[base+11])
		bv0, bv1, bv2, bv3 := uint32(b[base]), uint32(b[base+1]), uint32(b[base+2]), uint32(b[base+3])
		bv4, bv5, bv6, bv7 := uint32(b[base+4]), uint32(b[base+5]), uint32(b[base+6]), uint32(b[base+7])
		bv8, bv9, bv10, bv11 := uint32(b[base+8]), uint32(b[base+9]), uint32(b[base+10]), uint32(b[base+11])
		a0 += q6ValueTable[byte(av0&0x3F)]*x0 + q6ValueTable[byte(av6&0x3F)]*x8
		a1 += q6ValueTable[byte(((av0>>6)|((av1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((av6>>6)|((av7&0x0F)<<2))&0x3F)]*x9
		a2 += q6ValueTable[byte(((av1>>4)|((av2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((av7>>4)|((av8&0x03)<<4))&0x3F)]*x10
		a3 += q6ValueTable[byte((av2>>2)&0x3F)]*x3 + q6ValueTable[byte((av8>>2)&0x3F)]*x11
		a0 += q6ValueTable[byte(av3&0x3F)]*x4 + q6ValueTable[byte(av9&0x3F)]*x12
		a1 += q6ValueTable[byte(((av3>>6)|((av4&0x0F)<<2))&0x3F)]*x5 + q6ValueTable[byte(((av9>>6)|((av10&0x0F)<<2))&0x3F)]*x13
		a2 += q6ValueTable[byte(((av4>>4)|((av5&0x03)<<4))&0x3F)]*x6 + q6ValueTable[byte(((av10>>4)|((av11&0x03)<<4))&0x3F)]*x14
		a3 += q6ValueTable[byte((av5>>2)&0x3F)]*x7 + q6ValueTable[byte((av11>>2)&0x3F)]*x15
		b0 += q6ValueTable[byte(bv0&0x3F)]*x0 + q6ValueTable[byte(bv6&0x3F)]*x8
		b1 += q6ValueTable[byte(((bv0>>6)|((bv1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((bv6>>6)|((bv7&0x0F)<<2))&0x3F)]*x9
		b2 += q6ValueTable[byte(((bv1>>4)|((bv2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((bv7>>4)|((bv8&0x03)<<4))&0x3F)]*x10
		b3 += q6ValueTable[byte((bv2>>2)&0x3F)]*x3 + q6ValueTable[byte((bv8>>2)&0x3F)]*x11
		b0 += q6ValueTable[byte(bv3&0x3F)]*x4 + q6ValueTable[byte(bv9&0x3F)]*x12
		b1 += q6ValueTable[byte(((bv3>>6)|((bv4&0x0F)<<2))&0x3F)]*x5 + q6ValueTable[byte(((bv9>>6)|((bv10&0x0F)<<2))&0x3F)]*x13
		b2 += q6ValueTable[byte(((bv4>>4)|((bv5&0x03)<<4))&0x3F)]*x6 + q6ValueTable[byte(((bv10>>4)|((bv11&0x03)<<4))&0x3F)]*x14
		b3 += q6ValueTable[byte((bv5>>2)&0x3F)]*x7 + q6ValueTable[byte((bv11>>2)&0x3F)]*x15
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		aa0 := uint32(a[base])
		aa1 := uint32(a[base+1])
		aa2 := uint32(a[base+2])
		aa3 := uint32(a[base+3])
		aa4 := uint32(a[base+4])
		aa5 := uint32(a[base+5])
		bb0 := uint32(b[base])
		bb1 := uint32(b[base+1])
		bb2 := uint32(b[base+2])
		bb3 := uint32(b[base+3])
		bb4 := uint32(b[base+4])
		bb5 := uint32(b[base+5])
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q6ValueTable[byte(aa0&0x3F)]*x0 + q6ValueTable[byte(aa3&0x3F)]*x4
		a1 += q6ValueTable[byte(((aa0>>6)|((aa1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((aa3>>6)|((aa4&0x0F)<<2))&0x3F)]*x5
		a2 += q6ValueTable[byte(((aa1>>4)|((aa2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((aa4>>4)|((aa5&0x03)<<4))&0x3F)]*x6
		a3 += q6ValueTable[byte((aa2>>2)&0x3F)]*x3 + q6ValueTable[byte((aa5>>2)&0x3F)]*x7
		b0 += q6ValueTable[byte(bb0&0x3F)]*x0 + q6ValueTable[byte(bb3&0x3F)]*x4
		b1 += q6ValueTable[byte(((bb0>>6)|((bb1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((bb3>>6)|((bb4&0x0F)<<2))&0x3F)]*x5
		b2 += q6ValueTable[byte(((bb1>>4)|((bb2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((bb4>>4)|((bb5&0x03)<<4))&0x3F)]*x6
		b3 += q6ValueTable[byte((bb2>>2)&0x3F)]*x3 + q6ValueTable[byte((bb5>>2)&0x3F)]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	for ; i < cols; i++ {
		xi := x[i]
		sa += q6ValueTable[getQ6(a, i)] * xi
		sb += q6ValueTable[getQ6(b, i)] * xi
	}
	return sa, sb
}

func dotQ6Triplet(a, b, c []byte, x []float32, cols int) (float32, float32, float32) {
	var a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3 float32
	i := 0
	for ; i+15 < cols; i += 16 {
		base := (i * 6) / 8
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		x8, x9, x10, x11 := x[i+8], x[i+9], x[i+10], x[i+11]
		x12, x13, x14, x15 := x[i+12], x[i+13], x[i+14], x[i+15]
		av0, av1, av2, av3 := uint32(a[base]), uint32(a[base+1]), uint32(a[base+2]), uint32(a[base+3])
		av4, av5, av6, av7 := uint32(a[base+4]), uint32(a[base+5]), uint32(a[base+6]), uint32(a[base+7])
		av8, av9, av10, av11 := uint32(a[base+8]), uint32(a[base+9]), uint32(a[base+10]), uint32(a[base+11])
		bv0, bv1, bv2, bv3 := uint32(b[base]), uint32(b[base+1]), uint32(b[base+2]), uint32(b[base+3])
		bv4, bv5, bv6, bv7 := uint32(b[base+4]), uint32(b[base+5]), uint32(b[base+6]), uint32(b[base+7])
		bv8, bv9, bv10, bv11 := uint32(b[base+8]), uint32(b[base+9]), uint32(b[base+10]), uint32(b[base+11])
		cv0, cv1, cv2, cv3 := uint32(c[base]), uint32(c[base+1]), uint32(c[base+2]), uint32(c[base+3])
		cv4, cv5, cv6, cv7 := uint32(c[base+4]), uint32(c[base+5]), uint32(c[base+6]), uint32(c[base+7])
		cv8, cv9, cv10, cv11 := uint32(c[base+8]), uint32(c[base+9]), uint32(c[base+10]), uint32(c[base+11])
		aq0 := q6ValueTable[byte(av0&0x3F)]
		aq1 := q6ValueTable[byte(((av0>>6)|((av1&0x0F)<<2))&0x3F)]
		aq2 := q6ValueTable[byte(((av1>>4)|((av2&0x03)<<4))&0x3F)]
		aq3 := q6ValueTable[byte((av2>>2)&0x3F)]
		aq4 := q6ValueTable[byte(av3&0x3F)]
		aq5 := q6ValueTable[byte(((av3>>6)|((av4&0x0F)<<2))&0x3F)]
		aq6 := q6ValueTable[byte(((av4>>4)|((av5&0x03)<<4))&0x3F)]
		aq7 := q6ValueTable[byte((av5>>2)&0x3F)]
		aq8 := q6ValueTable[byte(av6&0x3F)]
		aq9 := q6ValueTable[byte(((av6>>6)|((av7&0x0F)<<2))&0x3F)]
		aq10 := q6ValueTable[byte(((av7>>4)|((av8&0x03)<<4))&0x3F)]
		aq11 := q6ValueTable[byte((av8>>2)&0x3F)]
		aq12 := q6ValueTable[byte(av9&0x3F)]
		aq13 := q6ValueTable[byte(((av9>>6)|((av10&0x0F)<<2))&0x3F)]
		aq14 := q6ValueTable[byte(((av10>>4)|((av11&0x03)<<4))&0x3F)]
		aq15 := q6ValueTable[byte((av11>>2)&0x3F)]
		bq0 := q6ValueTable[byte(bv0&0x3F)]
		bq1 := q6ValueTable[byte(((bv0>>6)|((bv1&0x0F)<<2))&0x3F)]
		bq2 := q6ValueTable[byte(((bv1>>4)|((bv2&0x03)<<4))&0x3F)]
		bq3 := q6ValueTable[byte((bv2>>2)&0x3F)]
		bq4 := q6ValueTable[byte(bv3&0x3F)]
		bq5 := q6ValueTable[byte(((bv3>>6)|((bv4&0x0F)<<2))&0x3F)]
		bq6 := q6ValueTable[byte(((bv4>>4)|((bv5&0x03)<<4))&0x3F)]
		bq7 := q6ValueTable[byte((bv5>>2)&0x3F)]
		bq8 := q6ValueTable[byte(bv6&0x3F)]
		bq9 := q6ValueTable[byte(((bv6>>6)|((bv7&0x0F)<<2))&0x3F)]
		bq10 := q6ValueTable[byte(((bv7>>4)|((bv8&0x03)<<4))&0x3F)]
		bq11 := q6ValueTable[byte((bv8>>2)&0x3F)]
		bq12 := q6ValueTable[byte(bv9&0x3F)]
		bq13 := q6ValueTable[byte(((bv9>>6)|((bv10&0x0F)<<2))&0x3F)]
		bq14 := q6ValueTable[byte(((bv10>>4)|((bv11&0x03)<<4))&0x3F)]
		bq15 := q6ValueTable[byte((bv11>>2)&0x3F)]
		cq0 := q6ValueTable[byte(cv0&0x3F)]
		cq1 := q6ValueTable[byte(((cv0>>6)|((cv1&0x0F)<<2))&0x3F)]
		cq2 := q6ValueTable[byte(((cv1>>4)|((cv2&0x03)<<4))&0x3F)]
		cq3 := q6ValueTable[byte((cv2>>2)&0x3F)]
		cq4 := q6ValueTable[byte(cv3&0x3F)]
		cq5 := q6ValueTable[byte(((cv3>>6)|((cv4&0x0F)<<2))&0x3F)]
		cq6 := q6ValueTable[byte(((cv4>>4)|((cv5&0x03)<<4))&0x3F)]
		cq7 := q6ValueTable[byte((cv5>>2)&0x3F)]
		cq8 := q6ValueTable[byte(cv6&0x3F)]
		cq9 := q6ValueTable[byte(((cv6>>6)|((cv7&0x0F)<<2))&0x3F)]
		cq10 := q6ValueTable[byte(((cv7>>4)|((cv8&0x03)<<4))&0x3F)]
		cq11 := q6ValueTable[byte((cv8>>2)&0x3F)]
		cq12 := q6ValueTable[byte(cv9&0x3F)]
		cq13 := q6ValueTable[byte(((cv9>>6)|((cv10&0x0F)<<2))&0x3F)]
		cq14 := q6ValueTable[byte(((cv10>>4)|((cv11&0x03)<<4))&0x3F)]
		cq15 := q6ValueTable[byte((cv11>>2)&0x3F)]
		a0 += aq0*x0 + aq4*x4 + aq8*x8 + aq12*x12
		a1 += aq1*x1 + aq5*x5 + aq9*x9 + aq13*x13
		a2 += aq2*x2 + aq6*x6 + aq10*x10 + aq14*x14
		a3 += aq3*x3 + aq7*x7 + aq11*x11 + aq15*x15
		b0 += bq0*x0 + bq4*x4 + bq8*x8 + bq12*x12
		b1 += bq1*x1 + bq5*x5 + bq9*x9 + bq13*x13
		b2 += bq2*x2 + bq6*x6 + bq10*x10 + bq14*x14
		b3 += bq3*x3 + bq7*x7 + bq11*x11 + bq15*x15
		c0 += cq0*x0 + cq4*x4 + cq8*x8 + cq12*x12
		c1 += cq1*x1 + cq5*x5 + cq9*x9 + cq13*x13
		c2 += cq2*x2 + cq6*x6 + cq10*x10 + cq14*x14
		c3 += cq3*x3 + cq7*x7 + cq11*x11 + cq15*x15
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		av0, av1, av2 := uint32(a[base]), uint32(a[base+1]), uint32(a[base+2])
		av3, av4, av5 := uint32(a[base+3]), uint32(a[base+4]), uint32(a[base+5])
		bv0, bv1, bv2 := uint32(b[base]), uint32(b[base+1]), uint32(b[base+2])
		bv3, bv4, bv5 := uint32(b[base+3]), uint32(b[base+4]), uint32(b[base+5])
		cv0, cv1, cv2 := uint32(c[base]), uint32(c[base+1]), uint32(c[base+2])
		cv3, cv4, cv5 := uint32(c[base+3]), uint32(c[base+4]), uint32(c[base+5])
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q6ValueTable[byte(av0&0x3F)]*x0 + q6ValueTable[byte(av3&0x3F)]*x4
		a1 += q6ValueTable[byte(((av0>>6)|((av1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((av3>>6)|((av4&0x0F)<<2))&0x3F)]*x5
		a2 += q6ValueTable[byte(((av1>>4)|((av2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((av4>>4)|((av5&0x03)<<4))&0x3F)]*x6
		a3 += q6ValueTable[byte((av2>>2)&0x3F)]*x3 + q6ValueTable[byte((av5>>2)&0x3F)]*x7
		b0 += q6ValueTable[byte(bv0&0x3F)]*x0 + q6ValueTable[byte(bv3&0x3F)]*x4
		b1 += q6ValueTable[byte(((bv0>>6)|((bv1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((bv3>>6)|((bv4&0x0F)<<2))&0x3F)]*x5
		b2 += q6ValueTable[byte(((bv1>>4)|((bv2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((bv4>>4)|((bv5&0x03)<<4))&0x3F)]*x6
		b3 += q6ValueTable[byte((bv2>>2)&0x3F)]*x3 + q6ValueTable[byte((bv5>>2)&0x3F)]*x7
		c0 += q6ValueTable[byte(cv0&0x3F)]*x0 + q6ValueTable[byte(cv3&0x3F)]*x4
		c1 += q6ValueTable[byte(((cv0>>6)|((cv1&0x0F)<<2))&0x3F)]*x1 + q6ValueTable[byte(((cv3>>6)|((cv4&0x0F)<<2))&0x3F)]*x5
		c2 += q6ValueTable[byte(((cv1>>4)|((cv2&0x03)<<4))&0x3F)]*x2 + q6ValueTable[byte(((cv4>>4)|((cv5&0x03)<<4))&0x3F)]*x6
		c3 += q6ValueTable[byte((cv2>>2)&0x3F)]*x3 + q6ValueTable[byte((cv5>>2)&0x3F)]*x7
	}
	sa := (a0 + a1) + (a2 + a3)
	sb := (b0 + b1) + (b2 + b3)
	sc := (c0 + c1) + (c2 + c3)
	for ; i < cols; i++ {
		xi := x[i]
		sa += q6ValueTable[getQ6(a, i)] * xi
		sb += q6ValueTable[getQ6(b, i)] * xi
		sc += q6ValueTable[getQ6(c, i)] * xi
	}
	return sa, sb, sc
}

func q6Value(v byte) float32 {
	return q6ValueTable[v&0x3F]
}

var q6ValueTable = [64]float32{
	-32, -31, -30, -29, -28, -27, -26, -25,
	-24, -23, -22, -21, -20, -19, -18, -17,
	-16, -15, -14, -13, -12, -11, -10, -9,
	-8, -7, -6, -5, -4, -3, -2, -1,
	0, 1, 2, 3, 4, 5, 6, 7,
	8, 9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31,
}

func PackedQ6Cols(cols int) int {
	return (cols*6 + 7) / 8
}

func putQ6(row []byte, col int, v byte) {
	bit := col * 6
	idx := bit / 8
	shift := uint(bit % 8)
	x := uint16(v&0x3F) << shift
	row[idx] |= byte(x)
	if idx+1 < len(row) {
		row[idx+1] |= byte(x >> 8)
	}
}

func getQ6(row []byte, col int) byte {
	bit := col * 6
	idx := bit / 8
	shift := uint(bit % 8)
	x := uint16(row[idx])
	if idx+1 < len(row) {
		x |= uint16(row[idx+1]) << 8
	}
	return byte((x >> shift) & 0x3F)
}
