package tensor

import "runtime"

type Q8Matrix struct {
	Rows   int
	Cols   int
	Data   []int8
	Scale  []float32
	RowSum []int32 // sum of int8 weights per row, for VNNI offset correction
}

type Q4Matrix struct {
	Rows     int
	Cols     int
	Data     []byte
	Unpacked []int8
	Scale    []float32
	RowSum   []int32 // per-row sum of int8 weights, for VNNI offset correction
}

type Q6Matrix struct {
	Rows     int
	Cols     int
	Data     []byte
	Unpacked []int8
	Scale    []float32
	RowSum   []int32 // per-row sum of int8 weights, for VNNI offset correction
}

func QuantizeQ8Row(w []float32, rows, cols int) *Q8Matrix {
	q := &Q8Matrix{
		Rows:   rows,
		Cols:   cols,
		Data:   make([]int8, rows*cols),
		Scale:  make([]float32, rows),
		RowSum: make([]int32, rows),
	}
	if shouldParallelQuantizeRows(rows*cols, rows) {
		parallelForQuantizeRows(rows, func(start, end int) {
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
	if useDotQ8AVX2 && len(w) >= 8 {
		quantizeQ8RowAVX2(w, data, inv)
		return scale
	}
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
		if useVNNI {
			q.RowSum[r] = rowSumQ8(q.Data[base:base+q.Cols])
		}
	}
}

func QuantizeQ4Row(w []float32, rows, cols int) *Q4Matrix {
	q := &Q4Matrix{
		Rows:     rows,
		Cols:     cols,
		Data:     make([]byte, rows*((cols+1)/2)),
		Unpacked: make([]int8, rows*cols),
		Scale:    make([]float32, rows),
		RowSum:   make([]int32, rows),
	}
	if shouldParallelQuantizeRows(rows*cols, rows) {
		parallelForQuantizeRows(rows, func(start, end int) {
			quantizeQ4Rows(w, q, start, end)
		})
		UnpackQ4Matrix(q)
		return q
	}
	quantizeQ4Rows(w, q, 0, rows)
	UnpackQ4Matrix(q)
	return q
}

func QuantizeQ6Row(w []float32, rows, cols int) *Q6Matrix {
	q := &Q6Matrix{
		Rows:     rows,
		Cols:     cols,
		Data:     make([]byte, rows*PackedQ6Cols(cols)),
		Unpacked: make([]int8, rows*cols),
		Scale:    make([]float32, rows),
		RowSum:   make([]int32, rows),
	}
	if shouldParallelQuantizeRows(rows*cols, rows) {
		parallelForQuantizeRows(rows, func(start, end int) {
			quantizeQ6Rows(w, q, start, end)
		})
		UnpackQ6Matrix(q)
		return q
	}
	quantizeQ6Rows(w, q, 0, rows)
	UnpackQ6Matrix(q)
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
	if useDotQ8AVX2 && len(w) >= 8 {
		n := (len(w) / 8) * 8
		quantizeQ6RowAVX2(w[:n], data, inv)
		// Handle tail
		i := n
		j := (n * 6) / 8
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
		for ; i < len(w); i++ {
			putQ6(data, i, quantByte6(w[i]*inv))
		}
		return scale
	}
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
	if useDotQ4AVX2 && len(w) >= 8 {
		quantizeQ4RowAVX2(w, data, inv)
		return scale
	}
	i := 0
	j := 0
	for ; i+15 < len(w); i, j = i+16, j+8 {
		v0 := q4Nib(w[i] * inv)
		v1 := q4Nib(w[i+1] * inv)
		v2 := q4Nib(w[i+2] * inv)
		v3 := q4Nib(w[i+3] * inv)
		v4 := q4Nib(w[i+4] * inv)
		v5 := q4Nib(w[i+5] * inv)
		v6 := q4Nib(w[i+6] * inv)
		v7 := q4Nib(w[i+7] * inv)
		v8 := q4Nib(w[i+8] * inv)
		v9 := q4Nib(w[i+9] * inv)
		v10 := q4Nib(w[i+10] * inv)
		v11 := q4Nib(w[i+11] * inv)
		v12 := q4Nib(w[i+12] * inv)
		v13 := q4Nib(w[i+13] * inv)
		v14 := q4Nib(w[i+14] * inv)
		v15 := q4Nib(w[i+15] * inv)
		data[j] = v0 | (v1 << 4)
		data[j+1] = v2 | (v3 << 4)
		data[j+2] = v4 | (v5 << 4)
		data[j+3] = v6 | (v7 << 4)
		data[j+4] = v8 | (v9 << 4)
		data[j+5] = v10 | (v11 << 4)
		data[j+6] = v12 | (v13 << 4)
		data[j+7] = v14 | (v15 << 4)
	}
	for ; i+7 < len(w); i, j = i+8, j+4 {
		v0 := q4Nib(w[i] * inv)
		v1 := q4Nib(w[i+1] * inv)
		v2 := q4Nib(w[i+2] * inv)
		v3 := q4Nib(w[i+3] * inv)
		v4 := q4Nib(w[i+4] * inv)
		v5 := q4Nib(w[i+5] * inv)
		v6 := q4Nib(w[i+6] * inv)
		v7 := q4Nib(w[i+7] * inv)
		data[j] = v0 | (v1 << 4)
		data[j+1] = v2 | (v3 << 4)
		data[j+2] = v4 | (v5 << 4)
		data[j+3] = v6 | (v7 << 4)
	}
	for ; i+1 < len(w); i, j = i+2, j+1 {
		lo := q4Nib(w[i] * inv)
		hi := q4Nib(w[i+1] * inv)
		data[j] = lo | (hi << 4)
	}
	if i < len(w) {
		data[j] = q4Nib(w[i] * inv)
	}
	return scale
}

// q4Nib is a branchless inline version of quantNibble4.
func q4Nib(x float32) byte {
	v := roundToInt(x)
	if v > 7 {
		v = 7
	} else if v < -7 {
		v = -7
	}
	return byte(v + 8)
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
	if useDotQ8AVX2 && len(x) >= 8 {
		return maxAbsFloat32AVX2(x)
	}
	return maxAbsFloat32Scalar(x)
}

func maxAbsFloat32Scalar(x []float32) float32 {
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
	matVecQ8Rows(out, x, q, q.Rows)
}

func matVecQ8Rows(out, x []float32, q *Q8Matrix, rows int) {
	if shouldParallelQuantMatVec(rows*q.Cols, rows) {
		parallelForQuantMatVec(rows, func(start, end int) {
			matVecQ8Serial(out, x, q, start, end)
		})
		return
	}
	matVecQ8Serial(out, x, q, 0, rows)
}

func MatVecQ8Bias(out, x []float32, q *Q8Matrix, bias []float32) {
	if shouldParallelQuantMatVec(q.Rows*q.Cols, q.Rows) {
		parallelForQuantMatVec(q.Rows, func(start, end int) {
			matVecQ8BiasSerial(out, x, q, bias, start, end)
		})
		return
	}
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		parallelFor(q.Rows, func(start, end int) {
			matVecQ8BiasSerial(out, x, q, bias, start, end)
		})
		return
	}
	matVecQ8BiasSerial(out, x, q, bias, 0, q.Rows)
}

func MatVecQ4(out, x []float32, q *Q4Matrix) {
	matVecQ4Rows(out, x, q, q.Rows)
}

func matVecQ4Rows(out, x []float32, q *Q4Matrix, rows int) {
	if shouldParallelQuantMatVec(rows*q.Cols, rows) {
		parallelForQuantMatVec(rows, func(start, end int) {
			matVecQ4Serial(out, x, q, start, end)
		})
		return
	}
	matVecQ4Serial(out, x, q, 0, rows)
}

func MatVecQ6(out, x []float32, q *Q6Matrix) {
	matVecQ6Rows(out, x, q, q.Rows)
}

func matVecQ6Rows(out, x []float32, q *Q6Matrix, rows int) {
	if shouldParallelQuantMatVec(rows*q.Cols, rows) {
		parallelForQuantMatVec(rows, func(start, end int) {
			matVecQ6Serial(out, x, q, start, end)
		})
		return
	}
	matVecQ6Serial(out, x, q, 0, rows)
}

func shouldParallelQuantMatVec(work, rows int) bool {
	if work >= parallelWork/2 && rows >= 512 {
		return shouldParallel(parallelWork, rows)
	}
	return false
}

func shouldParallelQuantMatVec3EqualRows(work, rows int) bool {
	if work >= parallelWork/2 && rows >= 256 && rows <= 512 {
		return shouldParallel(parallelWork, rows)
	}
	return false
}

func shouldParallelQuantizeRows(work, rows int) bool {
	if work >= parallelWork/2 && rows >= 512 {
		return shouldParallel(parallelWork, rows)
	}
	return shouldParallel(work, rows)
}

func parallelForQuantizeRows(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelRows(rows, fn)
}

func parallelForQuantMatVec(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func parallelForQuantMatVec3EqualRows(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func FusedMatVec3Q8(outA, outB, outC, x []float32, a, b, c *Q8Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if shouldParallelQuantMatVec3EqualRows(totalRows*a.Cols, a.Rows) {
			parallelForQuantMatVec3EqualRows(a.Rows, func(start, end int) {
				fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
			return
		}
		if shouldParallel(totalRows*a.Cols, a.Rows) {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if shouldParallel(totalRows*a.Cols, totalRows) {
		fusedMatVec3Q8Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q8Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedMatVec3Q6(outA, outB, outC, x []float32, a, b, c *Q6Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if shouldParallelQuantMatVec3EqualRows(totalRows*a.Cols, a.Rows) {
			parallelForQuantMatVec3EqualRows(a.Rows, func(start, end int) {
				fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
			return
		}
		if shouldParallel(totalRows*a.Cols, a.Rows) {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q6EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if shouldParallel(totalRows*a.Cols, totalRows) {
		fusedMatVec3Q6Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q6Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedMatVec3Q4(outA, outB, outC, x []float32, a, b, c *Q4Matrix) {
	totalRows := a.Rows + b.Rows + c.Rows
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		if shouldParallelQuantMatVec3EqualRows(totalRows*a.Cols, a.Rows) {
			parallelForQuantMatVec3EqualRows(a.Rows, func(start, end int) {
				fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
			return
		}
		if shouldParallel(totalRows*a.Cols, a.Rows) {
			parallelFor(a.Rows, func(start, end int) {
				fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x, a, b, c, start, end)
			})
		} else {
			fusedMatVec3Q4EqualRowsSerial(outA, outB, outC, x, a, b, c, 0, a.Rows)
		}
		return
	}
	if shouldParallel(totalRows*a.Cols, totalRows) {
		fusedMatVec3Q4Parallel(outA, outB, outC, x, a, b, c, totalRows)
		return
	}
	fusedMatVec3Q4Serial(outA, outB, outC, x, a, b, c, 0, totalRows)
}

func FusedSwiGLUQ8(out, x []float32, gate, up, down *Q8Matrix) {
	tmpG := make([]float32, swiGLUQuantScratchLen(gate.Rows, gate.Cols, up.Rows, up.Cols))
	FusedSwiGLUQ8Scratch(out, x, gate, up, down, tmpG)
}

func FusedSwiGLUQ4(out, x []float32, gate, up, down *Q4Matrix) {
	tmpG := make([]float32, swiGLUQuantScratchLen(gate.Rows, gate.Cols, up.Rows, up.Cols))
	FusedSwiGLUQ4Scratch(out, x, gate, up, down, tmpG)
}

func FusedSwiGLUQ6(out, x []float32, gate, up, down *Q6Matrix) {
	tmpG := make([]float32, swiGLUQuantScratchLen(gate.Rows, gate.Cols, up.Rows, up.Cols))
	FusedSwiGLUQ6Scratch(out, x, gate, up, down, tmpG)
}

func FusedSwiGLUQ8Scratch(out, x []float32, gate, up, down *Q8Matrix, tmpG []float32) {
	tmpG, tmpU := swiGLUQuantScratch(tmpG, gate.Rows, up.Rows)
	matVecQ8SwiGLUScratch(tmpG, x, gate, up, tmpU)
	MatVecQ8(out, tmpG, down)
}

func FusedSwiGLUQ4Scratch(out, x []float32, gate, up, down *Q4Matrix, tmpG []float32) {
	tmpG, tmpU := swiGLUQuantScratch(tmpG, gate.Rows, up.Rows)
	matVecQ4SwiGLUScratch(tmpG, x, gate, up, tmpU)
	MatVecQ4(out, tmpG, down)
}

func FusedSwiGLUQ6Scratch(out, x []float32, gate, up, down *Q6Matrix, tmpG []float32) {
	tmpG, tmpU := swiGLUQuantScratch(tmpG, gate.Rows, up.Rows)
	matVecQ6SwiGLUScratch(tmpG, x, gate, up, tmpU)
	MatVecQ6(out, tmpG, down)
}

// SwiGLUGateUpQ8Scratch computes the SwiGLU gate+up projection without the down matvec.
// Result is written to out[:gate.Rows]. tmpU is used for batched SiLU.
func SwiGLUGateUpQ8Scratch(out, x []float32, gate, up *Q8Matrix, tmpU []float32) {
	out, tmpU = swiGLUQuantScratch(out, gate.Rows, up.Rows)
	matVecQ8SwiGLUScratch(out, x, gate, up, tmpU)
}

// SwiGLUGateUpQ4Scratch computes the SwiGLU gate+up projection without the down matvec.
func SwiGLUGateUpQ4Scratch(out, x []float32, gate, up *Q4Matrix, tmpU []float32) {
	out, tmpU = swiGLUQuantScratch(out, gate.Rows, up.Rows)
	matVecQ4SwiGLUScratch(out, x, gate, up, tmpU)
}

// SwiGLUGateUpQ6Scratch computes the SwiGLU gate+up projection without the down matvec.
func SwiGLUGateUpQ6Scratch(out, x []float32, gate, up *Q6Matrix, tmpU []float32) {
	out, tmpU = swiGLUQuantScratch(out, gate.Rows, up.Rows)
	matVecQ6SwiGLUScratch(out, x, gate, up, tmpU)
}

func swiGLUQuantScratch(tmp []float32, gateRows, upRows int) ([]float32, []float32) {
	gateTmp := tmp[:gateRows]
	upScratchRows := min(gateRows, upRows)
	if gateRows+upScratchRows <= cap(tmp) {
		return gateTmp, tmp[:gateRows+upScratchRows][gateRows:]
	}
	return gateTmp, nil
}

func swiGLUQuantScratchLen(gateRows, gateCols, upRows, upCols int) int {
	if gateRows != upRows || gateCols != upCols {
		return gateRows + min(gateRows, upRows)
	}
	return gateRows
}

func matVecQ8SwiGLU(out, x []float32, gate, up *Q8Matrix) {
	matVecQ8SwiGLUScratch(out, x, gate, up, nil)
}

func matVecQ8SwiGLUScratch(out, x []float32, gate, up *Q8Matrix, tmpU []float32) {
	if gate.Rows != up.Rows || gate.Cols != up.Cols {
		rows, tmpU := swiGLUFallbackScratch(gate.Rows, up.Rows, tmpU)
		matVecQ8Rows(out, x, gate, rows)
		matVecQ8Rows(tmpU, x, up, rows)
		swiGLUFallbackInPlace(out, tmpU)
		return
	}
	if shouldParallel(gate.Rows*gate.Cols*2, gate.Rows) {
		parallelForQuantPair(gate.Rows, func(start, end int) {
			matVecQ8SwiGLUSerialBatched(out, tmpU, x, gate, up, start, end)
		})
		return
	}
	matVecQ8SwiGLUSerialBatched(out, tmpU, x, gate, up, 0, gate.Rows)
}

func matVecQ8SwiGLUSerialBatched(out, tmpU []float32, x []float32, gate, up *Q8Matrix, start, end int) {
	batchSize := end - start
	if useVNNI && gate.Cols >= 32 && gate.RowSum != nil && up.RowSum != nil {
		xq := getVNNIScratch(gate.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		gData := gate.Data
		uData := up.Data
		gScale := gate.Scale
		uScale := up.Scale
		gRowSum := gate.RowSum
		uRowSum := up.RowSum
		cols := gate.Cols
		if batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
			if useVNNI {
				for r := start; r < end; r++ {
					base := r * cols
					dotA, dotB := dotQ8PairVNNICore(&gData[base], &uData[base], &xq[0], cols)
					out[r] = float32(dotA-128*gRowSum[r]) * scaleX * gScale[r]
					tmpU[r] = float32(dotB-128*uRowSum[r]) * scaleX * uScale[r]
				}
				SiLUMulInPlace(out[start:end], tmpU[start:end])
				return
			}
			for r := start; r < end; r++ {
				base := r * cols
				g, u := dotQ8PairVNNI(gData[base:base+cols], uData[base:base+cols], xq, scaleX, gRowSum[r], uRowSum[r], gScale[r], uScale[r])
				out[r] = g
				tmpU[r] = u
			}
			SiLUMulInPlace(out[start:end], tmpU[start:end])
			return
		}
		for r := start; r < end; r++ {
			base := r * gate.Cols
			g, u := dotQ8PairVNNI(gate.Data[base:base+gate.Cols], up.Data[base:base+up.Cols], xq, scaleX, gate.RowSum[r], up.RowSum[r], gate.Scale[r], up.Scale[r])
			out[r] = SiLU(g) * u
		}
		return
	}
	if useDotQ8AVX2 && batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		for r := start; r < end; r++ {
			base := r * gate.Cols
			g, u := dotQ8Pair(gate.Data[base:base+gate.Cols], up.Data[base:base+up.Cols], x)
			out[r] = g * gate.Scale[r]
			tmpU[r] = u * up.Scale[r]
		}
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	for r := start; r < end; r++ {
		base := r * gate.Cols
		g, u := dotQ8Pair(gate.Data[base:base+gate.Cols], up.Data[base:base+up.Cols], x)
		out[r] = SiLU(g*gate.Scale[r]) * (u * up.Scale[r])
	}
}

func matVecQ4SwiGLU(out, x []float32, gate, up *Q4Matrix) {
	matVecQ4SwiGLUScratch(out, x, gate, up, nil)
}

func matVecQ4SwiGLUScratch(out, x []float32, gate, up *Q4Matrix, tmpU []float32) {
	if gate.Rows != up.Rows || gate.Cols != up.Cols {
		rows, tmpU := swiGLUFallbackScratch(gate.Rows, up.Rows, tmpU)
		matVecQ4Rows(out, x, gate, rows)
		matVecQ4Rows(tmpU, x, up, rows)
		swiGLUFallbackInPlace(out, tmpU)
		return
	}
	if shouldParallel(gate.Rows*gate.Cols*2, gate.Rows) {
		parallelForQuantPair(gate.Rows, func(start, end int) {
			matVecQ4SwiGLUSerialBatched(out, tmpU, x, gate, up, start, end)
		})
		return
	}
	matVecQ4SwiGLUSerialBatched(out, tmpU, x, gate, up, 0, gate.Rows)
}

func matVecQ4SwiGLUSerial(out, x []float32, gate, up *Q4Matrix, start, end int) {
	if useVNNI && gate.Unpacked != nil && up.Unpacked != nil && gate.Cols >= 32 && gate.RowSum != nil && up.RowSum != nil {
		xq := getVNNIScratch(gate.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		gData := gate.Unpacked
		uData := up.Unpacked
		gScale := gate.Scale
		uScale := up.Scale
		gRS := gate.RowSum
		uRS := up.RowSum
		cols := gate.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB := dotQ8PairVNNICore(&gData[base], &uData[base], &xq[0], cols)
				g := float32(dotA-128*gRS[r]) * scaleX * gScale[r]
				u := float32(dotB-128*uRS[r]) * scaleX * uScale[r]
				out[r] = SiLU(g) * u
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			g, u := dotQ8PairVNNI(gData[base:base+cols], uData[base:base+cols], xq, scaleX, gRS[r], uRS[r], gScale[r], uScale[r])
			out[r] = SiLU(g) * u
		}
		return
	}
	if gate.Unpacked != nil && up.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * gate.Cols
			g, u := dotQ4UnpackedPair(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], x)
			out[r] = SiLU(g*gate.Scale[r]) * (u * up.Scale[r])
		}
		return
	}
	packedCols := (gate.Cols + 1) / 2
	for r := start; r < end; r++ {
		base := r * packedCols
		g, u := dotQ4Pair(gate.Data[base:base+packedCols], up.Data[base:base+packedCols], x, gate.Cols)
		out[r] = SiLU(g*gate.Scale[r]) * (u * up.Scale[r])
	}
}

func matVecQ4SwiGLUSerialBatched(out, tmpU []float32, x []float32, gate, up *Q4Matrix, start, end int) {
	batchSize := end - start
	if useVNNI && gate.Unpacked != nil && up.Unpacked != nil && gate.Cols >= 32 && gate.RowSum != nil && up.RowSum != nil {
		xq := getVNNIScratch(gate.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		if batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
			if useVNNI {
				gData := gate.Unpacked
				uData := up.Unpacked
				gScale := gate.Scale
				uScale := up.Scale
				gRS := gate.RowSum
				uRS := up.RowSum
				cols := gate.Cols
				for r := start; r < end; r++ {
					base := r * cols
					dotA, dotB := dotQ8PairVNNICore(&gData[base], &uData[base], &xq[0], cols)
					out[r] = float32(dotA-128*gRS[r]) * scaleX * gScale[r]
					tmpU[r] = float32(dotB-128*uRS[r]) * scaleX * uScale[r]
				}
				SiLUMulInPlace(out[start:end], tmpU[start:end])
				return
			}
			for r := start; r < end; r++ {
				base := r * gate.Cols
				g, u := dotQ8PairVNNI(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], xq, scaleX, gate.RowSum[r], up.RowSum[r], gate.Scale[r], up.Scale[r])
				out[r] = g
				tmpU[r] = u
			}
			SiLUMulInPlace(out[start:end], tmpU[start:end])
			return
		}
		if useVNNI {
			gData := gate.Unpacked
			uData := up.Unpacked
			gScale := gate.Scale
			uScale := up.Scale
			gRS := gate.RowSum
			uRS := up.RowSum
			cols := gate.Cols
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB := dotQ8PairVNNICore(&gData[base], &uData[base], &xq[0], cols)
				g := float32(dotA-128*gRS[r]) * scaleX * gScale[r]
				u := float32(dotB-128*uRS[r]) * scaleX * uScale[r]
				out[r] = SiLU(g) * u
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * gate.Cols
			g, u := dotQ8PairVNNI(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], xq, scaleX, gate.RowSum[r], up.RowSum[r], gate.Scale[r], up.Scale[r])
			out[r] = SiLU(g) * u
		}
		return
	}
	if useDotQ4AVX2 && batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		if gate.Unpacked != nil && up.Unpacked != nil {
			for r := start; r < end; r++ {
				base := r * gate.Cols
				g, u := dotQ4UnpackedPair(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], x)
				out[r] = g * gate.Scale[r]
				tmpU[r] = u * up.Scale[r]
			}
		} else {
			packedCols := (gate.Cols + 1) / 2
			for r := start; r < end; r++ {
				base := r * packedCols
				g, u := dotQ4Pair(gate.Data[base:base+packedCols], up.Data[base:base+packedCols], x, gate.Cols)
				out[r] = g * gate.Scale[r]
				tmpU[r] = u * up.Scale[r]
			}
		}
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	matVecQ4SwiGLUSerial(out, x, gate, up, start, end)
}

func matVecQ6SwiGLU(out, x []float32, gate, up *Q6Matrix) {
	matVecQ6SwiGLUScratch(out, x, gate, up, nil)
}

func matVecQ6SwiGLUScratch(out, x []float32, gate, up *Q6Matrix, tmpU []float32) {
	if gate.Rows != up.Rows || gate.Cols != up.Cols {
		rows, tmpU := swiGLUFallbackScratch(gate.Rows, up.Rows, tmpU)
		matVecQ6Rows(out, x, gate, rows)
		matVecQ6Rows(tmpU, x, up, rows)
		swiGLUFallbackInPlace(out, tmpU)
		return
	}
	if shouldParallel(gate.Rows*gate.Cols*2, gate.Rows) {
		parallelForQuantPair(gate.Rows, func(start, end int) {
			matVecQ6SwiGLUSerialBatched(out, tmpU, x, gate, up, start, end)
		})
		return
	}
	matVecQ6SwiGLUSerialBatched(out, tmpU, x, gate, up, 0, gate.Rows)
}

func matVecQ6SwiGLUSerial(out, x []float32, gate, up *Q6Matrix, start, end int) {
	if useVNNI && gate.Unpacked != nil && up.Unpacked != nil && gate.Cols >= 32 && gate.RowSum != nil && up.RowSum != nil {
		xq := getVNNIScratch(gate.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		gData := gate.Unpacked
		uData := up.Unpacked
		gScale := gate.Scale
		uScale := up.Scale
		gRS := gate.RowSum
		uRS := up.RowSum
		cols := gate.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB := dotQ8PairVNNICore(&gData[base], &uData[base], &xq[0], cols)
				g := float32(dotA-128*gRS[r]) * scaleX * gScale[r]
				u := float32(dotB-128*uRS[r]) * scaleX * uScale[r]
				out[r] = SiLU(g) * u
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			g, u := dotQ8PairVNNI(gData[base:base+cols], uData[base:base+cols], xq, scaleX, gRS[r], uRS[r], gScale[r], uScale[r])
			out[r] = SiLU(g) * u
		}
		return
	}
	if gate.Unpacked != nil && up.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * gate.Cols
			g, u := dotQ6UnpackedPair(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], x)
			out[r] = SiLU(g*gate.Scale[r]) * (u * up.Scale[r])
		}
		return
	}
	packedCols := PackedQ6Cols(gate.Cols)
	for r := start; r < end; r++ {
		base := r * packedCols
		g, u := dotQ6Pair(gate.Data[base:base+packedCols], up.Data[base:base+packedCols], x, gate.Cols)
		out[r] = SiLU(g*gate.Scale[r]) * (u * up.Scale[r])
	}
}

func matVecQ6SwiGLUSerialBatched(out, tmpU []float32, x []float32, gate, up *Q6Matrix, start, end int) {
	batchSize := end - start
	if useDotQ8AVX2 && batchSize >= 8 && tmpU != nil && len(tmpU) >= end {
		if gate.Unpacked != nil && up.Unpacked != nil {
			for r := start; r < end; r++ {
				base := r * gate.Cols
				g, u := dotQ6UnpackedPair(gate.Unpacked[base:base+gate.Cols], up.Unpacked[base:base+up.Cols], x)
				out[r] = g * gate.Scale[r]
				tmpU[r] = u * up.Scale[r]
			}
		} else {
			packedCols := PackedQ6Cols(gate.Cols)
			for r := start; r < end; r++ {
				base := r * packedCols
				g, u := dotQ6Pair(gate.Data[base:base+packedCols], up.Data[base:base+packedCols], x, gate.Cols)
				out[r] = g * gate.Scale[r]
				tmpU[r] = u * up.Scale[r]
			}
		}
		SiLUMulInPlace(out[start:end], tmpU[start:end])
		return
	}
	matVecQ6SwiGLUSerial(out, x, gate, up, start, end)
}

func swiGLUFallbackInPlace(gate, up []float32) {
	n := min(len(gate), len(up))
	for i := 0; i < n; i++ {
		gate[i] = SiLU(gate[i]) * up[i]
	}
	clear(gate[n:])
}

func swiGLUFallbackScratch(gateRows, upRows int, tmpU []float32) (int, []float32) {
	rows := min(gateRows, upRows)
	if len(tmpU) < rows {
		tmpU = make([]float32, rows)
	} else {
		tmpU = tmpU[:rows]
	}
	return rows, tmpU
}

func matVecQ8Pair(outA, outB, x []float32, a, b *Q8Matrix) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		MatVecQ8(outA, x, a)
		MatVecQ8(outB, x, b)
		return
	}
	if shouldParallel(a.Rows*a.Cols*2, a.Rows) {
		parallelForQuantPair(a.Rows, func(start, end int) {
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

func fusedMatVec3Q8EqualRowsBiasSerial(outA, outB, outC, x []float32, a, b, c *Q8Matrix, ba, bb, bc []float32, start, end int) {
	if useVNNI && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Data, b.Data, c.Data
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		cols := a.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB, dotC := dotQ8TripletVNNICore(&aData[base], &bData[base], &cData[base], &xq[0], cols)
				outA[r] = float32(dotA-128*aRS[r])*scaleX*aScale[r] + ba[r]
				outB[r] = float32(dotB-128*bRS[r])*scaleX*bScale[r] + bb[r]
				outC[r] = float32(dotC-128*cRS[r])*scaleX*cScale[r] + bc[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			av, bv, cv := dotQ8TripletVNNI(aData[base:base+cols], bData[base:base+cols], cData[base:base+cols], xq, scaleX, aRS[r], bRS[r], cRS[r], aScale[r], bScale[r], cScale[r])
			outA[r] = av + ba[r]
			outB[r] = bv + bb[r]
			outC[r] = cv + bc[r]
		}
		return
	}
	for r := start; r < end; r++ {
		base := r * a.Cols
		av, bv, cv := dotQ8Triplet(a.Data[base:base+a.Cols], b.Data[base:base+b.Cols], c.Data[base:base+c.Cols], x)
		outA[r] = av * a.Scale[r] + ba[r]
		outB[r] = bv * b.Scale[r] + bb[r]
		outC[r] = cv * c.Scale[r] + bc[r]
	}
}

func fusedMatVec3Q8EqualRowsSerial(outA, outB, outC, x []float32, a, b, c *Q8Matrix, start, end int) {
	if useVNNI && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Data, b.Data, c.Data
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		cols := a.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB, dotC := dotQ8TripletVNNICore(&aData[base], &bData[base], &cData[base], &xq[0], cols)
				outA[r] = float32(dotA-128*aRS[r]) * scaleX * aScale[r]
				outB[r] = float32(dotB-128*bRS[r]) * scaleX * bScale[r]
				outC[r] = float32(dotC-128*cRS[r]) * scaleX * cScale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			av, bv, cv := dotQ8TripletVNNI(aData[base:base+cols], bData[base:base+cols], cData[base:base+cols], xq, scaleX, aRS[r], bRS[r], cRS[r], aScale[r], bScale[r], cScale[r])
			outA[r] = av
			outB[r] = bv
			outC[r] = cv
		}
		return
	}
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
	if useVNNI && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		for r := start; r < aEnd; r++ {
			base := r * a.Cols
			outA[r] = dotQ8VNNI(a.Data[base:base+a.Cols], xq, scaleX, a.Scale[r], a.RowSum[r])
		}
		bStart := max(start, a.Rows)
		bEnd := min(end, splitB)
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * b.Cols
			outB[br] = dotQ8VNNI(b.Data[base:base+b.Cols], xq, scaleX, b.Scale[br], b.RowSum[br])
		}
		cStart := max(start, splitB)
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * c.Cols
			outC[cr] = dotQ8VNNI(c.Data[base:base+c.Cols], xq, scaleX, c.Scale[cr], c.RowSum[cr])
		}
		return
	}
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
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		data := q.Data
		scale := q.Scale
		rowSum := q.RowSum
		cols := q.Cols
		if useVNNI {
			// Inline the VNNI dot product to avoid per-row function call overhead
			for r := start; r < end; r++ {
				base := r * cols
				dot := dotQ8VNNICoreZMM(&data[base], &xq[0], cols)
				out[r] = float32(dot-128*rowSum[r]) * scaleX * scale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			out[r] = dotQ8VNNI(data[base:base+cols], xq, scaleX, scale[r], rowSum[r])
		}
		return
	}
	for r := start; r < end; r++ {
		base := r * q.Cols
		out[r] = dotQ8(q.Data[base:base+q.Cols], x) * q.Scale[r]
	}
}

// MatRowsQ8Bias computes out[i] = q * x[i] + bias for n rows of x.
// Each row of x has cols elements; q is a rows x cols Q8 matrix.
// MatRowsQ8Bias3 computes outA[i], outB[i], outC[i] = a*x[i]+ba, b*x[i]+bb, c*x[i]+bc
// for n rows of x, using fused Q8 triplet dot products for efficiency.
func MatRowsQ8Bias3(outA, outB, outC [][]float32, xs [][]float32, a, b, c *Q8Matrix, ba, bb, bc []float32) {
	if a == nil || b == nil || c == nil || len(xs) == 0 {
		return
	}
	if a.Rows == b.Rows && b.Rows == c.Rows && a.Cols == b.Cols && b.Cols == c.Cols {
		work := len(xs) * a.Rows * a.Cols
		if shouldParallel(work, len(xs)) && len(xs) > 1 {
			workers := runtime.GOMAXPROCS(0)
			if a.Rows*a.Cols < 1<<16 {
				workers = min(workers, 8)
			}
			parallelForMax(len(xs), workers, func(start, end int) {
				for i := start; i < end; i++ {
					fusedMatVec3Q8EqualRowsBiasSerial(outA[i], outB[i], outC[i], xs[i], a, b, c, ba, bb, bc, 0, a.Rows)
				}
			})
			return
		}
		for i := range xs {
			fusedMatVec3Q8EqualRowsBiasSerial(outA[i], outB[i], outC[i], xs[i], a, b, c, ba, bb, bc, 0, a.Rows)
		}
		return
	}
	// Fallback: three separate calls
	MatRowsQ8Bias(outA, xs, a, ba)
	MatRowsQ8Bias(outB, xs, b, bb)
	MatRowsQ8Bias(outC, xs, c, bc)
}

func MatRowsQ8Bias(out [][]float32, xs [][]float32, q *Q8Matrix, bias []float32) {
	if q == nil || len(xs) == 0 {
		return
	}
	work := len(xs) * q.Rows * q.Cols
	if shouldParallel(work, len(xs)) && len(xs) > 1 {
		workers := runtime.GOMAXPROCS(0)
		if q.Rows*q.Cols < 1<<16 {
			workers = min(workers, 8)
		}
		parallelForMax(len(xs), workers, func(start, end int) {
			for i := start; i < end; i++ {
				matVecQ8BiasSerial(out[i], xs[i], q, bias, 0, q.Rows)
			}
		})
		return
	}
	for i := range xs {
		matVecQ8BiasSerial(out[i], xs[i], q, bias, 0, q.Rows)
	}
}

// MatVecQ8BiasSerial computes out[start:end] = q*x + bias for the specified rows,
// without parallelization. Safe for concurrent use across different output ranges.
func MatVecQ8BiasSerial(out, x []float32, q *Q8Matrix, bias []float32, start, end int) {
	matVecQ8BiasSerial(out, x, q, bias, start, end)
}

func matVecQ8BiasSerial(out, x []float32, q *Q8Matrix, bias []float32, start, end int) {
	if useVNNI && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		data := q.Data
		scale := q.Scale
		rowSum := q.RowSum
		cols := q.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dot := dotQ8VNNICoreZMM(&data[base], &xq[0], cols)
				out[r] = float32(dot-128*rowSum[r])*scaleX*scale[r] + bias[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			out[r] = dotQ8VNNI(data[base:base+cols], xq, scaleX, scale[r], rowSum[r]) + bias[r]
		}
		return
	}
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
	if shouldParallel(a.Rows*a.Cols*2, a.Rows) {
		parallelForQuantPair(a.Rows, func(start, end int) {
			matVecQ4PairSerial(outA, outB, x, a, b, start, end)
		})
		return
	}
	matVecQ4PairSerial(outA, outB, x, a, b, 0, a.Rows)
}

func matVecQ4PairSerial(outA, outB, x []float32, a, b *Q4Matrix, start, end int) {
	if useVNNI && a.Unpacked != nil && b.Unpacked != nil && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv := dotQ8PairVNNI(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], xq, scaleX, a.RowSum[r], b.RowSum[r], a.Scale[r], b.Scale[r])
			outA[r] = av
			outB[r] = bv
		}
		return
	}
	if a.Unpacked != nil && b.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv := dotQ4UnpackedPair(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], x)
			outA[r] = av * a.Scale[r]
			outB[r] = bv * b.Scale[r]
		}
		return
	}
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
	if shouldParallel(a.Rows*a.Cols*2, a.Rows) {
		parallelForQuantPair(a.Rows, func(start, end int) {
			matVecQ6PairSerial(outA, outB, x, a, b, start, end)
		})
		return
	}
	matVecQ6PairSerial(outA, outB, x, a, b, 0, a.Rows)
}

func parallelForQuantPair(rows int, fn func(start, end int)) {
	if rows <= 512 {
		parallelForMax(rows, 8, fn)
		return
	}
	parallelFor(rows, fn)
}

func matVecQ6PairSerial(outA, outB, x []float32, a, b *Q6Matrix, start, end int) {
	if useVNNI && a.Unpacked != nil && b.Unpacked != nil && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv := dotQ8PairVNNI(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], xq, scaleX, a.RowSum[r], b.RowSum[r], a.Scale[r], b.Scale[r])
			outA[r] = av
			outB[r] = bv
		}
		return
	}
	if a.Unpacked != nil && b.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv := dotQ6UnpackedPair(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], x)
			outA[r] = av * a.Scale[r]
			outB[r] = bv * b.Scale[r]
		}
		return
	}
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
	if useVNNI && a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		cols := a.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB, dotC := dotQ8TripletVNNICore(&aData[base], &bData[base], &cData[base], &xq[0], cols)
				outA[r] = float32(dotA-128*aRS[r]) * scaleX * aScale[r]
				outB[r] = float32(dotB-128*bRS[r]) * scaleX * bScale[r]
				outC[r] = float32(dotC-128*cRS[r]) * scaleX * cScale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv, cv := dotQ8TripletVNNI(aData[base:base+a.Cols], bData[base:base+b.Cols], cData[base:base+c.Cols], xq, scaleX, aRS[r], bRS[r], cRS[r], aScale[r], bScale[r], cScale[r])
			outA[r] = av
			outB[r] = bv
			outC[r] = cv
		}
		return
	}
	if a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv, cv := dotQ6UnpackedTriplet(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], c.Unpacked[base:base+c.Cols], x)
			outA[r] = av * a.Scale[r]
			outB[r] = bv * b.Scale[r]
			outC[r] = cv * c.Scale[r]
		}
		return
	}
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
	allUnpacked := a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil
	splitB := a.Rows + b.Rows
	aEnd := min(end, a.Rows)
	if useVNNI && allUnpacked && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		aCols := a.Cols
		bCols := b.Cols
		cCols := c.Cols
		if useVNNI {
			for r := start; r < aEnd; r++ {
				base := r * aCols
				dot := dotQ8VNNICoreZMM(&aData[base], &xq[0], aCols)
				outA[r] = float32(dot-128*aRS[r]) * scaleX * aScale[r]
			}
			bStart := max(start, a.Rows)
			bEnd := min(end, splitB)
			for r := bStart; r < bEnd; r++ {
				br := r - a.Rows
				base := br * bCols
				dot := dotQ8VNNICoreZMM(&bData[base], &xq[0], bCols)
				outB[br] = float32(dot-128*bRS[br]) * scaleX * bScale[br]
			}
			cStart := max(start, splitB)
			for r := cStart; r < end; r++ {
				cr := r - splitB
				base := cr * cCols
				dot := dotQ8VNNICoreZMM(&cData[base], &xq[0], cCols)
				outC[cr] = float32(dot-128*cRS[cr]) * scaleX * cScale[cr]
			}
			return
		}
		for r := start; r < aEnd; r++ {
			base := r * a.Cols
			outA[r] = dotQ8VNNI(aData[base:base+a.Cols], xq, scaleX, aScale[r], aRS[r])
		}
		bStart := max(start, a.Rows)
		bEnd := min(end, splitB)
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * b.Cols
			outB[br] = dotQ8VNNI(bData[base:base+b.Cols], xq, scaleX, bScale[br], bRS[br])
		}
		cStart := max(start, splitB)
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * c.Cols
			outC[cr] = dotQ8VNNI(cData[base:base+c.Cols], xq, scaleX, cScale[cr], cRS[cr])
		}
		return
	}
	if allUnpacked {
		for r := start; r < aEnd; r++ {
			base := r * a.Cols
			outA[r] = dotQ6Unpacked(a.Unpacked[base:base+a.Cols], x) * a.Scale[r]
		}
	} else {
		aPacked := PackedQ6Cols(a.Cols)
		for r := start; r < aEnd; r++ {
			base := r * aPacked
			outA[r] = dotQ6(a.Data[base:base+aPacked], x, a.Cols) * a.Scale[r]
		}
	}
	bStart := max(start, a.Rows)
	bEnd := min(end, splitB)
	if allUnpacked {
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * b.Cols
			outB[br] = dotQ6Unpacked(b.Unpacked[base:base+b.Cols], x) * b.Scale[br]
		}
	} else {
		bPacked := PackedQ6Cols(b.Cols)
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * bPacked
			outB[br] = dotQ6(b.Data[base:base+bPacked], x, b.Cols) * b.Scale[br]
		}
	}
	cStart := max(start, splitB)
	if allUnpacked {
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * c.Cols
			outC[cr] = dotQ6Unpacked(c.Unpacked[base:base+c.Cols], x) * c.Scale[cr]
		}
	} else {
		cPacked := PackedQ6Cols(c.Cols)
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * cPacked
			outC[cr] = dotQ6(c.Data[base:base+cPacked], x, c.Cols) * c.Scale[cr]
		}
	}
}

func matVecQ6Serial(out, x []float32, q *Q6Matrix, start, end int) {
	if useVNNI && q.Unpacked != nil && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		data := q.Unpacked
		scale := q.Scale
		rowSum := q.RowSum
		cols := q.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dot := dotQ8VNNICoreZMM(&data[base], &xq[0], cols)
				out[r] = float32(dot-128*rowSum[r]) * scaleX * scale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			out[r] = dotQ8VNNI(data[base:base+cols], xq, scaleX, scale[r], rowSum[r])
		}
		return
	}
	if q.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * q.Cols
			out[r] = dotQ6Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]
		}
		return
	}
	packedCols := PackedQ6Cols(q.Cols)
	for r := start; r < end; r++ {
		base := r * packedCols
		out[r] = dotQ6(q.Data[base:base+packedCols], x, q.Cols) * q.Scale[r]
	}
}

func matVecQ4Serial(out, x []float32, q *Q4Matrix, start, end int) {
	if useVNNI && q.Unpacked != nil && q.Cols >= 32 && q.RowSum != nil {
		xq := getVNNIScratch(q.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		data := q.Unpacked
		scale := q.Scale
		rowSum := q.RowSum
		cols := q.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dot := dotQ8VNNICoreZMM(&data[base], &xq[0], cols)
				out[r] = float32(dot-128*rowSum[r]) * scaleX * scale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * cols
			out[r] = dotQ8VNNI(data[base:base+cols], xq, scaleX, scale[r], rowSum[r])
		}
		return
	}
	if q.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * q.Cols
			out[r] = dotQ4Unpacked(q.Unpacked[base:base+q.Cols], x) * q.Scale[r]
		}
		return
	}
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
	if useVNNI && a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		cols := a.Cols
		if useVNNI {
			for r := start; r < end; r++ {
				base := r * cols
				dotA, dotB, dotC := dotQ8TripletVNNICore(&aData[base], &bData[base], &cData[base], &xq[0], cols)
				outA[r] = float32(dotA-128*aRS[r]) * scaleX * aScale[r]
				outB[r] = float32(dotB-128*bRS[r]) * scaleX * bScale[r]
				outC[r] = float32(dotC-128*cRS[r]) * scaleX * cScale[r]
			}
			return
		}
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv, cv := dotQ8TripletVNNI(aData[base:base+a.Cols], bData[base:base+b.Cols], cData[base:base+c.Cols], xq, scaleX, aRS[r], bRS[r], cRS[r], aScale[r], bScale[r], cScale[r])
			outA[r] = av
			outB[r] = bv
			outC[r] = cv
		}
		return
	}
	if a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil {
		for r := start; r < end; r++ {
			base := r * a.Cols
			av, bv, cv := dotQ4UnpackedTriplet(a.Unpacked[base:base+a.Cols], b.Unpacked[base:base+b.Cols], c.Unpacked[base:base+c.Cols], x)
			outA[r] = av * a.Scale[r]
			outB[r] = bv * b.Scale[r]
			outC[r] = cv * c.Scale[r]
		}
		return
	}
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
	allUnpacked := a.Unpacked != nil && b.Unpacked != nil && c.Unpacked != nil
	splitB := a.Rows + b.Rows
	aEnd := min(end, a.Rows)
	if useVNNI && allUnpacked && a.Cols >= 32 && a.RowSum != nil && b.RowSum != nil && c.RowSum != nil {
		xq := getVNNIScratch(a.Cols)
		defer putVNNIScratch(xq)
		scaleX := quantizeXForVNNI(x, xq)
		aData, bData, cData := a.Unpacked, b.Unpacked, c.Unpacked
		aScale, bScale, cScale := a.Scale, b.Scale, c.Scale
		aRS, bRS, cRS := a.RowSum, b.RowSum, c.RowSum
		aCols := a.Cols
		bCols := b.Cols
		cCols := c.Cols
		if useVNNI {
			for r := start; r < aEnd; r++ {
				base := r * aCols
				dot := dotQ8VNNICoreZMM(&aData[base], &xq[0], aCols)
				outA[r] = float32(dot-128*aRS[r]) * scaleX * aScale[r]
			}
			bStart := max(start, a.Rows)
			bEnd := min(end, splitB)
			for r := bStart; r < bEnd; r++ {
				br := r - a.Rows
				base := br * bCols
				dot := dotQ8VNNICoreZMM(&bData[base], &xq[0], bCols)
				outB[br] = float32(dot-128*bRS[br]) * scaleX * bScale[br]
			}
			cStart := max(start, splitB)
			for r := cStart; r < end; r++ {
				cr := r - splitB
				base := cr * cCols
				dot := dotQ8VNNICoreZMM(&cData[base], &xq[0], cCols)
				outC[cr] = float32(dot-128*cRS[cr]) * scaleX * cScale[cr]
			}
			return
		}
		for r := start; r < aEnd; r++ {
			base := r * a.Cols
			outA[r] = dotQ8VNNI(aData[base:base+a.Cols], xq, scaleX, aScale[r], aRS[r])
		}
		bStart := max(start, a.Rows)
		bEnd := min(end, splitB)
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * b.Cols
			outB[br] = dotQ8VNNI(bData[base:base+b.Cols], xq, scaleX, bScale[br], bRS[br])
		}
		cStart := max(start, splitB)
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * c.Cols
			outC[cr] = dotQ8VNNI(cData[base:base+c.Cols], xq, scaleX, cScale[cr], cRS[cr])
		}
		return
	}
	if allUnpacked {
		for r := start; r < aEnd; r++ {
			base := r * a.Cols
			outA[r] = dotQ4Unpacked(a.Unpacked[base:base+a.Cols], x) * a.Scale[r]
		}
	} else {
		aPacked := (a.Cols + 1) / 2
		for r := start; r < aEnd; r++ {
			base := r * aPacked
			outA[r] = dotQ4(a.Data[base:base+aPacked], x, a.Cols) * a.Scale[r]
		}
	}
	bStart := max(start, a.Rows)
	bEnd := min(end, splitB)
	if allUnpacked {
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * b.Cols
			outB[br] = dotQ4Unpacked(b.Unpacked[base:base+b.Cols], x) * b.Scale[br]
		}
	} else {
		bPacked := (b.Cols + 1) / 2
		for r := bStart; r < bEnd; r++ {
			br := r - a.Rows
			base := br * bPacked
			outB[br] = dotQ4(b.Data[base:base+bPacked], x, b.Cols) * b.Scale[br]
		}
	}
	cStart := max(start, splitB)
	if allUnpacked {
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * c.Cols
			outC[cr] = dotQ4Unpacked(c.Unpacked[base:base+c.Cols], x) * c.Scale[cr]
		}
	} else {
		cPacked := (c.Cols + 1) / 2
		for r := cStart; r < end; r++ {
			cr := r - splitB
			base := cr * cPacked
			outC[cr] = dotQ4(c.Data[base:base+cPacked], x, c.Cols) * c.Scale[cr]
		}
	}
}

func dotQ8(a []int8, b []float32) float32 {
	if useDotFMA && useDotQ8AVX2 && len(a) >= 8 {
		return dotQ8FMA(a, b)
	}
	if useDotQ8AVX2 && len(a) >= 8 {
		return dotQ8AVX2(a, b)
	}
	return dotQ8Scalar(a, b)
}

// rowSumQ8 computes the sum of int8 values in a row as int32.
func rowSumQ8(a []int8) int32 {
	if useVNNI && len(a) >= 8 {
		return rowSumQ8AVX2(a)
	}
	var s int32
	for _, v := range a {
		s += int32(v)
	}
	return s
}

// quantizeXForVNNI quantizes a float32 vector x into uint8 with offset 128.
// Returns (xq, scaleX) where scaleX = maxAbs(x)/127.
// The caller must provide xq with at least len(x) capacity.
func quantizeXForVNNI(x []float32, xq []uint8) float32 {
	if useVNNI && len(x) >= 8 {
		return quantizeXForVNNIAVX2(x, xq)
	}
	maxAbs := maxAbsFloat32(x)
	if maxAbs == 0 {
		for i := range xq[:len(x)] {
			xq[i] = 128
		}
		return 1
	}
	scale := maxAbs / 127
	inv := 1 / scale
	for i, v := range x {
		q := int(v*inv + 128)
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		xq[i] = byte(q)
	}
	return scale
}

// vnniScratchBuf returns a reusable uint8 buffer for x quantization.
// Uses a sync.Pool-like approach with a package-level buffer.
var vnniScratchPool = make(chan []uint8, 32)

func getVNNIScratch(n int) []uint8 {
	select {
	case buf := <-vnniScratchPool:
		if cap(buf) >= n {
			return buf[:n]
		}
	default:
	}
	return make([]uint8, n)
}

func putVNNIScratch(buf []uint8) {
	select {
	case vnniScratchPool <- buf:
	default:
	}
}

func dotQ8Scalar(a []int8, b []float32) float32 {
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
	if useDotFMA && useDotQ8AVX2 && len(x) >= 8 {
		return dotQ8PairFMA(a, b, x)
	}
	if useDotQ8AVX2 && len(x) >= 8 {
		return dotQ8PairAVX2(a, b, x)
	}
	return dotQ8PairScalar(a, b, x)
}

func dotQ8PairScalar(a, b []int8, x []float32) (float32, float32) {
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
	if useDotFMA && useDotQ8AVX2 && len(x) >= 8 {
		return dotQ8TripletFMA(a, b, c, x)
	}
	if useDotQ8AVX2 && len(x) >= 8 {
		return dotQ8TripletAVX2(a, b, c, x)
	}
	return dotQ8TripletScalar(a, b, c, x)
}

func dotQ8TripletScalar(a, b, c []int8, x []float32) (float32, float32, float32) {
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

// UnpackQ4Matrix populates the Unpacked field by extracting 4-bit nibble values
// from the packed Data into a contiguous int8 array, enabling reuse of the
// fast Q8 AVX2 dot product kernels.
func UnpackQ4Matrix(q *Q4Matrix) {
	if q.Unpacked == nil {
		q.Unpacked = make([]int8, q.Rows*q.Cols)
	}
	if q.RowSum == nil {
		q.RowSum = make([]int32, q.Rows)
	}
	packedCols := (q.Cols + 1) / 2
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		parallelFor(q.Rows, func(start, end int) {
			for r := start; r < end; r++ {
				row := q.Data[r*packedCols : (r+1)*packedCols]
				out := q.Unpacked[r*q.Cols : (r+1)*q.Cols]
				unpackQ4RowFast(row, out, q.Cols)
				if useVNNI {
					q.RowSum[r] = rowSumQ8(out)
				}
			}
		})
		return
	}
	for r := 0; r < q.Rows; r++ {
		row := q.Data[r*packedCols : (r+1)*packedCols]
		out := q.Unpacked[r*q.Cols : (r+1)*q.Cols]
		unpackQ4RowFast(row, out, q.Cols)
		if useVNNI {
			q.RowSum[r] = rowSumQ8(out)
		}
	}
}

var q4Int8Table = [16]int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}

func unpackQ4RowFast(row []byte, out []int8, cols int) {
	c := 0
	for ; c+2 <= cols; c += 2 {
		b := row[c/2]
		out[c] = q4Int8Table[b&0x0F]
		out[c+1] = q4Int8Table[b>>4]
	}
	if c < cols {
		out[c] = q4Int8Table[row[c/2]&0x0F]
	}
}

// dotQ4Unpacked computes the dot product using the pre-unpacked int8 data,
// reusing the same AVX2 kernel as Q8. No new assembly needed.
func dotQ4Unpacked(a []int8, b []float32) float32 {
	return dotQ8(a, b)
}

func dotQ4UnpackedPair(a, b []int8, x []float32) (float32, float32) {
	return dotQ8Pair(a, b, x)
}

func dotQ4UnpackedTriplet(a, b, c []int8, x []float32) (float32, float32, float32) {
	return dotQ8Triplet(a, b, c, x)
}

func dotQ4(a []byte, b []float32, cols int) float32 {
	if useDotFMA && useDotQ4AVX2 && cols >= 8 {
		return dotQ4FMA(a, b, cols)
	}
	if useDotQ4AVX2 && cols >= 8 {
		return dotQ4AVX2(a, b, cols)
	}
	return dotQ4Scalar(a, b, cols)
}

func dotQ4Scalar(a []byte, b []float32, cols int) float32 {
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
	if useDotFMA && useDotQ4AVX2 && cols >= 8 {
		return dotQ4PairFMA(a, b, x, cols)
	}
	if useDotQ4AVX2 && cols >= 8 {
		return dotQ4PairAVX2(a, b, x, cols)
	}
	return dotQ4PairScalar(a, b, x, cols)
}

func dotQ4PairScalar(a, b []byte, x []float32, cols int) (float32, float32) {
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
	if useDotFMA && useDotQ4AVX2 && cols >= 8 {
		return dotQ4TripletFMA(a, b, c, x, cols)
	}
	if useDotQ4AVX2 && cols >= 8 {
		return dotQ4TripletAVX2(a, b, c, x, cols)
	}
	return dotQ4TripletScalar(a, b, c, x, cols)
}

func dotQ4TripletScalar(a, b, c []byte, x []float32, cols int) (float32, float32, float32) {
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

var q6PairLo, q6PairHi = func() ([4096]float32, [4096]float32) {
	var lo, hi [4096]float32
	for i := range lo {
		v := uint32(i)
		lo[i] = q6ValueTable[byte(v&0x3F)]
		hi[i] = q6ValueTable[byte((v>>6)&0x3F)]
	}
	return lo, hi
}()


var q6Int8Table = [64]int8{
	-32, -31, -30, -29, -28, -27, -26, -25,
	-24, -23, -22, -21, -20, -19, -18, -17,
	-16, -15, -14, -13, -12, -11, -10, -9,
	-8, -7, -6, -5, -4, -3, -2, -1,
	0, 1, 2, 3, 4, 5, 6, 7,
	8, 9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23,
	24, 25, 26, 27, 28, 29, 30, 31,
}
// UnpackQ6Matrix populates the Unpacked field by extracting 6-bit values
// from the packed Data into a contiguous int8 array, enabling reuse of the
// fast Q8 AVX2 dot product kernels.
func UnpackQ6Matrix(q *Q6Matrix) {
	if q.Unpacked == nil {
		q.Unpacked = make([]int8, q.Rows*q.Cols)
	}
	if q.RowSum == nil {
		q.RowSum = make([]int32, q.Rows)
	}
	packedCols := PackedQ6Cols(q.Cols)
	if shouldParallel(q.Rows*q.Cols, q.Rows) {
		parallelFor(q.Rows, func(start, end int) {
			for r := start; r < end; r++ {
				row := q.Data[r*packedCols : (r+1)*packedCols]
				out := q.Unpacked[r*q.Cols : (r+1)*q.Cols]
				unpackQ6RowFast(row, out, q.Cols)
				if useVNNI {
					q.RowSum[r] = rowSumQ8(out)
				}
			}
		})
		return
	}
	for r := 0; r < q.Rows; r++ {
		row := q.Data[r*packedCols : (r+1)*packedCols]
		out := q.Unpacked[r*q.Cols : (r+1)*q.Cols]
		unpackQ6RowFast(row, out, q.Cols)
		if useVNNI {
			q.RowSum[r] = rowSumQ8(out)
		}
	}
}

// unpackQ6RowFast extracts 6-bit values from packed bytes into int8.
// Processes 4 values (24 bits = 3 bytes) per iteration using the int8 LUT,
// avoiding the per-element getQ6 bit-shift overhead.
func unpackQ6RowFast(row []byte, out []int8, cols int) {
	c := 0
	j := 0
	for ; c+4 <= cols; c += 4 {
		b0 := uint32(row[j])
		b1 := uint32(row[j+1])
		b2 := uint32(row[j+2])
		v := b0 | (b1 << 8) | (b2 << 16)
		out[c] = q6Int8Table[v&0x3F]
		out[c+1] = q6Int8Table[(v>>6)&0x3F]
		out[c+2] = q6Int8Table[(v>>12)&0x3F]
		out[c+3] = q6Int8Table[(v>>18)&0x3F]
		j += 3
	}
	for ; c < cols; c++ {
		out[c] = q6Int8Table[getQ6(row, c)]
	}
}

// dotQ6Unpacked computes the dot product using the pre-unpacked int8 data,
// reusing the same AVX2 kernel as Q8. No new assembly needed.
func dotQ6Unpacked(a []int8, b []float32) float32 {
	return dotQ8(a, b)
}

func dotQ6UnpackedPair(a, b []int8, x []float32) (float32, float32) {
	return dotQ8Pair(a, b, x)
}

func dotQ6UnpackedTriplet(a, b, c []int8, x []float32) (float32, float32, float32) {
	return dotQ8Triplet(a, b, c, x)
}

func dotQ6(a []byte, b []float32, cols int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+15 < cols; i += 16 {
		base := (i * 6) / 8
		v0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		v1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		v2 := uint32(a[base+6]) | uint32(a[base+7])<<8 | uint32(a[base+8])<<16
		v3 := uint32(a[base+9]) | uint32(a[base+10])<<8 | uint32(a[base+11])<<16
		p0, p1 := v0&0x0FFF, v0>>12
		p2, p3 := v1&0x0FFF, v1>>12
		p4, p5 := v2&0x0FFF, v2>>12
		p6, p7 := v3&0x0FFF, v3>>12
		s0 += q6PairLo[p0]*b[i] + q6PairLo[p4]*b[i+8]
		s1 += q6PairHi[p0]*b[i+1] + q6PairHi[p4]*b[i+9]
		s2 += q6PairLo[p1]*b[i+2] + q6PairLo[p5]*b[i+10]
		s3 += q6PairHi[p1]*b[i+3] + q6PairHi[p5]*b[i+11]
		s4 += q6PairLo[p2]*b[i+4] + q6PairLo[p6]*b[i+12]
		s5 += q6PairHi[p2]*b[i+5] + q6PairHi[p6]*b[i+13]
		s6 += q6PairLo[p3]*b[i+6] + q6PairLo[p7]*b[i+14]
		s7 += q6PairHi[p3]*b[i+7] + q6PairHi[p7]*b[i+15]
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		v0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		v1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		p0, p1 := v0&0x0FFF, v0>>12
		p2, p3 := v1&0x0FFF, v1>>12
		s0 += q6PairLo[p0] * b[i]
		s1 += q6PairHi[p0] * b[i+1]
		s2 += q6PairLo[p1] * b[i+2]
		s3 += q6PairHi[p1] * b[i+3]
		s4 += q6PairLo[p2] * b[i+4]
		s5 += q6PairHi[p2] * b[i+5]
		s6 += q6PairLo[p3] * b[i+6]
		s7 += q6PairHi[p3] * b[i+7]
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
		av0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		av1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		av2 := uint32(a[base+6]) | uint32(a[base+7])<<8 | uint32(a[base+8])<<16
		av3 := uint32(a[base+9]) | uint32(a[base+10])<<8 | uint32(a[base+11])<<16
		bv0 := uint32(b[base]) | uint32(b[base+1])<<8 | uint32(b[base+2])<<16
		bv1 := uint32(b[base+3]) | uint32(b[base+4])<<8 | uint32(b[base+5])<<16
		bv2 := uint32(b[base+6]) | uint32(b[base+7])<<8 | uint32(b[base+8])<<16
		bv3 := uint32(b[base+9]) | uint32(b[base+10])<<8 | uint32(b[base+11])<<16
		ap0, ap1 := av0&0x0FFF, av0>>12
		ap2, ap3 := av1&0x0FFF, av1>>12
		ap4, ap5 := av2&0x0FFF, av2>>12
		ap6, ap7 := av3&0x0FFF, av3>>12
		bp0, bp1 := bv0&0x0FFF, bv0>>12
		bp2, bp3 := bv1&0x0FFF, bv1>>12
		bp4, bp5 := bv2&0x0FFF, bv2>>12
		bp6, bp7 := bv3&0x0FFF, bv3>>12
		a0 += q6PairLo[ap0]*x0 + q6PairLo[ap2]*x4 + q6PairLo[ap4]*x8 + q6PairLo[ap6]*x12
		a1 += q6PairHi[ap0]*x1 + q6PairHi[ap2]*x5 + q6PairHi[ap4]*x9 + q6PairHi[ap6]*x13
		a2 += q6PairLo[ap1]*x2 + q6PairLo[ap3]*x6 + q6PairLo[ap5]*x10 + q6PairLo[ap7]*x14
		a3 += q6PairHi[ap1]*x3 + q6PairHi[ap3]*x7 + q6PairHi[ap5]*x11 + q6PairHi[ap7]*x15
		b0 += q6PairLo[bp0]*x0 + q6PairLo[bp2]*x4 + q6PairLo[bp4]*x8 + q6PairLo[bp6]*x12
		b1 += q6PairHi[bp0]*x1 + q6PairHi[bp2]*x5 + q6PairHi[bp4]*x9 + q6PairHi[bp6]*x13
		b2 += q6PairLo[bp1]*x2 + q6PairLo[bp3]*x6 + q6PairLo[bp5]*x10 + q6PairLo[bp7]*x14
		b3 += q6PairHi[bp1]*x3 + q6PairHi[bp3]*x7 + q6PairHi[bp5]*x11 + q6PairHi[bp7]*x15
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		av0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		av1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		bv0 := uint32(b[base]) | uint32(b[base+1])<<8 | uint32(b[base+2])<<16
		bv1 := uint32(b[base+3]) | uint32(b[base+4])<<8 | uint32(b[base+5])<<16
		ap0, ap1 := av0&0x0FFF, av0>>12
		ap2, ap3 := av1&0x0FFF, av1>>12
		bp0, bp1 := bv0&0x0FFF, bv0>>12
		bp2, bp3 := bv1&0x0FFF, bv1>>12
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q6PairLo[ap0]*x0 + q6PairLo[ap2]*x4
		a1 += q6PairHi[ap0]*x1 + q6PairHi[ap2]*x5
		a2 += q6PairLo[ap1]*x2 + q6PairLo[ap3]*x6
		a3 += q6PairHi[ap1]*x3 + q6PairHi[ap3]*x7
		b0 += q6PairLo[bp0]*x0 + q6PairLo[bp2]*x4
		b1 += q6PairHi[bp0]*x1 + q6PairHi[bp2]*x5
		b2 += q6PairLo[bp1]*x2 + q6PairLo[bp3]*x6
		b3 += q6PairHi[bp1]*x3 + q6PairHi[bp3]*x7
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
		av0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		av1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		av2 := uint32(a[base+6]) | uint32(a[base+7])<<8 | uint32(a[base+8])<<16
		av3 := uint32(a[base+9]) | uint32(a[base+10])<<8 | uint32(a[base+11])<<16
		bv0 := uint32(b[base]) | uint32(b[base+1])<<8 | uint32(b[base+2])<<16
		bv1 := uint32(b[base+3]) | uint32(b[base+4])<<8 | uint32(b[base+5])<<16
		bv2 := uint32(b[base+6]) | uint32(b[base+7])<<8 | uint32(b[base+8])<<16
		bv3 := uint32(b[base+9]) | uint32(b[base+10])<<8 | uint32(b[base+11])<<16
		cv0 := uint32(c[base]) | uint32(c[base+1])<<8 | uint32(c[base+2])<<16
		cv1 := uint32(c[base+3]) | uint32(c[base+4])<<8 | uint32(c[base+5])<<16
		cv2 := uint32(c[base+6]) | uint32(c[base+7])<<8 | uint32(c[base+8])<<16
		cv3 := uint32(c[base+9]) | uint32(c[base+10])<<8 | uint32(c[base+11])<<16
		ap0, ap1 := av0&0x0FFF, av0>>12
		ap2, ap3 := av1&0x0FFF, av1>>12
		ap4, ap5 := av2&0x0FFF, av2>>12
		ap6, ap7 := av3&0x0FFF, av3>>12
		bp0, bp1 := bv0&0x0FFF, bv0>>12
		bp2, bp3 := bv1&0x0FFF, bv1>>12
		bp4, bp5 := bv2&0x0FFF, bv2>>12
		bp6, bp7 := bv3&0x0FFF, bv3>>12
		cp0, cp1 := cv0&0x0FFF, cv0>>12
		cp2, cp3 := cv1&0x0FFF, cv1>>12
		cp4, cp5 := cv2&0x0FFF, cv2>>12
		cp6, cp7 := cv3&0x0FFF, cv3>>12
		a0 += q6PairLo[ap0]*x0 + q6PairLo[ap2]*x4 + q6PairLo[ap4]*x8 + q6PairLo[ap6]*x12
		a1 += q6PairHi[ap0]*x1 + q6PairHi[ap2]*x5 + q6PairHi[ap4]*x9 + q6PairHi[ap6]*x13
		a2 += q6PairLo[ap1]*x2 + q6PairLo[ap3]*x6 + q6PairLo[ap5]*x10 + q6PairLo[ap7]*x14
		a3 += q6PairHi[ap1]*x3 + q6PairHi[ap3]*x7 + q6PairHi[ap5]*x11 + q6PairHi[ap7]*x15
		b0 += q6PairLo[bp0]*x0 + q6PairLo[bp2]*x4 + q6PairLo[bp4]*x8 + q6PairLo[bp6]*x12
		b1 += q6PairHi[bp0]*x1 + q6PairHi[bp2]*x5 + q6PairHi[bp4]*x9 + q6PairHi[bp6]*x13
		b2 += q6PairLo[bp1]*x2 + q6PairLo[bp3]*x6 + q6PairLo[bp5]*x10 + q6PairLo[bp7]*x14
		b3 += q6PairHi[bp1]*x3 + q6PairHi[bp3]*x7 + q6PairHi[bp5]*x11 + q6PairHi[bp7]*x15
		c0 += q6PairLo[cp0]*x0 + q6PairLo[cp2]*x4 + q6PairLo[cp4]*x8 + q6PairLo[cp6]*x12
		c1 += q6PairHi[cp0]*x1 + q6PairHi[cp2]*x5 + q6PairHi[cp4]*x9 + q6PairHi[cp6]*x13
		c2 += q6PairLo[cp1]*x2 + q6PairLo[cp3]*x6 + q6PairLo[cp5]*x10 + q6PairLo[cp7]*x14
		c3 += q6PairHi[cp1]*x3 + q6PairHi[cp3]*x7 + q6PairHi[cp5]*x11 + q6PairHi[cp7]*x15
	}
	for ; i+7 < cols; i += 8 {
		base := (i * 6) / 8
		av0 := uint32(a[base]) | uint32(a[base+1])<<8 | uint32(a[base+2])<<16
		av1 := uint32(a[base+3]) | uint32(a[base+4])<<8 | uint32(a[base+5])<<16
		bv0 := uint32(b[base]) | uint32(b[base+1])<<8 | uint32(b[base+2])<<16
		bv1 := uint32(b[base+3]) | uint32(b[base+4])<<8 | uint32(b[base+5])<<16
		cv0 := uint32(c[base]) | uint32(c[base+1])<<8 | uint32(c[base+2])<<16
		cv1 := uint32(c[base+3]) | uint32(c[base+4])<<8 | uint32(c[base+5])<<16
		ap0, ap1 := av0&0x0FFF, av0>>12
		ap2, ap3 := av1&0x0FFF, av1>>12
		bp0, bp1 := bv0&0x0FFF, bv0>>12
		bp2, bp3 := bv1&0x0FFF, bv1>>12
		cp0, cp1 := cv0&0x0FFF, cv0>>12
		cp2, cp3 := cv1&0x0FFF, cv1>>12
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += q6PairLo[ap0]*x0 + q6PairLo[ap2]*x4
		a1 += q6PairHi[ap0]*x1 + q6PairHi[ap2]*x5
		a2 += q6PairLo[ap1]*x2 + q6PairLo[ap3]*x6
		a3 += q6PairHi[ap1]*x3 + q6PairHi[ap3]*x7
		b0 += q6PairLo[bp0]*x0 + q6PairLo[bp2]*x4
		b1 += q6PairHi[bp0]*x1 + q6PairHi[bp2]*x5
		b2 += q6PairLo[bp1]*x2 + q6PairLo[bp3]*x6
		b3 += q6PairHi[bp1]*x3 + q6PairHi[bp3]*x7
		c0 += q6PairLo[cp0]*x0 + q6PairLo[cp2]*x4
		c1 += q6PairHi[cp0]*x1 + q6PairHi[cp2]*x5
		c2 += q6PairLo[cp1]*x2 + q6PairLo[cp3]*x6
		c3 += q6PairHi[cp1]*x3 + q6PairHi[cp3]*x7
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
