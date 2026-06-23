package model

import (
	"math"
	"path/filepath"
	"testing"

	"paddleocrvl-go/internal/gguf"
	"paddleocrvl-go/internal/safetensors"
	"paddleocrvl-go/internal/tensor"
)

func TestDotAt128PairAtMatchesScalar(t *testing.T) {
	q := make([]float32, 128)
	keys := make([]float32, 3*160)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range keys {
		keys[i] = float32(i%19-9) / 19
	}
	offset0, offset1 := 16, 176
	got0, got1 := dotAt128PairAt(q, keys, offset0, offset1)
	want0 := dotAt128(q, keys, offset0)
	want1 := dotAt128(q, keys, offset1)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 {
		t.Fatalf("pair=(%f,%f) scalar=(%f,%f)", got0, got1, want0, want1)
	}
}

func TestDotAt64PairAtMatchesScalar(t *testing.T) {
	q := make([]float32, 64)
	keys := make([]float32, 3*96)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range keys {
		keys[i] = float32(i%19-9) / 19
	}
	offset0, offset1 := 16, 112
	got0, got1 := dotAt64PairAt(q, keys, offset0, offset1)
	want0 := dotAt(q, keys, offset0, 64)
	want1 := dotAt(q, keys, offset1, 64)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 {
		t.Fatalf("pair=(%f,%f) scalar=(%f,%f)", got0, got1, want0, want1)
	}
}

func TestDotAt128QuadAtMatchesScalar(t *testing.T) {
	q := make([]float32, 128)
	keys := make([]float32, 5*160)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range keys {
		keys[i] = float32(i%19-9) / 19
	}
	offset0, offset1, offset2, offset3 := 16, 176, 336, 496
	got0, got1, got2, got3 := dotAt128QuadAt(q, keys, offset0, offset1, offset2, offset3)
	want0 := dotAt128(q, keys, offset0)
	want1 := dotAt128(q, keys, offset1)
	want2 := dotAt128(q, keys, offset2)
	want3 := dotAt128(q, keys, offset3)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 || math.Abs(float64(got2-want2)) > 1e-5 || math.Abs(float64(got3-want3)) > 1e-5 {
		t.Fatalf("quad=(%f,%f,%f,%f) scalar=(%f,%f,%f,%f)", got0, got1, got2, got3, want0, want1, want2, want3)
	}
}

func TestDotAt64QuadAtMatchesScalar(t *testing.T) {
	q := make([]float32, 64)
	keys := make([]float32, 5*96)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range keys {
		keys[i] = float32(i%19-9) / 19
	}
	offset0, offset1, offset2, offset3 := 16, 112, 208, 304
	got0, got1, got2, got3 := dotAt64QuadAt(q, keys, offset0, offset1, offset2, offset3)
	want0 := dotAt(q, keys, offset0, 64)
	want1 := dotAt(q, keys, offset1, 64)
	want2 := dotAt(q, keys, offset2, 64)
	want3 := dotAt(q, keys, offset3, 64)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 || math.Abs(float64(got2-want2)) > 1e-5 || math.Abs(float64(got3-want3)) > 1e-5 {
		t.Fatalf("quad=(%f,%f,%f,%f) scalar=(%f,%f,%f,%f)", got0, got1, got2, got3, want0, want1, want2, want3)
	}
}

func TestDotAt128PairRowsMatchesScalar(t *testing.T) {
	q := make([]float32, 128)
	row0 := make([]float32, 192)
	row1 := make([]float32, 192)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range row0 {
		row0[i] = float32(i%19-9) / 19
		row1[i] = float32(i%23-11) / 23
	}
	offset := 32
	got0, got1 := dotAt128PairRows(q, row0, row1, offset)
	want0 := dotAt128(q, row0, offset)
	want1 := dotAt128(q, row1, offset)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 {
		t.Fatalf("pair=(%f,%f) scalar=(%f,%f)", got0, got1, want0, want1)
	}
}

func TestDotAt128QuadRowsMatchesScalar(t *testing.T) {
	q := make([]float32, 128)
	row0 := make([]float32, 192)
	row1 := make([]float32, 192)
	row2 := make([]float32, 192)
	row3 := make([]float32, 192)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range row0 {
		row0[i] = float32(i%19-9) / 19
		row1[i] = float32(i%23-11) / 23
		row2[i] = float32(i%29-14) / 29
		row3[i] = float32(i%31-15) / 31
	}
	offset := 32
	got0, got1, got2, got3 := dotAt128QuadRows(q, row0, row1, row2, row3, offset)
	want0 := dotAt128(q, row0, offset)
	want1 := dotAt128(q, row1, offset)
	want2 := dotAt128(q, row2, offset)
	want3 := dotAt128(q, row3, offset)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 || math.Abs(float64(got2-want2)) > 1e-5 || math.Abs(float64(got3-want3)) > 1e-5 {
		t.Fatalf("quad=(%f,%f,%f,%f) scalar=(%f,%f,%f,%f)", got0, got1, got2, got3, want0, want1, want2, want3)
	}
}

func TestDotAt64PairRowsMatchesScalar(t *testing.T) {
	q := make([]float32, 64)
	row0 := make([]float32, 96)
	row1 := make([]float32, 96)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range row0 {
		row0[i] = float32(i%19-9) / 19
		row1[i] = float32(i%23-11) / 23
	}
	offset := 32
	got0, got1 := dotAt64PairRows(q, row0, row1, offset)
	want0 := dotAt(q, row0, offset, 64)
	want1 := dotAt(q, row1, offset, 64)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 {
		t.Fatalf("pair=(%f,%f) scalar=(%f,%f)", got0, got1, want0, want1)
	}
}

func TestDotAt64QuadRowsMatchesScalar(t *testing.T) {
	q := make([]float32, 64)
	row0 := make([]float32, 96)
	row1 := make([]float32, 96)
	row2 := make([]float32, 96)
	row3 := make([]float32, 96)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	for i := range row0 {
		row0[i] = float32(i%19-9) / 19
		row1[i] = float32(i%23-11) / 23
		row2[i] = float32(i%29-14) / 29
		row3[i] = float32(i%31-15) / 31
	}
	offset := 32
	got0, got1, got2, got3 := dotAt64QuadRows(q, row0, row1, row2, row3, offset)
	want0 := dotAt(q, row0, offset, 64)
	want1 := dotAt(q, row1, offset, 64)
	want2 := dotAt(q, row2, offset, 64)
	want3 := dotAt(q, row3, offset, 64)
	if math.Abs(float64(got0-want0)) > 1e-5 || math.Abs(float64(got1-want1)) > 1e-5 || math.Abs(float64(got2-want2)) > 1e-5 || math.Abs(float64(got3-want3)) > 1e-5 {
		t.Fatalf("quad=(%f,%f,%f,%f) scalar=(%f,%f,%f,%f)", got0, got1, got2, got3, want0, want1, want2, want3)
	}
}

func TestBuildMRoPETableFastPaths(t *testing.T) {
	rt := &Runtime{
		ropeFreq: make([]float64, 4),
		ropeAxis: []byte{0, 1, 2, 0},
	}
	for i := range rt.ropeFreq {
		rt.ropeFreq[i] = math.Pow(1000000, -float64(2*i)/8)
	}
	cosTable := make([]float32, 4)
	sinTable := make([]float32, 4)
	rt.buildMRoPETable(cosTable, sinTable, ropePos{})
	for i := range cosTable {
		if cosTable[i] != 1 || sinTable[i] != 0 {
			t.Fatalf("zero table[%d]=(%f,%f)", i, cosTable[i], sinTable[i])
		}
	}
	rt.buildMRoPETable(cosTable, sinTable, ropePos{T: 7, H: 7, W: 7})
	for i := range cosTable {
		ang := float64(7) * rt.ropeFreq[i]
		if math.Abs(float64(cosTable[i])-math.Cos(ang)) > 1e-6 || math.Abs(float64(sinTable[i])-math.Sin(ang)) > 1e-6 {
			t.Fatalf("text table[%d]=(%f,%f)", i, cosTable[i], sinTable[i])
		}
	}
}

func BenchmarkWeightedCacheValueSum(b *testing.B) {
	tokens, heads, dim := 512, 8, 128
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := make([]float32, tokens)
	for i := range weights {
		weights[i] = float32((i%13)+1) / 91
	}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumLen1(b *testing.B) {
	heads, dim := 8, 128
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, heads*dim),
		v:     make([]float32, 0, heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for i := range v {
		v[i] = float32(i%17-8) / 17
	}
	cache.append(k, v)
	weights := []float32{1}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumLen2(b *testing.B) {
	heads, dim, tokens := 8, 128, 2
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := []float32{0.75, 0.25}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumLen4(b *testing.B) {
	heads, dim, tokens := 8, 128, 4
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := []float32{0.4, 0.3, 0.2, 0.1}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumDim64Len2(b *testing.B) {
	heads, dim, tokens := 8, 64, 2
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := []float32{0.75, 0.25}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumDim64Len3(b *testing.B) {
	heads, dim, tokens := 8, 64, 3
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := []float32{0.5, 0.3, 0.2}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumDim64Len4(b *testing.B) {
	heads, dim, tokens := 8, 64, 4
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := []float32{0.4, 0.3, 0.2, 0.1}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func TestWeightedCacheValueSumLen1CopiesValue(t *testing.T) {
	heads, dim := 2, 8
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i)
	}
	cache.len = 1
	dst := make([]float32, dim)
	weightedCacheValueSum(dst, cache, 1, dim, []float32{1})
	for i := range dst {
		want := cache.v[dim+i]
		if dst[i] != want {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestCopyCacheValue(t *testing.T) {
	heads, dim := 2, 8
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i)
	}
	dst := make([]float32, dim)
	copyCacheValue(dst, cache, 1, dim)
	for i := range dst {
		want := cache.v[dim+i]
		if dst[i] != want {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedCacheValueSumLen2MatchesManual(t *testing.T) {
	heads, dim, tokens := 2, 8, 2
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, tokens*heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i%11 - 5)
	}
	cache.len = tokens
	dst := make([]float32, dim)
	weights := []float32{0.25, 0.75}
	weightedCacheValueSum(dst, cache, 1, dim, weights)
	for i := range dst {
		v0 := cache.v[dim+i]
		v1 := cache.v[cache.kvDim+dim+i]
		want := weights[0]*v0 + weights[1]*v1
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedCacheValueSumLen4MatchesManual(t *testing.T) {
	heads, dim, tokens := 2, 8, 4
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, tokens*heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i%11 - 5)
	}
	cache.len = tokens
	dst := make([]float32, dim)
	weights := []float32{0.4, 0.3, 0.2, 0.1}
	weightedCacheValueSum(dst, cache, 1, dim, weights)
	for i := range dst {
		var want float32
		for t, w := range weights {
			want += w * cache.v[t*cache.kvDim+dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedCacheValueSumLen3MatchesManual(t *testing.T) {
	heads, dim, tokens := 2, 8, 3
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, tokens*heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i%11 - 5)
	}
	cache.len = tokens
	dst := make([]float32, dim)
	weights := []float32{0.5, 0.3, 0.2}
	weightedCacheValueSum(dst, cache, 1, dim, weights)
	for i := range dst {
		var want float32
		for t, w := range weights {
			want += w * cache.v[t*cache.kvDim+dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedCacheValueSumDim64MatchesManual(t *testing.T) {
	heads, dim, tokens := 4, 64, 9
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, tokens*heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i%17-8) / 17
	}
	cache.len = tokens
	weights := make([]float32, tokens)
	for i := range weights {
		weights[i] = float32(i+1) / 45
	}
	dst := make([]float32, dim)
	weightedCacheValueSum(dst, cache, 2, dim, weights)
	for i := range dst {
		var want float32
		for t, w := range weights {
			want += w * cache.v[t*cache.kvDim+2*dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedCacheValueSumDim128MatchesManual(t *testing.T) {
	heads, dim, tokens := 4, 128, 9
	cache := &kvCache{kvDim: heads * dim}
	cache.v = make([]float32, tokens*heads*dim)
	for i := range cache.v {
		cache.v[i] = float32(i%17-8) / 17
	}
	cache.len = tokens
	weights := make([]float32, tokens)
	for i := range weights {
		weights[i] = float32(i+1) / 45
	}
	dst := make([]float32, dim)
	weightedCacheValueSum(dst, cache, 2, dim, weights)
	for i := range dst {
		var want float32
		for t, w := range weights {
			want += w * cache.v[t*cache.kvDim+2*dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedValueSumDim64MatchesManual(t *testing.T) {
	rows, heads, dim := 9, 4, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%19-9) / 19
		}
	}
	weights := make([]float32, rows)
	for i := range weights {
		weights[i] = float32(i+1) / 45
	}
	dst := make([]float32, dim)
	weightedValueSum(dst, xs, 2*dim, dim, weights)
	for i := range dst {
		var want float32
		for r, w := range weights {
			want += w * xs[r][2*dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func TestWeightedValueSumDim128MatchesManual(t *testing.T) {
	rows, heads, dim := 9, 4, 128
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%19-9) / 19
		}
	}
	weights := make([]float32, rows)
	for i := range weights {
		weights[i] = float32(i+1) / 45
	}
	dst := make([]float32, dim)
	weightedValueSum(dst, xs, 2*dim, dim, weights)
	for i := range dst {
		var want float32
		for r, w := range weights {
			want += w * xs[r][2*dim+i]
		}
		if math.Abs(float64(dst[i]-want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want)
		}
	}
}

func BenchmarkCacheAttentionScores(b *testing.B) {
	tokens, heads, dim := 512, 8, 128
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range k {
			k[i] = float32((t+i)%17-8) / 17
			v[i] = float32((t+i)%13-6) / 13
		}
		cache.append(k, v)
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	scores := make([]float32, tokens)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheAttentionScores(scores, q, cache, 3, dim, 0.08838835)
	}
}

func BenchmarkCacheAttentionScoresDim64(b *testing.B) {
	tokens, heads, dim := 512, 8, 64
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range k {
			k[i] = float32((t+i)%17-8) / 17
			v[i] = float32((t+i)%13-6) / 13
		}
		cache.append(k, v)
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	scores := make([]float32, tokens)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheAttentionScores(scores, q, cache, 3, dim, 0.125)
	}
}

func BenchmarkCacheAttentionScoresLen1(b *testing.B) {
	heads, dim := 8, 128
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, heads*dim),
		v:     make([]float32, 0, heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for i := range k {
		k[i] = float32(i%17-8) / 17
	}
	cache.append(k, v)
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	scores := make([]float32, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheAttentionScores(scores, q, cache, 3, dim, 0.08838835)
	}
}

func TestCacheAttentionScoresLen1MatchesScalar(t *testing.T) {
	heads, dim := 2, 128
	cache := &kvCache{kvDim: heads * dim}
	cache.k = make([]float32, heads*dim)
	for i := range cache.k {
		cache.k[i] = float32(i%17-8) / 17
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%11-5) / 11
	}
	scores := make([]float32, 1)
	scale := float32(0.25)
	cacheAttentionScores(scores, q, cache, 1, dim, scale)
	want := dotAt128(q, cache.k, dim) * scale
	if math.Abs(float64(scores[0]-want)) > 1e-5 {
		t.Fatalf("score=%f want %f", scores[0], want)
	}
}

func TestCacheAttentionSmallMatchesSplit(t *testing.T) {
	for _, dim := range []int{64, 96, 128} {
		for _, tokens := range []int{2, 3, 4} {
			heads := 3
			cache := &kvCache{
				kvDim: heads * dim,
				k:     make([]float32, 0, tokens*heads*dim),
				v:     make([]float32, 0, tokens*heads*dim),
			}
			k := make([]float32, heads*dim)
			v := make([]float32, heads*dim)
			for t := 0; t < tokens; t++ {
				for i := range k {
					k[i] = float32((t+i)%17-8) / 17
					v[i] = float32((2*t+i)%19-9) / 19
				}
				cache.append(k, v)
			}
			q := make([]float32, dim)
			for i := range q {
				q[i] = float32(i%13-6) / 13
			}
			got := make([]float32, dim)
			want := make([]float32, dim)
			scores := make([]float32, tokens)
			scale := float32(0.125)
			if !cacheAttentionSmall(got, q, cache, 1, dim, scale) {
				t.Fatalf("cacheAttentionSmall rejected dim=%d tokens=%d", dim, tokens)
			}
			cacheAttentionScores(scores, q, cache, 1, dim, scale)
			tensor.SoftmaxInPlace(scores)
			weightedCacheValueSum(want, cache, 1, dim, scores)
			for i := range want {
				if math.Abs(float64(got[i]-want[i])) > 1e-5 {
					t.Fatalf("dim=%d tokens=%d dst[%d]=%f want %f", dim, tokens, i, got[i], want[i])
				}
			}
		}
	}
}

func BenchmarkCacheAttentionSmallLen2(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 128, 2)
}

func BenchmarkCacheAttentionSplitLen2(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 128, 2)
}

func BenchmarkCacheAttentionSmallLen3(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 128, 3)
}

func BenchmarkCacheAttentionSplitLen3(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 128, 3)
}

func BenchmarkCacheAttentionDispatchLen3(b *testing.B) {
	benchmarkCacheAttentionDispatch(b, 128, 3)
}

func BenchmarkCacheAttentionSmallLen4(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 128, 4)
}

func BenchmarkCacheAttentionSplitLen4(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 128, 4)
}

func BenchmarkCacheAttentionSmallDim64Len2(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 64, 2)
}

func BenchmarkCacheAttentionSplitDim64Len2(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 64, 2)
}

func BenchmarkCacheAttentionSmallDim64Len3(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 64, 3)
}

func BenchmarkCacheAttentionSplitDim64Len3(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 64, 3)
}

func BenchmarkCacheAttentionDispatchDim64Len3(b *testing.B) {
	benchmarkCacheAttentionDispatch(b, 64, 3)
}

func BenchmarkCacheAttentionSmallDim64Len4(b *testing.B) {
	benchmarkCacheAttentionSmall(b, 64, 4)
}

func BenchmarkCacheAttentionSplitDim64Len4(b *testing.B) {
	benchmarkCacheAttentionSplit(b, 64, 4)
}

func BenchmarkCacheAttentionDispatchLen4(b *testing.B) {
	benchmarkCacheAttentionDispatch(b, 128, 4)
}

func BenchmarkCacheAttentionDispatchDim64Len4(b *testing.B) {
	benchmarkCacheAttentionDispatch(b, 64, 4)
}

func benchmarkCacheAttentionSmall(b *testing.B, dim, tokens int) {
	cache, q := buildCacheAttentionBenchInput(tokens, 8, dim)
	dst := make([]float32, dim)
	scale := float32(1 / math.Sqrt(float64(dim)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheAttentionSmall(dst, q, cache, 3, dim, scale)
	}
}

func benchmarkCacheAttentionSplit(b *testing.B, dim, tokens int) {
	cache, q := buildCacheAttentionBenchInput(tokens, 8, dim)
	dst := make([]float32, dim)
	scores := make([]float32, tokens)
	scale := float32(1 / math.Sqrt(float64(dim)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cacheAttentionScores(scores, q, cache, 3, dim, scale)
		tensor.SoftmaxInPlace(scores)
		weightedCacheValueSum(dst, cache, 3, dim, scores)
	}
}

func benchmarkCacheAttentionDispatch(b *testing.B, dim, tokens int) {
	cache, q := buildCacheAttentionBenchInput(tokens, 8, dim)
	dst := make([]float32, dim)
	scores := make([]float32, tokens)
	scale := float32(1 / math.Sqrt(float64(dim)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		switch {
		case cache.len == 2:
			cacheAttentionLen2(dst, q, cache, 3, dim, scale)
		case cache.len == 3:
			cacheAttentionLen3(dst, q, cache, 3, dim, scale)
		case cache.len == 4 && (dim == 128 || dim == 64):
			cacheAttentionLen4(dst, q, cache, 3, dim, scale)
		default:
			cacheAttentionScores(scores, q, cache, 3, dim, scale)
			tensor.SoftmaxInPlace(scores)
			weightedCacheValueSum(dst, cache, 3, dim, scores)
		}
	}
}

func buildCacheAttentionBenchInput(tokens, heads, dim int) (*kvCache, []float32) {
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range k {
			k[i] = float32((t+i)%17-8) / 17
			v[i] = float32((2*t+i)%19-9) / 19
		}
		cache.append(k, v)
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%13-6) / 13
	}
	return cache, q
}

func BenchmarkKVCacheAppend(b *testing.B) {
	dim := 1024
	k := make([]float32, dim)
	v := make([]float32, dim)
	for i := range k {
		k[i] = float32(i%17-8) / 17
		v[i] = float32(i%13-6) / 13
	}
	cache := &kvCache{
		kvDim: dim,
		k:     make([]float32, 0, b.N*dim),
		v:     make([]float32, 0, b.N*dim),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.append(k, v)
	}
}

func BenchmarkWeightedValueSum(b *testing.B) {
	rows, heads, dim := 512, 8, 128
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := make([]float32, rows)
	for i := range weights {
		weights[i] = float32((i%13)+1) / 91
	}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumDim64(b *testing.B) {
	rows, heads, dim := 512, 8, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := make([]float32, rows)
	for i := range weights {
		weights[i] = float32((i%13)+1) / 91
	}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumDim64Len2(b *testing.B) {
	rows, heads, dim := 2, 8, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := []float32{0.75, 0.25}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumLen3(b *testing.B) {
	rows, heads, dim := 3, 8, 128
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := []float32{0.5, 0.3, 0.2}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumDim64Len3(b *testing.B) {
	rows, heads, dim := 3, 8, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := []float32{0.5, 0.3, 0.2}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumLen4(b *testing.B) {
	rows, heads, dim := 4, 8, 128
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := []float32{0.4, 0.3, 0.2, 0.1}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedValueSumDim64Len4(b *testing.B) {
	rows, heads, dim := 4, 8, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	weights := []float32{0.4, 0.3, 0.2, 0.1}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedValueSum(dst, xs, 3*dim, dim, weights)
	}
}

func BenchmarkWeightedCacheValueSumDim64(b *testing.B) {
	tokens, heads, dim := 512, 8, 64
	cache := &kvCache{
		kvDim: heads * dim,
		k:     make([]float32, 0, tokens*heads*dim),
		v:     make([]float32, 0, tokens*heads*dim),
	}
	k := make([]float32, heads*dim)
	v := make([]float32, heads*dim)
	for t := 0; t < tokens; t++ {
		for i := range v {
			v[i] = float32((t+i)%17-8) / 17
		}
		cache.append(k, v)
	}
	weights := make([]float32, tokens)
	for i := range weights {
		weights[i] = float32((i%13)+1) / 91
	}
	dst := make([]float32, dim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		weightedCacheValueSum(dst, cache, 3, dim, weights)
	}
}

func BenchmarkVisionAttentionScores(b *testing.B) {
	rows, heads, dim := 512, 8, 128
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	scores := make([]float32, rows)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		visionAttentionScores(scores, q, xs, 3*dim, dim, 0.08838835)
	}
}

func BenchmarkVisionAttentionScoresDim64(b *testing.B) {
	rows, heads, dim := 512, 8, 64
	width := heads * dim
	xs := make([][]float32, rows)
	for r := range xs {
		xs[r] = make([]float32, width)
		for i := range xs[r] {
			xs[r][i] = float32((r+i)%17-8) / 17
		}
	}
	q := make([]float32, dim)
	for i := range q {
		q[i] = float32(i%17-8) / 17
	}
	scores := make([]float32, rows)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		visionAttentionScores(scores, q, xs, 3*dim, dim, 0.125)
	}
}

func BenchmarkQuantizeRowsFromGGUFF32Store(b *testing.B) {
	dir := b.TempDir()
	writeTinyConfig(b, dir)
	values := make([]float32, 1024*1024)
	for i := range values {
		values[i] = float32(i%17-8) / 17
	}
	src := filepath.Join(dir, "model.safetensors")
	dst := filepath.Join(dir, "model.gguf")
	writeTinySafetensors(b, src, "lm_head.weight", []int64{1024, 1024}, values)
	if err := gguf.ConvertSafetensorsWithOptions(src, dst, dir, gguf.ConvertOptions{}); err != nil {
		b.Fatal(err)
	}
	gf, err := gguf.Open(dst)
	if err != nil {
		b.Fatal(err)
	}
	defer gf.Close()
	var rowFloatBuf []float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt := &Runtime{sf: gf, q6w: map[string]*tensor.Q6Matrix{}, quantization: "q6"}
		if err := rt.quantizeRowsFromStore("lm_head.weight", gf, &rowFloatBuf, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantizeRowsFromSafetensorsBF16Store(b *testing.B) {
	dir := b.TempDir()
	writeTinyConfig(b, dir)
	values := make([]float32, 1024*1024)
	for i := range values {
		values[i] = float32(i%17-8) / 17
	}
	src := filepath.Join(dir, "model.safetensors")
	writeTinySafetensorsBF16(b, src, "lm_head.weight", []int64{1024, 1024}, values)
	sf, err := safetensors.OpenModel(dir)
	if err != nil {
		b.Fatal(err)
	}
	defer sf.Close()
	var rowFloatBuf []float32
	var rowRawBuf []byte
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt := &Runtime{sf: sf, q6w: map[string]*tensor.Q6Matrix{}, quantization: "q6"}
		if err := rt.quantizeRowsFromStore("lm_head.weight", sf, &rowFloatBuf, &rowRawBuf); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkApplyMRoPEWithTable(b *testing.B) {
	heads, dim := 12, 128
	x := make([]float32, heads*dim)
	cosTable := make([]float32, dim/2)
	sinTable := make([]float32, dim/2)
	for i := range x {
		x[i] = float32(i%17-8) / 17
	}
	for i := range cosTable {
		cosTable[i] = 0.99
		sinTable[i] = 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyMRoPEWithTable(x, heads, dim, cosTable, sinTable)
	}
}

func BenchmarkBuildMRoPETableText(b *testing.B) {
	rt := &Runtime{
		ropeFreq: make([]float64, 64),
		ropeAxis: make([]byte, 64),
	}
	for i := range rt.ropeFreq {
		rt.ropeFreq[i] = math.Pow(1000000, -float64(2*i)/128)
		rt.ropeAxis[i] = byte(i % 3)
	}
	cosTable := make([]float32, 64)
	sinTable := make([]float32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt.buildMRoPETable(cosTable, sinTable, ropePos{T: 128, H: 128, W: 128})
	}
}

func BenchmarkBuildMRoPETableZero(b *testing.B) {
	rt := &Runtime{
		ropeFreq: make([]float64, 64),
		ropeAxis: make([]byte, 64),
	}
	for i := range rt.ropeFreq {
		rt.ropeFreq[i] = math.Pow(1000000, -float64(2*i)/128)
	}
	cosTable := make([]float32, 64)
	sinTable := make([]float32, 64)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rt.buildMRoPETable(cosTable, sinTable, ropePos{})
	}
}
