package model

import (
	"math"
	"testing"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/tensor"
)

func TestGenerationScratchPoolReusesScratch(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	rt.scratchPool.New = func() any { return rt.newGenerationScratch() }
	first := rt.getGenerationScratch()
	if len(first.hidden) != 4 || len(first.logits) != 8 || len(first.layers) != 1 {
		t.Fatalf("bad scratch shape: hidden=%d logits=%d layers=%d", len(first.hidden), len(first.logits), len(first.layers))
	}
	rt.putGenerationScratch(first)
	if second := rt.getGenerationScratch(); second == nil {
		t.Fatal("nil scratch")
	}
}

func TestGenerationScratchFallbackWithoutPoolNew(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	got := rt.getGenerationScratch()
	if got == nil || len(got.hidden) != 4 || len(got.logits) != 8 {
		t.Fatalf("bad scratch: %+v", got)
	}
}

func TestGenerationScratchDropsLargeTopKCandidates(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	large := rt.newGenerationScratch()
	large.candidates = make([]tokenScore, 0, 8193)
	rt.putGenerationScratch(large)
	if got := rt.getGenerationScratch(); cap(got.candidates) != 0 {
		t.Fatalf("large candidates retained cap=%d", cap(got.candidates))
	}
	small := rt.newGenerationScratch()
	small.candidates = make([]tokenScore, 0, 8192)
	rt.putGenerationScratch(small)
	if got := rt.getGenerationScratch(); cap(got.candidates) != 8192 {
		t.Fatalf("small candidates dropped cap=%d", cap(got.candidates))
	}
}

func TestGenerationScratchDropsLargeScoreBlock(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	large := rt.newGenerationScratch()
	rt.ensureScoreCapacity(large, 8193)
	rt.putGenerationScratch(large)
	got := rt.getGenerationScratch()
	if cap(got.scoreBlock) != 0 {
		t.Fatalf("large score block retained cap=%d", cap(got.scoreBlock))
	}
	for i := range got.layers {
		if cap(got.layers[i].scores) != 0 {
			t.Fatalf("layer %d scores retained cap=%d", i, cap(got.layers[i].scores))
		}
	}
	small := rt.newGenerationScratch()
	rt.ensureScoreCapacity(small, 8192)
	rt.putGenerationScratch(small)
	got = rt.getGenerationScratch()
	if cap(got.scoreBlock) != 2*8192 {
		t.Fatalf("small score block dropped cap=%d", cap(got.scoreBlock))
	}
}

func TestGenerationScratchDropsLargeGenerationBuffers(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	large := rt.newGenerationScratch()
	large.positions = make([]ropePos, 0, 8193)
	large.inputIDs = make([]int, 0, 8193)
	rt.putGenerationScratch(large)
	got := rt.getGenerationScratch()
	if cap(got.positions) != 0 || cap(got.inputIDs) != 0 {
		t.Fatalf("large generation buffers retained positions=%d inputIDs=%d", cap(got.positions), cap(got.inputIDs))
	}
	small := rt.newGenerationScratch()
	small.positions = make([]ropePos, 0, 8192)
	small.inputIDs = make([]int, 0, 8192)
	rt.putGenerationScratch(small)
	got = rt.getGenerationScratch()
	if cap(got.positions) != 8192 || cap(got.inputIDs) != 8192 {
		t.Fatalf("small generation buffers dropped positions=%d inputIDs=%d", cap(got.positions), cap(got.inputIDs))
	}
}

func TestGenerationScratchDropsOversizedWeights(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         9000,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   1,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	large := rt.newGenerationScratch()
	large.weights = make([]float32, 0, 9001)
	rt.putGenerationScratch(large)
	if got := rt.getGenerationScratch(); cap(got.weights) != 0 {
		t.Fatalf("large weights retained cap=%d", cap(got.weights))
	}
	small := rt.newGenerationScratch()
	small.weights = make([]float32, 0, 9000)
	rt.putGenerationScratch(small)
	if got := rt.getGenerationScratch(); cap(got.weights) != 9000 {
		t.Fatalf("vocab-sized weights dropped cap=%d", cap(got.weights))
	}
}

func TestEnsureScoreCapacity(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		VocabSize:         8,
		HiddenSize:        4,
		IntermediateSize:  6,
		NumHiddenLayers:   2,
		NumAttentionHeads: 1,
		NumKeyValueHeads:  1,
		HeadDim:           4,
	}}
	scratch := rt.newGenerationScratch()
	rt.ensureScoreCapacity(scratch, 16)
	for i := range scratch.layers {
		if cap(scratch.layers[i].scores) < 16 {
			t.Fatalf("layer %d score cap=%d want >=16", i, cap(scratch.layers[i].scores))
		}
		if cap(scratch.layers[i].scores) != 16 {
			t.Fatalf("layer %d score cap=%d want exact packed cap 16", i, cap(scratch.layers[i].scores))
		}
	}
	if cap(scratch.scoreBlock) < 32 {
		t.Fatalf("score block cap=%d want >=32", cap(scratch.scoreBlock))
	}
	scratch.layers[0].scores = scratch.layers[0].scores[:8]
	rt.scratchPool.New = func() any { return scratch }
	got := rt.getGenerationScratch()
	if len(got.layers[0].scores) != 0 {
		t.Fatalf("score len=%d want 0", len(got.layers[0].scores))
	}
}

func TestKVCachePoolShapes(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		NumHiddenLayers:  2,
		NumKeyValueHeads: 1,
		HeadDim:          4,
	}}
	rt.kvPool.New = func() any { return rt.newKVCaches(0) }
	caches, cachePtr := rt.getKVCaches(3)
	if len(caches) != 2 {
		t.Fatalf("len=%d want 2", len(caches))
	}
	for i := range caches {
		if caches[i].kvDim != 4 || cap(caches[i].k) < 12 || cap(caches[i].v) < 12 {
			t.Fatalf("cache[%d]=%+v", i, caches[i])
		}
	}
	caches[0].append([]float32{1, 2, 3, 4}, []float32{5, 6, 7, 8})
	rt.putKVCaches(caches, cachePtr)
	next, _ := rt.getKVCaches(1)
	if next[0].len != 0 || len(next[0].k) != 0 || len(next[0].v) != 0 {
		t.Fatalf("cache not reset: %+v", next[0])
	}
}

func TestKVCacheFallbackWithoutPoolNew(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{
		NumHiddenLayers:  2,
		NumKeyValueHeads: 1,
		HeadDim:          4,
	}}
	got, _ := rt.getKVCaches(3)
	if len(got) != 2 {
		t.Fatalf("len=%d want 2", len(got))
	}
	if cap(got[0].k) < 12 || cap(got[0].v) < 12 {
		t.Fatalf("bad cache: %+v", got[0])
	}
}

func BenchmarkEnsureScoreCapacity(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{
		NumHiddenLayers: 24,
	}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scratch := &generationScratch{layers: make([]layerScratch, 24)}
		rt.ensureScoreCapacity(scratch, 2048)
	}
}

func TestWeightedCacheValueSumClearsAndUsesHead(t *testing.T) {
	cache := &kvCache{kvDim: 8, k: make([]float32, 0, 16), v: make([]float32, 0, 16)}
	cache.append([]float32{0, 0, 0, 0, 0, 0, 0, 0}, []float32{1, 2, 3, 4, 10, 20, 30, 40})
	cache.append([]float32{0, 0, 0, 0, 0, 0, 0, 0}, []float32{5, 6, 7, 8, 50, 60, 70, 80})
	dst := []float32{99, 99, 99, 99}
	weightedCacheValueSum(dst, cache, 1, 4, []float32{0.25, 0.75})
	want := []float32{40, 50, 60, 70}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want[i])
		}
	}
}

func TestCacheAttentionScoresMatchesDot(t *testing.T) {
	cache := &kvCache{kvDim: 8, k: make([]float32, 0, 24), v: make([]float32, 0, 24)}
	cache.append([]float32{1, 2, 3, 4, 10, 20, 30, 40}, make([]float32, 8))
	cache.append([]float32{5, 6, 7, 8, 50, 60, 70, 80}, make([]float32, 8))
	cache.append([]float32{-1, -2, -3, -4, -10, -20, -30, -40}, make([]float32, 8))
	q := []float32{0.5, -1, 0.25, 2}
	got := make([]float32, cache.len)
	const scale = 0.125
	cacheAttentionScores(got, q, cache, 1, 4, scale)
	for i := range got {
		want := tensor.Dot(q, cache.key(i, 1, 4)) * scale
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("score[%d]=%f want %f", i, got[i], want)
		}
	}
}

func TestCacheAttentionScoresDim128MatchesDot(t *testing.T) {
	const dim = 128
	cache := &kvCache{kvDim: dim, k: make([]float32, 0, dim*2), v: make([]float32, 0, dim*2)}
	k0 := make([]float32, dim)
	k1 := make([]float32, dim)
	q := make([]float32, dim)
	for i := 0; i < dim; i++ {
		k0[i] = float32(i%11-5) / 11
		k1[i] = float32(i%17-8) / 17
		q[i] = float32(i%13-6) / 13
	}
	cache.append(k0, make([]float32, dim))
	cache.append(k1, make([]float32, dim))
	got := make([]float32, cache.len)
	const scale = 0.08838835
	cacheAttentionScores(got, q, cache, 0, dim, scale)
	for i := range got {
		want := tensor.Dot(q, cache.key(i, 0, dim)) * scale
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("score[%d]=%f want %f", i, got[i], want)
		}
	}
}
