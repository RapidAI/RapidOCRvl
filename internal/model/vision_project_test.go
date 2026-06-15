package model

import (
	"math"
	"strconv"
	"testing"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/tensor"
	"paddleocrvl-go/internal/vision"
)

func TestProjectImageUsesAllTemporalFrames(t *testing.T) {
	rt := &Runtime{
		cfg: &config.Config{
			HiddenSize: 3,
			VisionConfig: config.Vision{
				HiddenSize: 2,
			},
		},
	}
	rt.vision.projNormW = []float32{1, 1}
	rt.vision.projNormB = []float32{0, 0}
	rt.vision.proj1W = identityWeights(8)
	rt.vision.proj1B = make([]float32, 8)
	rt.vision.proj2W = []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
	}
	rt.vision.proj2B = make([]float32, 3)
	x := [][]float32{
		{1, 2}, {3, 5}, {7, 11}, {13, 17},
		{19, 29}, {31, 43}, {47, 61}, {67, 83},
	}
	out := rt.projectImage(x, vision.Grid{T: 2, H: 2, W: 2})
	if len(out) != 2 {
		t.Fatalf("rows=%d want 2", len(out))
	}
	if allZero(out[1]) {
		t.Fatalf("second temporal frame was not projected: %+v", out)
	}
	if equalVec(out[0], out[1]) {
		t.Fatalf("temporal frames collapsed into same output: %+v", out)
	}
}

func TestVisionEmbeddingsRepeatsPositionForTemporalFrames(t *testing.T) {
	rt := &Runtime{
		cfg: &config.Config{
			VisionConfig: config.Vision{
				HiddenSize: 2,
			},
		},
	}
	rt.vision.patchW = []float32{
		1, 0,
		0, 1,
	}
	rt.vision.patchB = []float32{0, 0}
	rt.vision.pos = make([]float32, 27*27*2)
	rt.vision.pos[0] = 10
	rt.vision.pos[1] = 20
	pp := &vision.Preprocessed{
		Patches: [][]float32{{1, 2}, {3, 4}},
		Grid:    vision.Grid{T: 2, H: 1, W: 1},
	}
	out := rt.visionEmbeddings(pp)
	if len(out) != 2 {
		t.Fatalf("rows=%d want 2", len(out))
	}
	if out[0][0] != 11 || out[0][1] != 22 || out[1][0] != 13 || out[1][1] != 24 {
		t.Fatalf("embeddings=%+v", out)
	}
}

func TestInterpolateVisionPosCachesBySize(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 2}}}
	rt.vision.pos = make([]float32, 27*27*2)
	first := rt.interpolateVisionPos(2, 2)
	second := rt.interpolateVisionPos(2, 2)
	if len(first) == 0 || len(second) == 0 || &first[0][0] != &second[0][0] {
		t.Fatalf("expected cached position table")
	}
	third := rt.interpolateVisionPos(1, 1)
	if len(third) == 0 || &first[0][0] == &third[0][0] {
		t.Fatalf("expected separate cache entry")
	}
	if got := rt.CacheStats().VisionPositionTables; got != 2 {
		t.Fatalf("cache tables=%d want 2", got)
	}
}

func TestInterpolateVisionPosBaseUsesPositionView(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 2}}}
	rt.vision.pos = make([]float32, 27*27*2)
	for i := range rt.vision.pos {
		rt.vision.pos[i] = float32(i)
	}
	got := rt.interpolateVisionPos(27, 27)
	if len(got) != 27*27 || len(got[0]) != 2 {
		t.Fatalf("shape rows=%d cols=%d", len(got), len(got[0]))
	}
	if &got[0][0] != &rt.vision.pos[0] {
		t.Fatal("base grid should use position embedding view")
	}
	if got[10][1] != rt.vision.pos[10*2+1] {
		t.Fatalf("got=%f want=%f", got[10][1], rt.vision.pos[10*2+1])
	}
	if got := rt.CacheStats().VisionPositionTables; got != 0 {
		t.Fatalf("cache tables=%d want 0", got)
	}
}

func TestWeightedValueSumClearsAndUsesHeadOffset(t *testing.T) {
	rows := [][]float32{
		{1, 2, 10, 20, 30},
		{3, 4, 40, 50, 60},
		{5, 6, 70, 80, 90},
	}
	dst := []float32{99, 99, 99}
	weightedValueSum(dst, rows, 2, 3, []float32{0.25, 0.5, 0.25})
	want := []float32{40, 50, 60}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%f want %f", i, dst[i], want[i])
		}
	}
}

func TestVisionAttentionScoresMatchesDot(t *testing.T) {
	rows := [][]float32{
		{1, 2, 10, 20, 30},
		{3, 4, 40, 50, 60},
		{5, 6, 70, 80, 90},
	}
	q := []float32{0.5, -1, 0.25}
	got := make([]float32, len(rows))
	const scale = 0.25
	visionAttentionScores(got, q, rows, 2, 3, scale)
	for i := range got {
		want := tensor.Dot(q, rows[i][2:5]) * scale
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("score[%d]=%f want %f", i, got[i], want)
		}
	}
}

func TestApplyVisionRoPEPairMatchesSeparate(t *testing.T) {
	grid := vision.Grid{T: 1, H: 2, W: 2}
	heads, hd := 2, 8
	rope := newVisionRoPETables(grid, hd)
	q := makeRowsForModelTest(grid.H*grid.W, heads*hd)
	k := makeRowsForModelTest(grid.H*grid.W, heads*hd)
	for i := range q {
		fillBenchFloat32(q[i])
		fillBenchFloat32(k[i])
	}
	wantQ := cloneRowsForModelTest(q)
	wantK := cloneRowsForModelTest(k)
	applyVisionRoPE(wantQ, grid, heads, hd, rope)
	applyVisionRoPE(wantK, grid, heads, hd, rope)
	applyVisionRoPEPair(q, k, grid, heads, hd, rope)
	for i := range q {
		assertModelCloseVec(t, "vision-rope-q", q[i], wantQ[i], 0)
		assertModelCloseVec(t, "vision-rope-k", k[i], wantK[i], 0)
	}
}

func TestVisionLayerChainMatchesUnfused(t *testing.T) {
	rt := newVisionLayerTestRuntime()
	grid := vision.Grid{T: 1, H: 2, W: 2}
	rope := newVisionRoPETables(grid, rt.cfg.VisionConfig.HiddenSize/rt.cfg.VisionConfig.NumAttentionHeads)
	src := makeRowsForModelTest(grid.H*grid.W, rt.cfg.VisionConfig.HiddenSize)
	for i := range src {
		fillBenchFloat32(src[i])
	}
	fused := cloneRowsForModelTest(src)
	unfused := cloneRowsForModelTest(src)

	scratch := rt.newVisionScratch(len(src))
	normReady := false
	for i := range rt.vision.layers {
		var next *visionLayerWeights
		if i+1 < len(rt.vision.layers) {
			next = &rt.vision.layers[i+1]
		}
		fused = rt.visionLayer(fused, rt.vision.layers[i], next, normReady, grid, rope, scratch)
		normReady = next != nil
	}

	legacyScratch := rt.newVisionScratch(len(src))
	for _, layer := range rt.vision.layers {
		unfusedVisionLayer(rt, unfused, layer, grid, rope, legacyScratch)
	}
	rt.layerNormRows(unfused, rt.vision.postNormW, rt.vision.postNormB, float32(rt.cfg.VisionConfig.LayerNormEps))

	for i := range fused {
		assertModelCloseVec(t, "vision-layer-chain", fused[i], unfused[i], 1e-5)
	}
}

func TestNewVisionScratchShapes(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 4, IntermediateSize: 6}}}
	s := rt.newVisionScratch(3)
	if len(s.norm) != 3 || len(s.norm[0]) != 4 || len(s.hids) != 3 || len(s.hids[0]) != 6 || len(s.scores) != 3 {
		t.Fatalf("scratch shapes norm=%dx%d hids=%dx%d scores=%d", len(s.norm), len(s.norm[0]), len(s.hids), len(s.hids[0]), len(s.scores))
	}
	if len(s.q[0]) != 4 || len(s.k[0]) != 4 || len(s.v[0]) != 4 || len(s.headOut[0]) != 4 || len(s.attOut[0]) != 4 || len(s.mlp[0]) != 4 {
		t.Fatalf("bad matrix widths")
	}
}

func TestVisionScratchPoolReusesMatchingShape(t *testing.T) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 4, IntermediateSize: 6}}}
	first := rt.getVisionScratch(3)
	rt.putVisionScratch(first)
	second := rt.getVisionScratch(3)
	if second == nil || second.hidden != 4 || len(second.norm) != 3 {
		t.Fatalf("expected matching scratch, got %+v", second)
	}
	rt.putVisionScratch(second)
	rt.cfg.VisionConfig.HiddenSize = 8
	third := rt.getVisionScratch(3)
	if third == first || third.hidden != 8 {
		t.Fatalf("expected replacement scratch, got hidden=%d", third.hidden)
	}
}

func TestReleaseCachedVisionWeightMapEntriesKeepsCachedSlices(t *testing.T) {
	rt := newVisionLayerTestRuntime()
	rt.w = map[string][]float32{
		"visual.vision_model.embeddings.patch_embedding.weight":    {1},
		"visual.vision_model.embeddings.patch_embedding.bias":      {2},
		"visual.vision_model.embeddings.position_embedding.weight": make([]float32, 27*27*rt.cfg.VisionConfig.HiddenSize),
		"visual.vision_model.post_layernorm.weight":                onesForModelTest(rt.cfg.VisionConfig.HiddenSize),
		"visual.vision_model.post_layernorm.bias":                  biasForModelTest(rt.cfg.VisionConfig.HiddenSize),
		"mlp_AR.pre_norm.weight":                                   onesForModelTest(rt.cfg.VisionConfig.HiddenSize),
		"mlp_AR.pre_norm.bias":                                     biasForModelTest(rt.cfg.VisionConfig.HiddenSize),
		"mlp_AR.linear_1.weight":                                   {3},
		"mlp_AR.linear_1.bias":                                     {4},
		"mlp_AR.linear_2.weight":                                   {5},
		"mlp_AR.linear_2.bias":                                     {6},
	}
	for i := 0; i < rt.cfg.VisionConfig.NumHiddenLayers; i++ {
		p := "visual.vision_model.encoder.layers." + strconv.Itoa(i) + "."
		rt.w[p+"layer_norm1.weight"] = rt.vision.layers[i].ln1w
		rt.w[p+"layer_norm1.bias"] = rt.vision.layers[i].ln1b
		rt.w[p+"layer_norm2.weight"] = rt.vision.layers[i].ln2w
		rt.w[p+"layer_norm2.bias"] = rt.vision.layers[i].ln2b
		rt.w[p+"self_attn.q_proj.weight"] = rt.vision.layers[i].qw
		rt.w[p+"self_attn.q_proj.bias"] = rt.vision.layers[i].qb
		rt.w[p+"self_attn.k_proj.weight"] = rt.vision.layers[i].kw
		rt.w[p+"self_attn.k_proj.bias"] = rt.vision.layers[i].kb
		rt.w[p+"self_attn.v_proj.weight"] = rt.vision.layers[i].vw
		rt.w[p+"self_attn.v_proj.bias"] = rt.vision.layers[i].vb
		rt.w[p+"self_attn.out_proj.weight"] = rt.vision.layers[i].ow
		rt.w[p+"self_attn.out_proj.bias"] = rt.vision.layers[i].ob
		rt.w[p+"mlp.fc1.weight"] = rt.vision.layers[i].fc1w
		rt.w[p+"mlp.fc1.bias"] = rt.vision.layers[i].fc1b
		rt.w[p+"mlp.fc2.weight"] = rt.vision.layers[i].fc2w
		rt.w[p+"mlp.fc2.bias"] = rt.vision.layers[i].fc2b
	}
	rt.cacheVisionWeights()
	patch := rt.vision.patchW
	ln1 := rt.vision.layers[0].ln1w
	rt.releaseCachedVisionWeightMapEntries()
	if len(rt.w) != 0 {
		t.Fatalf("w map len=%d want 0", len(rt.w))
	}
	if &rt.vision.patchW[0] != &patch[0] || &rt.vision.layers[0].ln1w[0] != &ln1[0] {
		t.Fatal("cached vision slices changed")
	}
}

func identityWeights(n int) []float32 {
	w := make([]float32, n*n)
	for i := 0; i < n; i++ {
		w[i*n+i] = 1
	}
	return w
}

func allZero(x []float32) bool {
	for _, v := range x {
		if v != 0 {
			return false
		}
	}
	return true
}

func equalVec(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func newVisionLayerTestRuntime() *Runtime {
	cfg := &config.Config{
		VisionConfig: config.Vision{
			HiddenSize:        16,
			IntermediateSize:  24,
			NumAttentionHeads: 4,
			NumHiddenLayers:   2,
			LayerNormEps:      1e-5,
		},
	}
	rt := &Runtime{cfg: cfg}
	d := cfg.VisionConfig.HiddenSize
	inter := cfg.VisionConfig.IntermediateSize
	rt.vision.postNormW = onesForModelTest(d)
	rt.vision.postNormB = biasForModelTest(d)
	rt.vision.layers = make([]visionLayerWeights, cfg.VisionConfig.NumHiddenLayers)
	for i := range rt.vision.layers {
		rt.vision.layers[i] = visionLayerWeights{
			ln1w: onesForModelTest(d), ln1b: biasForModelTest(d),
			ln2w: onesForModelTest(d), ln2b: biasForModelTest(d),
			qw: make([]float32, d*d), qb: biasForModelTest(d),
			kw: make([]float32, d*d), kb: biasForModelTest(d),
			vw: make([]float32, d*d), vb: biasForModelTest(d),
			ow: make([]float32, d*d), ob: biasForModelTest(d),
			fc1w: make([]float32, inter*d), fc1b: biasForModelTest(inter),
			fc2w: make([]float32, d*inter), fc2b: biasForModelTest(d),
		}
		fillBenchFloat32(rt.vision.layers[i].qw)
		fillBenchFloat32(rt.vision.layers[i].kw)
		fillBenchFloat32(rt.vision.layers[i].vw)
		fillBenchFloat32(rt.vision.layers[i].ow)
		fillBenchFloat32(rt.vision.layers[i].fc1w)
		fillBenchFloat32(rt.vision.layers[i].fc2w)
	}
	return rt
}

func unfusedVisionLayer(rt *Runtime, x [][]float32, lw visionLayerWeights, grid vision.Grid, rope visionRoPETables, scratch *visionScratch) {
	d := rt.cfg.VisionConfig.HiddenSize
	eps := float32(rt.cfg.VisionConfig.LayerNormEps)
	norm := scratch.norm
	for i := range x {
		tensor.LayerNorm(norm[i], x[i], lw.ln1w, lw.ln1b, eps)
	}
	att := rt.visionAttention(norm, lw, grid, rope, scratch)
	for i := range x {
		tensor.AddInPlace(x[i], att[i])
		tensor.LayerNorm(norm[i], x[i], lw.ln2w, lw.ln2b, eps)
	}
	tensor.MatRowsBias(scratch.hids, norm, lw.fc1w, lw.fc1b, rt.cfg.VisionConfig.IntermediateSize, d)
	tensor.GELUTanhRowsInPlace(scratch.hids)
	tensor.MatRowsBias(scratch.mlp, scratch.hids, lw.fc2w, lw.fc2b, d, rt.cfg.VisionConfig.IntermediateSize)
	for i := range x {
		tensor.AddInPlace(x[i], scratch.mlp[i])
	}
}

func onesForModelTest(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = 1
	}
	return x
}

func biasForModelTest(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%5-2) / 31
	}
	return x
}

func makeRowsForModelTest(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := range out {
		out[i] = make([]float32, cols)
	}
	return out
}

func cloneRowsForModelTest(src [][]float32) [][]float32 {
	out := make([][]float32, len(src))
	for i := range src {
		out[i] = append([]float32(nil), src[i]...)
	}
	return out
}

func assertModelCloseVec(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("%s[%d] got %f want %f", name, i, got[i], want[i])
		}
	}
}
