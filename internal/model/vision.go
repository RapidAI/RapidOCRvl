package model

import (
	"bytes"
	"fmt"
	"io"
	"runtime"
	"strconv"
	"sync"

	"paddleocrvl-go/internal/tensor"
	"paddleocrvl-go/internal/vision"
)

const maxVisionShapeCacheEntries = 16

type visionLayerWeights struct {
	ln1w, ln1b []float32
	ln2w, ln2b []float32
	qw, qb     []float32
	kw, kb     []float32
	vw, vb     []float32
	ow, ob     []float32
	fc1w, fc1b []float32
	fc2w, fc2b []float32
}

type visionWeights struct {
	patchW, patchB       []float32
	pos                  []float32
	postNormW, postNormB []float32
	projNormW, projNormB []float32
	proj1W, proj1B       []float32
	proj2W, proj2B       []float32
	layers               []visionLayerWeights
	basePosRows          [][]float32
}

type visionRoPETables struct {
	h    [][]ropePair
	w    [][]ropePair
	hCos [][]float32
	hSin [][]float32
	wCos [][]float32
	wSin [][]float32
}

type visionScratch struct {
	tokens  int
	hidden  int
	inter   int
	embed   [][]float32
	norm    [][]float32
	q       [][]float32
	k       [][]float32
	v       [][]float32
	headOut [][]float32
	attOut  [][]float32
	mlp     [][]float32
	hids    [][]float32
	scores  []float32
}

func (rt *Runtime) EncodeImage(path string) ([][]float32, error) {
	embeds, _, err := rt.EncodeImageWithGrid(path)
	return embeds, err
}

func (rt *Runtime) PreloadVision() error {
	return rt.ensureVisionWeights()
}

func (rt *Runtime) validateVisionReadyConfig() error {
	if rt == nil || rt.cfg == nil {
		return fmt.Errorf("runtime config not initialized")
	}
	if rt.cfg.VisionConfig.NumHiddenLayers <= 0 {
		return fmt.Errorf("config vision_config.num_hidden_layers must be > 0 for image encoding")
	}
	return validateRuntimeVisionConfig(rt.cfg.VisionConfig)
}

func (rt *Runtime) VisionLoaded() bool {
	rt.visionMu.RLock()
	defer rt.visionMu.RUnlock()
	return rt.visionLoaded
}

func (rt *Runtime) EncodeImageBytes(data []byte) ([][]float32, [3]int, error) {
	return rt.EncodeImageReader(bytes.NewReader(data))
}

func (rt *Runtime) EncodeImageReader(r io.Reader) ([][]float32, [3]int, error) {
	if err := rt.ensureVisionWeights(); err != nil {
		return nil, [3]int{}, err
	}
	pp, err := vision.LoadImageReader(r)
	if err != nil {
		return nil, [3]int{}, err
	}
	return rt.encodePreprocessedImage(pp)
}

func (rt *Runtime) EncodeImageWithGrid(path string) ([][]float32, [3]int, error) {
	if err := rt.ensureVisionWeights(); err != nil {
		return nil, [3]int{}, err
	}
	pp, err := vision.LoadImage(path)
	if err != nil {
		return nil, [3]int{}, err
	}
	return rt.encodePreprocessedImage(pp)
}

func (rt *Runtime) encodePreprocessedImage(pp *vision.Preprocessed) ([][]float32, [3]int, error) {
	scratch := rt.getVisionScratch(len(pp.Patches))
	hidden := rt.visionEmbeddingsInto(scratch.embed, pp)
	rope := rt.cachedVisionRoPETables(pp.Grid, rt.cfg.VisionConfig.HiddenSize/rt.cfg.VisionConfig.NumAttentionHeads)
	layers := rt.cfg.VisionConfig.NumHiddenLayers
	normReady := false
	for i := 0; i < layers; i++ {
		var next *visionLayerWeights
		if i+1 < layers {
			next = &rt.vision.layers[i+1]
		}
		hidden = rt.visionLayer(hidden, rt.vision.layers[i], next, normReady, pp.Grid, rope, scratch)
		normReady = next != nil
	}
	if layers == 0 {
		rt.layerNormRows(hidden,
			rt.vision.postNormW,
			rt.vision.postNormB,
			float32(rt.cfg.VisionConfig.LayerNormEps),
		)
	}
	out := rt.projectImage(hidden, pp.Grid)
	grid := [3]int{pp.Grid.T, pp.Grid.H, pp.Grid.W}
	rt.putVisionScratch(scratch)
	return out, grid, nil
}

func (rt *Runtime) newVisionScratch(tokens int) *visionScratch {
	d := rt.cfg.VisionConfig.HiddenSize
	inter := rt.cfg.VisionConfig.IntermediateSize
	hiddenData := make([]float32, tokens*d*8)
	hiddenRows := make([][]float32, tokens*8)
	nextHiddenRows := func() [][]float32 {
		rows := hiddenRows[:tokens]
		makeRowsViewInto(rows, hiddenData[:tokens*d], d)
		hiddenData = hiddenData[tokens*d:]
		hiddenRows = hiddenRows[tokens:]
		return rows
	}
	return &visionScratch{
		tokens:  tokens,
		hidden:  d,
		inter:   inter,
		embed:   nextHiddenRows(),
		norm:    nextHiddenRows(),
		q:       nextHiddenRows(),
		k:       nextHiddenRows(),
		v:       nextHiddenRows(),
		headOut: nextHiddenRows(),
		attOut:  nextHiddenRows(),
		mlp:     nextHiddenRows(),
		hids:    makeRows(tokens, inter),
		scores:  make([]float32, tokens),
	}
}

func (rt *Runtime) getVisionScratch(tokens int) *visionScratch {
	if v := rt.visionScratchPool.Get(); v != nil {
		s := v.(*visionScratch)
		d := rt.cfg.VisionConfig.HiddenSize
		inter := rt.cfg.VisionConfig.IntermediateSize
		if s.tokens == tokens && s.hidden == d && s.inter == inter {
			rt.ensureVisionScratchEmbed(s)
			return s
		}
	}
	s := rt.newVisionScratch(tokens)
	rt.ensureVisionScratchEmbed(s)
	return s
}

func (rt *Runtime) putVisionScratch(s *visionScratch) {
	if s != nil {
		rt.visionScratchPool.Put(s)
	}
}

func (rt *Runtime) ensureVisionScratchEmbed(s *visionScratch) {
	if s == nil || len(s.embed) == s.tokens {
		return
	}
	d := rt.cfg.VisionConfig.HiddenSize
	s.embed = makeRows(s.tokens, d)
}

func (rt *Runtime) ensureVisionWeights() error {
	if err := rt.validateVisionReadyConfig(); err != nil {
		return err
	}
	rt.visionMu.Lock()
	defer rt.visionMu.Unlock()
	if rt.visionLoaded {
		return nil
	}
	names := make([]string, 0, 11+rt.cfg.VisionConfig.NumHiddenLayers*16)
	names = append(names,
		"visual.vision_model.embeddings.patch_embedding.weight",
		"visual.vision_model.embeddings.patch_embedding.bias",
		"visual.vision_model.embeddings.position_embedding.weight",
		"visual.vision_model.post_layernorm.weight",
		"visual.vision_model.post_layernorm.bias",
		"mlp_AR.pre_norm.weight",
		"mlp_AR.pre_norm.bias",
		"mlp_AR.linear_1.weight",
		"mlp_AR.linear_1.bias",
		"mlp_AR.linear_2.weight",
		"mlp_AR.linear_2.bias",
	)
	for i := 0; i < rt.cfg.VisionConfig.NumHiddenLayers; i++ {
		p := "visual.vision_model.encoder.layers." + strconv.Itoa(i) + "."
		names = append(names,
			p+"layer_norm1.weight", p+"layer_norm1.bias",
			p+"layer_norm2.weight", p+"layer_norm2.bias",
			p+"self_attn.q_proj.weight", p+"self_attn.q_proj.bias",
			p+"self_attn.k_proj.weight", p+"self_attn.k_proj.bias",
			p+"self_attn.v_proj.weight", p+"self_attn.v_proj.bias",
			p+"self_attn.out_proj.weight", p+"self_attn.out_proj.bias",
			p+"mlp.fc1.weight", p+"mlp.fc1.bias",
			p+"mlp.fc2.weight", p+"mlp.fc2.bias",
		)
	}
	for i, name := range names {
		if rt.progress != nil {
			rt.progress(i, len(names), name, "LOAD-VISION")
		}
		if _, ok := rt.w[name]; ok {
			continue
		}
		v, _, err := rt.sf.Float32(name)
		if err != nil {
			return err
		}
		rt.w[name] = v
	}
	rt.cacheVisionWeights()
	rt.releaseCachedVisionWeightMapEntries()
	rt.visionLoaded = true
	if rt.progress != nil {
		rt.progress(len(names), len(names), "", "LOAD-VISION")
	}
	return nil
}

func (rt *Runtime) cacheVisionWeights() {
	rt.vision = visionWeights{
		patchW:    rt.w["visual.vision_model.embeddings.patch_embedding.weight"],
		patchB:    rt.w["visual.vision_model.embeddings.patch_embedding.bias"],
		pos:       rt.w["visual.vision_model.embeddings.position_embedding.weight"],
		postNormW: rt.w["visual.vision_model.post_layernorm.weight"],
		postNormB: rt.w["visual.vision_model.post_layernorm.bias"],
		projNormW: rt.w["mlp_AR.pre_norm.weight"],
		projNormB: rt.w["mlp_AR.pre_norm.bias"],
		proj1W:    rt.w["mlp_AR.linear_1.weight"],
		proj1B:    rt.w["mlp_AR.linear_1.bias"],
		proj2W:    rt.w["mlp_AR.linear_2.weight"],
		proj2B:    rt.w["mlp_AR.linear_2.bias"],
		layers:    make([]visionLayerWeights, rt.cfg.VisionConfig.NumHiddenLayers),
	}
	for i := range rt.vision.layers {
		rt.vision.layers[i] = rt.vlw(i)
	}
}

func (rt *Runtime) releaseCachedVisionWeightMapEntries() {
	delete(rt.w, "visual.vision_model.embeddings.patch_embedding.weight")
	delete(rt.w, "visual.vision_model.embeddings.patch_embedding.bias")
	delete(rt.w, "visual.vision_model.embeddings.position_embedding.weight")
	delete(rt.w, "visual.vision_model.post_layernorm.weight")
	delete(rt.w, "visual.vision_model.post_layernorm.bias")
	delete(rt.w, "mlp_AR.pre_norm.weight")
	delete(rt.w, "mlp_AR.pre_norm.bias")
	delete(rt.w, "mlp_AR.linear_1.weight")
	delete(rt.w, "mlp_AR.linear_1.bias")
	delete(rt.w, "mlp_AR.linear_2.weight")
	delete(rt.w, "mlp_AR.linear_2.bias")
	if rt.cfg == nil {
		return
	}
	for i := 0; i < rt.cfg.VisionConfig.NumHiddenLayers; i++ {
		p := "visual.vision_model.encoder.layers." + strconv.Itoa(i) + "."
		delete(rt.w, p+"layer_norm1.weight")
		delete(rt.w, p+"layer_norm1.bias")
		delete(rt.w, p+"layer_norm2.weight")
		delete(rt.w, p+"layer_norm2.bias")
		delete(rt.w, p+"self_attn.q_proj.weight")
		delete(rt.w, p+"self_attn.q_proj.bias")
		delete(rt.w, p+"self_attn.k_proj.weight")
		delete(rt.w, p+"self_attn.k_proj.bias")
		delete(rt.w, p+"self_attn.v_proj.weight")
		delete(rt.w, p+"self_attn.v_proj.bias")
		delete(rt.w, p+"self_attn.out_proj.weight")
		delete(rt.w, p+"self_attn.out_proj.bias")
		delete(rt.w, p+"mlp.fc1.weight")
		delete(rt.w, p+"mlp.fc1.bias")
		delete(rt.w, p+"mlp.fc2.weight")
		delete(rt.w, p+"mlp.fc2.bias")
	}
}

func (rt *Runtime) vlw(i int) visionLayerWeights {
	p := "visual.vision_model.encoder.layers." + strconv.Itoa(i) + "."
	return visionLayerWeights{
		ln1w: rt.w[p+"layer_norm1.weight"], ln1b: rt.w[p+"layer_norm1.bias"],
		ln2w: rt.w[p+"layer_norm2.weight"], ln2b: rt.w[p+"layer_norm2.bias"],
		qw: rt.w[p+"self_attn.q_proj.weight"], qb: rt.w[p+"self_attn.q_proj.bias"],
		kw: rt.w[p+"self_attn.k_proj.weight"], kb: rt.w[p+"self_attn.k_proj.bias"],
		vw: rt.w[p+"self_attn.v_proj.weight"], vb: rt.w[p+"self_attn.v_proj.bias"],
		ow: rt.w[p+"self_attn.out_proj.weight"], ob: rt.w[p+"self_attn.out_proj.bias"],
		fc1w: rt.w[p+"mlp.fc1.weight"], fc1b: rt.w[p+"mlp.fc1.bias"],
		fc2w: rt.w[p+"mlp.fc2.weight"], fc2b: rt.w[p+"mlp.fc2.bias"],
	}
}

func (rt *Runtime) visionEmbeddings(pp *vision.Preprocessed) [][]float32 {
	d := rt.cfg.VisionConfig.HiddenSize
	out := makeRows(len(pp.Patches), d)
	return rt.visionEmbeddingsInto(out, pp)
}

func (rt *Runtime) visionEmbeddingsInto(out [][]float32, pp *vision.Preprocessed) [][]float32 {
	pos := rt.interpolateVisionPos(pp.Grid.H, pp.Grid.W)
	if len(out) == len(pos) {
		tensor.MatRowsBiasAddRows(out, pp.Patches, rt.vision.patchW, rt.vision.patchB, pos, rt.cfg.VisionConfig.HiddenSize, len(pp.Patches[0]))
		return out
	}
	tensor.MatRowsBias(out, pp.Patches, rt.vision.patchW, rt.vision.patchB, rt.cfg.VisionConfig.HiddenSize, len(pp.Patches[0]))
	for i, row := range out {
		tensor.AddInPlace(row, pos[i%len(pos)])
	}
	return out
}

func (rt *Runtime) interpolateVisionPos(h, w int) [][]float32 {
	d := rt.cfg.VisionConfig.HiddenSize
	const base = 27
	if h == base && w == base {
		rt.visionPosMu.RLock()
		out := rt.vision.basePosRows
		rt.visionPosMu.RUnlock()
		if out != nil {
			return out
		}
		rt.visionPosMu.Lock()
		if rt.vision.basePosRows == nil {
			rt.vision.basePosRows = makeRowsView(rt.vision.pos, h*w, d)
		}
		out = rt.vision.basePosRows
		rt.visionPosMu.Unlock()
		return out
	}
	key := [2]int{h, w}
	rt.visionPosMu.RLock()
	if rt.visionPosCache != nil {
		if cached := rt.visionPosCache[key]; cached != nil {
			rt.visionPosMu.RUnlock()
			return cached
		}
	}
	rt.visionPosMu.RUnlock()

	src := rt.vision.pos
	out := makeRows(h*w, d)
	for y := 0; y < h; y++ {
		fy := float64(y) * float64(base-1) / float64(max(h-1, 1))
		y0 := int(fy)
		y1 := min(y0+1, base-1)
		wy := float32(fy - float64(y0))
		for x := 0; x < w; x++ {
			fx := float64(x) * float64(base-1) / float64(max(w-1, 1))
			x0 := int(fx)
			x1 := min(x0+1, base-1)
			wx := float32(fx - float64(x0))
			row := out[y*w+x]
			a := src[(y0*base+x0)*d : (y0*base+x0+1)*d]
			b := src[(y0*base+x1)*d : (y0*base+x1+1)*d]
			c := src[(y1*base+x0)*d : (y1*base+x0+1)*d]
			e := src[(y1*base+x1)*d : (y1*base+x1+1)*d]
			for i := 0; i < d; i++ {
				top := a[i]*(1-wx) + b[i]*wx
				bot := c[i]*(1-wx) + e[i]*wx
				row[i] = top*(1-wy) + bot*wy
			}
			out[y*w+x] = row
		}
	}
	rt.visionPosMu.Lock()
	if rt.visionPosCache == nil {
		rt.visionPosCache = make(map[[2]int][][]float32, maxVisionShapeCacheEntries)
	}
	if cached := rt.visionPosCache[key]; cached != nil {
		rt.visionPosMu.Unlock()
		return cached
	}
	if len(rt.visionPosCache) >= maxVisionShapeCacheEntries {
		rt.visionPosCache = make(map[[2]int][][]float32, maxVisionShapeCacheEntries)
	}
	rt.visionPosCache[key] = out
	rt.visionPosMu.Unlock()
	return out
}

func (rt *Runtime) visionLayer(x [][]float32, lw visionLayerWeights, next *visionLayerWeights, normReady bool, grid vision.Grid, rope visionRoPETables, scratch *visionScratch) [][]float32 {
	d := rt.cfg.VisionConfig.HiddenSize
	norm := scratch.norm
	eps := float32(rt.cfg.VisionConfig.LayerNormEps)
	if !normReady {
		tensor.LayerNormRows(norm, x, lw.ln1w, lw.ln1b, eps)
	}
	att := rt.visionAttention(norm, lw, grid, rope, scratch)
	tensor.AddThenLayerNormRows(norm, x, att, lw.ln2w, lw.ln2b, eps)
	mlp := scratch.mlp
	hids := scratch.hids
	tensor.MatRowsBias(hids, norm, lw.fc1w, lw.fc1b, rt.cfg.VisionConfig.IntermediateSize, d)
	tensor.GELUTanhRowsInPlace(hids)
	tensor.MatRowsBias(mlp, hids, lw.fc2w, lw.fc2b, d, rt.cfg.VisionConfig.IntermediateSize)
	if next != nil {
		tensor.AddThenLayerNormRows(norm, x, mlp, next.ln1w, next.ln1b, eps)
	} else {
		tensor.AddThenLayerNormRows(x, x, mlp, rt.vision.postNormW, rt.vision.postNormB, eps)
	}
	return x
}

func (rt *Runtime) visionAttention(x [][]float32, lw visionLayerWeights, grid vision.Grid, rope visionRoPETables, scratch *visionScratch) [][]float32 {
	n := len(x)
	d := rt.cfg.VisionConfig.HiddenSize
	heads := rt.cfg.VisionConfig.NumAttentionHeads
	hd := d / heads
	q, k, v := scratch.q, scratch.k, scratch.v
	tensor.MatRowsBias3(q, k, v, x, lw.qw, lw.qb, lw.kw, lw.kb, lw.vw, lw.vb, d, d, d, d)
	applyVisionRoPEPair(q, k, grid, heads, hd, rope)
	headOut := scratch.headOut
	scale := invSqrt(hd)
	scores := scratch.scores[:n]
	for h := 0; h < heads; h++ {
		for i := 0; i < n; i++ {
			qi := q[i][h*hd : (h+1)*hd]
			visionAttentionScores(scores, qi, k, h*hd, hd, scale)
			tensor.SoftmaxInPlace(scores)
			dst := headOut[i][h*hd : (h+1)*hd]
			weightedValueSum(dst, v, h*hd, hd, scores)
		}
	}
	out := scratch.attOut
	tensor.MatRowsBias(out, headOut, lw.ow, lw.ob, d, d)
	return out
}

func visionAttentionScores(scores, q []float32, rows [][]float32, offset, dim int, scale float32) {
	if dim == 128 {
		i := 0
		for ; i+3 < len(scores); i += 4 {
			s0, s1, s2, s3 := dotAt128QuadRows(q, rows[i], rows[i+1], rows[i+2], rows[i+3], offset)
			scores[i] = s0 * scale
			scores[i+1] = s1 * scale
			scores[i+2] = s2 * scale
			scores[i+3] = s3 * scale
		}
		for ; i+1 < len(scores); i += 2 {
			s0, s1 := dotAt128PairRows(q, rows[i], rows[i+1], offset)
			scores[i] = s0 * scale
			scores[i+1] = s1 * scale
		}
		for ; i < len(scores); i++ {
			scores[i] = dotAt128(q, rows[i], offset) * scale
		}
		return
	}
	if dim == 64 {
		i := 0
		for ; i+3 < len(scores); i += 4 {
			s0, s1, s2, s3 := dotAt64QuadRows(q, rows[i], rows[i+1], rows[i+2], rows[i+3], offset)
			scores[i] = s0 * scale
			scores[i+1] = s1 * scale
			scores[i+2] = s2 * scale
			scores[i+3] = s3 * scale
		}
		for ; i+1 < len(scores); i += 2 {
			s0, s1 := dotAt64PairRows(q, rows[i], rows[i+1], offset)
			scores[i] = s0 * scale
			scores[i+1] = s1 * scale
		}
		for ; i < len(scores); i++ {
			scores[i] = dotAt64(q, rows[i], offset) * scale
		}
		return
	}
	for i := range scores {
		scores[i] = dotAt(q, rows[i], offset, dim) * scale
	}
}

func dotAt128QuadRows(a, row0, row1, row2, row3 []float32, offset int) (float32, float32, float32, float32) {
	return tensor.DotQuad(
		row0[offset:offset+128],
		row1[offset:offset+128],
		row2[offset:offset+128],
		row3[offset:offset+128],
		a,
	)
}

func dotAt64QuadRows(a, row0, row1, row2, row3 []float32, offset int) (float32, float32, float32, float32) {
	return tensor.DotQuad(
		row0[offset:offset+64],
		row1[offset:offset+64],
		row2[offset:offset+64],
		row3[offset:offset+64],
		a,
	)
}

func dotAt128PairRows(a, row0, row1 []float32, offset int) (float32, float32) {
	return tensor.DotPair(
		row0[offset:offset+128],
		row1[offset:offset+128],
		a,
	)
}

func dotAt64PairRows(a, row0, row1 []float32, offset int) (float32, float32) {
	return tensor.DotPair(
		row0[offset:offset+64],
		row1[offset:offset+64],
		a,
	)
}

func newVisionRoPETables(grid vision.Grid, hd int) visionRoPETables {
	half := hd / 2
	if half == 0 {
		return visionRoPETables{
			h: make([][]ropePair, grid.H),
			w: make([][]ropePair, grid.W),
		}
	}
	rows := make([][]ropePair, grid.H+grid.W)
	h := rows[:grid.H]
	w := rows[grid.H:]
	data := make([]ropePair, (grid.H+grid.W)*(half/2))
	fillAxisRoPETable(h, data[:grid.H*(half/2)], half)
	fillAxisRoPETable(w, data[grid.H*(half/2):], half)
	// Pre-build deinterleaved cos/sin tables for SIMD RoPE
	hCos, hSin := deinterleaveRoPETable(h)
	wCos, wSin := deinterleaveRoPETable(w)
	return visionRoPETables{
		h:    h,
		w:    w,
		hCos: hCos,
		hSin: hSin,
		wCos: wCos,
		wSin: wSin,
	}
}

func deinterleaveRoPETable(table [][]ropePair) (cos, sin [][]float32) {
	cos = make([][]float32, len(table))
	sin = make([][]float32, len(table))
	for i, row := range table {
		n := len(row)
		c := make([]float32, n)
		s := make([]float32, n)
		for j, p := range row {
			c[j] = p.cos
			s[j] = p.sin
		}
		cos[i] = c
		sin[i] = s
	}
	return cos, sin
}

func (rt *Runtime) cachedVisionRoPETables(grid vision.Grid, hd int) visionRoPETables {
	key := [3]int{grid.H, grid.W, hd}
	rt.visionRoPEMu.RLock()
	if rt.visionRoPECache != nil {
		if cached := rt.visionRoPECache[key]; cached.h != nil || cached.w != nil {
			rt.visionRoPEMu.RUnlock()
			return cached
		}
	}
	rt.visionRoPEMu.RUnlock()

	tables := newVisionRoPETables(grid, hd)
	rt.visionRoPEMu.Lock()
	if rt.visionRoPECache == nil {
		rt.visionRoPECache = make(map[[3]int]visionRoPETables, maxVisionShapeCacheEntries)
	}
	if cached := rt.visionRoPECache[key]; cached.h != nil || cached.w != nil {
		rt.visionRoPEMu.Unlock()
		return cached
	}
	if len(rt.visionRoPECache) >= maxVisionShapeCacheEntries {
		rt.visionRoPECache = make(map[[3]int]visionRoPETables, maxVisionShapeCacheEntries)
	}
	rt.visionRoPECache[key] = tables
	rt.visionRoPEMu.Unlock()
	return tables
}

func applyVisionRoPE(x [][]float32, grid vision.Grid, heads, hd int, rope visionRoPETables) {
	half := hd / 2
	hy, wx := 0, 0
	for _, row := range x {
		for h := 0; h < heads; h++ {
			base := h * hd
			applyAxisRoPEWithTable(row[base:base+half], rope.h[hy])
			applyAxisRoPEWithTable(row[base+half:base+hd], rope.w[wx])
		}
		wx++
		if wx == grid.W {
			wx = 0
			hy++
			if hy == grid.H {
				hy = 0
			}
		}
	}
}

func applyVisionRoPEPair(q, k [][]float32, grid vision.Grid, heads, hd int, rope visionRoPETables) {
	half := hd / 2
	hy, wx := 0, 0
	for idx, qr := range q {
		kr := k[idx]
		cosH, sinH := rope.hCos[hy], rope.hSin[hy]
		cosW, sinW := rope.wCos[wx], rope.wSin[wx]
		for h := 0; h < heads; h++ {
			base := h * hd
			tensor.RoPEPairAxis(qr, kr, base, half, cosH, sinH)
			tensor.RoPEPairAxis(qr, kr, base+half, half, cosW, sinW)
		}
		wx++
		if wx == grid.W {
			wx = 0
			hy++
			if hy == grid.H {
				hy = 0
			}
		}
	}
}

func applyAxisRoPEPairWithTableAt(q, k []float32, start, axisLen int, table []ropePair) {
	half := axisLen / 2
	if half <= 0 {
		return
	}
	// For small half (typical vision: half=32), the deinterleave overhead
	// exceeds SIMD gains. Use direct struct access instead.
	other := start + half
	q0 := q[start:other]
	q1 := q[other : other+half]
	k0 := k[start:other]
	k1 := k[other : other+half]
	for i := 0; i < half; i++ {
		cs, sn := table[i].cos, table[i].sin
		qa, qb := q0[i], q1[i]
		ka, kb := k0[i], k1[i]
		q0[i] = qa*cs - qb*sn
		q1[i] = qb*cs + qa*sn
		k0[i] = ka*cs - kb*sn
		k1[i] = kb*cs + ka*sn
	}
}

type ropePair struct {
	cos float32
	sin float32
}

var ropePairScratchBufs struct {
	cos [128]float32
	sin [128]float32
}

// ropePairScratch deinterleaves a []ropePair into separate cos and sin float32 slices.
// Uses a fixed-size thread-local buffer (safe for single-threaded vision encoding).
func ropePairScratch(table []ropePair) ([]float32, []float32) {
	n := len(table)
	if n > len(ropePairScratchBufs.cos) {
		cos := make([]float32, n)
		sin := make([]float32, n)
		for i, p := range table {
			cos[i] = p.cos
			sin[i] = p.sin
		}
		return cos, sin
	}
	cos := ropePairScratchBufs.cos[:n]
	sin := ropePairScratchBufs.sin[:n]
	for i, p := range table {
		cos[i] = p.cos
		sin[i] = p.sin
	}
	return cos, sin
}

func fillAxisRoPETable(table [][]ropePair, data []ropePair, dim int) {
	half := dim / 2
	if len(table) == 0 || half == 0 {
		return
	}
	var smallFreq [128]float64
	freqs := smallFreq[:half]
	if half > len(smallFreq) {
		freqs = make([]float64, half)
	}
	for i := 0; i < half; i++ {
		freqs[i] = pow(10000, -float64(2*i)/float64(dim))
	}
	for pos := 0; pos < len(table); pos++ {
		row := data[pos*half : (pos+1)*half]
		for i := 0; i < half; i++ {
			ang := float64(pos) * freqs[i]
			row[i] = ropePair{cos: cos(ang), sin: sin(ang)}
		}
		table[pos] = row
	}
}

func applyAxisRoPEWithTable(x []float32, table []ropePair) {
	half := len(x) / 2
	for i := 0; i < half; i++ {
		cs, sn := table[i].cos, table[i].sin
		a, b := x[i], x[half+i]
		x[i] = a*cs - b*sn
		x[half+i] = b*cs + a*sn
	}
}

func applyAxisRoPEWithTableAt(x []float32, start, axisLen int, table []ropePair) {
	half := axisLen / 2
	other := start + half
	i := 0
	for ; i+1 < half; i += 2 {
		cs, sn := table[i].cos, table[i].sin
		a, b := x[start+i], x[other+i]
		x[start+i] = a*cs - b*sn
		x[other+i] = b*cs + a*sn
		cs, sn = table[i+1].cos, table[i+1].sin
		a, b = x[start+i+1], x[other+i+1]
		x[start+i+1] = a*cs - b*sn
		x[other+i+1] = b*cs + a*sn
	}
	for ; i < half; i++ {
		cs, sn := table[i].cos, table[i].sin
		a, b := x[start+i], x[other+i]
		x[start+i] = a*cs - b*sn
		x[other+i] = b*cs + a*sn
	}
}

func weightedValueSum(dst []float32, rows [][]float32, offset, dim int, weights []float32) {
	if len(weights) == 0 {
		clear(dst[:dim])
		return
	}
	a := weights[0]
	x := rows[0][offset : offset+dim]
	if len(weights) == 1 {
		if a == 1 {
			copy(dst[:dim], x)
			return
		}
		tensor.ScaleCopy(dst[:dim], x, a)
		return
	}
	if len(weights) == 2 {
		a1 := weights[1]
		x1 := rows[1][offset : offset+dim]
		if dim == 128 {
			weightedValueSum2_128(dst, x, x1, a, a1)
			return
		}
		if dim == 64 {
			weightedValueSum2_64(dst, x, x1, a, a1)
			return
		}
		tensor.WeightedSum2(dst[:dim], x, x1, a, a1)
		return
	}
	if len(weights) == 3 {
		a1, a2 := weights[1], weights[2]
		x1 := rows[1][offset : offset+dim]
		x2 := rows[2][offset : offset+dim]
		if dim == 128 {
			weightedValueSum3_128(dst, x, x1, x2, a, a1, a2)
			return
		}
		if dim == 64 {
			weightedValueSum3_64(dst, x, x1, x2, a, a1, a2)
			return
		}
		tensor.WeightedSum3(dst[:dim], x, x1, x2, a, a1, a2)
		return
	}
	if len(weights) == 4 {
		a1, a2, a3 := weights[1], weights[2], weights[3]
		x1 := rows[1][offset : offset+dim]
		x2 := rows[2][offset : offset+dim]
		x3 := rows[3][offset : offset+dim]
		if dim == 128 {
			weightedValueSum4_128(dst, x, x1, x2, x3, a, a1, a2, a3)
			return
		}
		if dim == 64 {
			weightedValueSum4_64(dst, x, x1, x2, x3, a, a1, a2, a3)
			return
		}
		tensor.WeightedSum4(dst[:dim], x, x1, x2, x3, a, a1, a2, a3)
		return
	}
	if dim == 64 {
		weightedValueSum64(dst, rows, offset, weights)
		return
	}
	if dim == 128 {
		weightedValueSum128(dst, rows, offset, weights)
		return
	}
	tensor.ScaleCopy(dst[:dim], x, a)
	r := 1
	for ; r+3 < len(weights); r += 4 {
		a0, a1, a2, a3 := weights[r], weights[r+1], weights[r+2], weights[r+3]
		x0 := rows[r][offset : offset+dim]
		x1 := rows[r+1][offset : offset+dim]
		x2 := rows[r+2][offset : offset+dim]
		x3 := rows[r+3][offset : offset+dim]
		tensor.WeightedSumAdd4(dst[:dim], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; r+1 < len(weights); r += 2 {
		a0, a1 := weights[r], weights[r+1]
		x0 := rows[r][offset : offset+dim]
		x1 := rows[r+1][offset : offset+dim]
		tensor.WeightedSumAdd2(dst[:dim], x0, x1, a0, a1)
	}
	for ; r < len(weights); r++ {
		a := weights[r]
		x := rows[r][offset : offset+dim]
		tensor.ScaleAdd(dst[:dim], x, a)
	}
}

func weightedValueSum128(dst []float32, rows [][]float32, offset int, weights []float32) {
	a := weights[0]
	x := rows[0][offset : offset+128]
	tensor.ScaleCopy(dst[:128], x, a)
	r := 1
	for ; r+3 < len(weights); r += 4 {
		a0, a1, a2, a3 := weights[r], weights[r+1], weights[r+2], weights[r+3]
		x0 := rows[r][offset : offset+128]
		x1 := rows[r+1][offset : offset+128]
		x2 := rows[r+2][offset : offset+128]
		x3 := rows[r+3][offset : offset+128]
		tensor.WeightedSumAdd4(dst[:128], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; r+1 < len(weights); r += 2 {
		a0, a1 := weights[r], weights[r+1]
		x0 := rows[r][offset : offset+128]
		x1 := rows[r+1][offset : offset+128]
		tensor.WeightedSumAdd2(dst[:128], x0, x1, a0, a1)
	}
	for ; r < len(weights); r++ {
		a := weights[r]
		x := rows[r][offset : offset+128]
		tensor.ScaleAdd(dst[:128], x, a)
	}
}

func weightedValueSum64(dst []float32, rows [][]float32, offset int, weights []float32) {
	a := weights[0]
	x := rows[0][offset : offset+64]
	tensor.ScaleCopy(dst[:64], x, a)
	r := 1
	for ; r+3 < len(weights); r += 4 {
		a0, a1, a2, a3 := weights[r], weights[r+1], weights[r+2], weights[r+3]
		x0 := rows[r][offset : offset+64]
		x1 := rows[r+1][offset : offset+64]
		x2 := rows[r+2][offset : offset+64]
		x3 := rows[r+3][offset : offset+64]
		tensor.WeightedSumAdd4(dst[:64], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; r+1 < len(weights); r += 2 {
		a0, a1 := weights[r], weights[r+1]
		x0 := rows[r][offset : offset+64]
		x1 := rows[r+1][offset : offset+64]
		tensor.WeightedSumAdd2(dst[:64], x0, x1, a0, a1)
	}
	for ; r < len(weights); r++ {
		a := weights[r]
		x := rows[r][offset : offset+64]
		tensor.ScaleAdd(dst[:64], x, a)
	}
}

func (rt *Runtime) projectImage(x [][]float32, grid vision.Grid) [][]float32 {
	vd := rt.cfg.VisionConfig.HiddenSize
	td := rt.cfg.HiddenSize
	rows := (grid.H / 2) * (grid.W / 2) * grid.T
	out := makeRows(rows, td)
	if rows == 0 {
		return out
	}
	work := rows * vd * td
	gomax := runtime.GOMAXPROCS(0)
	if work >= 1<<18 && rows >= 4 && gomax > 1 {
		workers := min(gomax, rows)
		if vd*td < 1<<20 {
			workers = min(workers, 8)
		}
		if workers <= 1 {
			rt.projectImageRows(out, x, grid, 0, rows)
			return out
		}
		var wg sync.WaitGroup
		for worker := 1; worker < workers; worker++ {
			start := worker * rows / workers
			end := (worker + 1) * rows / workers
			wg.Add(1)
			go func() {
				rt.projectImageRows(out, x, grid, start, end)
				wg.Done()
			}()
		}
		rt.projectImageRows(out, x, grid, 0, rows/workers)
		wg.Wait()
		return out
	}
	rt.projectImageRows(out, x, grid, 0, rows)
	return out
}

func (rt *Runtime) projectImageRows(out, x [][]float32, grid vision.Grid, start, end int) {
	vd := rt.cfg.VisionConfig.HiddenSize
	td := rt.cfg.HiddenSize
	blocksW := grid.W / 2
	blocksPerT := (grid.H / 2) * blocksW
	scratch, scratchPtr := rt.getProjectScratch(vd * 8)
	merged := scratch[:vd*4]
	hid := scratch[vd*4:]
	blocksH := grid.H / 2
	t := start / blocksPerT
	local := start - t*blocksPerT
	by := local / blocksW
	bx := local - by*blocksW
	base := t*grid.H*grid.W + by*2*grid.W + bx*2
	for row := start; row < end; row++ {
		tensor.LayerNorm(merged[:vd], x[base], rt.vision.projNormW, rt.vision.projNormB, 1e-5)
		tensor.LayerNorm(merged[vd:2*vd], x[base+1], rt.vision.projNormW, rt.vision.projNormB, 1e-5)
		tensor.LayerNorm(merged[2*vd:3*vd], x[base+grid.W], rt.vision.projNormW, rt.vision.projNormB, 1e-5)
		tensor.LayerNorm(merged[3*vd:4*vd], x[base+grid.W+1], rt.vision.projNormW, rt.vision.projNormB, 1e-5)
		tensor.MatVecBiasSerial(hid, merged, rt.vision.proj1W, rt.vision.proj1B, vd*4, vd*4)
		tensor.GELUTanhInPlace(hid)
		tensor.MatVecBiasSerial(out[row], hid, rt.vision.proj2W, rt.vision.proj2B, td, vd*4)
		bx++
		base += 2
		if bx == blocksW {
			bx = 0
			base += grid.W
			by++
			if by == blocksH {
				by = 0
			}
		}
	}
	rt.putProjectScratch(scratch, scratchPtr)
}

func (rt *Runtime) getProjectScratch(n int) ([]float32, *[]float32) {
	if n <= 0 {
		return nil, nil
	}
	if v := rt.projectScratchPool.Get(); v != nil {
		p := v.(*[]float32)
		if cap(*p) >= n {
			return (*p)[:n], p
		}
	}
	buf := make([]float32, n)
	return buf, &buf
}

func (rt *Runtime) putProjectScratch(buf []float32, p *[]float32) {
	const maxProjectScratchFloats = 1 << 20
	if p == nil || cap(buf) == 0 || cap(buf) > maxProjectScratchFloats {
		return
	}
	*p = buf[:0]
	rt.projectScratchPool.Put(p)
}

func (rt *Runtime) layerNormRows(x [][]float32, w, b []float32, eps float32) {
	tensor.LayerNormRows(x, x, w, b, eps)
}

func makeRows(rows, cols int) [][]float32 {
	if rows == 0 || cols == 0 {
		return make([][]float32, rows)
	}
	data := make([]float32, rows*cols)
	return makeRowsView(data, rows, cols)
}

func makeRowsView(data []float32, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	makeRowsViewInto(out, data, cols)
	return out
}

func makeRowsViewInto(out [][]float32, data []float32, cols int) {
	for i := range out {
		out[i] = data[i*cols : (i+1)*cols]
	}
}
