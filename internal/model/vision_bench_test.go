package model

import (
	"runtime"
	"testing"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/tensor"
	"paddleocrvl-go/internal/vision"
)

func BenchmarkProjectImage(b *testing.B) {
	vd, td := 64, 128
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkProjectImageShape(b, vd, td, grid)
}

func BenchmarkProjectImageLarge(b *testing.B) {
	vd, td := 1024, 4096
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkProjectImageShape(b, vd, td, grid)
}

func BenchmarkProjectImageSingleProc(b *testing.B) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	vd, td := 64, 128
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkProjectImageShape(b, vd, td, grid)
}

func BenchmarkProjectImageLargeSingleProc(b *testing.B) {
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)
	vd, td := 1024, 4096
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkProjectImageShape(b, vd, td, grid)
}

func benchmarkProjectImageShape(b *testing.B, vd, td int, grid vision.Grid) {
	tokens := grid.T * grid.H * grid.W
	rt := newProjectImageBenchRuntime(vd, td)
	x := makeRows(tokens, vd)
	for i := range x {
		fillBenchFloat32(x[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.projectImage(x, grid)
	}
}

func BenchmarkVisionEmbeddings(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 16, vision.Grid{T: 1, H: 14, W: 14})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddings(pp)
	}
}

func BenchmarkVisionEmbeddingsInto(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 16, vision.Grid{T: 1, H: 14, W: 14})
	out := makeRows(len(pp.Patches), rt.cfg.VisionConfig.HiddenSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddingsInto(out, pp)
	}
}

func BenchmarkVisionEmbeddingsIntoRepeatedPos(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 16, vision.Grid{T: 2, H: 14, W: 14})
	out := makeRows(len(pp.Patches), rt.cfg.VisionConfig.HiddenSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddingsInto(out, pp)
	}
}

func BenchmarkVisionEmbeddingsRepeatedPos(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 16, vision.Grid{T: 2, H: 14, W: 14})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddings(pp)
	}
}

func BenchmarkVisionEmbeddingsPatch14(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 14*14*3, vision.Grid{T: 1, H: 14, W: 14})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddings(pp)
	}
}

func BenchmarkVisionEmbeddingsIntoPatch14(b *testing.B) {
	rt, pp := newVisionEmbeddingsBenchCase(1024, 14*14*3, vision.Grid{T: 1, H: 14, W: 14})
	out := makeRows(len(pp.Patches), rt.cfg.VisionConfig.HiddenSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.visionEmbeddingsInto(out, pp)
	}
}

func newVisionEmbeddingsBenchCase(vd, patch int, grid vision.Grid) (*Runtime, *vision.Preprocessed) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: vd}}}
	rt.vision.patchW = make([]float32, vd*patch)
	rt.vision.patchB = make([]float32, vd)
	rt.vision.pos = make([]float32, 27*27*vd)
	fillBenchFloat32(rt.vision.patchW)
	fillBenchFloat32(rt.vision.pos)
	pp := &vision.Preprocessed{Grid: grid, Patches: makeRows(grid.T*grid.H*grid.W, patch)}
	for i := range pp.Patches {
		fillBenchFloat32(pp.Patches[i])
	}
	return rt, pp
}

func BenchmarkEncodePreprocessedImageNoLayers(b *testing.B) {
	vd, td, patch := 1024, 4096, 16
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkEncodePreprocessedImageNoLayers(b, vd, td, patch, grid)
}

func BenchmarkEncodePreprocessedImageNoLayersPatch14(b *testing.B) {
	vd, td, patch := 1024, 4096, 14*14*3
	grid := vision.Grid{T: 1, H: 14, W: 14}
	benchmarkEncodePreprocessedImageNoLayers(b, vd, td, patch, grid)
}

func benchmarkEncodePreprocessedImageNoLayers(b *testing.B, vd, td, patch int, grid vision.Grid) {
	rt := newProjectImageBenchRuntime(vd, td)
	rt.cfg.VisionConfig.PatchSize = patch
	rt.cfg.VisionConfig.NumAttentionHeads = 8
	rt.cfg.VisionConfig.IntermediateSize = 4096
	rt.cfg.VisionConfig.LayerNormEps = 1e-5
	rt.vision.patchW = make([]float32, vd*patch)
	rt.vision.patchB = make([]float32, vd)
	rt.vision.pos = make([]float32, 27*27*vd)
	rt.vision.postNormW = make([]float32, vd)
	rt.vision.postNormB = make([]float32, vd)
	for i := range rt.vision.postNormW {
		rt.vision.postNormW[i] = 1
	}
	fillBenchFloat32(rt.vision.patchW)
	fillBenchFloat32(rt.vision.pos)
	pp := &vision.Preprocessed{Grid: grid, Patches: makeRows(grid.T*grid.H*grid.W, patch)}
	for i := range pp.Patches {
		fillBenchFloat32(pp.Patches[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := rt.encodePreprocessedImage(pp); err != nil {
			b.Fatal(err)
		}
	}
}

func newProjectImageBenchRuntime(vd, td int) *Runtime {
	rt := &Runtime{cfg: &config.Config{HiddenSize: td, VisionConfig: config.Vision{HiddenSize: vd}}}
	rt.vision.projNormW = make([]float32, vd)
	rt.vision.projNormB = make([]float32, vd)
	rt.vision.proj1W = make([]float32, vd*4*vd*4)
	rt.vision.proj1B = make([]float32, vd*4)
	rt.vision.proj2W = make([]float32, td*vd*4)
	rt.vision.proj2B = make([]float32, td)
	for i := range rt.vision.projNormW {
		rt.vision.projNormW[i] = 1
	}
	fillBenchFloat32(rt.vision.proj1W)
	fillBenchFloat32(rt.vision.proj2W)
	rt.vision.proj1Q8 = tensor.QuantizeQ8Row(rt.vision.proj1W, vd*4, vd*4)
	rt.vision.proj2Q8 = tensor.QuantizeQ8Row(rt.vision.proj2W, td, vd*4)
	return rt
}

func BenchmarkVisionLayerChain(b *testing.B) {
	rt := newVisionLayerTestRuntime()
	grid := vision.Grid{T: 1, H: 8, W: 8}
	rope := newVisionRoPETables(grid, rt.cfg.VisionConfig.HiddenSize/rt.cfg.VisionConfig.NumAttentionHeads)
	src := makeRowsForModelTest(grid.H*grid.W, rt.cfg.VisionConfig.HiddenSize)
	for i := range src {
		fillBenchFloat32(src[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := cloneRowsForModelTest(src)
		scratch := rt.newVisionScratch(len(x))
		normReady := false
		for j := range rt.vision.layers {
			var next *visionLayerWeights
			if j+1 < len(rt.vision.layers) {
				next = &rt.vision.layers[j+1]
			}
			x = rt.visionLayer(x, rt.vision.layers[j], next, normReady, grid, rope, scratch)
			normReady = next != nil
		}
	}
}

func BenchmarkVisionLayerChainReuse(b *testing.B) {
	rt := newVisionLayerTestRuntime()
	grid := vision.Grid{T: 1, H: 8, W: 8}
	rope := newVisionRoPETables(grid, rt.cfg.VisionConfig.HiddenSize/rt.cfg.VisionConfig.NumAttentionHeads)
	src := makeRowsForModelTest(grid.H*grid.W, rt.cfg.VisionConfig.HiddenSize)
	x := makeRowsForModelTest(grid.H*grid.W, rt.cfg.VisionConfig.HiddenSize)
	for i := range src {
		fillBenchFloat32(src[i])
	}
	scratch := rt.newVisionScratch(len(x))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyRowsForVisionBench(x, src)
		normReady := false
		for j := range rt.vision.layers {
			var next *visionLayerWeights
			if j+1 < len(rt.vision.layers) {
				next = &rt.vision.layers[j+1]
			}
			x = rt.visionLayer(x, rt.vision.layers[j], next, normReady, grid, rope, scratch)
			normReady = next != nil
		}
	}
}

func copyRowsForVisionBench(dst, src [][]float32) {
	for i := range src {
		copy(dst[i], src[i])
	}
}

func BenchmarkNewVisionRoPETables(b *testing.B) {
	grid := vision.Grid{T: 1, H: 14, W: 14}
	for i := 0; i < b.N; i++ {
		_ = newVisionRoPETables(grid, 128)
	}
}

func BenchmarkApplyVisionRoPEPair(b *testing.B) {
	grid := vision.Grid{T: 1, H: 14, W: 14}
	heads, hd := 8, 128
	width := heads * hd
	q := make([][]float32, grid.H*grid.W)
	k := make([][]float32, grid.H*grid.W)
	for i := range q {
		q[i] = make([]float32, width)
		k[i] = make([]float32, width)
		fillBenchFloat32(q[i])
		fillBenchFloat32(k[i])
	}
	rope := newVisionRoPETables(grid, hd)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyVisionRoPEPair(q, k, grid, heads, hd, rope)
	}
}

func BenchmarkInterpolateVisionPosBase(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 1024}}}
	rt.vision.pos = make([]float32, 27*27*1024)
	fillBenchFloat32(rt.vision.pos)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.interpolateVisionPos(27, 27)
	}
}

func BenchmarkInterpolateVisionPosScaled(b *testing.B) {
	rt := &Runtime{cfg: &config.Config{VisionConfig: config.Vision{HiddenSize: 1024}}}
	rt.vision.pos = make([]float32, 27*27*1024)
	fillBenchFloat32(rt.vision.pos)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.interpolateVisionPos(14, 14)
	}
}

func fillBenchFloat32(x []float32) {
	for i := range x {
		x[i] = float32(i%17-8) / 17
	}
}

func BenchmarkVisionLayerChainLarge(b *testing.B) {
	vd := 1024
	rt := newVisionLayerTestRuntime()
	rt.cfg.VisionConfig.HiddenSize = vd
	rt.cfg.VisionConfig.IntermediateSize = 4096
	rt.cfg.VisionConfig.NumAttentionHeads = 8
	d := vd
	inter := rt.cfg.VisionConfig.IntermediateSize
	rt.vision.layers = make([]visionLayerWeights, 2)
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
		rt.vision.layers[i].qQ8 = tensor.QuantizeQ8Row(rt.vision.layers[i].qw, d, d)
		rt.vision.layers[i].kQ8 = tensor.QuantizeQ8Row(rt.vision.layers[i].kw, d, d)
		rt.vision.layers[i].vQ8 = tensor.QuantizeQ8Row(rt.vision.layers[i].vw, d, d)
		rt.vision.layers[i].oQ8 = tensor.QuantizeQ8Row(rt.vision.layers[i].ow, d, d)
		rt.vision.layers[i].fc1Q8 = tensor.QuantizeQ8Row(rt.vision.layers[i].fc1w, inter, d)
		rt.vision.layers[i].fc2Q8 = tensor.QuantizeQ8Row(rt.vision.layers[i].fc2w, d, inter)
	}
	grid := vision.Grid{T: 1, H: 14, W: 14}
	rope := newVisionRoPETables(grid, vd/rt.cfg.VisionConfig.NumAttentionHeads)
	src := makeRowsForModelTest(grid.H*grid.W, vd)
	for i := range src {
		fillBenchFloat32(src[i])
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		x := cloneRowsForModelTest(src)
		scratch := rt.newVisionScratch(len(x))
		for j := range rt.vision.layers {
			x = rt.visionLayer(x, rt.vision.layers[j], nil, false, grid, rope, scratch)
		}
	}
}
