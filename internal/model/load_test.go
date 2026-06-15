package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/gguf"
	"paddleocrvl-go/internal/tensor"
)

func TestOpenOrConvertGGUFAutoPrefersQ4ThenQ6ThenQ8(t *testing.T) {
	dir := t.TempDir()
	writeTinyConfig(t, dir)
	values := []float32{1, -2, 3, 0.5, 0.25, -0.75}
	src := filepath.Join(dir, "model.safetensors")
	writeTinySafetensors(t, src, "lm_head.weight", []int64{2, 3}, values)
	if err := gguf.ConvertSafetensorsWithOptions(src, filepath.Join(dir, "model-q8.gguf"), dir, gguf.ConvertOptions{Quantization: "q8"}); err != nil {
		t.Fatal(err)
	}
	store, quant, path, source, err := openOrConvertGGUF(dir, "auto", nil)
	if err != nil {
		t.Fatal(err)
	}
	store.Close()
	if quant != "q8" {
		t.Fatalf("auto quant=%q want q8", quant)
	}
	if filepath.Base(path) != "model-q8.gguf" {
		t.Fatalf("path=%q", path)
	}
	if source != "existing_gguf" {
		t.Fatalf("source=%q", source)
	}
	if err := gguf.ConvertSafetensorsWithOptions(src, filepath.Join(dir, "model-q6.gguf"), dir, gguf.ConvertOptions{Quantization: "q6"}); err != nil {
		t.Fatal(err)
	}
	store, quant, path, source, err = openOrConvertGGUF(dir, "auto", nil)
	if err != nil {
		t.Fatal(err)
	}
	store.Close()
	if quant != "q6" {
		t.Fatalf("auto quant=%q want q6", quant)
	}
	if filepath.Base(path) != "model-q6.gguf" {
		t.Fatalf("path=%q", path)
	}
	if source != "existing_gguf" {
		t.Fatalf("source=%q", source)
	}
	if err := gguf.ConvertSafetensorsWithOptions(src, filepath.Join(dir, "model-q4.gguf"), dir, gguf.ConvertOptions{Quantization: "q4"}); err != nil {
		t.Fatal(err)
	}
	store, quant, path, source, err = openOrConvertGGUF(dir, "auto", nil)
	if err != nil {
		t.Fatal(err)
	}
	store.Close()
	if quant != "q4" {
		t.Fatalf("auto quant=%q want q4", quant)
	}
	if filepath.Base(path) != "model-q4.gguf" {
		t.Fatalf("path=%q", path)
	}
	if source != "existing_gguf" {
		t.Fatalf("source=%q", source)
	}
}

func TestAutoQuantModes(t *testing.T) {
	if got := autoQuantBuildTarget("auto-fast"); got != "q4" {
		t.Fatalf("auto-fast target=%q", got)
	}
	if got := autoQuantBuildTarget("auto-quality"); got != "q8" {
		t.Fatalf("auto-quality target=%q", got)
	}
	if got := autoQuantBuildTarget("auto"); got != "q6" {
		t.Fatalf("auto target=%q", got)
	}
	fast := autoQuantCandidates("auto-fast")
	if fast[0].quant != "q4" {
		t.Fatalf("fast candidates=%v", fast)
	}
	quality := autoQuantCandidates("auto-quality")
	if quality[0].quant != "q8" {
		t.Fatalf("quality candidates=%v", quality)
	}
}

func TestOpenOrConvertGGUFExistingBadFileReturnsError(t *testing.T) {
	dir := t.TempDir()
	writeTinyConfig(t, dir)
	if err := os.WriteFile(filepath.Join(dir, "model-q4.gguf"), []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeTinySafetensors(t, filepath.Join(dir, "model.safetensors"), "lm_head.weight", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	store, _, _, _, err := openOrConvertGGUF(dir, "auto", nil)
	if store != nil {
		store.Close()
	}
	if err == nil {
		t.Fatalf("err=%v", err)
	}
}

func TestNormalizeQuantization(t *testing.T) {
	cases := map[string]string{
		"":               "f32",
		"\tQ8\n":         "q8",
		" F32 ":          "f32",
		" Q4 ":           "q4",
		" AUTO-QUALITY ": "auto-quality",
		"Custom-Q":       "custom-q",
	}
	for in, want := range cases {
		if got := NormalizeQuantization(in); got != want {
			t.Fatalf("%q -> %q want %q", in, got, want)
		}
	}
}

func BenchmarkNormalizeQuantizationKnown(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if got := NormalizeQuantization(" AUTO-QUALITY "); got != "auto-quality" {
			b.Fatal(got)
		}
	}
}

func BenchmarkNormalizeQuantizationUnknown(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if got := NormalizeQuantization(" Custom-Q "); got != "custom-q" {
			b.Fatal(got)
		}
	}
}

func TestQuantizeRowsFromGGUFF32Store(t *testing.T) {
	dir := t.TempDir()
	writeTinyConfig(t, dir)
	values := []float32{1, -2, 3, 0.5, 0.25, -0.75}
	src := filepath.Join(dir, "model.safetensors")
	writeTinySafetensors(t, src, "lm_head.weight", []int64{2, 3}, values)
	dst := filepath.Join(dir, "model.gguf")
	if err := gguf.ConvertSafetensorsWithOptions(src, dst, dir, gguf.ConvertOptions{}); err != nil {
		t.Fatal(err)
	}
	gf, err := gguf.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer gf.Close()
	rt := &Runtime{sf: gf, q6w: map[string]*tensor.Q6Matrix{}, quantization: "q6"}
	var rowFloatBuf []float32
	if err := rt.quantizeRowsFromStore("lm_head.weight", gf, &rowFloatBuf, nil); err != nil {
		t.Fatal(err)
	}
	q := rt.q6w["lm_head.weight"]
	if q == nil || q.Rows != 2 || q.Cols != 3 || len(q.Data) != 6 || len(q.Scale) != 2 {
		t.Fatalf("bad q6=%+v", q)
	}
}

type dtypeStoreForTest struct {
	tensorStore
	dtype string
	ok    bool
}

func (s dtypeStoreForTest) DType(string) (string, bool) {
	return s.dtype, s.ok
}

func TestNeedsRawFloatDecodeBuffer(t *testing.T) {
	if needsRawFloatDecodeBuffer(dtypeStoreForTest{dtype: "F32", ok: true}, "x") {
		t.Fatal("F32 should not need raw decode buffer")
	}
	if !needsRawFloatDecodeBuffer(dtypeStoreForTest{dtype: "BF16", ok: true}, "x") {
		t.Fatal("BF16 should need raw decode buffer")
	}
	if !needsRawFloatDecodeBuffer(dtypeStoreForTest{}, "x") {
		t.Fatal("unknown dtype should conservatively need raw decode buffer")
	}
}

func TestProgressLoggerDoesNotPanic(t *testing.T) {
	p := ProgressLogger("test")
	p(0, 2, "x.weight", "LOAD")
	p(2, 2, "", "LOAD")
	p(0, 2, "x.weight", "LOAD-VISION")
	p(2, 2, "", "LOAD-VISION")
	p(0, 2, "x.weight", "F32")
	p(2, 2, "", "")
}

func TestRequestedQuantizationDefault(t *testing.T) {
	rt := &Runtime{}
	if got := rt.RequestedQuantization(); got != "f32" {
		t.Fatalf("got %q", got)
	}
	rt.requestedQuantization = "auto-fast"
	if got := rt.RequestedQuantization(); got != "auto-fast" {
		t.Fatalf("got %q", got)
	}
}

func TestCloseNilRuntime(t *testing.T) {
	var rt *Runtime
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := (&Runtime{}).Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCacheStatsEmptyRuntime(t *testing.T) {
	rt := &Runtime{}
	if got := rt.CacheStats(); got != (CacheStats{}) {
		t.Fatalf("stats=%+v", got)
	}
}

func TestWeightPath(t *testing.T) {
	rt := &Runtime{weightPath: "model-q4.gguf", weightSource: "existing_gguf", loadStats: LoadStats{TotalMS: 7}}
	if got := rt.WeightPath(); got != "model-q4.gguf" {
		t.Fatalf("got %q", got)
	}
	if got := rt.WeightSource(); got != "existing_gguf" {
		t.Fatalf("source=%q", got)
	}
	if got := (&Runtime{}).WeightSource(); got != "unknown" {
		t.Fatalf("empty source=%q", got)
	}
	if got := rt.LoadStats().TotalMS; got != 7 {
		t.Fatalf("load total=%d", got)
	}
}

func TestWeightStatsUsesCachedValueWhenAvailable(t *testing.T) {
	rt := &Runtime{
		w: map[string][]float32{
			"a": make([]float32, 4),
		},
		weightStats: WeightStats{F32Tensors: 9, F32Bytes: 99, TotalBytes: 99},
	}
	rt.w["b"] = make([]float32, 100)
	got := rt.WeightStats()
	if got.F32Tensors != 9 || got.TotalBytes != 99 {
		t.Fatalf("WeightStats=%+v", got)
	}
}

func TestRuntimeVulkanPlansUseConfigShape(t *testing.T) {
	rt := &Runtime{
		quantization: "q6",
		cfg: &config.Config{
			VocabSize:         1000,
			HiddenSize:        128,
			IntermediateSize:  512,
			NumAttentionHeads: 4,
			NumKeyValueHeads:  2,
			HeadDim:           32,
			VisionConfig: config.Vision{
				HiddenSize:       64,
				IntermediateSize: 256,
				PatchSize:        14,
			},
		},
	}
	plans := rt.VulkanPlans()
	if len(plans) != 10 {
		t.Fatalf("plans=%+v", plans)
	}
	if plans[0].Name != "text.qkv" || plans[0].Rows != 256 || plans[0].Cols != 128 || plans[0].Quant != "q6" {
		t.Fatalf("first plan=%+v", plans[0])
	}
	if plans[5].Name != "vision.patch" || plans[5].Cols != 14*14*3 || plans[5].Quant != "f32" {
		t.Fatalf("vision patch plan=%+v", plans[5])
	}
	st := rt.VulkanPlanSummary()
	if st.PlanCount != 10 || st.Dispatches <= 0 || st.WeightBytes <= 0 {
		t.Fatalf("summary=%+v", st)
	}
	if st.TextLayers != 0 || st.VisionLayers != 0 {
		t.Fatalf("empty layer counts should stay omitted-like, got %+v", st)
	}
	graph := rt.VulkanExecutionGraph()
	if len(graph.Stages) != 2 || graph.PlanCount != st.PlanCount || graph.Dispatches != st.Dispatches {
		t.Fatalf("graph=%+v summary=%+v", graph, st)
	}
	pipes := rt.VulkanPipelinePlan()
	if len(pipes) == 0 || len(pipes) != graph.PipelineCount {
		t.Fatalf("pipes=%+v graph=%+v", pipes, graph)
	}
	cmdPlan := rt.VulkanCommandPlan()
	if cmdPlan.CommandCount == 0 || cmdPlan.PipelineCount != len(pipes) || cmdPlan.Dispatches != graph.Dispatches {
		t.Fatalf("command plan=%+v pipes=%+v graph=%+v", cmdPlan, pipes, graph)
	}
}

func TestReleaseCachedTextWeightMapEntriesKeepsCachedSlices(t *testing.T) {
	rt := &Runtime{
		cfg: &configForReleaseTest,
		w:   map[string][]float32{},
	}
	rt.w["model.embed_tokens.weight"] = []float32{1, 2}
	rt.w["model.norm.weight"] = []float32{3, 4}
	rt.w["lm_head.weight"] = []float32{5, 6}
	rt.w["model.layers.0.input_layernorm.weight"] = []float32{7, 8}
	rt.w["model.layers.0.post_attention_layernorm.weight"] = []float32{9, 10}
	rt.w["model.layers.0.self_attn.q_proj.weight"] = []float32{11}
	rt.w["model.layers.0.self_attn.k_proj.weight"] = []float32{12}
	rt.w["model.layers.0.self_attn.v_proj.weight"] = []float32{13}
	rt.w["model.layers.0.self_attn.o_proj.weight"] = []float32{14}
	rt.w["model.layers.0.mlp.gate_proj.weight"] = []float32{15}
	rt.w["model.layers.0.mlp.up_proj.weight"] = []float32{16}
	rt.w["model.layers.0.mlp.down_proj.weight"] = []float32{17}
	rt.cacheTextWeights()
	embed := rt.embed
	ln1 := rt.textLayers[0].w.ln1
	rt.releaseCachedTextWeightMapEntries()
	if len(rt.w) != 0 {
		t.Fatalf("w map len=%d want 0", len(rt.w))
	}
	if &rt.embed[0] != &embed[0] || &rt.textLayers[0].w.ln1[0] != &ln1[0] {
		t.Fatal("cached slices changed")
	}
}

var configForReleaseTest = config.Config{
	VocabSize:        2,
	HiddenSize:       2,
	IntermediateSize: 1,
	NumHiddenLayers:  1,
}

func BenchmarkWeightStatsCached(b *testing.B) {
	rt := &Runtime{weightStats: WeightStats{F32Tensors: 42, F32Bytes: 1024, TotalBytes: 1024}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.WeightStats()
	}
}

func BenchmarkWeightStatsComputed(b *testing.B) {
	rt := &Runtime{w: map[string][]float32{}, qw: map[string]*tensor.Q8Matrix{}, q6w: map[string]*tensor.Q6Matrix{}, q4w: map[string]*tensor.Q4Matrix{}}
	for i := 0; i < 256; i++ {
		rt.w[fmt.Sprintf("w%d", i)] = make([]float32, 1024)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rt.computeWeightStats()
	}
}

func TestCleanWeightPath(t *testing.T) {
	got := cleanWeightPath(filepath.Join(".", "model.gguf"))
	if !filepath.IsAbs(got) {
		t.Fatalf("path %q is not absolute", got)
	}
	if filepath.Base(got) != "model.gguf" {
		t.Fatalf("path=%q", got)
	}
}

func writeTinyConfig(t testing.TB, dir string) {
	t.Helper()
	body := `{
		"vocab_size": 2,
		"hidden_size": 3,
		"intermediate_size": 4,
		"num_hidden_layers": 0,
		"num_attention_heads": 1,
		"num_key_value_heads": 1,
		"head_dim": 3,
		"vision_config": {"num_hidden_layers": 0}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTinySafetensors(t testing.TB, path, name string, shape []int64, values []float32) {
	t.Helper()
	raw := map[string]any{
		name: map[string]any{
			"dtype":        "F32",
			"shape":        shape,
			"data_offsets": []int{0, len(values) * 4},
		},
	}
	header, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8+len(header)+len(values)*4)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	pos := 8 + len(header)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[pos+i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTinySafetensorsBF16(t testing.TB, path, name string, shape []int64, values []float32) {
	t.Helper()
	raw := map[string]any{
		name: map[string]any{
			"dtype":        "BF16",
			"shape":        shape,
			"data_offsets": []int{0, len(values) * 2},
		},
	}
	header, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8+len(header)+len(values)*2)
	binary.LittleEndian.PutUint64(data[:8], uint64(len(header)))
	copy(data[8:], header)
	pos := 8 + len(header)
	for i, v := range values {
		binary.LittleEndian.PutUint16(data[pos+i*2:], uint16(math.Float32bits(v)>>16))
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
