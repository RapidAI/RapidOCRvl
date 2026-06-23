package model

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"paddleocrvl-go/internal/backend"
	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/fileutil"
	"paddleocrvl-go/internal/gguf"
	"paddleocrvl-go/internal/tensor"
)

type tensorStore interface {
	Float32(name string) ([]float32, []int64, error)
	Close() error
}

type q8TensorStore interface {
	Q8Row(name string) ([]int8, []float32, []int64, error)
}

type q4TensorStore interface {
	Q4Row(name string) ([]byte, []float32, []int64, error)
}

type q6TensorStore interface {
	Q6Row(name string) ([]byte, []float32, []int64, error)
}

type rowTensorStore interface {
	Float32Rows(name string, fn func(row int, values []float32) error) ([]int64, error)
}

type rowTensorFloatBufferStore interface {
	Float32RowsBuffer(name string, floatBuf []float32, fn func(row int, values []float32) error) ([]int64, error)
}

type rowTensorRawBufferStore interface {
	Float32RowsBuffer(name string, floatBuf []float32, rawBuf []byte, fn func(row int, values []float32) error) ([]int64, error)
}

type shapeTensorStore interface {
	Shape(name string) ([]int64, error)
}

type dtypeTensorStore interface {
	DType(name string) (string, bool)
}

type Runtime struct {
	cfg                   *config.Config
	sf                    tensorStore
	w                     map[string][]float32
	qw                    map[string]*tensor.Q8Matrix
	q6w                   map[string]*tensor.Q6Matrix
	q4w                   map[string]*tensor.Q4Matrix
	embed                 []float32
	finalNorm             []float32
	lmHead                []float32
	q8LmHead              *tensor.Q8Matrix
	q6LmHead              *tensor.Q6Matrix
	q4LmHead              *tensor.Q4Matrix
	textLayers            []textLayer
	vision                visionWeights
	visionPosMu           sync.RWMutex
	visionPosCache        map[[2]int][][]float32
	visionRoPEMu          sync.RWMutex
	visionRoPECache       map[[3]int]visionRoPETables
	quantization          string
	backend               string
	ropeFreq              []float64
	ropeAxis              []byte
	scratchPool           sync.Pool
	kvPool                sync.Pool
	visionScratchPool     sync.Pool
	projectScratchPool    sync.Pool
	visionLoaded          bool
	visionMu              sync.RWMutex
	progress              func(done, total int, name string, typ string)
	requestedQuantization string
	weightPath            string
	weightSource          string
	loadStats             LoadStats
	weightStats           WeightStats
	vulkanPlanMu          sync.RWMutex
	vulkanPlanCache       vulkanPlanCache
}

type vulkanPlanCache struct {
	shape             backend.VulkanModelShape
	valid             bool
	plans             []backend.VulkanPlan
	summary           backend.VulkanPlanSummary
	graph             backend.VulkanExecutionGraph
	pipes             []backend.VulkanPipelinePlan
	command           backend.VulkanCommandPlan
	commandValidation string
}

type layerWeights struct {
	q, k, v, o     []float32
	gate, up, down []float32
	ln1, ln2       []float32
}

type qLayerWeights struct {
	q, k, v, o     *tensor.Q8Matrix
	gate, up, down *tensor.Q8Matrix
}

type q4LayerWeights struct {
	q, k, v, o     *tensor.Q4Matrix
	gate, up, down *tensor.Q4Matrix
}

type q6LayerWeights struct {
	q, k, v, o     *tensor.Q6Matrix
	gate, up, down *tensor.Q6Matrix
}

type textLayer struct {
	w  layerWeights
	q8 qLayerWeights
	q6 q6LayerWeights
	q4 q4LayerWeights
}

type LoadOptions struct {
	Quantization string
	Backend      string
	Progress     func(done, total int, name string, typ string)
}

type kvCache struct {
	k     []float32
	v     []float32
	len   int
	kvDim int
}

type layerScratch struct {
	norm    []float32
	att     []float32
	mlp     []float32
	q       []float32
	k       []float32
	v       []float32
	headOut []float32
	gate    []float32
	scores  []float32
}

type generationScratch struct {
	layers     []layerScratch
	hidden     []float32
	norm       []float32
	logits     []float32
	ropeCos    []float32
	ropeSin    []float32
	scoreBlock []float32
	candidates []tokenScore
	weights    []float32
	positions  []ropePos
	inputIDs   []int
	rng        fastRNG
}

type GenerateOptions struct {
	MaxNewTokens int
	Temperature  float64
	TopK         int
	Seed         int64
	EOSTokenIDs  []int
}

var defaultEOSTokenIDs = []int{2}

const maxEOSTokenIDs = 1024

type GenerateResult struct {
	Tokens       []int
	PromptTokens int
}

func ValidateGenerateOptions(opts GenerateOptions) error {
	if opts.MaxNewTokens < 0 {
		return fmt.Errorf("max_new_tokens must be >= 0")
	}
	if opts.Temperature < 0 {
		return fmt.Errorf("temperature must be >= 0")
	}
	if math.IsNaN(opts.Temperature) || math.IsInf(opts.Temperature, 0) {
		return fmt.Errorf("temperature must be finite")
	}
	if opts.TopK < 0 {
		return fmt.Errorf("top_k must be >= 0")
	}
	if len(opts.EOSTokenIDs) > maxEOSTokenIDs {
		return fmt.Errorf("eos_token_ids length %d exceeds limit %d", len(opts.EOSTokenIDs), maxEOSTokenIDs)
	}
	for _, id := range opts.EOSTokenIDs {
		if id < 0 {
			return fmt.Errorf("eos_token_ids must not contain negative ids")
		}
	}
	return nil
}

func validateInputTokenIDs(ids []int) error {
	for _, id := range ids {
		if id < 0 {
			return fmt.Errorf("input tokens must not contain negative ids")
		}
	}
	return nil
}

func validateGenerationInput(input []int, opts GenerateOptions) error {
	if err := validateInputTokenIDs(input); err != nil {
		return err
	}
	if opts.MaxNewTokens > 0 && len(input) == 0 {
		return fmt.Errorf("input tokens must not be empty when max_new_tokens > 0")
	}
	return nil
}

type WeightStats struct {
	F32Tensors int   `json:"f32_tensors"`
	F32Bytes   int64 `json:"f32_bytes"`
	Q8Tensors  int   `json:"q8_tensors"`
	Q8Bytes    int64 `json:"q8_bytes"`
	Q6Tensors  int   `json:"q6_tensors"`
	Q6Bytes    int64 `json:"q6_bytes"`
	Q4Tensors  int   `json:"q4_tensors"`
	Q4Bytes    int64 `json:"q4_bytes"`
	TotalBytes int64 `json:"total_bytes"`
}

type LoadStats struct {
	TotalMS       int64 `json:"total_ms"`
	OpenWeightMS  int64 `json:"open_weight_ms"`
	PreloadTextMS int64 `json:"preload_text_ms"`
	QuantizeMS    int64 `json:"quantize_ms"`
}

type CacheStats struct {
	VisionPositionTables int `json:"vision_position_tables"`
	VisionRoPETables     int `json:"vision_rope_tables"`
}

type ropePos struct {
	T int
	H int
	W int
}

func Load(dir string) (*Runtime, error) {
	return LoadWithOptions(dir, LoadOptions{})
}

func ProgressLogger(prefix string) func(done, total int, name string, typ string) {
	var last time.Time
	return func(done, total int, name string, typ string) {
		now := time.Now()
		action := "converting"
		doneAction := "converted"
		if typ == "LOAD" || typ == "LOAD-VISION" {
			action = "loading"
			doneAction = "loaded"
		}
		if done == total {
			log.Printf("%s %s %d/%d tensors", prefix, doneAction, done, total)
			return
		}
		if done == 0 || now.Sub(last) >= 2*time.Second {
			log.Printf("%s %s %d/%d %s %s", prefix, action, done+1, total, typ, name)
			last = now
		}
	}
}

func LoadWithOptions(dir string, opts LoadOptions) (*Runtime, error) {
	totalStart := time.Now()
	opts.Quantization = NormalizeQuantization(opts.Quantization)
	cfg, err := config.Load(dir)
	if err != nil {
		return nil, err
	}
	if err := ValidateRuntimeConfig(cfg); err != nil {
		return nil, err
	}
	openStart := time.Now()
	store, activeQuant, weightPath, weightSource, err := openOrConvertGGUF(dir, opts.Quantization, opts.Progress)
	if err != nil {
		return nil, err
	}
	openMS := elapsedMillis(openStart)
	textWeightCount := 3 + cfg.NumHiddenLayers*9
	visionWeightCount := 11 + cfg.VisionConfig.NumHiddenLayers*16
	quantWeightCount := 1 + cfg.NumHiddenLayers*7
	rt := &Runtime{
		cfg:                   cfg,
		sf:                    store,
		w:                     make(map[string][]float32, textWeightCount+visionWeightCount),
		qw:                    make(map[string]*tensor.Q8Matrix, quantWeightCount),
		q6w:                   make(map[string]*tensor.Q6Matrix, quantWeightCount),
		q4w:                   make(map[string]*tensor.Q4Matrix, quantWeightCount),
		quantization:          activeQuant,
		backend:               opts.Backend,
		progress:              opts.Progress,
		requestedQuantization: opts.Quantization,
		weightPath:            weightPath,
		weightSource:          weightSource,
	}
	rt.scratchPool.New = func() any { return rt.newGenerationScratch() }
	rt.kvPool.New = func() any { return rt.newKVCaches(0) }
	rt.initTextRoPE()
	preloadStart := time.Now()
	if err := rt.preloadTextWeights(); err != nil {
		store.Close()
		return nil, err
	}
	preloadMS := elapsedMillis(preloadStart)
	quantStart := time.Now()
	if activeQuant == "q8" {
		rt.quantizeTextWeights()
	} else if activeQuant == "q6" {
		rt.quantizeTextWeightsQ6()
	} else if activeQuant == "q4" {
		rt.quantizeTextWeightsQ4()
	} else if activeQuant != "" && activeQuant != "f32" {
		store.Close()
		return nil, fmt.Errorf("unsupported quantization %q", activeQuant)
	}
	quantMS := elapsedMillis(quantStart)
	rt.cacheTextWeights()
	rt.weightStats = rt.computeWeightStats()
	rt.releaseCachedTextWeightMapEntries()
	rt.loadStats = LoadStats{TotalMS: elapsedMillis(totalStart), OpenWeightMS: openMS, PreloadTextMS: preloadMS, QuantizeMS: quantMS}
	return rt, nil
}

func ValidateRuntimeConfig(c *config.Config) error {
	if c == nil {
		return fmt.Errorf("config is nil")
	}
	if c.VocabSize <= 0 {
		return fmt.Errorf("config vocab_size must be > 0")
	}
	if c.HiddenSize <= 0 {
		return fmt.Errorf("config hidden_size must be > 0")
	}
	if c.IntermediateSize <= 0 {
		return fmt.Errorf("config intermediate_size must be > 0")
	}
	if c.NumHiddenLayers < 0 {
		return fmt.Errorf("config num_hidden_layers must be >= 0")
	}
	if c.MaxPositionEmb < 0 {
		return fmt.Errorf("config max_position_embeddings must be >= 0")
	}
	if c.ImageTokenID < 0 || c.VisionStartTokenID < 0 || c.VisionEndTokenID < 0 || c.VideoTokenID < 0 {
		return fmt.Errorf("config vision token ids must be >= 0")
	}
	if c.NumAttentionHeads <= 0 {
		return fmt.Errorf("config num_attention_heads must be > 0")
	}
	if c.NumKeyValueHeads <= 0 {
		return fmt.Errorf("config num_key_value_heads must be > 0")
	}
	if c.HeadDim <= 0 {
		return fmt.Errorf("config head_dim must be > 0")
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return fmt.Errorf("config num_attention_heads must be divisible by num_key_value_heads")
	}
	if !finitePositive(c.RMSNormEps) {
		return fmt.Errorf("config rms_norm_eps must be finite and > 0")
	}
	if !finitePositive(c.RopeTheta) {
		return fmt.Errorf("config rope_theta must be finite and > 0")
	}
	attentionSize, ok := checkedMulInt(c.NumAttentionHeads, c.HeadDim)
	if !ok {
		return fmt.Errorf("config attention shape overflows int")
	}
	if attentionSize != c.HiddenSize {
		return fmt.Errorf("config num_attention_heads * head_dim must equal hidden_size")
	}
	if _, ok := checkedMulInt(c.NumKeyValueHeads, c.HeadDim); !ok {
		return fmt.Errorf("config key/value shape overflows int")
	}
	if err := validateRuntimeDerivedShapes(c); err != nil {
		return err
	}
	if c.VisionConfig.NumHiddenLayers < 0 {
		return fmt.Errorf("config vision_config.num_hidden_layers must be >= 0")
	}
	if c.VisionConfig.NumHiddenLayers > 0 {
		if err := validateRuntimeVisionConfig(c.VisionConfig); err != nil {
			return err
		}
	}
	if c.RopeScaling != nil {
		for _, section := range c.RopeScaling.MropeSection {
			if section < 0 {
				return fmt.Errorf("config rope_scaling.mrope_section must not contain negative values")
			}
		}
	}
	return nil
}

func validateRuntimeDerivedShapes(c *config.Config) error {
	if _, ok := checkedMulInt(c.VocabSize, c.HiddenSize); !ok {
		return fmt.Errorf("config vocab_size * hidden_size overflows int")
	}
	if _, ok := checkedMulInt(c.NumHiddenLayers, c.HiddenSize); !ok {
		return fmt.Errorf("config num_hidden_layers * hidden_size overflows int")
	}
	if _, ok := checkedMulInt(c.NumHiddenLayers, c.IntermediateSize); !ok {
		return fmt.Errorf("config num_hidden_layers * intermediate_size overflows int")
	}
	if _, ok := checkedMulInt(c.HiddenSize, c.IntermediateSize); !ok {
		return fmt.Errorf("config hidden_size * intermediate_size overflows int")
	}
	return nil
}

func validateRuntimeVisionConfig(v config.Vision) error {
	if v.HiddenSize <= 0 {
		return fmt.Errorf("config vision_config.hidden_size must be > 0")
	}
	if v.ImageSize < 0 {
		return fmt.Errorf("config vision_config.image_size must be >= 0")
	}
	if v.IntermediateSize <= 0 {
		return fmt.Errorf("config vision_config.intermediate_size must be > 0")
	}
	if v.NumAttentionHeads <= 0 {
		return fmt.Errorf("config vision_config.num_attention_heads must be > 0")
	}
	if v.HiddenSize%v.NumAttentionHeads != 0 {
		return fmt.Errorf("config vision_config.hidden_size must be divisible by vision_config.num_attention_heads")
	}
	if v.PatchSize <= 0 {
		return fmt.Errorf("config vision_config.patch_size must be > 0")
	}
	if v.SpatialMergeSize <= 0 {
		return fmt.Errorf("config vision_config.spatial_merge_size must be > 0")
	}
	if !finitePositive(v.LayerNormEps) {
		return fmt.Errorf("config vision_config.layer_norm_eps must be finite and > 0")
	}
	if _, ok := checkedMulInt(v.PatchSize, v.PatchSize); !ok {
		return fmt.Errorf("config vision_config.patch_size overflows int")
	}
	patchChannels, ok := checkedMulInt(3, v.PatchSize)
	if !ok {
		return fmt.Errorf("config vision_config.patch_size overflows int")
	}
	patchElements, ok := checkedMulInt(patchChannels, v.PatchSize)
	if !ok {
		return fmt.Errorf("config vision_config.patch elements overflow int")
	}
	if _, ok := checkedMulInt(v.HiddenSize, patchElements); !ok {
		return fmt.Errorf("config vision_config patch embedding shape overflows int")
	}
	if _, ok := checkedMulInt(v.HiddenSize, v.HiddenSize); !ok {
		return fmt.Errorf("config vision_config hidden shape overflows int")
	}
	if _, ok := checkedMulInt(v.HiddenSize, v.IntermediateSize); !ok {
		return fmt.Errorf("config vision_config mlp shape overflows int")
	}
	if _, ok := checkedMulInt(v.HiddenSize, 8); !ok {
		return fmt.Errorf("config vision_config scratch shape overflows int")
	}
	if _, ok := checkedMulInt(v.HiddenSize, 4); !ok {
		return fmt.Errorf("config vision_config projection shape overflows int")
	}
	projHidden, ok := checkedMulInt(v.HiddenSize, 4)
	if ok {
		_, ok = checkedMulInt(projHidden, projHidden)
	}
	if !ok {
		return fmt.Errorf("config vision_config projection shape overflows int")
	}
	return nil
}

func finitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func elapsedMillis(start time.Time) int64 {
	return time.Since(start).Milliseconds()
}

func NormalizeQuantization(quantization string) string {
	q := trimASCIIWhitespace(quantization)
	if q == "" {
		return "f32"
	}
	switch len(q) {
	case 2:
		if asciiEqualFoldModel(q, "q4") {
			return "q4"
		}
		if asciiEqualFoldModel(q, "q6") {
			return "q6"
		}
		if asciiEqualFoldModel(q, "q8") {
			return "q8"
		}
	case 3:
		if asciiEqualFoldModel(q, "f32") {
			return "f32"
		}
	case 4:
		if asciiEqualFoldModel(q, "auto") {
			return "auto"
		}
	case 9:
		if asciiEqualFoldModel(q, "auto-fast") {
			return "auto-fast"
		}
	case 12:
		if asciiEqualFoldModel(q, "auto-quality") {
			return "auto-quality"
		}
	}
	return lowerASCII(q)
}

func trimASCIIWhitespace(s string) string {
	if len(s) == 0 || (!isASCIIWhitespace(s[0]) && !isASCIIWhitespace(s[len(s)-1])) {
		return s
	}
	start, end := 0, len(s)
	for start < end && isASCIIWhitespace(s[start]) {
		start++
	}
	for end > start && isASCIIWhitespace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isASCIIWhitespace(c byte) bool {
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f'
}

func asciiEqualFoldModel(s, lower string) bool {
	if len(s) != len(lower) {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != lower[i] {
			return false
		}
	}
	return true
}

func lowerASCII(s string) string {
	firstUpper := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			firstUpper = i
			break
		}
	}
	if firstUpper < 0 {
		return s
	}
	if len(s) <= 64 {
		var small [64]byte
		copy(small[:], s[:firstUpper])
		for i := firstUpper; i < len(s); i++ {
			c := s[i]
			if 'A' <= c && c <= 'Z' {
				c += 'a' - 'A'
			}
			small[i] = c
		}
		return string(small[:len(s)])
	}
	out := []byte(s)
	for i := firstUpper; i < len(out); i++ {
		c := out[i]
		if 'A' <= c && c <= 'Z' {
			out[i] = c + ('a' - 'A')
		}
	}
	return unsafe.String(&out[0], len(out))
}

func (rt *Runtime) initTextRoPE() {
	c := rt.cfg
	half := c.HeadDim / 2
	rt.ropeFreq = make([]float64, half)
	rt.ropeAxis = make([]byte, half)
	sections := []int{16, 24, 24}
	if c.RopeScaling != nil && len(c.RopeScaling.MropeSection) > 0 {
		sections = c.RopeScaling.MropeSection
	}
	sectionEnd := 0
	axis := 0
	if len(sections) > 0 {
		sectionEnd = sections[0]
	}
	for i := 0; i < half; i++ {
		for axis+1 < len(sections) && i >= sectionEnd {
			axis++
			sectionEnd += sections[axis]
		}
		if axis > 2 {
			axis = 2
		}
		rt.ropeAxis[i] = byte(axis)
		rt.ropeFreq[i] = math.Pow(c.RopeTheta, -float64(2*i)/float64(c.HeadDim))
	}
}

func openOrConvertGGUF(dir, quantization string, progress func(done, total int, name string, typ string)) (tensorStore, string, string, string, error) {
	quantization = NormalizeQuantization(quantization)
	var safetensorsChecked bool
	var safetensorsAvailable bool
	hasSafetensorsCached := func() bool {
		if !safetensorsChecked {
			safetensorsAvailable = hasSafetensors(dir)
			safetensorsChecked = true
		}
		return safetensorsAvailable
	}
	if quantization == "auto" || quantization == "auto-fast" || quantization == "auto-quality" {
		for _, candidate := range autoQuantCandidates(quantization) {
			path := filepath.Join(dir, candidate.file)
			if store, ok, err := openGGUFIfExists(path); ok || err != nil {
				return store, candidate.quant, cleanWeightPath(path), "existing_gguf", err
			}
		}
		if hasSafetensorsCached() {
			target := autoQuantBuildTarget(quantization)
			path := filepath.Join(dir, "model-"+target+".gguf")
			if err := gguf.ConvertSafetensorsWithOptions(dir, path, dir, gguf.ConvertOptions{Quantization: target, Progress: progress}); err != nil {
				return nil, "", "", "", err
			}
			store, err := gguf.Open(path)
			return store, target, cleanWeightPath(path), "converted_safetensors", err
		}
		return nil, "", "", "", fmt.Errorf("no model-q4.gguf, model-q6.gguf, model-q8.gguf, model.gguf, or model.safetensors found in %s", dir)
	}
	if quantization == "q8" {
		ggufQ8Path := filepath.Join(dir, "model-q8.gguf")
		if store, ok, err := openGGUFIfExists(ggufQ8Path); ok || err != nil {
			return store, "q8", cleanWeightPath(ggufQ8Path), "existing_gguf", err
		}
		if hasSafetensorsCached() {
			if err := gguf.ConvertSafetensorsWithOptions(dir, ggufQ8Path, dir, gguf.ConvertOptions{Quantization: "q8", Progress: progress}); err != nil {
				return nil, "", "", "", err
			}
			store, err := gguf.Open(ggufQ8Path)
			return store, "q8", cleanWeightPath(ggufQ8Path), "converted_safetensors", err
		}
	}
	if quantization == "q4" {
		ggufQ4Path := filepath.Join(dir, "model-q4.gguf")
		if store, ok, err := openGGUFIfExists(ggufQ4Path); ok || err != nil {
			return store, "q4", cleanWeightPath(ggufQ4Path), "existing_gguf", err
		}
		if hasSafetensorsCached() {
			if err := gguf.ConvertSafetensorsWithOptions(dir, ggufQ4Path, dir, gguf.ConvertOptions{Quantization: "q4", Progress: progress}); err != nil {
				return nil, "", "", "", err
			}
			store, err := gguf.Open(ggufQ4Path)
			return store, "q4", cleanWeightPath(ggufQ4Path), "converted_safetensors", err
		}
	}
	if quantization == "q6" {
		ggufQ6Path := filepath.Join(dir, "model-q6.gguf")
		if store, ok, err := openGGUFIfExists(ggufQ6Path); ok || err != nil {
			return store, "q6", cleanWeightPath(ggufQ6Path), "existing_gguf", err
		}
		if hasSafetensorsCached() {
			if err := gguf.ConvertSafetensorsWithOptions(dir, ggufQ6Path, dir, gguf.ConvertOptions{Quantization: "q6", Progress: progress}); err != nil {
				return nil, "", "", "", err
			}
			store, err := gguf.Open(ggufQ6Path)
			return store, "q6", cleanWeightPath(ggufQ6Path), "converted_safetensors", err
		}
	}
	ggufPath := filepath.Join(dir, "model.gguf")
	if store, ok, err := openGGUFIfExists(ggufPath); ok || err != nil {
		return store, quantization, cleanWeightPath(ggufPath), "existing_gguf", err
	}
	if !hasSafetensorsCached() {
		return nil, "", "", "", fmt.Errorf("no model.gguf, model.safetensors, or model.safetensors.index.json found in %s", dir)
	}
	if err := gguf.ConvertSafetensorsWithOptions(dir, ggufPath, dir, gguf.ConvertOptions{Progress: progress}); err != nil {
		return nil, "", "", "", err
	}
	store, err := gguf.Open(ggufPath)
	return store, quantization, cleanWeightPath(ggufPath), "converted_safetensors", err
}

func openGGUFIfExists(path string) (tensorStore, bool, error) {
	store, err := gguf.Open(path)
	if err == nil {
		return store, true, nil
	}
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	return nil, true, err
}

func cleanWeightPath(path string) string {
	return fileutil.Abs(path)
}

type autoQuantCandidate struct {
	file  string
	quant string
}

func autoQuantCandidates(mode string) []autoQuantCandidate {
	switch mode {
	case "auto-fast":
		return []autoQuantCandidate{
			{"model-q4.gguf", "q4"},
			{"model-q6.gguf", "q6"},
			{"model-q8.gguf", "q8"},
			{"model.gguf", "f32"},
		}
	case "auto-quality":
		return []autoQuantCandidate{
			{"model-q8.gguf", "q8"},
			{"model-q6.gguf", "q6"},
			{"model-q4.gguf", "q4"},
			{"model.gguf", "f32"},
		}
	default:
		return []autoQuantCandidate{
			{"model-q4.gguf", "q4"},
			{"model-q6.gguf", "q6"},
			{"model-q8.gguf", "q8"},
			{"model.gguf", "f32"},
		}
	}
}

func autoQuantBuildTarget(mode string) string {
	switch mode {
	case "auto-fast":
		return "q4"
	case "auto-quality":
		return "q8"
	default:
		return "q6"
	}
}

func hasSafetensors(dir string) bool {
	for _, name := range []string{"model.safetensors", "model.safetensors.index.json"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return true
		}
	}
	return false
}

func (rt *Runtime) Close() error {
	if rt == nil || rt.sf == nil {
		return nil
	}
	return rt.sf.Close()
}

func (rt *Runtime) Config() *config.Config {
	c := *rt.cfg
	return &c
}

// ConfigView returns runtime config for read-only callers.
// Callers must not mutate returned config.
func (rt *Runtime) ConfigView() *config.Config {
	return rt.cfg
}

func (rt *Runtime) Quantization() string {
	if rt.quantization == "" {
		return "f32"
	}
	return rt.quantization
}

func (rt *Runtime) RequestedQuantization() string {
	if rt.requestedQuantization == "" {
		return "f32"
	}
	return rt.requestedQuantization
}

func (rt *Runtime) WeightPath() string {
	return rt.weightPath
}

func (rt *Runtime) WeightSource() string {
	if rt.weightSource == "" {
		return "unknown"
	}
	return rt.weightSource
}

func (rt *Runtime) LoadStats() LoadStats {
	return rt.loadStats
}

func (rt *Runtime) CacheStats() CacheStats {
	rt.visionPosMu.RLock()
	pos := len(rt.visionPosCache)
	rt.visionPosMu.RUnlock()
	rt.visionRoPEMu.RLock()
	rope := len(rt.visionRoPECache)
	rt.visionRoPEMu.RUnlock()
	return CacheStats{VisionPositionTables: pos, VisionRoPETables: rope}
}

func (rt *Runtime) Backend() string {
	if rt.backend == "" {
		return "cpu"
	}
	return rt.backend
}

func (rt *Runtime) VulkanPlans() []backend.VulkanPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return nil
	}
	return append([]backend.VulkanPlan(nil), cache.plans...)
}

func (rt *Runtime) VulkanPlanSummary() backend.VulkanPlanSummary {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanPlanSummary{}
	}
	out := cache.summary
	out.Plans = append([]backend.VulkanPlan(nil), cache.summary.Plans...)
	return out
}

func (rt *Runtime) VulkanExecutionGraph() backend.VulkanExecutionGraph {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanExecutionGraph{}
	}
	out := cache.graph
	out.Stages = append([]backend.VulkanStageSummary(nil), cache.graph.Stages...)
	return out
}

func (rt *Runtime) VulkanPipelinePlan() []backend.VulkanPipelinePlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return nil
	}
	return append([]backend.VulkanPipelinePlan(nil), cache.pipes...)
}

func (rt *Runtime) VulkanCommandPlan() backend.VulkanCommandPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanCommandPlan{}
	}
	return cloneVulkanCommandPlan(cache.command)
}

func (rt *Runtime) VulkanArtifacts() backend.VulkanModelArtifacts {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanModelArtifacts{}
	}
	pipes := append([]backend.VulkanPipelinePlan(nil), cache.pipes...)
	out := backend.VulkanModelArtifacts{
		Plans:          append([]backend.VulkanPlan(nil), cache.plans...),
		Summary:        cache.summary,
		ExecutionGraph: cache.graph,
		PipelinePlan:   pipes,
		CommandPlan:    cloneVulkanCommandPlanWithPipelines(cache.command, pipes),
	}
	out.Summary.Plans = out.Plans
	out.ExecutionGraph.Stages = append([]backend.VulkanStageSummary(nil), cache.graph.Stages...)
	return out
}

// VulkanArtifactsView returns cached Vulkan planning metadata for read-only use.
// Callers must not mutate returned slices or nested slices.
func (rt *Runtime) VulkanArtifactsView() backend.VulkanModelArtifacts {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanModelArtifacts{}
	}
	return backend.VulkanModelArtifacts{
		Plans:          cache.plans,
		Summary:        cache.summary,
		ExecutionGraph: cache.graph,
		PipelinePlan:   cache.pipes,
		CommandPlan:    cache.command,
	}
}

func (rt *Runtime) VulkanCommandPlanValidation() string {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return ""
	}
	return cache.commandValidation
}

func (rt *Runtime) cachedVulkanPlans() (vulkanPlanCache, bool) {
	shape, ok := rt.vulkanModelShape()
	if !ok {
		return vulkanPlanCache{}, false
	}
	rt.vulkanPlanMu.RLock()
	if rt.vulkanPlanCache.valid && rt.vulkanPlanCache.shape == shape {
		cache := rt.vulkanPlanCache
		rt.vulkanPlanMu.RUnlock()
		return cache, true
	}
	rt.vulkanPlanMu.RUnlock()

	rt.vulkanPlanMu.Lock()
	defer rt.vulkanPlanMu.Unlock()
	if rt.vulkanPlanCache.valid && rt.vulkanPlanCache.shape == shape {
		return rt.vulkanPlanCache, true
	}
	artifacts := backend.VulkanModelArtifactsForShape(shape)
	rt.vulkanPlanCache = vulkanPlanCache{
		shape:             shape,
		valid:             true,
		plans:             artifacts.Plans,
		summary:           artifacts.Summary,
		graph:             artifacts.ExecutionGraph,
		pipes:             artifacts.PipelinePlan,
		command:           artifacts.CommandPlan,
		commandValidation: backend.ValidateVulkanCommandPlan(artifacts.CommandPlan),
	}
	return rt.vulkanPlanCache, true
}

func cloneVulkanCommandPlan(in backend.VulkanCommandPlan) backend.VulkanCommandPlan {
	return cloneVulkanCommandPlanWithPipelines(in, append([]backend.VulkanPipelinePlan(nil), in.Pipelines...))
}

func cloneVulkanCommandPlanWithPipelines(in backend.VulkanCommandPlan, pipelines []backend.VulkanPipelinePlan) backend.VulkanCommandPlan {
	out := in
	out.Pipelines = pipelines
	out.ShaderModules = append([]backend.VulkanShaderModulePlan(nil), in.ShaderModules...)
	out.Layouts = append([]backend.VulkanPipelineLayoutPlan(nil), in.Layouts...)
	bindingCount := 0
	for i := range in.Layouts {
		bindingCount += len(in.Layouts[i].Bindings)
	}
	bindings := make([]backend.VulkanBinding, 0, bindingCount)
	for i := range out.Layouts {
		start := len(bindings)
		bindings = append(bindings, in.Layouts[i].Bindings...)
		out.Layouts[i].Bindings = bindings[start:len(bindings):len(bindings)]
	}
	out.Commands = append([]backend.VulkanCommand(nil), in.Commands...)
	resourceCount := 0
	descriptorCount := 0
	for i := range in.Commands {
		resourceCount += len(in.Commands[i].Resources)
		descriptorCount += len(in.Commands[i].DescriptorWrites)
	}
	resources := make([]backend.VulkanResource, 0, resourceCount)
	descriptors := make([]backend.VulkanDescriptorWrite, 0, descriptorCount)
	for i := range out.Commands {
		resourceStart := len(resources)
		resources = append(resources, in.Commands[i].Resources...)
		out.Commands[i].Resources = resources[resourceStart:len(resources):len(resources)]
		descriptorStart := len(descriptors)
		descriptors = append(descriptors, in.Commands[i].DescriptorWrites...)
		out.Commands[i].DescriptorWrites = descriptors[descriptorStart:len(descriptors):len(descriptors)]
	}
	out.DispatchBatches = append([]backend.VulkanDispatchBatch(nil), in.DispatchBatches...)
	out.Records = append([]backend.VulkanCommandRecord(nil), in.Records...)
	out.Barriers = append([]backend.VulkanBufferBarrier(nil), in.Barriers...)
	out.Allocations = append([]backend.VulkanBufferAllocation(nil), in.Allocations...)
	out.Uploads = append([]backend.VulkanBufferTransfer(nil), in.Uploads...)
	out.Readbacks = append([]backend.VulkanBufferTransfer(nil), in.Readbacks...)
	return out
}

func (rt *Runtime) vulkanModelShape() (backend.VulkanModelShape, bool) {
	if rt.cfg == nil {
		return backend.VulkanModelShape{}, false
	}
	cfg := rt.cfg
	patchElements := cfg.VisionConfig.PatchSize * cfg.VisionConfig.PatchSize * 3
	return backend.VulkanModelShape{
		Quant:               rt.quantization,
		VocabSize:           cfg.VocabSize,
		HiddenSize:          cfg.HiddenSize,
		IntermediateSize:    cfg.IntermediateSize,
		NumAttentionHeads:   cfg.NumAttentionHeads,
		NumKeyValueHeads:    cfg.NumKeyValueHeads,
		HeadDim:             cfg.HeadDim,
		VisionHiddenSize:    cfg.VisionConfig.HiddenSize,
		VisionIntermediate:  cfg.VisionConfig.IntermediateSize,
		VisionPatchElements: patchElements,
		TextLayers:          cfg.NumHiddenLayers,
		VisionLayers:        cfg.VisionConfig.NumHiddenLayers,
	}, true
}

func (rt *Runtime) WeightStats() WeightStats {
	if rt.weightStats.TotalBytes != 0 || rt.weightStats.F32Tensors != 0 || rt.weightStats.Q8Tensors != 0 || rt.weightStats.Q6Tensors != 0 || rt.weightStats.Q4Tensors != 0 {
		return rt.weightStats
	}
	return rt.computeWeightStats()
}

func (rt *Runtime) computeWeightStats() WeightStats {
	var st WeightStats
	for _, v := range rt.w {
		st.F32Tensors++
		st.F32Bytes += int64(len(v)) << 2
	}
	for _, v := range rt.qw {
		st.Q8Tensors++
		st.Q8Bytes += int64(len(v.Data)) + (int64(len(v.Scale)) << 2)
	}
	for _, v := range rt.q6w {
		st.Q6Tensors++
		st.Q6Bytes += int64(len(v.Data)) + (int64(len(v.Scale)) << 2)
	}
	for _, v := range rt.q4w {
		st.Q4Tensors++
		st.Q4Bytes += int64(len(v.Data)) + (int64(len(v.Scale)) << 2)
	}
	st.TotalBytes = st.F32Bytes + st.Q8Bytes + st.Q6Bytes + st.Q4Bytes
	return st
}

func (rt *Runtime) Generate(ctx context.Context, input []int, maxNew int) ([]int, error) {
	res, err := rt.GenerateWithOptions(ctx, input, GenerateOptions{MaxNewTokens: maxNew})
	if err != nil {
		return nil, err
	}
	return res.Tokens, nil
}

func (rt *Runtime) GenerateWithOptions(ctx context.Context, input []int, opts GenerateOptions) (GenerateResult, error) {
	if err := ValidateGenerateOptions(opts); err != nil {
		return GenerateResult{}, err
	}
	if err := validateGenerationInput(input, opts); err != nil {
		return GenerateResult{}, err
	}
	opts = normalizeGenerateOptions(opts)
	if opts.MaxNewTokens == 0 {
		return GenerateResult{Tokens: append([]int(nil), input...), PromptTokens: len(input)}, nil
	}
	caches, cachePtr := rt.getKVCaches(len(input) + opts.MaxNewTokens)
	scratch := rt.getGenerationScratch()
	rt.ensureScoreCapacity(scratch, len(input)+opts.MaxNewTokens)
	out := make([]int, 0, len(input)+opts.MaxNewTokens)
	out = append(out, input...)
	var logits []float32
	for pos, id := range input {
		select {
		case <-ctx.Done():
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, ctx.Err()
		default:
		}
		if err := rt.tokenEmbeddingInto(scratch.hidden, id); err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		l, err := rt.forwardEmbedding(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = l
	}
	rng := samplingRNGInto(opts, &scratch.rng)
	for i := 0; i < opts.MaxNewTokens; i++ {
		next := sampleTokenScratch(logits, opts, rng, scratch)
		out = append(out, next)
		if isEOS(next, opts.EOSTokenIDs) {
			break
		}
		if err := rt.tokenEmbeddingInto(scratch.hidden, next); err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		pos := len(out) - 1
		l, err := rt.forwardEmbedding(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = l
	}
	res := GenerateResult{Tokens: out, PromptTokens: len(input)}
	rt.putGenerationResources(scratch, caches, cachePtr)
	return res, nil
}

func (rt *Runtime) GenerateWithImage(ctx context.Context, input []int, imagePath string, maxNew int) ([]int, error) {
	res, err := rt.GenerateWithImageOptions(ctx, input, imagePath, GenerateOptions{MaxNewTokens: maxNew})
	if err != nil {
		return nil, err
	}
	return res.Tokens, nil
}

func (rt *Runtime) GenerateWithImageOptions(ctx context.Context, input []int, imagePath string, opts GenerateOptions) (GenerateResult, error) {
	if err := ValidateGenerateOptions(opts); err != nil {
		return GenerateResult{}, err
	}
	if err := validateGenerationInput(input, opts); err != nil {
		return GenerateResult{}, err
	}
	opts = normalizeGenerateOptions(opts)
	if opts.MaxNewTokens == 0 {
		return GenerateResult{Tokens: append([]int(nil), input...), PromptTokens: len(input)}, nil
	}
	imageEmbeds, imageGrid, err := rt.EncodeImageWithGrid(imagePath)
	if err != nil {
		return GenerateResult{}, err
	}
	return rt.generateWithImageEmbeds(ctx, input, imageEmbeds, imageGrid, opts)
}

func (rt *Runtime) GenerateWithImageBytesOptions(ctx context.Context, input []int, imageData []byte, opts GenerateOptions) (GenerateResult, error) {
	if err := ValidateGenerateOptions(opts); err != nil {
		return GenerateResult{}, err
	}
	if err := validateGenerationInput(input, opts); err != nil {
		return GenerateResult{}, err
	}
	opts = normalizeGenerateOptions(opts)
	if opts.MaxNewTokens == 0 {
		return GenerateResult{Tokens: append([]int(nil), input...), PromptTokens: len(input)}, nil
	}
	imageEmbeds, imageGrid, err := rt.EncodeImageBytes(imageData)
	if err != nil {
		return GenerateResult{}, err
	}
	return rt.generateWithImageEmbeds(ctx, input, imageEmbeds, imageGrid, opts)
}

func (rt *Runtime) generateWithImageEmbeds(ctx context.Context, input []int, imageEmbeds [][]float32, imageGrid [3]int, opts GenerateOptions) (GenerateResult, error) {
	scratch := rt.getGenerationScratch()
	expandedInput, expanded := rt.expandSingleImagePlaceholderInto(input, len(imageEmbeds), scratch.inputIDs)
	input = expandedInput
	if expanded {
		scratch.inputIDs = input
	}
	caches, cachePtr := rt.getKVCaches(len(input) + opts.MaxNewTokens)
	rt.ensureScoreCapacity(scratch, len(input)+opts.MaxNewTokens)
	positions, ropeDelta := rt.multimodalPositionsInto(input, imageGrid, scratch.positions)
	scratch.positions = positions
	imageCursor := 0
	out := make([]int, 0, len(input)+opts.MaxNewTokens)
	out = append(out, input...)
	var logits []float32
	for pos, id := range input {
		select {
		case <-ctx.Done():
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, ctx.Err()
		default:
		}
		var emb []float32
		if id == rt.cfg.ImageTokenID {
			if imageCursor >= len(imageEmbeds) {
				rt.putGenerationResources(scratch, caches, cachePtr)
				return GenerateResult{}, fmt.Errorf("too many image placeholder tokens: need %d", len(imageEmbeds))
			}
			emb = imageEmbeds[imageCursor]
			imageCursor++
		} else {
			if err := rt.tokenEmbeddingInto(scratch.hidden, id); err != nil {
				rt.putGenerationResources(scratch, caches, cachePtr)
				return GenerateResult{}, err
			}
			emb = scratch.hidden
		}
		l, err := rt.forwardEmbedding(emb, positions[pos], caches, scratch)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = l
	}
	if imageCursor != len(imageEmbeds) {
		rt.putGenerationResources(scratch, caches, cachePtr)
		return GenerateResult{}, fmt.Errorf("not enough image placeholder tokens: got %d need %d", imageCursor, len(imageEmbeds))
	}
	rng := samplingRNGInto(opts, &scratch.rng)
	for i := 0; i < opts.MaxNewTokens; i++ {
		next := sampleTokenScratch(logits, opts, rng, scratch)
		out = append(out, next)
		if isEOS(next, opts.EOSTokenIDs) {
			break
		}
		if err := rt.tokenEmbeddingInto(scratch.hidden, next); err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		pos := len(out) - 1 + ropeDelta
		l, err := rt.forwardEmbedding(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = l
	}
	res := GenerateResult{Tokens: out, PromptTokens: len(input)}
	rt.putGenerationResources(scratch, caches, cachePtr)
	return res, nil
}

func (rt *Runtime) multimodalPositions(input []int, imageGrid [3]int) ([]ropePos, int) {
	return rt.multimodalPositionsInto(input, imageGrid, nil)
}

func (rt *Runtime) multimodalPositionsInto(input []int, imageGrid [3]int, buf []ropePos) ([]ropePos, int) {
	positions := buf[:0]
	if cap(positions) < len(input) {
		positions = make([]ropePos, 0, len(input))
	}
	st := 0
	nextPos := 0
	for {
		ed := -1
		for i := st; i < len(input); i++ {
			if input[i] == rt.cfg.ImageTokenID {
				ed = i
				break
			}
		}
		if ed < 0 {
			break
		}
		stIdx := nextPos
		for i := 0; i < ed-st; i++ {
			p := stIdx + i
			positions = append(positions, ropePos{p, p, p})
		}
		textLen := ed - st
		textEnd := stIdx + textLen - 1
		gridT := imageGrid[0]
		gridH := imageGrid[1] / rt.cfg.VisionConfig.SpatialMergeSize
		gridW := imageGrid[2] / rt.cfg.VisionConfig.SpatialMergeSize
		maxPos := textEnd
		for t := 0; t < gridT; t++ {
			for h := 0; h < gridH; h++ {
				for w := 0; w < gridW; w++ {
					pt := textLen + stIdx + t
					ph := textLen + stIdx + h
					pw := textLen + stIdx + w
					positions = append(positions, ropePos{
						T: pt,
						H: ph,
						W: pw,
					})
					maxPos = max(maxPos, pt, ph, pw)
				}
			}
		}
		nextPos = maxPos + 1
		st = ed + gridT*gridH*gridW
	}
	if st < len(input) {
		stIdx := nextPos
		for i := 0; i < len(input)-st; i++ {
			p := stIdx + i
			positions = append(positions, ropePos{p, p, p})
		}
		nextPos = stIdx + len(input) - st
	}
	if len(positions) == 0 {
		return positions, 0
	}
	return positions, nextPos - len(input)
}

func normalizeGenerateOptions(opts GenerateOptions) GenerateOptions {
	if opts.MaxNewTokens < 0 {
		opts.MaxNewTokens = 0
	}
	if opts.EOSTokenIDs == nil {
		opts.EOSTokenIDs = defaultEOSTokenIDs
	}
	return opts
}

func isEOS(id int, ids []int) bool {
	if len(ids) == 1 {
		return id == ids[0]
	}
	if len(ids) == 2 {
		return id == ids[0] || id == ids[1]
	}
	if len(ids) == 3 {
		return id == ids[0] || id == ids[1] || id == ids[2]
	}
	if len(ids) == 4 {
		return id == ids[0] || id == ids[1] || id == ids[2] || id == ids[3]
	}
	for _, eos := range ids {
		if id == eos {
			return true
		}
	}
	return false
}

type tokenScore struct {
	id    int
	score float32
}

func sampleToken(logits []float32, opts GenerateOptions, rng *rand.Rand) int {
	return sampleTokenScratch(logits, opts, rng, nil)
}

type float64RNG interface {
	Float64() float64
}

type fastRNG struct {
	state uint64
}

func (r *fastRNG) Float64() float64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return float64(z>>11) * (1.0 / (1 << 53))
}

func samplingRNG(opts GenerateOptions) *fastRNG {
	return samplingRNGInto(opts, &fastRNG{})
}

func samplingRNGInto(opts GenerateOptions, rng *fastRNG) *fastRNG {
	if opts.MaxNewTokens <= 0 || opts.Temperature <= 0 || opts.TopK == 1 {
		return nil
	}
	rng.state = uint64(opts.Seed)
	return rng
}

func sampleTokenScratch(logits []float32, opts GenerateOptions, rng float64RNG, scratch *generationScratch) int {
	if opts.Temperature <= 0 || opts.TopK == 1 {
		return tensor.Argmax(logits)
	}
	if opts.TopK <= 0 || opts.TopK >= len(logits) {
		return sampleFullLogitsScratch(logits, float32(opts.Temperature), rng, scratch)
	}
	if opts.Temperature == 1 {
		return sampleTopKTemp1Scratch(logits, opts.TopK, rng, scratch)
	}
	candidates, maxScoreRaw := topKCandidatesUnsortedWithMax(logits, opts.TopK, scratch)
	invTemp := float32(1 / opts.Temperature)
	maxScore := maxScoreRaw * invTemp
	weights := makeSampleWeights(len(candidates), scratch)
	var sum float64
	for i, c := range candidates {
		w := math.Exp(float64(c.score*invTemp - maxScore))
		if weights != nil {
			weights[i] = float32(w)
		}
		sum += w
	}
	pick := rng.Float64() * sum
	if weights != nil {
		if idx := pickWeightedFloat32(weights, pick); idx >= 0 {
			return candidates[idx].id
		}
	} else {
		var acc float64
		for _, c := range candidates {
			acc += math.Exp(float64(c.score*invTemp - maxScore))
			if pick <= acc {
				return c.id
			}
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleTopKTemp1Scratch(logits []float32, topK int, rng float64RNG, scratch *generationScratch) int {
	candidates, maxScore := topKCandidatesUnsortedWithMax(logits, topK, scratch)
	weights := makeSampleWeights(len(candidates), scratch)
	var sum float32
	for i, c := range candidates {
		w := float32(math.Exp(float64(c.score - maxScore)))
		if weights != nil {
			weights[i] = w
		}
		sum += w
	}
	pick := float32(rng.Float64()) * sum
	if weights != nil {
		if idx := pickWeightedRawFloat32(weights, pick); idx >= 0 {
			return candidates[idx].id
		}
	} else {
		var acc float32
		for _, c := range candidates {
			acc += float32(math.Exp(float64(c.score - maxScore)))
			if pick <= acc {
				return c.id
			}
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleFullLogits(logits []float32, temp float32, rng *rand.Rand) int {
	return sampleFullLogitsScratch(logits, temp, rng, nil)
}

func sampleFullLogitsScratch(logits []float32, temp float32, rng float64RNG, scratch *generationScratch) int {
	if temp == 1 {
		return sampleFullLogitsTemp1Scratch(logits, rng, scratch)
	}
	invTemp := 1 / temp
	maxScore := maxLogit(logits) * invTemp
	weights := makeSampleWeights(len(logits), scratch)
	if weights == nil {
		var sum float64
		i := 0
		for ; i+3 < len(logits); i += 4 {
			w0 := math.Exp(float64(logits[i]*invTemp - maxScore))
			w1 := math.Exp(float64(logits[i+1]*invTemp - maxScore))
			w2 := math.Exp(float64(logits[i+2]*invTemp - maxScore))
			w3 := math.Exp(float64(logits[i+3]*invTemp - maxScore))
			sum += (w0 + w1) + (w2 + w3)
		}
		for ; i < len(logits); i++ {
			sum += math.Exp(float64(logits[i]*invTemp - maxScore))
		}
		pick := rng.Float64() * sum
		var acc float64
		for i, score := range logits {
			acc += math.Exp(float64(score*invTemp - maxScore))
			if pick <= acc {
				return i
			}
		}
		return len(logits) - 1
	}
	var sum float32
	i := 0
	for ; i+3 < len(logits); i += 4 {
		w0 := float32(math.Exp(float64(logits[i]*invTemp - maxScore)))
		w1 := float32(math.Exp(float64(logits[i+1]*invTemp - maxScore)))
		w2 := float32(math.Exp(float64(logits[i+2]*invTemp - maxScore)))
		w3 := float32(math.Exp(float64(logits[i+3]*invTemp - maxScore)))
		sum += w0
		weights[i] = sum
		sum += w1
		weights[i+1] = sum
		sum += w2
		weights[i+2] = sum
		sum += w3
		weights[i+3] = sum
	}
	for ; i < len(logits); i++ {
		w := float32(math.Exp(float64(logits[i]*invTemp - maxScore)))
		sum += w
		weights[i] = sum
	}
	pick := float32(rng.Float64()) * sum
	if idx := pickCumulativeFloat32(weights, pick); idx >= 0 {
		return idx
	}
	return len(logits) - 1
}

func sampleFullLogitsTemp1Scratch(logits []float32, rng float64RNG, scratch *generationScratch) int {
	maxScore := maxLogit(logits)
	weights := makeSampleWeights(len(logits), scratch)
	if weights == nil {
		var sum float32
		i := 0
		for ; i+3 < len(logits); i += 4 {
			w0 := float32(math.Exp(float64(logits[i] - maxScore)))
			w1 := float32(math.Exp(float64(logits[i+1] - maxScore)))
			w2 := float32(math.Exp(float64(logits[i+2] - maxScore)))
			w3 := float32(math.Exp(float64(logits[i+3] - maxScore)))
			sum += (w0 + w1) + (w2 + w3)
		}
		for ; i < len(logits); i++ {
			sum += float32(math.Exp(float64(logits[i] - maxScore)))
		}
		pick := float32(rng.Float64()) * sum
		var acc float32
		for i, score := range logits {
			acc += float32(math.Exp(float64(score - maxScore)))
			if pick <= acc {
				return i
			}
		}
		return len(logits) - 1
	}
	var sum float32
	i := 0
	for ; i+3 < len(logits); i += 4 {
		w0 := float32(math.Exp(float64(logits[i] - maxScore)))
		w1 := float32(math.Exp(float64(logits[i+1] - maxScore)))
		w2 := float32(math.Exp(float64(logits[i+2] - maxScore)))
		w3 := float32(math.Exp(float64(logits[i+3] - maxScore)))
		sum += w0
		weights[i] = sum
		sum += w1
		weights[i+1] = sum
		sum += w2
		weights[i+2] = sum
		sum += w3
		weights[i+3] = sum
	}
	for ; i < len(logits); i++ {
		w := float32(math.Exp(float64(logits[i] - maxScore)))
		sum += w
		weights[i] = sum
	}
	pick := float32(rng.Float64()) * sum
	if idx := pickCumulativeFloat32(weights, pick); idx >= 0 {
		return idx
	}
	return len(logits) - 1
}

func pickCumulativeFloat32(cdf []float32, pick float32) int {
	lo, hi := 0, len(cdf)
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if pick <= cdf[mid] {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	if lo < len(cdf) {
		return lo
	}
	return -1
}

func pickWeightedFloat32(weights []float32, pick float64) int {
	var acc float64
	i := 0
	for ; i+7 < len(weights); i += 8 {
		w0 := float64(weights[i])
		w1 := float64(weights[i+1])
		w2 := float64(weights[i+2])
		w3 := float64(weights[i+3])
		w4 := float64(weights[i+4])
		w5 := float64(weights[i+5])
		w6 := float64(weights[i+6])
		w7 := float64(weights[i+7])
		acc += w0
		if pick <= acc {
			return i
		}
		acc += w1
		if pick <= acc {
			return i + 1
		}
		acc += w2
		if pick <= acc {
			return i + 2
		}
		acc += w3
		if pick <= acc {
			return i + 3
		}
		acc += w4
		if pick <= acc {
			return i + 4
		}
		acc += w5
		if pick <= acc {
			return i + 5
		}
		acc += w6
		if pick <= acc {
			return i + 6
		}
		acc += w7
		if pick <= acc {
			return i + 7
		}
	}
	for ; i+3 < len(weights); i += 4 {
		w0 := float64(weights[i])
		w1 := float64(weights[i+1])
		w2 := float64(weights[i+2])
		w3 := float64(weights[i+3])
		acc += w0
		if pick <= acc {
			return i
		}
		acc += w1
		if pick <= acc {
			return i + 1
		}
		acc += w2
		if pick <= acc {
			return i + 2
		}
		acc += w3
		if pick <= acc {
			return i + 3
		}
	}
	for ; i < len(weights); i++ {
		acc += float64(weights[i])
		if pick <= acc {
			return i
		}
	}
	return -1
}

func pickWeightedRawFloat32(weights []float32, pick float32) int {
	var acc float32
	i := 0
	for ; i+7 < len(weights); i += 8 {
		acc += weights[i]
		if pick <= acc {
			return i
		}
		acc += weights[i+1]
		if pick <= acc {
			return i + 1
		}
		acc += weights[i+2]
		if pick <= acc {
			return i + 2
		}
		acc += weights[i+3]
		if pick <= acc {
			return i + 3
		}
		acc += weights[i+4]
		if pick <= acc {
			return i + 4
		}
		acc += weights[i+5]
		if pick <= acc {
			return i + 5
		}
		acc += weights[i+6]
		if pick <= acc {
			return i + 6
		}
		acc += weights[i+7]
		if pick <= acc {
			return i + 7
		}
	}
	for ; i < len(weights); i++ {
		acc += weights[i]
		if pick <= acc {
			return i
		}
	}
	return -1
}

func maxLogit(logits []float32) float32 {
	m := float32(math.Inf(-1))
	i := 0
	n := len(logits)
	var m0, m1, m2, m3, m4, m5, m6, m7 = m, m, m, m, m, m, m, m
	var m8, m9, m10, m11, m12, m13, m14, m15 = m, m, m, m, m, m, m, m
	for ; i+15 < n; i += 16 {
		m0 = max32Local(m0, logits[i])
		m1 = max32Local(m1, logits[i+1])
		m2 = max32Local(m2, logits[i+2])
		m3 = max32Local(m3, logits[i+3])
		m4 = max32Local(m4, logits[i+4])
		m5 = max32Local(m5, logits[i+5])
		m6 = max32Local(m6, logits[i+6])
		m7 = max32Local(m7, logits[i+7])
		m8 = max32Local(m8, logits[i+8])
		m9 = max32Local(m9, logits[i+9])
		m10 = max32Local(m10, logits[i+10])
		m11 = max32Local(m11, logits[i+11])
		m12 = max32Local(m12, logits[i+12])
		m13 = max32Local(m13, logits[i+13])
		m14 = max32Local(m14, logits[i+14])
		m15 = max32Local(m15, logits[i+15])
	}
	m = max32Local(max32Local(max32Local(m0, m1), max32Local(m2, m3)), max32Local(max32Local(m4, m5), max32Local(m6, m7)))
	m = max32Local(m, max32Local(max32Local(max32Local(m8, m9), max32Local(m10, m11)), max32Local(max32Local(m12, m13), max32Local(m14, m15))))
	for ; i < n; i++ {
		m = max32Local(m, logits[i])
	}
	return m
}

func max32Local(a, b float32) float32 {
	if b > a {
		return b
	}
	return a
}

func rankedCandidates(logits []float32, topK int) []tokenScore {
	return rankedCandidatesScratch(logits, topK, nil)
}

func rankedCandidatesScratch(logits []float32, topK int, scratch *generationScratch) []tokenScore {
	if topK > 0 && topK < len(logits) {
		return topKCandidates(logits, topK, scratch)
	}
	candidates := makeTokenScores(len(logits), scratch)
	for i, v := range logits {
		candidates[i] = tokenScore{id: i, score: v}
	}
	slices.SortFunc(candidates, compareTokenScoreDesc)
	if topK > 0 && topK < len(candidates) {
		candidates = candidates[:topK]
	}
	return candidates
}

func topKCandidates(logits []float32, topK int, scratch *generationScratch) []tokenScore {
	candidates := topKCandidatesUnsorted(logits, topK, scratch)
	sortTokenScoresDesc(candidates)
	return candidates
}

func sortTokenScoresDesc(x []tokenScore) {
	if len(x) <= 32 {
		insertionSortTokenScoresDesc(x)
		return
	}
	slices.SortFunc(x, compareTokenScoreDesc)
}

func insertionSortTokenScoresDesc(x []tokenScore) {
	for i := 1; i < len(x); i++ {
		v := x[i]
		j := i - 1
		for ; j >= 0 && x[j].score < v.score; j-- {
			x[j+1] = x[j]
		}
		x[j+1] = v
	}
}

func compareTokenScoreDesc(a, b tokenScore) int {
	if a.score > b.score {
		return -1
	}
	if a.score < b.score {
		return 1
	}
	return 0
}

func topKCandidatesUnsorted(logits []float32, topK int, scratch *generationScratch) []tokenScore {
	candidates, _ := topKCandidatesUnsortedWithMax(logits, topK, scratch)
	return candidates
}

func topKCandidatesUnsortedWithMax(logits []float32, topK int, scratch *generationScratch) ([]tokenScore, float32) {
	candidates := makeTokenScores(topK, scratch)
	n := min(topK, len(logits))
	maxScore := float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		v := logits[i]
		candidates[i] = tokenScore{id: i, score: v}
		maxScore = max32Local(maxScore, v)
	}
	if n < topK {
		return candidates[:n], maxScore
	}
	buildMinTokenHeap(candidates[:n])
	rootScore := candidates[0].score
	i := topK
	for ; i+7 < len(logits); i += 8 {
		v0, v1, v2, v3 := logits[i], logits[i+1], logits[i+2], logits[i+3]
		v4, v5, v6, v7 := logits[i+4], logits[i+5], logits[i+6], logits[i+7]
		if v0 <= rootScore && v1 <= rootScore && v2 <= rootScore && v3 <= rootScore &&
			v4 <= rootScore && v5 <= rootScore && v6 <= rootScore && v7 <= rootScore {
			continue
		}
		if v0 > rootScore {
			candidates[0] = tokenScore{id: i, score: v0}
			maxScore = max32Local(maxScore, v0)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v1 > rootScore {
			candidates[0] = tokenScore{id: i + 1, score: v1}
			maxScore = max32Local(maxScore, v1)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v2 > rootScore {
			candidates[0] = tokenScore{id: i + 2, score: v2}
			maxScore = max32Local(maxScore, v2)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v3 > rootScore {
			candidates[0] = tokenScore{id: i + 3, score: v3}
			maxScore = max32Local(maxScore, v3)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v4 > rootScore {
			candidates[0] = tokenScore{id: i + 4, score: v4}
			maxScore = max32Local(maxScore, v4)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v5 > rootScore {
			candidates[0] = tokenScore{id: i + 5, score: v5}
			maxScore = max32Local(maxScore, v5)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v6 > rootScore {
			candidates[0] = tokenScore{id: i + 6, score: v6}
			maxScore = max32Local(maxScore, v6)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
		if v7 > rootScore {
			candidates[0] = tokenScore{id: i + 7, score: v7}
			maxScore = max32Local(maxScore, v7)
			siftDownTokenHeap(candidates[:n], 0)
			rootScore = candidates[0].score
		}
	}
	for ; i < len(logits); i++ {
		v := logits[i]
		if v <= rootScore {
			continue
		}
		candidates[0] = tokenScore{id: i, score: v}
		maxScore = max32Local(maxScore, v)
		siftDownTokenHeap(candidates[:n], 0)
		rootScore = candidates[0].score
	}
	return candidates[:n], maxScore
}

func buildMinTokenHeap(x []tokenScore) {
	for i := len(x)/2 - 1; i >= 0; i-- {
		siftDownTokenHeap(x, i)
	}
}

func siftDownTokenHeap(x []tokenScore, i int) {
	v := x[i]
	for {
		left := 2*i + 1
		if left >= len(x) {
			x[i] = v
			return
		}
		small := left
		if right := left + 1; right < len(x) && x[right].score < x[left].score {
			small = right
		}
		if v.score <= x[small].score {
			x[i] = v
			return
		}
		x[i] = x[small]
		i = small
	}
}

func makeTokenScores(n int, scratch *generationScratch) []tokenScore {
	if scratch == nil {
		return make([]tokenScore, n)
	}
	if cap(scratch.candidates) < n {
		scratch.candidates = make([]tokenScore, n)
	}
	return scratch.candidates[:n]
}

func makeSampleWeights(n int, scratch *generationScratch) []float32 {
	if scratch == nil {
		return nil
	}
	if cap(scratch.weights) < n {
		scratch.weights = make([]float32, n)
	}
	return scratch.weights[:n]
}

func (rt *Runtime) expandSingleImagePlaceholder(input []int, n int) []int {
	out, _ := rt.expandSingleImagePlaceholderInto(input, n, nil)
	return out
}

func (rt *Runtime) expandSingleImagePlaceholderInto(input []int, n int, buf []int) ([]int, bool) {
	count := 0
	for _, id := range input {
		if id == rt.cfg.ImageTokenID {
			count++
		}
	}
	if count != 1 || n <= 1 {
		return input, false
	}
	need := len(input) + n - 1
	out := buf[:0]
	if cap(out) < need {
		out = make([]int, 0, need)
	}
	for _, id := range input {
		if id == rt.cfg.ImageTokenID {
			for i := 0; i < n; i++ {
				out = append(out, id)
			}
			continue
		}
		out = append(out, id)
	}
	return out, true
}

func (rt *Runtime) preloadTextWeights() error {
	type preloadName struct {
		name        string
		quantizable bool
	}
	names := make([]preloadName, 0, 3+rt.cfg.NumHiddenLayers*9)
	names = append(names,
		preloadName{name: "model.embed_tokens.weight"},
		preloadName{name: "model.norm.weight"},
		preloadName{name: "lm_head.weight", quantizable: true},
	)
	for i := 0; i < rt.cfg.NumHiddenLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + "."
		names = append(names,
			preloadName{name: p + "input_layernorm.weight"},
			preloadName{name: p + "post_attention_layernorm.weight"},
			preloadName{name: p + "self_attn.q_proj.weight", quantizable: true},
			preloadName{name: p + "self_attn.k_proj.weight", quantizable: true},
			preloadName{name: p + "self_attn.v_proj.weight", quantizable: true},
			preloadName{name: p + "self_attn.o_proj.weight", quantizable: true},
			preloadName{name: p + "mlp.gate_proj.weight", quantizable: true},
			preloadName{name: p + "mlp.up_proj.weight", quantizable: true},
			preloadName{name: p + "mlp.down_proj.weight", quantizable: true},
		)
	}
	q8s, hasQ8 := rt.sf.(q8TensorStore)
	q4s, hasQ4 := rt.sf.(q4TensorStore)
	q6s, hasQ6 := rt.sf.(q6TensorStore)
	rs, hasRows := rt.sf.(rowTensorStore)
	var rowFloatBuf []float32
	var rowRawBuf []byte
	for i, item := range names {
		name := item.name
		if rt.progress != nil {
			rt.progress(i, len(names), name, "LOAD")
		}
		if rt.quantization == "q8" && item.quantizable {
			if hasQ8 {
				data, scales, shape, err := q8s.Q8Row(name)
				if err == nil {
					rt.qw[name] = &tensor.Q8Matrix{Rows: int(shape[0]), Cols: int(shape[1]), Data: data, Scale: scales}
					continue
				}
			}
		}
		if rt.quantization == "q4" && item.quantizable {
			if hasQ4 {
				data, scales, shape, err := q4s.Q4Row(name)
				if err == nil {
					rt.q4w[name] = &tensor.Q4Matrix{Rows: int(shape[0]), Cols: int(shape[1]), Data: data, Scale: scales}
					continue
				}
			}
		}
		if rt.quantization == "q6" && item.quantizable {
			if hasQ6 {
				data, scales, shape, err := q6s.Q6Row(name)
				if err == nil {
					rt.q6w[name] = &tensor.Q6Matrix{Rows: int(shape[0]), Cols: int(shape[1]), Data: data, Scale: scales}
					continue
				}
			}
		}
		if rt.quantization != "" && rt.quantization != "f32" && item.quantizable {
			if hasRows {
				if rt.quantizeRowsFromStore(name, rs, &rowFloatBuf, &rowRawBuf) == nil {
					continue
				}
			}
		}
		v, _, err := rt.sf.Float32(name)
		if err != nil {
			return err
		}
		rt.w[name] = v
	}
	if rt.progress != nil {
		rt.progress(len(names), len(names), "", "LOAD")
	}
	return nil
}

func (rt *Runtime) quantizeRowsFromStore(name string, rs rowTensorStore, rowFloatBuf *[]float32, rowRawBuf *[]byte) error {
	ss, ok := rt.sf.(shapeTensorStore)
	if !ok {
		return fmt.Errorf("row store lacks shape lookup")
	}
	shape, err := ss.Shape(name)
	if err != nil {
		return err
	}
	if len(shape) != 2 {
		return fmt.Errorf("%s must be 2D", name)
	}
	rows, cols := int(shape[0]), int(shape[1])
	if rowFloatBuf != nil {
		need := rowBufferFloats(rows, cols)
		if cap(*rowFloatBuf) < need {
			*rowFloatBuf = make([]float32, need)
		} else {
			*rowFloatBuf = (*rowFloatBuf)[:need]
		}
	}
	floatRows := func(fn func(row int, values []float32) error) ([]int64, error) {
		if bs, ok := rs.(rowTensorFloatBufferStore); ok && rowFloatBuf != nil {
			return bs.Float32RowsBuffer(name, *rowFloatBuf, fn)
		}
		if bs, ok := rs.(rowTensorRawBufferStore); ok && rowFloatBuf != nil {
			var raw []byte
			if rowRawBuf != nil && needsRawFloatDecodeBuffer(rt.sf, name) {
				needRaw := rowBufferRawBytes(rows, cols, 2)
				if cap(*rowRawBuf) < needRaw {
					*rowRawBuf = make([]byte, needRaw)
				} else {
					*rowRawBuf = (*rowRawBuf)[:needRaw]
				}
				raw = *rowRawBuf
			}
			return bs.Float32RowsBuffer(name, *rowFloatBuf, raw, fn)
		}
		return rs.Float32Rows(name, fn)
	}
	switch rt.quantization {
	case "q8":
		q := &tensor.Q8Matrix{Rows: rows, Cols: cols, Data: make([]int8, rows*cols), Scale: make([]float32, rows)}
		_, err = floatRows(func(row int, values []float32) error {
			base := row * cols
			q.Scale[row] = tensor.QuantizeQ8RowInto(values, q.Data[base:base+cols])
			return nil
		})
		if err == nil {
			rt.qw[name] = q
		}
	case "q6":
		packed := tensor.PackedQ6Cols(cols)
		q := &tensor.Q6Matrix{Rows: rows, Cols: cols, Data: make([]byte, rows*packed), Scale: make([]float32, rows)}
		_, err = floatRows(func(row int, values []float32) error {
			base := row * packed
			q.Scale[row] = tensor.QuantizeQ6RowInto(values, q.Data[base:base+packed])
			return nil
		})
		if err == nil {
			rt.q6w[name] = q
		}
	case "q4":
		packed := (cols + 1) / 2
		q := &tensor.Q4Matrix{Rows: rows, Cols: cols, Data: make([]byte, rows*packed), Scale: make([]float32, rows)}
		_, err = floatRows(func(row int, values []float32) error {
			base := row * packed
			q.Scale[row] = tensor.QuantizeQ4RowInto(values, q.Data[base:base+packed])
			return nil
		})
		if err == nil {
			rt.q4w[name] = q
		}
	default:
		return fmt.Errorf("unsupported quantization %q", rt.quantization)
	}
	return err
}

func needsRawFloatDecodeBuffer(ts tensorStore, name string) bool {
	ds, ok := ts.(dtypeTensorStore)
	if !ok {
		return true
	}
	dtype, ok := ds.DType(name)
	return !ok || dtype != "F32"
}

func rowBufferFloats(rows, cols int) int {
	if rows <= 0 || cols <= 0 {
		return 0
	}
	const targetBlockBytes = 1 << 20
	rowsPerBlock := targetBlockBytes / (cols * 4)
	if rowsPerBlock < 1 {
		rowsPerBlock = 1
	}
	if rowsPerBlock > rows {
		rowsPerBlock = rows
	}
	return rowsPerBlock * cols
}

func rowBufferRawBytes(rows, cols, elemSize int) int {
	if rows <= 0 || cols <= 0 || elemSize <= 0 {
		return 0
	}
	const targetBlockBytes = 1 << 20
	rowBytes := cols * elemSize
	rowsPerBlock := targetBlockBytes / rowBytes
	if rowsPerBlock < 1 {
		rowsPerBlock = 1
	}
	if rowsPerBlock > rows {
		rowsPerBlock = rows
	}
	return rowsPerBlock * rowBytes
}

func (rt *Runtime) cacheTextWeights() {
	rt.embed = rt.w["model.embed_tokens.weight"]
	rt.finalNorm = rt.w["model.norm.weight"]
	rt.lmHead = rt.w["lm_head.weight"]
	rt.q8LmHead = rt.qw["lm_head.weight"]
	rt.q6LmHead = rt.q6w["lm_head.weight"]
	rt.q4LmHead = rt.q4w["lm_head.weight"]
	rt.textLayers = make([]textLayer, rt.cfg.NumHiddenLayers)
	for i := range rt.textLayers {
		rt.textLayers[i] = textLayer{
			w:  rt.lw(i),
			q8: rt.qlw(i),
			q6: rt.q6lw(i),
			q4: rt.q4lw(i),
		}
	}
}

func (rt *Runtime) releaseCachedTextWeightMapEntries() {
	delete(rt.w, "model.embed_tokens.weight")
	delete(rt.w, "model.norm.weight")
	delete(rt.w, "lm_head.weight")
	if rt.cfg == nil {
		return
	}
	visionWeightCount := 11 + rt.cfg.VisionConfig.NumHiddenLayers*16
	if visionWeightCount < 0 {
		visionWeightCount = 0
	}
	if len(rt.w) == 0 {
		rt.w = make(map[string][]float32, visionWeightCount)
		return
	}
	for i := 0; i < rt.cfg.NumHiddenLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + "."
		delete(rt.w, p+"input_layernorm.weight")
		delete(rt.w, p+"post_attention_layernorm.weight")
		delete(rt.w, p+"self_attn.q_proj.weight")
		delete(rt.w, p+"self_attn.k_proj.weight")
		delete(rt.w, p+"self_attn.v_proj.weight")
		delete(rt.w, p+"self_attn.o_proj.weight")
		delete(rt.w, p+"mlp.gate_proj.weight")
		delete(rt.w, p+"mlp.up_proj.weight")
		delete(rt.w, p+"mlp.down_proj.weight")
	}
	if len(rt.w) == 0 {
		rt.w = make(map[string][]float32, visionWeightCount)
	}
}

func (rt *Runtime) lw(i int) layerWeights {
	p := "model.layers." + strconv.Itoa(i) + "."
	return layerWeights{
		ln1:  rt.w[p+"input_layernorm.weight"],
		ln2:  rt.w[p+"post_attention_layernorm.weight"],
		q:    rt.w[p+"self_attn.q_proj.weight"],
		k:    rt.w[p+"self_attn.k_proj.weight"],
		v:    rt.w[p+"self_attn.v_proj.weight"],
		o:    rt.w[p+"self_attn.o_proj.weight"],
		gate: rt.w[p+"mlp.gate_proj.weight"],
		up:   rt.w[p+"mlp.up_proj.weight"],
		down: rt.w[p+"mlp.down_proj.weight"],
	}
}

func (rt *Runtime) qlw(i int) qLayerWeights {
	p := "model.layers." + strconv.Itoa(i) + "."
	return qLayerWeights{
		q:    rt.qw[p+"self_attn.q_proj.weight"],
		k:    rt.qw[p+"self_attn.k_proj.weight"],
		v:    rt.qw[p+"self_attn.v_proj.weight"],
		o:    rt.qw[p+"self_attn.o_proj.weight"],
		gate: rt.qw[p+"mlp.gate_proj.weight"],
		up:   rt.qw[p+"mlp.up_proj.weight"],
		down: rt.qw[p+"mlp.down_proj.weight"],
	}
}

func (rt *Runtime) q4lw(i int) q4LayerWeights {
	p := "model.layers." + strconv.Itoa(i) + "."
	return q4LayerWeights{
		q:    rt.q4w[p+"self_attn.q_proj.weight"],
		k:    rt.q4w[p+"self_attn.k_proj.weight"],
		v:    rt.q4w[p+"self_attn.v_proj.weight"],
		o:    rt.q4w[p+"self_attn.o_proj.weight"],
		gate: rt.q4w[p+"mlp.gate_proj.weight"],
		up:   rt.q4w[p+"mlp.up_proj.weight"],
		down: rt.q4w[p+"mlp.down_proj.weight"],
	}
}

func (rt *Runtime) q6lw(i int) q6LayerWeights {
	p := "model.layers." + strconv.Itoa(i) + "."
	return q6LayerWeights{
		q:    rt.q6w[p+"self_attn.q_proj.weight"],
		k:    rt.q6w[p+"self_attn.k_proj.weight"],
		v:    rt.q6w[p+"self_attn.v_proj.weight"],
		o:    rt.q6w[p+"self_attn.o_proj.weight"],
		gate: rt.q6w[p+"mlp.gate_proj.weight"],
		up:   rt.q6w[p+"mlp.up_proj.weight"],
		down: rt.q6w[p+"mlp.down_proj.weight"],
	}
}

func (rt *Runtime) quantizeTextWeights() {
	if rt.qw["lm_head.weight"] == nil {
		rt.qw["lm_head.weight"] = tensor.QuantizeQ8Row(rt.w["lm_head.weight"], rt.cfg.VocabSize, rt.cfg.HiddenSize)
	}
	delete(rt.w, "lm_head.weight")
	for i := 0; i < rt.cfg.NumHiddenLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + "."
		quantizeMissing := func(name string, rows, cols int) {
			if rt.qw[name] == nil {
				rt.qw[name] = tensor.QuantizeQ8Row(rt.w[name], rows, cols)
			}
			delete(rt.w, name)
		}
		quantizeMissing(p+"self_attn.q_proj.weight", rt.cfg.NumAttentionHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.k_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.v_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.o_proj.weight", rt.cfg.HiddenSize, rt.cfg.NumAttentionHeads*rt.cfg.HeadDim)
		quantizeMissing(p+"mlp.gate_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.up_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.down_proj.weight", rt.cfg.HiddenSize, rt.cfg.IntermediateSize)
	}
}

func (rt *Runtime) quantizeTextWeightsQ6() {
	if rt.q6w["lm_head.weight"] == nil {
		rt.q6w["lm_head.weight"] = tensor.QuantizeQ6Row(rt.w["lm_head.weight"], rt.cfg.VocabSize, rt.cfg.HiddenSize)
	}
	delete(rt.w, "lm_head.weight")
	for i := 0; i < rt.cfg.NumHiddenLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + "."
		quantizeMissing := func(name string, rows, cols int) {
			if rt.q6w[name] == nil {
				rt.q6w[name] = tensor.QuantizeQ6Row(rt.w[name], rows, cols)
			}
			delete(rt.w, name)
		}
		quantizeMissing(p+"self_attn.q_proj.weight", rt.cfg.NumAttentionHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.k_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.v_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.o_proj.weight", rt.cfg.HiddenSize, rt.cfg.NumAttentionHeads*rt.cfg.HeadDim)
		quantizeMissing(p+"mlp.gate_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.up_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.down_proj.weight", rt.cfg.HiddenSize, rt.cfg.IntermediateSize)
	}
}

func (rt *Runtime) quantizeTextWeightsQ4() {
	if rt.q4w["lm_head.weight"] == nil {
		rt.q4w["lm_head.weight"] = tensor.QuantizeQ4Row(rt.w["lm_head.weight"], rt.cfg.VocabSize, rt.cfg.HiddenSize)
	}
	delete(rt.w, "lm_head.weight")
	for i := 0; i < rt.cfg.NumHiddenLayers; i++ {
		p := "model.layers." + strconv.Itoa(i) + "."
		quantizeMissing := func(name string, rows, cols int) {
			if rt.q4w[name] == nil {
				rt.q4w[name] = tensor.QuantizeQ4Row(rt.w[name], rows, cols)
			}
			delete(rt.w, name)
		}
		quantizeMissing(p+"self_attn.q_proj.weight", rt.cfg.NumAttentionHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.k_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.v_proj.weight", rt.cfg.NumKeyValueHeads*rt.cfg.HeadDim, rt.cfg.HiddenSize)
		quantizeMissing(p+"self_attn.o_proj.weight", rt.cfg.HiddenSize, rt.cfg.NumAttentionHeads*rt.cfg.HeadDim)
		quantizeMissing(p+"mlp.gate_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.up_proj.weight", rt.cfg.IntermediateSize, rt.cfg.HiddenSize)
		quantizeMissing(p+"mlp.down_proj.weight", rt.cfg.HiddenSize, rt.cfg.IntermediateSize)
	}
}

func (rt *Runtime) tokenEmbedding(id int) ([]float32, error) {
	c := rt.cfg
	if id < 0 || id >= c.VocabSize {
		return nil, fmt.Errorf("token id %d outside vocab", id)
	}
	h := make([]float32, c.HiddenSize)
	copy(h, rt.embed[id*c.HiddenSize:(id+1)*c.HiddenSize])
	return h, nil
}

func (rt *Runtime) tokenEmbeddingInto(out []float32, id int) error {
	c := rt.cfg
	if id < 0 || id >= c.VocabSize {
		return fmt.Errorf("token id %d outside vocab", id)
	}
	copy(out[:c.HiddenSize], rt.embed[id*c.HiddenSize:(id+1)*c.HiddenSize])
	return nil
}

func (rt *Runtime) newKVCaches(maxTokens int) []kvCache {
	c := rt.cfg
	kvDim := c.NumKeyValueHeads * c.HeadDim
	out := make([]kvCache, c.NumHiddenLayers)
	if maxTokens <= 0 || kvDim <= 0 {
		return out
	}
	for i := range out {
		out[i] = kvCache{
			k:     make([]float32, 0, maxTokens*kvDim),
			v:     make([]float32, 0, maxTokens*kvDim),
			kvDim: kvDim,
		}
	}
	return out
}

func (rt *Runtime) getKVCaches(maxTokens int) ([]kvCache, *[]kvCache) {
	v := rt.kvPool.Get()
	if v == nil {
		caches := rt.newKVCaches(maxTokens)
		return caches, &caches
	}
	var caches []kvCache
	var cachePtr *[]kvCache
	switch x := v.(type) {
	case *[]kvCache:
		cachePtr = x
		caches = *x
	case []kvCache:
		caches = x
		cachePtr = &caches
	default:
		caches = rt.newKVCaches(maxTokens)
		return caches, &caches
	}
	if len(caches) != rt.cfg.NumHiddenLayers {
		caches = rt.newKVCaches(maxTokens)
	}
	kvDim := rt.cfg.NumKeyValueHeads * rt.cfg.HeadDim
	need := maxTokens * kvDim
	for i := range caches {
		if cap(caches[i].k) < need {
			caches[i].k = make([]float32, 0, need)
		} else {
			caches[i].k = caches[i].k[:0]
		}
		if cap(caches[i].v) < need {
			caches[i].v = make([]float32, 0, need)
		} else {
			caches[i].v = caches[i].v[:0]
		}
		caches[i].len = 0
		caches[i].kvDim = kvDim
	}
	return caches, cachePtr
}

func (rt *Runtime) putKVCaches(caches []kvCache, cachePtr *[]kvCache) {
	if cachePtr == nil || len(caches) != rt.cfg.NumHiddenLayers {
		return
	}
	kvDim := rt.cfg.NumKeyValueHeads * rt.cfg.HeadDim
	const maxPooledKVTokens = 8192
	for i := range caches {
		if kvDim > 0 && cap(caches[i].k)/kvDim > maxPooledKVTokens {
			caches[i].k = nil
			caches[i].v = nil
		}
		caches[i].len = 0
	}
	*cachePtr = caches
	rt.kvPool.Put(cachePtr)
}

func (rt *Runtime) newGenerationScratch() *generationScratch {
	c := rt.cfg
	return &generationScratch{
		layers:  rt.newLayerScratch(),
		hidden:  make([]float32, c.HiddenSize),
		norm:    make([]float32, c.HiddenSize),
		logits:  make([]float32, c.VocabSize),
		ropeCos: make([]float32, c.HeadDim/2),
		ropeSin: make([]float32, c.HeadDim/2),
	}
}

func (rt *Runtime) getGenerationScratch() *generationScratch {
	v := rt.scratchPool.Get()
	if v == nil {
		return rt.newGenerationScratch()
	}
	scratch := v.(*generationScratch)
	for i := range scratch.layers {
		scratch.layers[i].scores = scratch.layers[i].scores[:0]
	}
	return scratch
}

func (rt *Runtime) putGenerationScratch(scratch *generationScratch) {
	if scratch == nil {
		return
	}
	const maxPooledTopKCandidates = 8192
	if cap(scratch.candidates) > maxPooledTopKCandidates {
		scratch.candidates = nil
	}
	const maxPooledScoreTokens = 8192
	if layers := len(scratch.layers); layers > 0 && cap(scratch.scoreBlock)/layers > maxPooledScoreTokens {
		scratch.scoreBlock = nil
		for i := range scratch.layers {
			scratch.layers[i].scores = nil
		}
	}
	const maxPooledGenerationTokens = 8192
	if cap(scratch.positions) > maxPooledGenerationTokens {
		scratch.positions = nil
	}
	if cap(scratch.inputIDs) > maxPooledGenerationTokens {
		scratch.inputIDs = nil
	}
	maxPooledWeights := max(rt.cfg.VocabSize, maxPooledGenerationTokens)
	if cap(scratch.weights) > maxPooledWeights {
		scratch.weights = nil
	}
	rt.scratchPool.Put(scratch)
}

func (rt *Runtime) putGenerationResources(scratch *generationScratch, caches []kvCache, cachePtr *[]kvCache) {
	rt.putGenerationScratch(scratch)
	rt.putKVCaches(caches, cachePtr)
}

func (rt *Runtime) ensureScoreCapacity(scratch *generationScratch, tokens int) {
	layers := len(scratch.layers)
	if layers == 0 || tokens <= 0 {
		return
	}
	need := layers * tokens
	if cap(scratch.scoreBlock) < need {
		scratch.scoreBlock = make([]float32, need)
	}
	block := scratch.scoreBlock[:need]
	for i := range scratch.layers {
		start := i * tokens
		scratch.layers[i].scores = block[start : start : start+tokens]
	}
}

func (rt *Runtime) newScratch() []layerScratch {
	return rt.newLayerScratch()
}

func (rt *Runtime) newLayerScratch() []layerScratch {
	c := rt.cfg
	out := make([]layerScratch, c.NumHiddenLayers)
	qRows := c.NumAttentionHeads * c.HeadDim
	kvRows := c.NumKeyValueHeads * c.HeadDim
	for i := range out {
		out[i] = layerScratch{
			norm:    make([]float32, c.HiddenSize),
			att:     make([]float32, c.HiddenSize),
			mlp:     make([]float32, c.HiddenSize),
			q:       make([]float32, qRows),
			k:       make([]float32, kvRows),
			v:       make([]float32, kvRows),
			headOut: make([]float32, qRows),
			gate:    make([]float32, c.IntermediateSize),
		}
	}
	return out
}

func (rt *Runtime) forwardEmbedding(embedding []float32, pos ropePos, caches []kvCache, scratch *generationScratch) ([]float32, error) {
	c := rt.cfg
	h := scratch.hidden[:c.HiddenSize]
	if len(embedding) < c.HiddenSize {
		return nil, fmt.Errorf("embedding length %d smaller than hidden size %d", len(embedding), c.HiddenSize)
	}
	if len(h) == 0 || len(embedding) == 0 || &h[0] != &embedding[0] {
		copy(h, embedding[:c.HiddenSize])
	}
	layers := scratch.layers
	hasRoPE := pos.T != 0 || pos.H != 0 || pos.W != 0
	ropeCos := scratch.ropeCos[:c.HeadDim/2]
	ropeSin := scratch.ropeSin[:c.HeadDim/2]
	if hasRoPE {
		rt.buildMRoPETable(ropeCos, ropeSin, pos)
	}

	for li := 0; li < c.NumHiddenLayers; li++ {
		tl := &rt.textLayers[li]
		sc := &layers[li]
		if li == 0 {
			tensor.RMSNorm(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
		}
		att := rt.attention(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin)
		tensor.AddRMSNorm(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
		mlp := rt.mlp(sc.norm, tl, sc)
		if li+1 < c.NumHiddenLayers {
			tensor.AddRMSNorm(layers[li+1].norm[:c.HiddenSize], h, mlp, rt.textLayers[li+1].w.ln1, float32(c.RMSNormEps))
		} else {
			tensor.AddRMSNorm(scratch.norm[:c.HiddenSize], h, mlp, rt.finalNorm, float32(c.RMSNormEps))
		}
	}
	norm := scratch.norm[:c.HiddenSize]
	if c.NumHiddenLayers == 0 {
		tensor.RMSNorm(norm, h, rt.finalNorm, float32(c.RMSNormEps))
	}
	logits := scratch.logits[:c.VocabSize]
	if q := rt.q8LmHead; q != nil {
		tensor.MatVecQ8(logits, norm, q)
	} else if q := rt.q6LmHead; q != nil {
		tensor.MatVecQ6(logits, norm, q)
	} else if q := rt.q4LmHead; q != nil {
		tensor.MatVecQ4(logits, norm, q)
	} else {
		tensor.MatVec(logits, norm, rt.lmHead, c.VocabSize, c.HiddenSize)
	}
	return logits, nil
}

func (rt *Runtime) attention(x []float32, cache *kvCache, tl *textLayer, sc *layerScratch, hasRoPE bool, ropeCos, ropeSin []float32) []float32 {
	c := rt.cfg
	qRows := c.NumAttentionHeads * c.HeadDim
	kvRows := c.NumKeyValueHeads * c.HeadDim
	q := sc.q[:qRows]
	k := sc.k[:kvRows]
	v := sc.v[:kvRows]
	if !fusedQKV(q, k, v, x, tl, qRows, kvRows, c.HiddenSize) {
		matVecMaybeQuant(q, x, tl.w.q, tl.q8.q, tl.q6.q, tl.q4.q, qRows, c.HiddenSize)
		matVecMaybeQuant(k, x, tl.w.k, tl.q8.k, tl.q6.k, tl.q4.k, kvRows, c.HiddenSize)
		matVecMaybeQuant(v, x, tl.w.v, tl.q8.v, tl.q6.v, tl.q4.v, kvRows, c.HiddenSize)
	}
	if hasRoPE {
		applyMRoPEWithTable(q, c.NumAttentionHeads, c.HeadDim, ropeCos, ropeSin)
		applyMRoPEWithTable(k, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin)
	}
	cache.append(k, v)

	headOut := sc.headOut[:qRows]
	group := c.NumAttentionHeads / c.NumKeyValueHeads
	scale := invSqrt(c.HeadDim)
	for h := 0; h < c.NumAttentionHeads; h++ {
		kvh := h / group
		dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
		if cache.len == 1 {
			copyCacheValue(dst, cache, kvh, c.HeadDim)
			continue
		}
		qv := q[h*c.HeadDim : (h+1)*c.HeadDim]
		if cache.len == 2 {
			cacheAttentionLen2(dst, qv, cache, kvh, c.HeadDim, scale)
			continue
		}
		if cache.len == 3 {
			cacheAttentionLen3(dst, qv, cache, kvh, c.HeadDim, scale)
			continue
		}
		if cache.len == 4 && (c.HeadDim == 128 || c.HeadDim == 64) {
			cacheAttentionLen4(dst, qv, cache, kvh, c.HeadDim, scale)
			continue
		}
		if cap(sc.scores) < cache.len {
			sc.scores = make([]float32, cache.len)
		}
		scores := sc.scores[:cache.len]
		cacheAttentionScores(scores, qv, cache, kvh, c.HeadDim, scale)
		tensor.SoftmaxInPlace(scores)
		weightedCacheValueSum(dst, cache, kvh, c.HeadDim, scores)
	}
	out := sc.att[:c.HiddenSize]
	matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
	return out
}

func copyCacheValue(dst []float32, cache *kvCache, head, dim int) {
	base := head * dim
	copy(dst[:dim], cache.v[base:base+dim])
}

func cacheAttentionScores(scores, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	if len(scores) == 1 {
		if dim == 128 {
			scores[0] = dotAt128(q, cache.k, headBase) * scale
		} else if dim == 64 {
			scores[0] = dotAt64(q, cache.k, headBase) * scale
		} else {
			scores[0] = dotAt(q, cache.k, headBase, dim) * scale
		}
		return
	}
	if dim == 128 {
		t := 0
		for ; t+3 < len(scores); t += 4 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			base2 := base1 + cache.kvDim
			base3 := base2 + cache.kvDim
			s0, s1, s2, s3 := dotAt128QuadAt(q, cache.k, base0, base1, base2, base3)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
		}
		for ; t+1 < len(scores); t += 2 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			s0, s1 := dotAt128PairAt(q, cache.k, base0, base1)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
		}
		for ; t < len(scores); t++ {
			base := t*cache.kvDim + headBase
			scores[t] = dotAt128(q, cache.k, base) * scale
		}
		return
	}
	if dim == 64 {
		t := 0
		for ; t+3 < len(scores); t += 4 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			base2 := base1 + cache.kvDim
			base3 := base2 + cache.kvDim
			s0, s1, s2, s3 := dotAt64QuadAt(q, cache.k, base0, base1, base2, base3)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
		}
		for ; t+1 < len(scores); t += 2 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			s0, s1 := dotAt64PairAt(q, cache.k, base0, base1)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
		}
		for ; t < len(scores); t++ {
			base := t*cache.kvDim + headBase
			scores[t] = dotAt64(q, cache.k, base) * scale
		}
		return
	}
	for t := 0; t < len(scores); t++ {
		base := t*cache.kvDim + headBase
		scores[t] = dotAt(q, cache.k, base, dim) * scale
	}
}

func cacheAttentionSmall(dst, q []float32, cache *kvCache, head, dim int, scale float32) bool {
	switch cache.len {
	case 2:
		cacheAttentionLen2(dst, q, cache, head, dim, scale)
		return true
	case 3:
		cacheAttentionLen3(dst, q, cache, head, dim, scale)
		return true
	case 4:
		cacheAttentionLen4(dst, q, cache, head, dim, scale)
		return true
	default:
		return false
	}
}

func cacheAttentionLen3(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	base2 := base1 + cache.kvDim
	var s0, s1 float32
	if dim == 128 {
		s0, s1 = dotAt128PairAt(q, cache.k, headBase, base1)
	} else if dim == 64 {
		s0, s1 = dotAt64PairAt(q, cache.k, headBase, base1)
	} else {
		s0 = dotAt(q, cache.k, headBase, dim)
		s1 = dotAt(q, cache.k, base1, dim)
	}
	var s2 float32
	if dim == 128 {
		s2 = dotAt128(q, cache.k, base2)
	} else if dim == 64 {
		s2 = dotAt64(q, cache.k, base2)
	} else {
		s2 = dotAt(q, cache.k, base2, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	m := max32Local(max32Local(s0, s1), s2)
	e0 := float32(math.Exp(float64(s0 - m)))
	e1 := float32(math.Exp(float64(s1 - m)))
	e2 := float32(math.Exp(float64(s2 - m)))
	inv := 1 / (e0 + e1 + e2)
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	x2 := cache.v[base2 : base2+dim]
	w0, w1, w2 := e0*inv, e1*inv, e2*inv
	if dim == 128 {
		weightedValueSum3_128(dst, x0, x1, x2, w0, w1, w2)
		return
	}
	if dim == 64 {
		weightedValueSum3_64(dst, x0, x1, x2, w0, w1, w2)
		return
	}
	weightedCacheValueSum3(dst, x0, x1, x2, w0, w1, w2, dim)
}

func cacheAttentionLen4(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	base2 := base1 + cache.kvDim
	base3 := base2 + cache.kvDim
	var s0, s1, s2, s3 float32
	if dim == 128 {
		s0, s1, s2, s3 = dotAt128QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 64 {
		s0, s1, s2, s3 = dotAt64QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else {
		s0 = dotAt(q, cache.k, headBase, dim)
		s1 = dotAt(q, cache.k, base1, dim)
		s2 = dotAt(q, cache.k, base2, dim)
		s3 = dotAt(q, cache.k, base3, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	s3 *= scale
	m := max32Local(max32Local(s0, s1), max32Local(s2, s3))
	e0 := float32(math.Exp(float64(s0 - m)))
	e1 := float32(math.Exp(float64(s1 - m)))
	e2 := float32(math.Exp(float64(s2 - m)))
	e3 := float32(math.Exp(float64(s3 - m)))
	inv := 1 / ((e0 + e1) + (e2 + e3))
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	x2 := cache.v[base2 : base2+dim]
	x3 := cache.v[base3 : base3+dim]
	w0, w1, w2, w3 := e0*inv, e1*inv, e2*inv, e3*inv
	if dim == 128 {
		weightedValueSum4_128(dst, x0, x1, x2, x3, w0, w1, w2, w3)
		return
	}
	if dim == 64 {
		weightedValueSum4_64(dst, x0, x1, x2, x3, w0, w1, w2, w3)
		return
	}
	weightedCacheValueSum4(dst, x0, x1, x2, x3, w0, w1, w2, w3, dim)
}

func cacheAttentionLen2(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	var s0, s1 float32
	if dim == 128 {
		s0, s1 = dotAt128PairAt(q, cache.k, headBase, base1)
	} else if dim == 64 {
		s0, s1 = dotAt64PairAt(q, cache.k, headBase, base1)
	} else {
		s0 = dotAt(q, cache.k, headBase, dim)
		s1 = dotAt(q, cache.k, base1, dim)
	}
	s0 *= scale
	s1 *= scale
	var w0, w1 float32
	if s0 >= s1 {
		e := float32(math.Exp(float64(s1 - s0)))
		inv := 1 / (1 + e)
		w0 = inv
		w1 = e * inv
	} else {
		e := float32(math.Exp(float64(s0 - s1)))
		inv := 1 / (1 + e)
		w0 = e * inv
		w1 = inv
	}
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	if dim == 128 {
		weightedValueSum2_128(dst, x0, x1, w0, w1)
		return
	}
	if dim == 64 {
		weightedValueSum2_64(dst, x0, x1, w0, w1)
		return
	}
	weightedCacheValueSum2(dst, x0, x1, w0, w1, dim)
}

func dotAt(a, b []float32, offset, n int) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+15 < n; i += 16 {
		j := offset + i
		s0 += a[i]*b[j] + a[i+8]*b[j+8]
		s1 += a[i+1]*b[j+1] + a[i+9]*b[j+9]
		s2 += a[i+2]*b[j+2] + a[i+10]*b[j+10]
		s3 += a[i+3]*b[j+3] + a[i+11]*b[j+11]
		s4 += a[i+4]*b[j+4] + a[i+12]*b[j+12]
		s5 += a[i+5]*b[j+5] + a[i+13]*b[j+13]
		s6 += a[i+6]*b[j+6] + a[i+14]*b[j+14]
		s7 += a[i+7]*b[j+7] + a[i+15]*b[j+15]
	}
	sum := (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
	for ; i < n; i++ {
		sum += a[i] * b[offset+i]
	}
	return sum
}

func dotAt128(a, b []float32, offset int) float32 {
	b = b[offset : offset+128]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for i := 0; i < 128; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

func dotAt64(a, b []float32, offset int) float32 {
	b = b[offset : offset+64]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	for i := 0; i < 64; i += 16 {
		s0 += a[i]*b[i] + a[i+8]*b[i+8]
		s1 += a[i+1]*b[i+1] + a[i+9]*b[i+9]
		s2 += a[i+2]*b[i+2] + a[i+10]*b[i+10]
		s3 += a[i+3]*b[i+3] + a[i+11]*b[i+11]
		s4 += a[i+4]*b[i+4] + a[i+12]*b[i+12]
		s5 += a[i+5]*b[i+5] + a[i+13]*b[i+13]
		s6 += a[i+6]*b[i+6] + a[i+14]*b[i+14]
		s7 += a[i+7]*b[i+7] + a[i+15]*b[i+15]
	}
	return (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

func dotAt128PairAt(a, b []float32, offset0, offset1 int) (float32, float32) {
	b0 := b[offset0 : offset0+128]
	b1 := b[offset1 : offset1+128]
	var s00, s01, s02, s03, s10, s11, s12, s13 float32
	for i := 0; i < 128; i += 8 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		a4, a5, a6, a7 := a[i+4], a[i+5], a[i+6], a[i+7]
		s00 += a0*b0[i] + a4*b0[i+4]
		s01 += a1*b0[i+1] + a5*b0[i+5]
		s02 += a2*b0[i+2] + a6*b0[i+6]
		s03 += a3*b0[i+3] + a7*b0[i+7]
		s10 += a0*b1[i] + a4*b1[i+4]
		s11 += a1*b1[i+1] + a5*b1[i+5]
		s12 += a2*b1[i+2] + a6*b1[i+6]
		s13 += a3*b1[i+3] + a7*b1[i+7]
	}
	return (s00 + s01) + (s02 + s03), (s10 + s11) + (s12 + s13)
}

func dotAt64PairAt(a, b []float32, offset0, offset1 int) (float32, float32) {
	b0 := b[offset0 : offset0+64]
	b1 := b[offset1 : offset1+64]
	var s00, s01, s02, s03, s10, s11, s12, s13 float32
	for i := 0; i < 64; i += 8 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		a4, a5, a6, a7 := a[i+4], a[i+5], a[i+6], a[i+7]
		s00 += a0*b0[i] + a4*b0[i+4]
		s01 += a1*b0[i+1] + a5*b0[i+5]
		s02 += a2*b0[i+2] + a6*b0[i+6]
		s03 += a3*b0[i+3] + a7*b0[i+7]
		s10 += a0*b1[i] + a4*b1[i+4]
		s11 += a1*b1[i+1] + a5*b1[i+5]
		s12 += a2*b1[i+2] + a6*b1[i+6]
		s13 += a3*b1[i+3] + a7*b1[i+7]
	}
	return (s00 + s01) + (s02 + s03), (s10 + s11) + (s12 + s13)
}

func dotAt128QuadAt(a, b []float32, offset0, offset1, offset2, offset3 int) (float32, float32, float32, float32) {
	b0 := b[offset0 : offset0+128]
	b1 := b[offset1 : offset1+128]
	b2 := b[offset2 : offset2+128]
	b3 := b[offset3 : offset3+128]
	var s00, s01, s10, s11, s20, s21, s30, s31 float32
	for i := 0; i < 128; i += 8 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		a4, a5, a6, a7 := a[i+4], a[i+5], a[i+6], a[i+7]
		s00 += a0*b0[i] + a1*b0[i+1] + a2*b0[i+2] + a3*b0[i+3]
		s01 += a4*b0[i+4] + a5*b0[i+5] + a6*b0[i+6] + a7*b0[i+7]
		s10 += a0*b1[i] + a1*b1[i+1] + a2*b1[i+2] + a3*b1[i+3]
		s11 += a4*b1[i+4] + a5*b1[i+5] + a6*b1[i+6] + a7*b1[i+7]
		s20 += a0*b2[i] + a1*b2[i+1] + a2*b2[i+2] + a3*b2[i+3]
		s21 += a4*b2[i+4] + a5*b2[i+5] + a6*b2[i+6] + a7*b2[i+7]
		s30 += a0*b3[i] + a1*b3[i+1] + a2*b3[i+2] + a3*b3[i+3]
		s31 += a4*b3[i+4] + a5*b3[i+5] + a6*b3[i+6] + a7*b3[i+7]
	}
	return s00 + s01, s10 + s11, s20 + s21, s30 + s31
}

func dotAt64QuadAt(a, b []float32, offset0, offset1, offset2, offset3 int) (float32, float32, float32, float32) {
	b0 := b[offset0 : offset0+64]
	b1 := b[offset1 : offset1+64]
	b2 := b[offset2 : offset2+64]
	b3 := b[offset3 : offset3+64]
	var s00, s01, s10, s11, s20, s21, s30, s31 float32
	for i := 0; i < 64; i += 8 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		a4, a5, a6, a7 := a[i+4], a[i+5], a[i+6], a[i+7]
		s00 += a0*b0[i] + a1*b0[i+1] + a2*b0[i+2] + a3*b0[i+3]
		s01 += a4*b0[i+4] + a5*b0[i+5] + a6*b0[i+6] + a7*b0[i+7]
		s10 += a0*b1[i] + a1*b1[i+1] + a2*b1[i+2] + a3*b1[i+3]
		s11 += a4*b1[i+4] + a5*b1[i+5] + a6*b1[i+6] + a7*b1[i+7]
		s20 += a0*b2[i] + a1*b2[i+1] + a2*b2[i+2] + a3*b2[i+3]
		s21 += a4*b2[i+4] + a5*b2[i+5] + a6*b2[i+6] + a7*b2[i+7]
		s30 += a0*b3[i] + a1*b3[i+1] + a2*b3[i+2] + a3*b3[i+3]
		s31 += a4*b3[i+4] + a5*b3[i+5] + a6*b3[i+6] + a7*b3[i+7]
	}
	return s00 + s01, s10 + s11, s20 + s21, s30 + s31
}

func weightedCacheValueSum(dst []float32, cache *kvCache, head, dim int, weights []float32) {
	if len(weights) == 0 {
		clear(dst[:dim])
		return
	}
	headBase := head * dim
	if len(weights) == 1 {
		copy(dst[:dim], cache.v[headBase:headBase+dim])
		return
	}
	if len(weights) == 2 {
		a0, a1 := weights[0], weights[1]
		base1 := cache.kvDim + headBase
		x0 := cache.v[headBase : headBase+dim]
		x1 := cache.v[base1 : base1+dim]
		if dim == 128 {
			weightedValueSum2_128(dst, x0, x1, a0, a1)
			return
		}
		if dim == 64 {
			weightedValueSum2_64(dst, x0, x1, a0, a1)
			return
		}
		weightedCacheValueSum2(dst, x0, x1, a0, a1, dim)
		return
	}
	if len(weights) == 3 {
		a0, a1, a2 := weights[0], weights[1], weights[2]
		base1 := cache.kvDim + headBase
		base2 := base1 + cache.kvDim
		x0 := cache.v[headBase : headBase+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		if dim == 128 {
			weightedValueSum3_128(dst, x0, x1, x2, a0, a1, a2)
			return
		}
		if dim == 64 {
			weightedValueSum3_64(dst, x0, x1, x2, a0, a1, a2)
			return
		}
		weightedCacheValueSum3(dst, x0, x1, x2, a0, a1, a2, dim)
		return
	}
	if len(weights) == 4 {
		a0, a1, a2, a3 := weights[0], weights[1], weights[2], weights[3]
		base1 := cache.kvDim + headBase
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[headBase : headBase+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		x3 := cache.v[base3 : base3+dim]
		if dim == 128 {
			weightedValueSum4_128(dst, x0, x1, x2, x3, a0, a1, a2, a3)
			return
		}
		if dim == 64 {
			weightedValueSum4_64(dst, x0, x1, x2, x3, a0, a1, a2, a3)
			return
		}
		weightedCacheValueSum4(dst, x0, x1, x2, x3, a0, a1, a2, a3, dim)
		return
	}
	if dim == 64 {
		weightedCacheValueSum64(dst, cache, head, weights)
		return
	}
	if dim == 128 {
		weightedCacheValueSum128(dst, cache, head, weights)
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+dim]
	i := 0
	for ; i+7 < dim; i += 8 {
		dst[i] = a * x[i]
		dst[i+1] = a * x[i+1]
		dst[i+2] = a * x[i+2]
		dst[i+3] = a * x[i+3]
		dst[i+4] = a * x[i+4]
		dst[i+5] = a * x[i+5]
		dst[i+6] = a * x[i+6]
		dst[i+7] = a * x[i+7]
	}
	for ; i < dim; i++ {
		dst[i] = a * x[i]
	}
	t := 1
	for ; t+3 < len(weights); t += 4 {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		x3 := cache.v[base3 : base3+dim]
		i := 0
		for ; i+7 < dim; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
		}
		for ; i < dim; i++ {
			dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		}
	}
	for ; t+1 < len(weights); t += 2 {
		a0, a1 := weights[t], weights[t+1]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+dim]
		x1 := cache.v[base1 : base1+dim]
		i := 0
		for ; i+7 < dim; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7]
		}
		for ; i < dim; i++ {
			dst[i] += a0*x0[i] + a1*x1[i]
		}
	}
	for ; t < len(weights); t++ {
		a := weights[t]
		base := t*cache.kvDim + headBase
		x := cache.v[base : base+dim]
		i := 0
		for ; i+7 < dim; i += 8 {
			dst[i] += a * x[i]
			dst[i+1] += a * x[i+1]
			dst[i+2] += a * x[i+2]
			dst[i+3] += a * x[i+3]
			dst[i+4] += a * x[i+4]
			dst[i+5] += a * x[i+5]
			dst[i+6] += a * x[i+6]
			dst[i+7] += a * x[i+7]
		}
		for ; i < dim; i++ {
			dst[i] += a * x[i]
		}
	}
}

func weightedCacheValueSum2(dst, x0, x1 []float32, a0, a1 float32, dim int) {
	i := 0
	for ; i+7 < dim; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7]
	}
	for ; i < dim; i++ {
		dst[i] = a0*x0[i] + a1*x1[i]
	}
}

func weightedValueSum2_128(dst, x0, x1 []float32, a0, a1 float32) {
	for i := 0; i < 128; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7]
	}
}

func weightedValueSum2_64(dst, x0, x1 []float32, a0, a1 float32) {
	for i := 0; i < 64; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7]
	}
}

func weightedValueSum3_128(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	for i := 0; i < 128; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7]
	}
}

func weightedValueSum3_64(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	for i := 0; i < 64; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7]
	}
}

func weightedCacheValueSum3(dst, x0, x1, x2 []float32, a0, a1, a2 float32, dim int) {
	i := 0
	for ; i+7 < dim; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7]
	}
	for ; i < dim; i++ {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i]
	}
}

func weightedValueSum4_128(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	for i := 0; i < 128; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
	}
}

func weightedValueSum4_64(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	for i := 0; i < 64; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
	}
}

func weightedCacheValueSum4(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32, dim int) {
	i := 0
	for ; i+7 < dim; i += 8 {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
		dst[i+1] = a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
		dst[i+2] = a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
		dst[i+3] = a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
		dst[i+4] = a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
		dst[i+5] = a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
		dst[i+6] = a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
		dst[i+7] = a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
	}
	for ; i < dim; i++ {
		dst[i] = a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
	}
}

func weightedCacheValueSum128(dst []float32, cache *kvCache, head int, weights []float32) {
	headBase := head * 128
	if len(weights) == 1 {
		copy(dst[:128], cache.v[headBase:headBase+128])
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+128]
	for i := 0; i < 128; i += 8 {
		dst[i] = a * x[i]
		dst[i+1] = a * x[i+1]
		dst[i+2] = a * x[i+2]
		dst[i+3] = a * x[i+3]
		dst[i+4] = a * x[i+4]
		dst[i+5] = a * x[i+5]
		dst[i+6] = a * x[i+6]
		dst[i+7] = a * x[i+7]
	}
	t := 1
	for ; t+3 < len(weights); t += 4 {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+128]
		x1 := cache.v[base1 : base1+128]
		x2 := cache.v[base2 : base2+128]
		x3 := cache.v[base3 : base3+128]
		for i := 0; i < 128; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
		}
	}
	for ; t+1 < len(weights); t += 2 {
		a0, a1 := weights[t], weights[t+1]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+128]
		x1 := cache.v[base1 : base1+128]
		for i := 0; i < 128; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7]
		}
	}
	for ; t < len(weights); t++ {
		a := weights[t]
		base := t*cache.kvDim + headBase
		x := cache.v[base : base+128]
		for i := 0; i < 128; i += 8 {
			dst[i] += a * x[i]
			dst[i+1] += a * x[i+1]
			dst[i+2] += a * x[i+2]
			dst[i+3] += a * x[i+3]
			dst[i+4] += a * x[i+4]
			dst[i+5] += a * x[i+5]
			dst[i+6] += a * x[i+6]
			dst[i+7] += a * x[i+7]
		}
	}
}

func weightedCacheValueSum64(dst []float32, cache *kvCache, head int, weights []float32) {
	headBase := head * 64
	if len(weights) == 1 {
		copy(dst[:64], cache.v[headBase:headBase+64])
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+64]
	for i := 0; i < 64; i += 8 {
		dst[i] = a * x[i]
		dst[i+1] = a * x[i+1]
		dst[i+2] = a * x[i+2]
		dst[i+3] = a * x[i+3]
		dst[i+4] = a * x[i+4]
		dst[i+5] = a * x[i+5]
		dst[i+6] = a * x[i+6]
		dst[i+7] = a * x[i+7]
	}
	t := 1
	for ; t+3 < len(weights); t += 4 {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+64]
		x1 := cache.v[base1 : base1+64]
		x2 := cache.v[base2 : base2+64]
		x3 := cache.v[base3 : base3+64]
		for i := 0; i < 64; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i] + a2*x2[i] + a3*x3[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1] + a2*x2[i+1] + a3*x3[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2] + a2*x2[i+2] + a3*x3[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3] + a2*x2[i+3] + a3*x3[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4] + a2*x2[i+4] + a3*x3[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5] + a2*x2[i+5] + a3*x3[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6] + a2*x2[i+6] + a3*x3[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7] + a2*x2[i+7] + a3*x3[i+7]
		}
	}
	for ; t+1 < len(weights); t += 2 {
		a0, a1 := weights[t], weights[t+1]
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+64]
		x1 := cache.v[base1 : base1+64]
		for i := 0; i < 64; i += 8 {
			dst[i] += a0*x0[i] + a1*x1[i]
			dst[i+1] += a0*x0[i+1] + a1*x1[i+1]
			dst[i+2] += a0*x0[i+2] + a1*x1[i+2]
			dst[i+3] += a0*x0[i+3] + a1*x1[i+3]
			dst[i+4] += a0*x0[i+4] + a1*x1[i+4]
			dst[i+5] += a0*x0[i+5] + a1*x1[i+5]
			dst[i+6] += a0*x0[i+6] + a1*x1[i+6]
			dst[i+7] += a0*x0[i+7] + a1*x1[i+7]
		}
	}
	for ; t < len(weights); t++ {
		a := weights[t]
		base := t*cache.kvDim + headBase
		x := cache.v[base : base+64]
		for i := 0; i < 64; i += 8 {
			dst[i] += a * x[i]
			dst[i+1] += a * x[i+1]
			dst[i+2] += a * x[i+2]
			dst[i+3] += a * x[i+3]
			dst[i+4] += a * x[i+4]
			dst[i+5] += a * x[i+5]
			dst[i+6] += a * x[i+6]
			dst[i+7] += a * x[i+7]
		}
	}
}

func (c *kvCache) append(k, v []float32) {
	if c.kvDim == 0 {
		c.kvDim = len(k)
	}
	c.k = append(c.k, k...)
	c.v = append(c.v, v...)
	c.len++
}

func (c *kvCache) key(t, head, headDim int) []float32 {
	base := t*c.kvDim + head*headDim
	return c.k[base : base+headDim]
}

func (c *kvCache) value(t, head, headDim int) []float32 {
	base := t*c.kvDim + head*headDim
	return c.v[base : base+headDim]
}

func (rt *Runtime) mlp(x []float32, tl *textLayer, sc *layerScratch) []float32 {
	c := rt.cfg
	out := sc.mlp[:c.HiddenSize]
	if tl.q8.gate != nil && tl.q8.up != nil && tl.q8.down != nil {
		tensor.FusedSwiGLUQ8Scratch(out, x, tl.q8.gate, tl.q8.up, tl.q8.down, sc.gate)
		return out
	}
	if tl.q6.gate != nil && tl.q6.up != nil && tl.q6.down != nil {
		tensor.FusedSwiGLUQ6Scratch(out, x, tl.q6.gate, tl.q6.up, tl.q6.down, sc.gate)
		return out
	}
	if tl.q4.gate != nil && tl.q4.up != nil && tl.q4.down != nil {
		tensor.FusedSwiGLUQ4Scratch(out, x, tl.q4.gate, tl.q4.up, tl.q4.down, sc.gate)
		return out
	}
	g := sc.gate[:c.IntermediateSize]
	tensor.FusedSwiGLUF32Scratch(out, x, tl.w.gate, tl.w.up, tl.w.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize, g)
	return out
}

func matVecMaybeQ8(out, x, w []float32, q *tensor.Q8Matrix, rows, cols int) {
	if q != nil {
		tensor.MatVecQ8(out, x, q)
		return
	}
	tensor.MatVec(out, x, w, rows, cols)
}

func fusedQKV(q, k, v, x []float32, tl *textLayer, qRows, kvRows, hidden int) bool {
	if tl.q8.q != nil && tl.q8.k != nil && tl.q8.v != nil && sameColsQ8(hidden, tl.q8.q, tl.q8.k, tl.q8.v) {
		tensor.FusedMatVec3Q8(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v)
		return true
	}
	if tl.q6.q != nil && tl.q6.k != nil && tl.q6.v != nil && sameColsQ6(hidden, tl.q6.q, tl.q6.k, tl.q6.v) {
		tensor.FusedMatVec3Q6(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v)
		return true
	}
	if tl.q4.q != nil && tl.q4.k != nil && tl.q4.v != nil && sameColsQ4(hidden, tl.q4.q, tl.q4.k, tl.q4.v) {
		tensor.FusedMatVec3Q4(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v)
		return true
	}
	if len(tl.w.q) >= qRows*hidden && len(tl.w.k) >= kvRows*hidden && len(tl.w.v) >= kvRows*hidden {
		tensor.FusedMatVec3(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden)
		return true
	}
	return false
}

func sameColsQ8(cols int, a, b, c *tensor.Q8Matrix) bool {
	return a.Cols == cols && b.Cols == cols && c.Cols == cols
}

func sameColsQ6(cols int, a, b, c *tensor.Q6Matrix) bool {
	return a.Cols == cols && b.Cols == cols && c.Cols == cols
}

func sameColsQ4(cols int, a, b, c *tensor.Q4Matrix) bool {
	return a.Cols == cols && b.Cols == cols && c.Cols == cols
}

func matVecMaybeQuant(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) {
	if q8 != nil {
		tensor.MatVecQ8(out, x, q8)
		return
	}
	if q6 != nil {
		tensor.MatVecQ6(out, x, q6)
		return
	}
	if q4 != nil {
		tensor.MatVecQ4(out, x, q4)
		return
	}
	tensor.MatVec(out, x, w, rows, cols)
}

func (rt *Runtime) applyMRoPE(x []float32, heads, dim int, pos ropePos) {
	half := dim / 2
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	rt.buildMRoPETable(cosTable, sinTable, pos)
	applyMRoPEWithTable(x, heads, dim, cosTable, sinTable)
}

func (rt *Runtime) buildMRoPETable(cosTable, sinTable []float32, pos ropePos) {
	freqs := rt.ropeFreq
	if pos.T == 0 && pos.H == 0 && pos.W == 0 {
		i := 0
		for ; i+7 < len(cosTable); i += 8 {
			cosTable[i] = 1
			cosTable[i+1] = 1
			cosTable[i+2] = 1
			cosTable[i+3] = 1
			cosTable[i+4] = 1
			cosTable[i+5] = 1
			cosTable[i+6] = 1
			cosTable[i+7] = 1
		}
		for ; i < len(cosTable); i++ {
			cosTable[i] = 1
		}
		clear(sinTable)
		return
	}
	if pos.T == pos.H && pos.H == pos.W {
		p := float64(pos.T)
		for i := range cosTable {
			ang := p * freqs[i]
			cosTable[i], sinTable[i] = cos(ang), sin(ang)
		}
		return
	}
	axes := rt.ropeAxis
	for i := range cosTable {
		p := pos.T
		if axes[i] == 1 {
			p = pos.H
		} else if axes[i] == 2 {
			p = pos.W
		}
		ang := float64(p) * freqs[i]
		cosTable[i], sinTable[i] = cos(ang), sin(ang)
	}
}

func applyMRoPEWithTable(x []float32, heads, dim int, cosTable, sinTable []float32) {
	half := dim / 2
	for h := 0; h < heads; h++ {
		base := h * dim
		i := 0
		for ; i+3 < half; i += 4 {
			cs, sn := cosTable[i], sinTable[i]
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
			cs, sn = cosTable[i+1], sinTable[i+1]
			a, b = x[base+i+1], x[base+half+i+1]
			x[base+i+1] = a*cs - b*sn
			x[base+half+i+1] = b*cs + a*sn
			cs, sn = cosTable[i+2], sinTable[i+2]
			a, b = x[base+i+2], x[base+half+i+2]
			x[base+i+2] = a*cs - b*sn
			x[base+half+i+2] = b*cs + a*sn
			cs, sn = cosTable[i+3], sinTable[i+3]
			a, b = x[base+i+3], x[base+half+i+3]
			x[base+i+3] = a*cs - b*sn
			x[base+half+i+3] = b*cs + a*sn
		}
		for ; i < half; i++ {
			cs, sn := cosTable[i], sinTable[i]
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
		}
	}
}

func applyMRoPE(x []float32, heads, dim int, pos ropePos, theta float64) {
	half := dim / 2
	sections := []int{16, 24, 24}
	for h := 0; h < heads; h++ {
		base := h * dim
		sectionEnd := sections[0]
		axis := 0
		for i := 0; i < half; i++ {
			for axis+1 < len(sections) && i >= sectionEnd {
				axis++
				sectionEnd += sections[axis]
			}
			p := pos.T
			if axis == 1 {
				p = pos.H
			} else if axis == 2 {
				p = pos.W
			}
			freq := pow(theta, -float64(2*i)/float64(dim))
			ang := float64(p) * freq
			cs, sn := cos(ang), sin(ang)
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
		}
	}
}

func invSqrt(n int) float32 { return float32(1 / sqrt(float64(n))) }

func sqrt(x float64) float64 { return pow(x, 0.5) }

func pow(x, y float64) float64 {
	return math.Pow(x, y)
}

func sin(x float64) float32 { return float32(math.Sin(x)) }
func cos(x float64) float32 { return float32(math.Cos(x)) }
