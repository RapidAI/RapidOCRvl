package model

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
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
	projectRowsPool       sync.Pool
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
	vulkanPrewarmRunMu    sync.Mutex
	vulkanPrewarmMu       sync.RWMutex
	vulkanPrewarm         VulkanPrewarmStats
	vulkanDisabledMu      sync.RWMutex
	vulkanDisabledReasons map[vulkanOp]string
	zeroBiasMu            sync.Mutex
	zeroBiasBuf           []float32
	zeroBiasSmall         [maxZeroBiasSmall]float32
	vulkanDisabledOps     [2]atomic.Uint64
	kvCacheEpoch          atomic.Uint64
}

type vulkanOp uint8

const (
	vulkanOpMatVecF32 vulkanOp = iota
	vulkanOpMatVecArgmaxF32
	vulkanOpMatVecQ8
	vulkanOpMatVecQ6
	vulkanOpMatVecQ4
	vulkanOpFusedQKVF32
	vulkanOpFusedQKVQ8
	vulkanOpFusedQKVQ6
	vulkanOpFusedQKVQ4
	vulkanOpTextAttentionF32
	vulkanOpTextAttentionOutF32
	vulkanOpTextAttentionOutQ8
	vulkanOpTextAttentionOutQ6
	vulkanOpTextAttentionOutQ4
	vulkanOpSwiGLUGateUpF32
	vulkanOpSwiGLUGateUpQ8
	vulkanOpSwiGLUGateUpQ6
	vulkanOpSwiGLUGateUpQ4
	vulkanOpSwiGLUDownF32
	vulkanOpSwiGLUDownQ8
	vulkanOpSwiGLUDownQ6
	vulkanOpSwiGLUDownQ4
	vulkanOpVisionMatRowsBiasF32
	vulkanOpVisionMatRowsBiasAddRowsF32
	vulkanOpVisionMatRowsBias3F32
	vulkanOpVisionAttentionF32
	vulkanOpVisionAttentionOutF32
	vulkanOpVisionMatRowsGELU2F32
	vulkanOpVisionProjectImageF32
	vulkanOpVisionRoPEPairF32
	vulkanOpVisionRoPEAttentionOutF32
	vulkanOpVisionQKVRoPEAttentionOutF32
	vulkanOpRMSNormF32
	vulkanOpAddRMSNormF32
	vulkanOpAddRMSNormOutOnlyF32
	vulkanOpMRoPEF32
	vulkanOpMRoPEPairF32
	vulkanOpFusedQKVMRoPEF32
	vulkanOpFusedQKVMRoPEQ8
	vulkanOpFusedQKVMRoPEQ6
	vulkanOpFusedQKVMRoPEQ4
	vulkanOpTextAttentionOutAddRMSNormF32
	vulkanOpTextAttentionOutAddRMSNormQ8
	vulkanOpTextAttentionOutAddRMSNormQ6
	vulkanOpTextAttentionOutAddRMSNormQ4
	vulkanOpSwiGLUDownAddRMSNormF32
	vulkanOpSwiGLUDownAddRMSNormQ8
	vulkanOpSwiGLUDownAddRMSNormQ6
	vulkanOpSwiGLUDownAddRMSNormQ4
	vulkanOpSwiGLUDownAddRMSNormOutOnlyF32
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ8
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ6
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ4
	vulkanOpMatVecAddRMSNormF32
	vulkanOpMatVecAddRMSNormQ8
	vulkanOpMatVecAddRMSNormQ6
	vulkanOpMatVecAddRMSNormQ4
	vulkanOpTextFirstTokenValueOutAddRMSNormF32
	vulkanOpTextFirstTokenValueOutAddRMSNormQ8
	vulkanOpTextFirstTokenValueOutAddRMSNormQ6
	vulkanOpTextFirstTokenValueOutAddRMSNormQ4
	vulkanOpFusedKVF32
	vulkanOpFusedKVQ8
	vulkanOpFusedKVQ6
	vulkanOpFusedKVQ4
	vulkanOpFusedKVMRoPEF32
	vulkanOpFusedKVMRoPEQ8
	vulkanOpFusedKVMRoPEQ6
	vulkanOpFusedKVMRoPEQ4
	vulkanOpVisionLayerNormRowsF32
	vulkanOpVisionAddLayerNormRowsF32
	vulkanOpVisionProjectImageFusedF32
	vulkanOpVisionMatRowsGELU2AddLayerNormF32
	vulkanOpMatVecTopKF32
	vulkanOpMatVecArgmaxQ8
	vulkanOpMatVecArgmaxQ6
	vulkanOpMatVecArgmaxQ4
	vulkanOpMatVecTopKQ8
	vulkanOpMatVecTopKQ6
	vulkanOpMatVecTopKQ4
	vulkanOpChainedRMSNormMatVecF32
	vulkanOpChainedMatVecAddRMSNormMatVecF32
	vulkanOpChainedRMSNormQKVMRoPEF32
	vulkanOpChainedRMSNormQKVMRoPEQ8
	vulkanOpChainedQKVAttentionOutAddRMSNormF32
	vulkanOpChainedQKVAttentionOutAddRMSNormQ8
	vulkanOpChainedSwiGLUDownAddRMSNormQ8
	vulkanOpLayerChainQ8
)

var vulkanOpNames = [...]string{
	vulkanOpMatVecF32:                           "matvec_f32",
	vulkanOpMatVecArgmaxF32:                     "matvec_argmax_f32",
	vulkanOpMatVecQ8:                            "matvec_q8",
	vulkanOpMatVecQ6:                            "matvec_q6",
	vulkanOpMatVecQ4:                            "matvec_q4",
	vulkanOpFusedQKVF32:                         "fused_qkv_f32",
	vulkanOpFusedQKVQ8:                          "fused_qkv_q8",
	vulkanOpFusedQKVQ6:                          "fused_qkv_q6",
	vulkanOpFusedQKVQ4:                          "fused_qkv_q4",
	vulkanOpTextAttentionF32:                    "text_attention_f32",
	vulkanOpTextAttentionOutF32:                 "text_attention_out_f32",
	vulkanOpTextAttentionOutQ8:                  "text_attention_out_q8",
	vulkanOpTextAttentionOutQ6:                  "text_attention_out_q6",
	vulkanOpTextAttentionOutQ4:                  "text_attention_out_q4",
	vulkanOpSwiGLUGateUpF32:                     "swiglu_gate_up_f32",
	vulkanOpSwiGLUGateUpQ8:                      "swiglu_gate_up_q8",
	vulkanOpSwiGLUGateUpQ6:                      "swiglu_gate_up_q6",
	vulkanOpSwiGLUGateUpQ4:                      "swiglu_gate_up_q4",
	vulkanOpSwiGLUDownF32:                       "swiglu_down_f32",
	vulkanOpSwiGLUDownQ8:                        "swiglu_down_q8",
	vulkanOpSwiGLUDownQ6:                        "swiglu_down_q6",
	vulkanOpSwiGLUDownQ4:                        "swiglu_down_q4",
	vulkanOpVisionMatRowsBiasF32:                "vision_matrows_bias_f32",
	vulkanOpVisionMatRowsBiasAddRowsF32:         "vision_matrows_bias_add_rows_f32",
	vulkanOpVisionMatRowsBias3F32:               "vision_matrows_bias3_f32",
	vulkanOpVisionAttentionF32:                  "vision_attention_f32",
	vulkanOpVisionAttentionOutF32:               "vision_attention_out_f32",
	vulkanOpVisionMatRowsGELU2F32:               "vision_matrows_gelu2_f32",
	vulkanOpVisionProjectImageF32:               "vision_project_image_f32",
	vulkanOpVisionRoPEPairF32:                   "vision_rope_pair_f32",
	vulkanOpVisionRoPEAttentionOutF32:           "vision_rope_attention_out_f32",
	vulkanOpVisionQKVRoPEAttentionOutF32:        "vision_qkv_rope_attention_out_f32",
	vulkanOpRMSNormF32:                          "rmsnorm_f32",
	vulkanOpAddRMSNormF32:                       "add_rmsnorm_f32",
	vulkanOpAddRMSNormOutOnlyF32:                "add_rmsnorm_out_only_f32",
	vulkanOpMRoPEF32:                            "mrope_f32",
	vulkanOpMRoPEPairF32:                        "mrope_pair_f32",
	vulkanOpFusedQKVMRoPEF32:                    "fused_qkv_mrope_f32",
	vulkanOpFusedQKVMRoPEQ8:                     "fused_qkv_mrope_q8",
	vulkanOpFusedQKVMRoPEQ6:                     "fused_qkv_mrope_q6",
	vulkanOpFusedQKVMRoPEQ4:                     "fused_qkv_mrope_q4",
	vulkanOpTextAttentionOutAddRMSNormF32:       "text_attention_out_add_rmsnorm_f32",
	vulkanOpTextAttentionOutAddRMSNormQ8:        "text_attention_out_add_rmsnorm_q8",
	vulkanOpTextAttentionOutAddRMSNormQ6:        "text_attention_out_add_rmsnorm_q6",
	vulkanOpTextAttentionOutAddRMSNormQ4:        "text_attention_out_add_rmsnorm_q4",
	vulkanOpSwiGLUDownAddRMSNormF32:             "swiglu_down_add_rmsnorm_f32",
	vulkanOpSwiGLUDownAddRMSNormQ8:              "swiglu_down_add_rmsnorm_q8",
	vulkanOpSwiGLUDownAddRMSNormQ6:              "swiglu_down_add_rmsnorm_q6",
	vulkanOpSwiGLUDownAddRMSNormQ4:              "swiglu_down_add_rmsnorm_q4",
	vulkanOpSwiGLUDownAddRMSNormOutOnlyF32:      "swiglu_down_add_rmsnorm_out_only_f32",
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ8:       "swiglu_down_add_rmsnorm_out_only_q8",
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ6:       "swiglu_down_add_rmsnorm_out_only_q6",
	vulkanOpSwiGLUDownAddRMSNormOutOnlyQ4:       "swiglu_down_add_rmsnorm_out_only_q4",
	vulkanOpMatVecAddRMSNormF32:                 "matvec_add_rmsnorm_f32",
	vulkanOpMatVecAddRMSNormQ8:                  "matvec_add_rmsnorm_q8",
	vulkanOpMatVecAddRMSNormQ6:                  "matvec_add_rmsnorm_q6",
	vulkanOpMatVecAddRMSNormQ4:                  "matvec_add_rmsnorm_q4",
	vulkanOpTextFirstTokenValueOutAddRMSNormF32: "text_first_token_value_out_add_rmsnorm_f32",
	vulkanOpTextFirstTokenValueOutAddRMSNormQ8:  "text_first_token_value_out_add_rmsnorm_q8",
	vulkanOpTextFirstTokenValueOutAddRMSNormQ6:  "text_first_token_value_out_add_rmsnorm_q6",
	vulkanOpTextFirstTokenValueOutAddRMSNormQ4:  "text_first_token_value_out_add_rmsnorm_q4",
	vulkanOpFusedKVF32:                          "fused_kv_f32",
	vulkanOpFusedKVQ8:                           "fused_kv_q8",
	vulkanOpFusedKVQ6:                           "fused_kv_q6",
	vulkanOpFusedKVQ4:                           "fused_kv_q4",
	vulkanOpFusedKVMRoPEF32:                     "fused_kv_mrope_f32",
	vulkanOpFusedKVMRoPEQ8:                      "fused_kv_mrope_q8",
	vulkanOpFusedKVMRoPEQ6:                      "fused_kv_mrope_q6",
	vulkanOpFusedKVMRoPEQ4:                      "fused_kv_mrope_q4",
	vulkanOpVisionLayerNormRowsF32:              "vision_layernorm_rows_f32",
	vulkanOpVisionAddLayerNormRowsF32:           "vision_add_layernorm_rows_f32",
	vulkanOpVisionProjectImageFusedF32:          "vision_project_image_fused_f32",
	vulkanOpVisionMatRowsGELU2AddLayerNormF32:   "vision_matrows_gelu2_add_layernorm_f32",
	vulkanOpMatVecTopKF32:                       "matvec_topk_f32",
	vulkanOpMatVecArgmaxQ8:                      "matvec_argmax_q8",
	vulkanOpMatVecArgmaxQ6:                      "matvec_argmax_q6",
	vulkanOpMatVecArgmaxQ4:                      "matvec_argmax_q4",
	vulkanOpMatVecTopKQ8:                        "matvec_topk_q8",
	vulkanOpMatVecTopKQ6:                        "matvec_topk_q6",
	vulkanOpMatVecTopKQ4:                        "matvec_topk_q4",
	vulkanOpChainedRMSNormMatVecF32:             "chained_rmsnorm_matvec_f32",
	vulkanOpChainedMatVecAddRMSNormMatVecF32:    "chained_matvec_add_rmsnorm_matvec_f32",
	vulkanOpChainedRMSNormQKVMRoPEF32:           "chained_rmsnorm_qkv_mrope_f32",
	vulkanOpChainedRMSNormQKVMRoPEQ8:            "chained_rmsnorm_qkv_mrope_q8",
	vulkanOpChainedQKVAttentionOutAddRMSNormF32: "chained_qkv_attention_out_norm_f32",
	vulkanOpChainedQKVAttentionOutAddRMSNormQ8: "chained_qkv_attention_out_norm_q8",
	vulkanOpChainedSwiGLUDownAddRMSNormQ8: "chained_swiglu_down_norm_q8",
	vulkanOpLayerChainQ8:                       "layer_chain_q8",
}

type vulkanPlanCache struct {
	shape              backend.VulkanModelShape
	valid              bool
	plans              []backend.VulkanPlan
	optionalPlans      []backend.VulkanPlan
	summary            backend.VulkanPlanSummary
	graph              backend.VulkanExecutionGraph
	pipes              []backend.VulkanPipelinePlan
	optionalPipes      []backend.VulkanPipelinePlan
	command            backend.VulkanCommandPlan
	optionalCommand    backend.VulkanCommandPlan
	commandValidation  string
	optionalValidation string
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

type VulkanPrewarmStats struct {
	Backend                  string                              `json:"backend"`
	DurationMS               int64                               `json:"duration_ms"`
	Planned                  bool                                `json:"planned"`
	PlanCount                int                                 `json:"plan_count"`
	OptionalPlanCount        int                                 `json:"optional_plan_count"`
	PipelineCount            int                                 `json:"pipeline_count"`
	OptionalPipelineCount    int                                 `json:"optional_pipeline_count"`
	CommandCount             int                                 `json:"command_count"`
	OptionalCommandCount     int                                 `json:"optional_command_count"`
	CommandPlanValid         bool                                `json:"command_plan_valid"`
	CommandPlanError         string                              `json:"command_plan_error,omitempty"`
	OptionalCommandPlanValid bool                                `json:"optional_command_plan_valid"`
	OptionalCommandPlanError string                              `json:"optional_command_plan_error,omitempty"`
	DispatchProbe            bool                                `json:"dispatch_probe"`
	DispatchReady            bool                                `json:"dispatch_ready"`
	Error                    string                              `json:"error,omitempty"`
	DispatchProbes           []backend.VulkanDispatchProbeResult `json:"dispatch_probes,omitempty"`
}

type VulkanDisabledOp struct {
	Name   string `json:"name"`
	Reason string `json:"reason,omitempty"`
}

type kvCache struct {
	k     []float32
	v     []float32
	len   int
	kvDim int
	epoch uint64
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
	up      []float32
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
	topKScores []tensor.TopKScore
	topKWork   []tensor.TopKScore
	weights    []float32
	positions  []ropePos
	inputIDs   []int
	rng        fastRNG
}

type generationForwardResult struct {
	logits        []float32
	candidates    []tokenScore
	candidateTemp float64
	token         int
	hasToken      bool
	hasCandidates bool
}

type GenerateOptions struct {
	MaxNewTokens int
	Temperature  float64
	TopK         int
	Seed         int64
	EOSTokenIDs  []int
}

var defaultEOSTokenIDs = []int{2}

var sampleFullLogitsWeightsPool sync.Pool

const maxEOSTokenIDs = 1024
const sampleFullLogitsAllocWeightsMin = 4
const sampleFullLogitsPooledWeightsMax = 256 * 1024
const vulkanMatVecMinWorkEnv = "RAPIDOCRVL_VULKAN_MATVEC_MIN_WORK"
const vulkanTextAttentionMinWorkEnv = "RAPIDOCRVL_VULKAN_TEXT_ATTENTION_MIN_WORK"
const vulkanVectorMinWorkEnv = "RAPIDOCRVL_VULKAN_VECTOR_MIN_WORK"
const vulkanVisionMinWorkEnv = "RAPIDOCRVL_VULKAN_VISION_MIN_WORK"
const defaultVulkanMatVecMinWork = 65536
const defaultVulkanVectorMinWork = 1024
const maxZeroBiasSmall = 8192

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
	return c == ' ' || c == '\n' || c == '\r' || c == '	' || c == '\v' || c == '\f'
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

func (rt *Runtime) VulkanDisabledOps() []string {
	var out []string
	for op, name := range vulkanOpNames {
		if name == "" {
			continue
		}
		bank, bit := vulkanOpBankBit(vulkanOp(op))
		if rt.vulkanDisabledOps[bank].Load()&bit != 0 {
			out = append(out, name)
		}
	}
	return out
}

func (rt *Runtime) VulkanDisabledOpReasons() []VulkanDisabledOp {
	rt.vulkanDisabledMu.RLock()
	defer rt.vulkanDisabledMu.RUnlock()
	var out []VulkanDisabledOp
	for op, name := range vulkanOpNames {
		if name == "" {
			continue
		}
		bank, bit := vulkanOpBankBit(vulkanOp(op))
		if rt.vulkanDisabledOps[bank].Load()&bit == 0 {
			continue
		}
		out = append(out, VulkanDisabledOp{
			Name:   name,
			Reason: rt.vulkanDisabledReasons[vulkanOp(op)],
		})
	}
	return out
}

func (rt *Runtime) ResetVulkanDisabledOps() {
	for i := range rt.vulkanDisabledOps {
		rt.vulkanDisabledOps[i].Store(0)
	}
	rt.vulkanDisabledMu.Lock()
	clear(rt.vulkanDisabledReasons)
	rt.vulkanDisabledMu.Unlock()
}

func (rt *Runtime) VulkanPrewarmStats() VulkanPrewarmStats {
	rt.vulkanPrewarmMu.RLock()
	defer rt.vulkanPrewarmMu.RUnlock()
	return rt.vulkanPrewarm
}

func (rt *Runtime) PrewarmVulkan() (VulkanPrewarmStats, error) {
	rt.vulkanPrewarmRunMu.Lock()
	defer rt.vulkanPrewarmRunMu.Unlock()

	start := time.Now()
	stats := VulkanPrewarmStats{Backend: rt.Backend()}
	artifacts := rt.VulkanArtifactsView()
	stats.Planned = len(artifacts.Plans) != 0 || len(artifacts.OptionalPlans) != 0 || artifacts.CommandPlan.CommandCount != 0 || artifacts.OptionalCommand.CommandCount != 0
	stats.PlanCount = len(artifacts.Plans)
	stats.OptionalPlanCount = len(artifacts.OptionalPlans)
	stats.PipelineCount = len(artifacts.PipelinePlan)
	stats.OptionalPipelineCount = len(artifacts.OptionalPipelinePlan)
	stats.CommandCount = artifacts.CommandPlan.CommandCount
	stats.OptionalCommandCount = artifacts.OptionalCommand.CommandCount
	stats.CommandPlanError = backend.ValidateVulkanCommandPlan(artifacts.CommandPlan)
	stats.OptionalCommandPlanError = backend.ValidateVulkanCommandPlan(artifacts.OptionalCommand)
	stats.CommandPlanValid = stats.CommandPlanError == ""
	stats.OptionalCommandPlanValid = stats.OptionalCommandPlanError == ""
	var err error
	if stats.Backend == "vulkan" {
		stats.DispatchProbe = true
		stats.DispatchProbes = backend.VulkanDispatchSmokeSuite()
		err = rt.recordVulkanProbeFailures(stats.DispatchProbes)
		if err == nil {
			stats.DispatchReady = true
		} else {
			stats.Error = err.Error()
		}
	}
	stats.DurationMS = elapsedMillis(start)
	rt.vulkanPrewarmMu.Lock()
	rt.vulkanPrewarm = stats
	rt.vulkanPrewarmMu.Unlock()
	return stats, err
}

func (rt *Runtime) recordVulkanProbeFailures(probes []backend.VulkanDispatchProbeResult) error {
	var firstErr error
	for _, probe := range probes {
		if probe.OK {
			continue
		}
		probeErr := fmt.Errorf("%s: %s", probe.Name, probe.Error)
		if op, ok := vulkanProbeOp(probe.Name); ok {
			rt.disableVulkanOp(op, probeErr)
		}
		if firstErr == nil {
			firstErr = probeErr
		}
	}
	return firstErr
}

func vulkanProbeOp(name string) (vulkanOp, bool) {
	for op, opName := range vulkanOpNames {
		if opName == name {
			return vulkanOp(op), true
		}
	}
	return 0, false
}

func (rt *Runtime) VulkanPlans() []backend.VulkanPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return nil
	}
	return append([]backend.VulkanPlan(nil), cache.plans...)
}

func (rt *Runtime) VulkanOptionalPlans() []backend.VulkanPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return nil
	}
	return append([]backend.VulkanPlan(nil), cache.optionalPlans...)
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

func (rt *Runtime) VulkanOptionalPipelinePlan() []backend.VulkanPipelinePlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return nil
	}
	return append([]backend.VulkanPipelinePlan(nil), cache.optionalPipes...)
}

func (rt *Runtime) VulkanCommandPlan() backend.VulkanCommandPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanCommandPlan{}
	}
	return cloneVulkanCommandPlan(cache.command)
}

func (rt *Runtime) VulkanOptionalCommandPlan() backend.VulkanCommandPlan {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanCommandPlan{}
	}
	return cloneVulkanCommandPlanWithPipelines(cache.optionalCommand, append([]backend.VulkanPipelinePlan(nil), cache.optionalPipes...))
}

func (rt *Runtime) VulkanArtifacts() backend.VulkanModelArtifacts {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return backend.VulkanModelArtifacts{}
	}
	pipes := append([]backend.VulkanPipelinePlan(nil), cache.pipes...)
	optionalPipes := append([]backend.VulkanPipelinePlan(nil), cache.optionalPipes...)
	out := backend.VulkanModelArtifacts{
		Plans:                append([]backend.VulkanPlan(nil), cache.plans...),
		OptionalPlans:        append([]backend.VulkanPlan(nil), cache.optionalPlans...),
		Summary:              cache.summary,
		ExecutionGraph:       cache.graph,
		PipelinePlan:         pipes,
		OptionalPipelinePlan: optionalPipes,
		CommandPlan:          cloneVulkanCommandPlanWithPipelines(cache.command, pipes),
		OptionalCommand:      cloneVulkanCommandPlanWithPipelines(cache.optionalCommand, optionalPipes),
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
		Plans:                cache.plans,
		OptionalPlans:        cache.optionalPlans,
		Summary:              cache.summary,
		ExecutionGraph:       cache.graph,
		PipelinePlan:         cache.pipes,
		OptionalPipelinePlan: cache.optionalPipes,
		CommandPlan:          cache.command,
		OptionalCommand:      cache.optionalCommand,
	}
}

func (rt *Runtime) VulkanCommandPlanValidation() string {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return ""
	}
	return cache.commandValidation
}

func (rt *Runtime) VulkanOptionalCommandPlanValidation() string {
	cache, ok := rt.cachedVulkanPlans()
	if !ok {
		return ""
	}
	return cache.optionalValidation
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
		shape:              shape,
		valid:              true,
		plans:              artifacts.Plans,
		optionalPlans:      artifacts.OptionalPlans,
		summary:            artifacts.Summary,
		graph:              artifacts.ExecutionGraph,
		pipes:              artifacts.PipelinePlan,
		optionalPipes:      artifacts.OptionalPipelinePlan,
		command:            artifacts.CommandPlan,
		optionalCommand:    artifacts.OptionalCommand,
		commandValidation:  backend.ValidateVulkanCommandPlan(artifacts.CommandPlan),
		optionalValidation: backend.ValidateVulkanCommandPlan(artifacts.OptionalCommand),
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
	rt.ensureSamplingScratchCapacity(scratch, opts, len(input)+opts.MaxNewTokens)
	out := make([]int, 0, len(input)+opts.MaxNewTokens)
	out = append(out, input...)
	var logits []float32
	var candidates []tokenScore
	var candidateTemp float64
	var nextToken int
	var hasNextToken bool
	var hasCandidates bool
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
		fr, err := rt.forwardEmbeddingForSampling(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch, pos == len(input)-1, opts)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = fr.logits
		candidates = fr.candidates
		candidateTemp = fr.candidateTemp
		nextToken = fr.token
		hasNextToken = fr.hasToken
		hasCandidates = fr.hasCandidates
	}
	rng := samplingRNGInto(opts, &scratch.rng)
	for i := 0; i < opts.MaxNewTokens; i++ {
		next := nextToken
		if hasCandidates {
			next = sampleCandidateScoresScratch(candidates, candidateTemp, rng, scratch)
		} else if !hasNextToken {
			next = sampleTokenScratch(logits, opts, rng, scratch)
		}
		out = append(out, next)
		if isEOS(next, opts.EOSTokenIDs) {
			break
		}
		if i+1 >= opts.MaxNewTokens {
			break
		}
		if err := rt.tokenEmbeddingInto(scratch.hidden, next); err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		pos := len(out) - 1
		fr, err := rt.forwardEmbeddingForSampling(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch, true, opts)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = fr.logits
		candidates = fr.candidates
		candidateTemp = fr.candidateTemp
		nextToken = fr.token
		hasNextToken = fr.hasToken
		hasCandidates = fr.hasCandidates
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
	rt.ensureSamplingScratchCapacity(scratch, opts, len(input)+opts.MaxNewTokens)
	positions, ropeDelta := rt.multimodalPositionsInto(input, imageGrid, scratch.positions)
	scratch.positions = positions
	imageCursor := 0
	out := make([]int, 0, len(input)+opts.MaxNewTokens)
	out = append(out, input...)
	var logits []float32
	var candidates []tokenScore
	var candidateTemp float64
	var nextToken int
	var hasNextToken bool
	var hasCandidates bool
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
		fr, err := rt.forwardEmbeddingForSampling(emb, positions[pos], caches, scratch, pos == len(input)-1, opts)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = fr.logits
		candidates = fr.candidates
		candidateTemp = fr.candidateTemp
		nextToken = fr.token
		hasNextToken = fr.hasToken
		hasCandidates = fr.hasCandidates
	}
	if imageCursor != len(imageEmbeds) {
		rt.putGenerationResources(scratch, caches, cachePtr)
		return GenerateResult{}, fmt.Errorf("not enough image placeholder tokens: got %d need %d", imageCursor, len(imageEmbeds))
	}
	rng := samplingRNGInto(opts, &scratch.rng)
	for i := 0; i < opts.MaxNewTokens; i++ {
		next := nextToken
		if hasCandidates {
			next = sampleCandidateScoresScratch(candidates, candidateTemp, rng, scratch)
		} else if !hasNextToken {
			next = sampleTokenScratch(logits, opts, rng, scratch)
		}
		out = append(out, next)
		if isEOS(next, opts.EOSTokenIDs) {
			break
		}
		if i+1 >= opts.MaxNewTokens {
			break
		}
		if err := rt.tokenEmbeddingInto(scratch.hidden, next); err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		pos := len(out) - 1 + ropeDelta
		fr, err := rt.forwardEmbeddingForSampling(scratch.hidden, ropePos{pos, pos, pos}, caches, scratch, true, opts)
		if err != nil {
			rt.putGenerationResources(scratch, caches, cachePtr)
			return GenerateResult{}, err
		}
		logits = fr.logits
		candidates = fr.candidates
		candidateTemp = fr.candidateTemp
		nextToken = fr.token
		hasNextToken = fr.hasToken
		hasCandidates = fr.hasCandidates
	}
	res := GenerateResult{Tokens: out, PromptTokens: len(input)}
	rt.putGenerationResources(scratch, caches, cachePtr)
	return res, nil
}

func (rt *Runtime) multimodalPositions(input []int, imageGrid [3]int) ([]ropePos, int) {
	return rt.multimodalPositionsInto(input, imageGrid, nil)
}

func (rt *Runtime) multimodalPositionsInto(input []int, imageGrid [3]int, buf []ropePos) ([]ropePos, int) {
	var positions []ropePos
	if cap(buf) < len(input) {
		positions = make([]ropePos, len(input))
	} else {
		positions = buf[:len(input)]
	}
	posIdx := 0
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
			positions[posIdx] = ropePos{p, p, p}
			posIdx++
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
					positions[posIdx] = ropePos{
						T: pt,
						H: ph,
						W: pw,
					}
					posIdx++
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
			positions[posIdx] = ropePos{p, p, p}
			posIdx++
		}
		nextPos = stIdx + len(input) - st
	}
	positions = positions[:posIdx]
	if posIdx == 0 {
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
	if !generationNeedsSamplingScratch(opts) {
		return nil
	}
	rng.state = uint64(opts.Seed)
	return rng
}

func generationNeedsSamplingScratch(opts GenerateOptions) bool {
	return opts.MaxNewTokens > 0 && opts.Temperature > 0 && opts.TopK != 1
}

func generationUsesArgmaxSampling(opts GenerateOptions) bool {
	return opts.Temperature <= 0 || opts.TopK == 1
}

func generationCanReturnArgmaxOnly(opts GenerateOptions) bool {
	return opts.MaxNewTokens > 0 && generationUsesArgmaxSampling(opts)
}

func generationCanReturnCandidatesOnly(opts GenerateOptions) bool {
	return opts.MaxNewTokens > 0 && opts.Temperature > 0 && opts.TopK > 1
}

func sampleTokenScratch(logits []float32, opts GenerateOptions, rng float64RNG, scratch *generationScratch) int {
	if scratch != nil && opts.TopK > 0 && opts.TopK < len(logits) && cap(scratch.weights) < opts.TopK {
		scratch.weights = make([]float32, opts.TopK)
	}
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
	if len(candidates) <= 4 {
		// Populate the weight scratch even on the small path so callers that
		// reuse generationScratch across samples see a stable backing array.
		weights := makeSampleWeights(len(candidates), scratch)
		for i, c := range candidates {
			weights[i] = tensor.FastExpF32(c.score*invTemp - maxScore)
		}
		return sampleCandidateScoresSmall(candidates, invTemp, maxScore, rng)
	}
	weights := makeSampleWeights(len(candidates), scratch)
	var sum float64
	for i, c := range candidates {
		w := float64(tensor.FastExpF32(c.score*invTemp - maxScore))
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
			acc += float64(tensor.FastExpF32(c.score*invTemp - maxScore))
			if pick <= acc {
				return c.id
			}
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleTopKTemp1Scratch(logits []float32, topK int, rng float64RNG, scratch *generationScratch) int {
	candidates, maxScore := topKCandidatesUnsortedWithMax(logits, topK, scratch)
	if len(candidates) <= 8 {
		return sampleCandidateScoresTemp1SmallScratch(candidates, maxScore, rng, scratch)
	}
	return sampleCandidateScoresTemp1Scratch(candidates, maxScore, rng, scratch)
}

func sampleCandidateScoresScratch(candidates []tokenScore, temp float64, rng float64RNG, scratch *generationScratch) int {
	if len(candidates) == 0 {
		return 0
	}
	if temp == 1 {
		maxScore := candidates[0].score
		for _, c := range candidates[1:] {
			maxScore = max32Local(maxScore, c.score)
		}
		if len(candidates) <= 4 {
			return sampleCandidateScoresTemp1SmallScratch(candidates, maxScore, rng, scratch)
		}
		return sampleCandidateScoresTemp1Scratch(candidates, maxScore, rng, scratch)
	}
	invTemp := float32(1 / temp)
	maxScoreRaw := candidates[0].score
	for _, c := range candidates[1:] {
		maxScoreRaw = max32Local(maxScoreRaw, c.score)
	}
	maxScore := maxScoreRaw * invTemp
	if len(candidates) <= 4 {
		return sampleCandidateScoresSmall(candidates, invTemp, maxScore, rng)
	}
	weights := makeSampleWeights(len(candidates), scratch)
	var sum float64
	for i, c := range candidates {
		w := float64(tensor.FastExpF32(c.score*invTemp - maxScore))
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
			acc += float64(tensor.FastExpF32(c.score*invTemp - maxScore))
			if pick <= acc {
				return c.id
			}
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleCandidateScoresSmall(candidates []tokenScore, invTemp, maxScore float32, rng float64RNG) int {
	switch len(candidates) {
	case 0:
		return 0
	case 1:
		return candidates[0].id
	case 2:
		s0 := candidates[0].score * invTemp
		s1 := candidates[1].score * invTemp
		if s0 >= s1 {
			w1 := float64(tensor.FastExpF32(s1 - s0))
			if rng.Float64()*(1+w1) <= 1 {
				return candidates[0].id
			}
			return candidates[1].id
		}
		w0 := float64(tensor.FastExpF32(s0 - s1))
		if rng.Float64()*(w0+1) <= w0 {
			return candidates[0].id
		}
		return candidates[1].id
	case 3:
		w0 := float64(tensor.FastExpF32(candidates[0].score*invTemp - maxScore))
		w1 := float64(tensor.FastExpF32(candidates[1].score*invTemp - maxScore))
		w2 := float64(tensor.FastExpF32(candidates[2].score*invTemp - maxScore))
		pick := rng.Float64() * ((w0 + w1) + w2)
		acc := w0
		if pick <= acc {
			return candidates[0].id
		}
		acc += w1
		if pick <= acc {
			return candidates[1].id
		}
		return candidates[2].id
	case 4:
		w0 := float64(tensor.FastExpF32(candidates[0].score*invTemp - maxScore))
		w1 := float64(tensor.FastExpF32(candidates[1].score*invTemp - maxScore))
		w2 := float64(tensor.FastExpF32(candidates[2].score*invTemp - maxScore))
		w3 := float64(tensor.FastExpF32(candidates[3].score*invTemp - maxScore))
		pick := rng.Float64() * ((w0 + w1) + (w2 + w3))
		acc := w0
		if pick <= acc {
			return candidates[0].id
		}
		acc += w1
		if pick <= acc {
			return candidates[1].id
		}
		acc += w2
		if pick <= acc {
			return candidates[2].id
		}
		return candidates[3].id
	}
	return candidates[len(candidates)-1].id
}

func sampleCandidateScoresTemp1Scratch(candidates []tokenScore, maxScore float32, rng float64RNG, scratch *generationScratch) int {
	weights := makeSampleWeights(len(candidates), scratch)
	var sum float32
	for i, c := range candidates {
		w := tensor.FastExpF32(c.score - maxScore)
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
			acc += tensor.FastExpF32(c.score - maxScore)
			if pick <= acc {
				return c.id
			}
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleCandidateScoresTemp1SmallScratch(candidates []tokenScore, maxScore float32, rng float64RNG, scratch *generationScratch) int {
	if scratch == nil || len(candidates) <= 4 {
		return sampleCandidateScoresTemp1Small(candidates, maxScore, rng)
	}
	weights := makeSampleWeights(len(candidates), scratch)
	switch len(candidates) {
	case 0:
		return 0
	case 1:
		weights[0] = 1
		return candidates[0].id
	case 2:
		w0 := tensor.FastExpF32(candidates[0].score - maxScore)
		w1 := tensor.FastExpF32(candidates[1].score - maxScore)
		weights[0] = w0
		weights[1] = w1
		pick := float32(rng.Float64()) * (w0 + w1)
		if pick <= w0 {
			return candidates[0].id
		}
		return candidates[1].id
	case 3:
		w0 := tensor.FastExpF32(candidates[0].score - maxScore)
		w1 := tensor.FastExpF32(candidates[1].score - maxScore)
		w2 := tensor.FastExpF32(candidates[2].score - maxScore)
		weights[0] = w0
		weights[1] = w1
		weights[2] = w2
		pick := float32(rng.Float64()) * ((w0 + w1) + w2)
		if pick <= w0 {
			return candidates[0].id
		}
		if pick <= w0+w1 {
			return candidates[1].id
		}
		return candidates[2].id
	case 4:
		w0 := tensor.FastExpF32(candidates[0].score - maxScore)
		w1 := tensor.FastExpF32(candidates[1].score - maxScore)
		w2 := tensor.FastExpF32(candidates[2].score - maxScore)
		w3 := tensor.FastExpF32(candidates[3].score - maxScore)
		weights[0] = w0
		weights[1] = w1
		weights[2] = w2
		weights[3] = w3
		pick := float32(rng.Float64()) * ((w0 + w1) + (w2 + w3))
		acc := w0
		if pick <= acc {
			return candidates[0].id
		}
		acc += w1
		if pick <= acc {
			return candidates[1].id
		}
		acc += w2
		if pick <= acc {
			return candidates[2].id
		}
		return candidates[3].id
	}
	var sum float32
	for i, c := range candidates {
		w := tensor.FastExpF32(c.score - maxScore)
		weights[i] = w
		sum += w
	}
	pick := float32(rng.Float64()) * sum
	var acc float32
	for i, w := range weights {
		acc += w
		if pick <= acc {
			return candidates[i].id
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleCandidateScoresTemp1Small(candidates []tokenScore, maxScore float32, rng float64RNG) int {
	switch len(candidates) {
	case 0:
		return 0
	case 1:
		return candidates[0].id
	case 2:
		s0 := candidates[0].score
		s1 := candidates[1].score
		if s0 >= s1 {
			w1 := float64(tensor.FastExpF32(s1 - s0))
			if rng.Float64()*(1+w1) <= 1 {
				return candidates[0].id
			}
			return candidates[1].id
		}
		w0 := float64(tensor.FastExpF32(s0 - s1))
		if rng.Float64()*(w0+1) <= w0 {
			return candidates[0].id
		}
		return candidates[1].id
	case 3:
		w0 := tensor.FastExpF32(candidates[0].score - maxScore)
		w1 := tensor.FastExpF32(candidates[1].score - maxScore)
		w2 := tensor.FastExpF32(candidates[2].score - maxScore)
		pick := float32(rng.Float64()) * ((w0 + w1) + w2)
		if pick <= w0 {
			return candidates[0].id
		}
		if pick <= w0+w1 {
			return candidates[1].id
		}
		return candidates[2].id
	case 4:
		w0 := tensor.FastExpF32(candidates[0].score - maxScore)
		w1 := tensor.FastExpF32(candidates[1].score - maxScore)
		w2 := tensor.FastExpF32(candidates[2].score - maxScore)
		w3 := tensor.FastExpF32(candidates[3].score - maxScore)
		pick := float32(rng.Float64()) * ((w0 + w1) + (w2 + w3))
		acc := w0
		if pick <= acc {
			return candidates[0].id
		}
		acc += w1
		if pick <= acc {
			return candidates[1].id
		}
		acc += w2
		if pick <= acc {
			return candidates[2].id
		}
		return candidates[3].id
	}
	var weights [8]float32
	var sum float32
	for i, c := range candidates {
		w := tensor.FastExpF32(c.score - maxScore)
		weights[i] = w
		sum += w
	}
	pick := float32(rng.Float64()) * sum
	var acc float32
	for i, w := range weights[:len(candidates)] {
		acc += w
		if pick <= acc {
			return candidates[i].id
		}
	}
	return candidates[len(candidates)-1].id
}

func sampleFullLogits(logits []float32, temp float32, rng *rand.Rand) int {
	return sampleFullLogitsScratch(logits, temp, rng, nil)
}

func sampleFullLogitsScratch(logits []float32, temp float32, rng float64RNG, scratch *generationScratch) int {
	if len(logits) == 2 {
		return sampleTwoLogits(logits, 1/temp, rng)
	}
	if temp == 1 {
		return sampleFullLogitsTemp1Scratch(logits, rng, scratch)
	}
	invTemp := 1 / temp
	maxScore := maxLogit(logits) * invTemp
	weights := makeSampleWeights(len(logits), scratch)
	if weights == nil && len(logits) >= sampleFullLogitsAllocWeightsMin {
		var handle *[]float32
		weights, handle = getSampleFullLogitsWeights(len(logits))
		defer putSampleFullLogitsWeights(handle, weights)
	}
	if weights == nil {
		var sum float64
		i := 0
		for ; i+3 < len(logits); i += 4 {
			w0 := float64(tensor.FastExpF32(logits[i]*invTemp - maxScore))
			w1 := float64(tensor.FastExpF32(logits[i+1]*invTemp - maxScore))
			w2 := float64(tensor.FastExpF32(logits[i+2]*invTemp - maxScore))
			w3 := float64(tensor.FastExpF32(logits[i+3]*invTemp - maxScore))
			sum += (w0 + w1) + (w2 + w3)
		}
		for ; i < len(logits); i++ {
			sum += float64(tensor.FastExpF32(logits[i]*invTemp - maxScore))
		}
		pick := rng.Float64() * sum
		var acc float64
		for i, score := range logits {
			acc += float64(tensor.FastExpF32(score*invTemp - maxScore))
			if pick <= acc {
				return i
			}
		}
		return len(logits) - 1
	}
	// Vectorized exp: weights[i] = exp(logits[i]*invTemp - maxScore)
	sum := tensor.ExpScaledVec(weights, logits, invTemp, -maxScore)
	// Build cumulative sum for sampling (parallel for large arrays)
	tensor.CumulativeSum(weights, weights)
	pick := float32(rng.Float64()) * sum
	if idx := pickCumulativeFloat32(weights, pick); idx >= 0 {
		return idx
	}
	return len(logits) - 1
}

func sampleFullLogitsTemp1Scratch(logits []float32, rng float64RNG, scratch *generationScratch) int {
	if len(logits) == 2 {
		return sampleTwoLogits(logits, 1, rng)
	}
	maxScore := maxLogit(logits)
	weights := makeSampleWeights(len(logits), scratch)
	if weights == nil && len(logits) >= sampleFullLogitsAllocWeightsMin {
		var handle *[]float32
		weights, handle = getSampleFullLogitsWeights(len(logits))
		defer putSampleFullLogitsWeights(handle, weights)
	}
	if weights == nil {
		var sum float32
		i := 0
		for ; i+3 < len(logits); i += 4 {
			w0 := tensor.FastExpF32(logits[i] - maxScore)
			w1 := tensor.FastExpF32(logits[i+1] - maxScore)
			w2 := tensor.FastExpF32(logits[i+2] - maxScore)
			w3 := tensor.FastExpF32(logits[i+3] - maxScore)
			sum += (w0 + w1) + (w2 + w3)
		}
		for ; i < len(logits); i++ {
			sum += tensor.FastExpF32(logits[i] - maxScore)
		}
		pick := float32(rng.Float64()) * sum
		var acc float32
		for i, score := range logits {
			acc += tensor.FastExpF32(score - maxScore)
			if pick <= acc {
				return i
			}
		}
		return len(logits) - 1
	}
	// Vectorized exp: weights[i] = exp(logits[i] - maxScore)
	sum := tensor.ExpScaledVec(weights, logits, 1, -maxScore)
	// Build cumulative sum for sampling (parallel for large arrays)
	tensor.CumulativeSum(weights, weights)
	pick := float32(rng.Float64()) * sum
	if idx := pickCumulativeFloat32(weights, pick); idx >= 0 {
		return idx
	}
	return len(logits) - 1
}

func sampleTwoLogits(logits []float32, invTemp float32, rng float64RNG) int {
	s0 := logits[0]
	s1 := logits[1]
	if s0 >= s1 {
		w1 := float64(tensor.FastExpF32((s1 - s0) * invTemp))
		if rng.Float64()*(1+w1) <= 1 {
			return 0
		}
		return 1
	}
	w0 := float64(tensor.FastExpF32((s0 - s1) * invTemp))
	if rng.Float64()*(w0+1) <= w0 {
		return 0
	}
	return 1
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
	return tensor.Max(logits)
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
	if topK > 0 && topK <= 8 {
		return topKCandidatesSmallUnsortedWithMax(logits, topK, scratch)
	}
	n := min(topK, len(logits))
	candidates := makeTokenScores(n, scratch)
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

func topKCandidatesSmallUnsortedWithMax(logits []float32, topK int, scratch *generationScratch) ([]tokenScore, float32) {
	n := min(topK, len(logits))
	candidates := makeTokenScores(n, scratch)
	maxScore := float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		v := logits[i]
		candidates[i] = tokenScore{id: i, score: v}
		maxScore = max32Local(maxScore, v)
	}
	if n < topK {
		return candidates[:n], maxScore
	}
	minIdx := 0
	minScore := candidates[0].score
	for i := 1; i < n; i++ {
		if candidates[i].score < minScore {
			minIdx = i
			minScore = candidates[i].score
		}
	}
	i := topK
	for ; i+7 < len(logits); i += 8 {
		v0, v1, v2, v3 := logits[i], logits[i+1], logits[i+2], logits[i+3]
		v4, v5, v6, v7 := logits[i+4], logits[i+5], logits[i+6], logits[i+7]
		if v0 <= minScore && v1 <= minScore && v2 <= minScore && v3 <= minScore &&
			v4 <= minScore && v5 <= minScore && v6 <= minScore && v7 <= minScore {
			continue
		}
		if v0 > minScore {
			candidates[minIdx] = tokenScore{id: i, score: v0}
			maxScore = max32Local(maxScore, v0)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v1 > minScore {
			candidates[minIdx] = tokenScore{id: i + 1, score: v1}
			maxScore = max32Local(maxScore, v1)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v2 > minScore {
			candidates[minIdx] = tokenScore{id: i + 2, score: v2}
			maxScore = max32Local(maxScore, v2)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v3 > minScore {
			candidates[minIdx] = tokenScore{id: i + 3, score: v3}
			maxScore = max32Local(maxScore, v3)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v4 > minScore {
			candidates[minIdx] = tokenScore{id: i + 4, score: v4}
			maxScore = max32Local(maxScore, v4)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v5 > minScore {
			candidates[minIdx] = tokenScore{id: i + 5, score: v5}
			maxScore = max32Local(maxScore, v5)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v6 > minScore {
			candidates[minIdx] = tokenScore{id: i + 6, score: v6}
			maxScore = max32Local(maxScore, v6)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
		if v7 > minScore {
			candidates[minIdx] = tokenScore{id: i + 7, score: v7}
			maxScore = max32Local(maxScore, v7)
			minIdx, minScore = minTokenScore(candidates[:n])
		}
	}
	for ; i < len(logits); i++ {
		v := logits[i]
		if v <= minScore {
			continue
		}
		candidates[minIdx] = tokenScore{id: i, score: v}
		maxScore = max32Local(maxScore, v)
		minIdx, minScore = minTokenScore(candidates[:n])
	}
	return candidates[:n], maxScore
}

func minTokenScore(x []tokenScore) (int, float32) {
	if len(x) == 4 {
		minIdx := 0
		minScore := x[0].score
		if x[1].score < minScore {
			minIdx = 1
			minScore = x[1].score
		}
		if x[2].score < minScore {
			minIdx = 2
			minScore = x[2].score
		}
		if x[3].score < minScore {
			return 3, x[3].score
		}
		return minIdx, minScore
	}
	if len(x) == 8 {
		minIdx := 0
		minScore := x[0].score
		if x[1].score < minScore {
			minIdx = 1
			minScore = x[1].score
		}
		if x[2].score < minScore {
			minIdx = 2
			minScore = x[2].score
		}
		if x[3].score < minScore {
			minIdx = 3
			minScore = x[3].score
		}
		if x[4].score < minScore {
			minIdx = 4
			minScore = x[4].score
		}
		if x[5].score < minScore {
			minIdx = 5
			minScore = x[5].score
		}
		if x[6].score < minScore {
			minIdx = 6
			minScore = x[6].score
		}
		if x[7].score < minScore {
			return 7, x[7].score
		}
		return minIdx, minScore
	}
	minIdx := 0
	minScore := x[0].score
	for i := 1; i < len(x); i++ {
		if x[i].score < minScore {
			minIdx = i
			minScore = x[i].score
		}
	}
	return minIdx, minScore
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

func getSampleFullLogitsWeights(n int) ([]float32, *[]float32) {
	if v := sampleFullLogitsWeightsPool.Get(); v != nil {
		p := v.(*[]float32)
		if cap(*p) >= n {
			return (*p)[:n], p
		}
	}
	p := new([]float32)
	*p = make([]float32, n)
	return *p, p
}

func putSampleFullLogitsWeights(p *[]float32, buf []float32) {
	if p == nil || cap(buf) == 0 || cap(buf) > sampleFullLogitsPooledWeightsMax {
		return
	}
	*p = buf[:0]
	sampleFullLogitsWeightsPool.Put(p)
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
					q4 := &tensor.Q4Matrix{Rows: int(shape[0]), Cols: int(shape[1]), Data: data, Scale: scales}
				tensor.UnpackQ4Matrix(q4)
				rt.q4w[name] = q4
					continue
				}
			}
		}
		if rt.quantization == "q6" && item.quantizable {
			if hasQ6 {
				data, scales, shape, err := q6s.Q6Row(name)
				if err == nil {
					q6 := &tensor.Q6Matrix{Rows: int(shape[0]), Cols: int(shape[1]), Data: data, Scale: scales}
				tensor.UnpackQ6Matrix(q6)
				rt.q6w[name] = q6
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
			tensor.UnpackQ6Matrix(q)
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
			tensor.UnpackQ4Matrix(q)
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
	epoch := rt.nextKVCacheEpoch()
	if maxTokens <= 0 || kvDim <= 0 {
		for i := range out {
			out[i].epoch = epoch
		}
		return out
	}
	for i := range out {
		out[i] = kvCache{
			k:     make([]float32, 0, maxTokens*kvDim),
			v:     make([]float32, 0, maxTokens*kvDim),
			kvDim: kvDim,
			epoch: epoch,
		}
	}
	return out
}

func (rt *Runtime) nextKVCacheEpoch() uint64 {
	return rt.kvCacheEpoch.Add(1)
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
	epoch := rt.nextKVCacheEpoch()
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
		caches[i].epoch = epoch
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
		weights: make([]float32, c.VocabSize),
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
	if cap(scratch.topKScores) > maxPooledTopKCandidates {
		scratch.topKScores = nil
	}
	if cap(scratch.topKWork) > maxPooledTopKCandidates {
		scratch.topKWork = nil
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

func (rt *Runtime) ensureSamplingScratchCapacity(scratch *generationScratch, opts GenerateOptions, tokens int) {
	if generationNeedsSamplingScratch(opts) {
		rt.ensureScoreCapacity(scratch, tokens)
	}
}

func (rt *Runtime) ensureLogitsCapacity(scratch *generationScratch) []float32 {
	n := rt.cfg.VocabSize
	if cap(scratch.logits) < n {
		scratch.logits = make([]float32, n)
	}
	return scratch.logits[:n]
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
			up:      make([]float32, c.IntermediateSize),
		}
	}
	return out
}

func (rt *Runtime) forwardEmbedding(embedding []float32, pos ropePos, caches []kvCache, scratch *generationScratch) ([]float32, error) {
	return rt.forwardEmbeddingMaybeLogits(embedding, pos, caches, scratch, true)
}

func (rt *Runtime) forwardEmbeddingMaybeLogits(embedding []float32, pos ropePos, caches []kvCache, scratch *generationScratch, needLogits bool) ([]float32, error) {
	fr, err := rt.forwardEmbeddingForSampling(embedding, pos, caches, scratch, needLogits, GenerateOptions{})
	return fr.logits, err
}

func (rt *Runtime) forwardEmbeddingForSampling(embedding []float32, pos ropePos, caches []kvCache, scratch *generationScratch, needLogits bool, opts GenerateOptions) (generationForwardResult, error) {
	c := rt.cfg
	h := scratch.hidden[:c.HiddenSize]
	if len(embedding) < c.HiddenSize {
		return generationForwardResult{}, fmt.Errorf("embedding length %d smaller than hidden size %d", len(embedding), c.HiddenSize)
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
		lastLayer := li+1 == c.NumHiddenLayers
		if li == 0 && lastLayer && !needLogits {
			rt.rmsNormMaybeVulkan(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
			rt.attentionCacheOnly(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin)
			return generationForwardResult{}, nil
		}
		// Try fused attention+MLP layer chain (single submit, device-local intermediates)
		if hasRoPE && !lastLayer && tl.q8.q != nil && rt.vulkanOpEnabled(vulkanOpLayerChainQ8) {
			nextNormWeight := rt.textLayers[li+1].w.ln1
			if rt.vulkanLayerChainQ8(layers[li+1].norm[:c.HiddenSize], h, h, tl.w.ln1, &caches[li], tl, ropeCos, ropeSin, tl.w.ln2, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, nextNormWeight, li > 0, true) {
				// Unified chain handled attention + MLP + AddRMSNorm for next layer
				continue
			}
		}
		// Try fused layer chain for last layer (with finalNorm, no residual readback)
		if hasRoPE && lastLayer && needLogits && tl.q8.q != nil && rt.vulkanOpEnabled(vulkanOpLayerChainQ8) {
			finalNormWeight := rt.finalNorm
			if rt.vulkanLayerChainQ8(scratch.norm[:c.HiddenSize], h, h, tl.w.ln1, &caches[li], tl, ropeCos, ropeSin, tl.w.ln2, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, finalNormWeight, li > 0, false) {
				// Unified chain handled attention + MLP + finalNorm for last layer
				continue
			}
		}
		if li == 0 {
			// Try chained RMSNorm+QKVMRoPE for F32 weights; falls back to separate calls
			if hasRoPE && rt.attentionChainedQKV(sc, &caches[li], tl, h, tl.w.ln1, hasRoPE, ropeCos, ropeSin, c) {
				// Chained path handled QKV; run attention+output+AddRMSNorm
				att, normDone := rt.attentionWithNormPostQKV(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
				if !normDone {
					rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
				}
			} else {
				// For Q8 weights, skip separate RMSNorm - the Q8 chain does it inline.
				q8ChainActive := hasRoPE && tl.q8.q != nil && rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormQ8)
				if q8ChainActive {
					// Q8 chain does RMSNorm(rawInput, ln1) + QKV + attention + output + AddRMSNorm(residual, ln2)
					// all in one submit with device-local buffers. No separate RMSNorm needed.
					if rt.vulkanChainedQKVAttentionOutAddRMSNormQ8(sc.norm, h, h, tl.w.ln1, &caches[li], tl, ropeCos, ropeSin, tl.w.ln2, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
						// chain handled everything including AddRMSNorm
					} else {
						// fallback: separate RMSNorm + attention
						rt.rmsNormMaybeVulkan(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
						att, _ := rt.attentionWithNorm(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
						rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
					}
				} else {
					rt.rmsNormMaybeVulkan(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
					if lastLayer && !needLogits {
						rt.attentionCacheOnly(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin)
						return generationForwardResult{}, nil
					}
					att, normDone := rt.attentionWithNorm(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
					if !normDone {
						rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
					}
				}
			}
		} else {
			// Try chained RMSNorm+QKVMRoPE (F32/Q8 weights); falls back to separate calls
			if hasRoPE && rt.attentionChainedQKV(sc, &caches[li], tl, h, tl.w.ln1, hasRoPE, ropeCos, ropeSin, c) {
				// Chained path handled QKV; run attention+output+AddRMSNorm
				att, normDone := rt.attentionWithNormPostQKV(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
				if !normDone {
					rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
				}
			} else {
				// For Q8 weights, skip separate RMSNorm - the Q8 chain does it inline.
				q8ChainActive := hasRoPE && tl.q8.q != nil && rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormQ8)
				if q8ChainActive {
					// Q8 chain does RMSNorm(rawInput, ln1) + QKV + attention + output + AddRMSNorm(residual, ln2)
					// all in one submit with device-local buffers. No separate RMSNorm needed.
					if rt.vulkanChainedQKVAttentionOutAddRMSNormQ8(sc.norm, h, h, tl.w.ln1, &caches[li], tl, ropeCos, ropeSin, tl.w.ln2, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
						// chain handled everything including AddRMSNorm
					} else {
						// fallback: separate RMSNorm + attention
						rt.rmsNormMaybeVulkan(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
						att, _ := rt.attentionWithNorm(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
						rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
					}
				} else {
					rt.rmsNormMaybeVulkan(sc.norm, h, tl.w.ln1, float32(c.RMSNormEps))
					if lastLayer && !needLogits {
						rt.attentionCacheOnly(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin)
						return generationForwardResult{}, nil
					}
					att, normDone := rt.attentionWithNorm(sc.norm, &caches[li], tl, sc, hasRoPE, ropeCos, ropeSin, h, tl.w.ln2, sc.norm)
					if !normDone {
						rt.addRMSNormMaybeVulkan(sc.norm, h, att, tl.w.ln2, float32(c.RMSNormEps))
					}
				}
			}
		}
		if !lastLayer {
			nextNorm := layers[li+1].norm[:c.HiddenSize]
			if !rt.mlpAddRMSNormMaybeVulkan(sc.norm, tl, sc, h, rt.textLayers[li+1].w.ln1, nextNorm, float32(c.RMSNormEps), true) {
				if !rt.mlpDownAddRMSNormCPU(nextNorm, h, sc.norm, tl, sc, rt.textLayers[li+1].w.ln1, true) {
					mlp := rt.mlp(sc.norm, tl, sc)
					rt.addRMSNormMaybeVulkan(nextNorm, h, mlp, rt.textLayers[li+1].w.ln1, float32(c.RMSNormEps))
				}
			}
		} else {
			finalNorm := scratch.norm[:c.HiddenSize]
			if !rt.mlpAddRMSNormMaybeVulkan(sc.norm, tl, sc, h, rt.finalNorm, finalNorm, float32(c.RMSNormEps), false) {
				if !rt.mlpDownAddRMSNormCPU(finalNorm, h, sc.norm, tl, sc, rt.finalNorm, false) {
					mlp := rt.mlp(sc.norm, tl, sc)
					rt.addRMSNormOutOnlyMaybeVulkan(finalNorm, h, mlp, rt.finalNorm, float32(c.RMSNormEps))
				}
			}
		}
	}
	norm := scratch.norm[:c.HiddenSize]
	if !needLogits {
		return generationForwardResult{}, nil
	}
	if c.NumHiddenLayers == 0 {
		rt.rmsNormMaybeVulkan(norm, h, rt.finalNorm, float32(c.RMSNormEps))
	}
	if generationCanReturnArgmaxOnly(opts) {
		if token, _, ok := rt.matVecArgmaxMaybeVulkan(norm, rt.lmHead, rt.q8LmHead, rt.q6LmHead, rt.q4LmHead, c.VocabSize, c.HiddenSize); ok {
			return generationForwardResult{token: token, hasToken: true}, nil
		}
		if rt.q8LmHead != nil {
			token, _ := tensor.MatVecArgmaxQ8(norm, rt.q8LmHead)
			return generationForwardResult{token: token, hasToken: true}, nil
		}
		if rt.q6LmHead != nil {
			token, _ := tensor.MatVecArgmaxQ6(norm, rt.q6LmHead)
			return generationForwardResult{token: token, hasToken: true}, nil
		}
		if rt.q4LmHead != nil {
			token, _ := tensor.MatVecArgmaxQ4(norm, rt.q4LmHead)
			return generationForwardResult{token: token, hasToken: true}, nil
		}
		if len(rt.lmHead) >= c.VocabSize*c.HiddenSize {
			token, _ := tensor.MatVecArgmax(norm, rt.lmHead, c.VocabSize, c.HiddenSize)
			return generationForwardResult{token: token, hasToken: true}, nil
		}
	}
	if generationCanReturnCandidatesOnly(opts) {
		if candidates, ok := rt.matVecTopKMaybeVulkan(norm, rt.lmHead, rt.q8LmHead, rt.q6LmHead, rt.q4LmHead, c.VocabSize, c.HiddenSize, opts.TopK, scratch); ok {
			return generationForwardResult{candidates: candidates, candidateTemp: opts.Temperature, hasCandidates: true}, nil
		}
		if candidates, ok := rt.matVecTopKMaybeCPU(norm, rt.lmHead, rt.q8LmHead, rt.q6LmHead, rt.q4LmHead, c.VocabSize, c.HiddenSize, opts.TopK, opts.Temperature, scratch); ok {
			return generationForwardResult{candidates: candidates, candidateTemp: opts.Temperature, hasCandidates: true}, nil
		}
	}
	logits := rt.ensureLogitsCapacity(scratch)
	rt.matVecMaybeQuant(logits, norm, rt.lmHead, rt.q8LmHead, rt.q6LmHead, rt.q4LmHead, c.VocabSize, c.HiddenSize)
	return generationForwardResult{logits: logits}, nil
}

func (rt *Runtime) attention(x []float32, cache *kvCache, tl *textLayer, sc *layerScratch, hasRoPE bool, ropeCos, ropeSin []float32) []float32 {
	out, _ := rt.attentionWithNorm(x, cache, tl, sc, hasRoPE, ropeCos, ropeSin, nil, nil, nil)
	return out
}

func (rt *Runtime) attentionCacheOnly(x []float32, cache *kvCache, tl *textLayer, sc *layerScratch, hasRoPE bool, ropeCos, ropeSin []float32) {
	c := rt.cfg
	kvRows := c.NumKeyValueHeads * c.HeadDim
	k := sc.k[:kvRows]
	v := sc.v[:kvRows]
	kvReady := false
	kHasRoPE := false
	if rt.backend == "vulkan" {
		kvReady, kHasRoPE = rt.fusedKV(k, v, x, tl, kvRows, c.HiddenSize, c.NumKeyValueHeads, c.HeadDim, hasRoPE, ropeCos, ropeSin)
	}
	if !kvReady {
		rt.matVecMaybeQuant(k, x, tl.w.k, tl.q8.k, tl.q6.k, tl.q4.k, kvRows, c.HiddenSize)
		rt.matVecMaybeQuant(v, x, tl.w.v, tl.q8.v, tl.q6.v, tl.q4.v, kvRows, c.HiddenSize)
	}
	if hasRoPE && !kHasRoPE {
		rt.mropeMaybeVulkan(k, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin)
	}
	cache.append(k, v)
}

// attentionChainedQKV attempts the chained RMSNorm+QKVMRoPE path for a text
// attention layer.  Fills sc.q, sc.k, sc.v and appends to cache.  Returns true
// on success, false to fall back to separate calls.
func (rt *Runtime) attentionChainedQKV(sc *layerScratch, cache *kvCache, tl *textLayer, rawInput, normWeight []float32, hasRoPE bool, ropeCos, ropeSin []float32, c *config.Config) bool {
	if !hasRoPE {
		return false
	}
	// For Q8 weights with the full chain enabled, skip the RMSNorm+QKV chain.
	// The full Q8 chain (QKV+attention+output+AddRMSNorm in one submit with
	// device-local buffers) is handled by attentionWithNorm instead.
	if tl.q8.q != nil && rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormQ8) {
		return false
	}
	qRows := c.NumAttentionHeads * c.HeadDim
	kvRows := c.NumKeyValueHeads * c.HeadDim
	q := sc.q[:qRows]
	k := sc.k[:kvRows]
	v := sc.v[:kvRows]
	eps := float32(c.RMSNormEps)
	// Try chained RMSNorm + fused Q8 QKV+MRoPE in a single command buffer
	if rt.fusedQKVMRoPEWithNormQ8(q, k, v, rawInput, normWeight, tl, qRows, kvRows, c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin, eps) {
		cache.append(k, v)
		return true
	}

	if rt.fusedQKVMRoPEWithNorm(q, k, v, rawInput, normWeight, tl, qRows, kvRows, c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin, eps) {
		cache.append(k, v)
		return true
	}
	return false
}

// attentionWithNormPostQKV runs the attention + output + AddRMSNorm portion
// of attentionWithNorm, assuming QKV has already been computed and cached.
// matVecOutAddRMSNormCPU tries the CPU fused matvec+AddRMSNorm for the attention
// output projection. Returns true if fused, false if caller should use separate calls.
func (rt *Runtime) matVecOutAddRMSNormCPU(normOut, residual, headOut []float32, tl *textLayer, hiddenSize, qRows int, normWeight []float32, eps float32) bool {
	if len(normOut) < hiddenSize || len(residual) < hiddenSize || len(normWeight) < hiddenSize {
		return false
	}
	if tl.q8.o != nil {
		tensor.MatVecQ8AddRMSNorm(normOut, residual, headOut, tl.q8.o, normWeight, eps)
		return true
	}
	if tl.q4.o != nil {
		tensor.MatVecQ4AddRMSNorm(normOut, residual, headOut, tl.q4.o, normWeight, eps)
		return true
	}
	if tl.q6.o != nil {
		tensor.MatVecQ6AddRMSNorm(normOut, residual, headOut, tl.q6.o, normWeight, eps)
		return true
	}
	if tl.w.o != nil {
		tensor.MatVecAddRMSNorm(normOut, residual, headOut, tl.w.o, hiddenSize, qRows, normWeight, eps)
		return true
	}
	return false
}

func (rt *Runtime) attentionWithNormPostQKV(x []float32, cache *kvCache, tl *textLayer, sc *layerScratch, hasRoPE bool, ropeCos, ropeSin, residual, normWeight, normOut []float32) ([]float32, bool) {
	c := rt.cfg
	qRows := c.NumAttentionHeads * c.HeadDim
	q := sc.q[:qRows]
	headOut := sc.headOut[:qRows]
	// Attention + output projection + AddRMSNorm (same as attentionWithNorm after QKV)
	if tl.q8.o == nil && tl.q6.o == nil && tl.q4.o == nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNorm(normOut, residual, q, cache, tl.w.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q8.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ8(normOut, residual, q, cache, tl.q8.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q6.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ6(normOut, residual, q, cache, tl.q6.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q4.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ4(normOut, residual, q, cache, tl.q4.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q8.o != nil &&
		rt.vulkanTextCacheAttentionOutQ8(sc.att[:c.HiddenSize], q, cache, tl.q8.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q6.o != nil &&
		rt.vulkanTextCacheAttentionOutQ6(sc.att[:c.HiddenSize], q, cache, tl.q6.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q4.o != nil &&
		rt.vulkanTextCacheAttentionOutQ4(sc.att[:c.HiddenSize], q, cache, tl.q4.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q8.o == nil && tl.q6.o == nil && tl.q4.o == nil &&
		rt.vulkanTextCacheAttentionOut(sc.att[:c.HiddenSize], q, cache, tl.w.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if rt.vulkanTextCacheAttention(headOut, q, cache, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		if rt.matVecOutAddRMSNormCPU(normOut, residual, headOut, tl, c.HiddenSize, qRows, normWeight, float32(c.RMSNormEps)) {
			return nil, true
		}
		out := sc.att[:c.HiddenSize]
		rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
		return out, false
	}
	if cache.len <= 6 {
		group := c.NumAttentionHeads / c.NumKeyValueHeads
		scale := invSqrt(c.HeadDim)
		for kvh := 0; kvh < c.NumKeyValueHeads; kvh++ {
			for g := 0; g < group; g++ {
				h := kvh*group + g
				dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
				if cache.len == 1 {
					copyCacheValue(dst, cache, kvh, c.HeadDim)
					continue
				}
				cacheAttentionSmall(dst, q[h*c.HeadDim:(h+1)*c.HeadDim], cache, kvh, c.HeadDim, scale)
			}
		}
		if rt.matVecOutAddRMSNormCPU(normOut, residual, headOut, tl, c.HiddenSize, qRows, normWeight, float32(c.RMSNormEps)) {
			return nil, true
		}
		out := sc.att[:c.HiddenSize]
		rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
		return out, false
	}
	group := c.NumAttentionHeads / c.NumKeyValueHeads
	scale := invSqrt(c.HeadDim)
	numHeads := c.NumAttentionHeads
	if cap(sc.scores) < numHeads*cache.len {
		sc.scores = make([]float32, numHeads*cache.len)
	}
	scoresBuf := sc.scores[:numHeads*cache.len]
	if numHeads >= 8 && cache.len >= 64 && runtime.GOMAXPROCS(0) > 1 {
		workers := min(numHeads, runtime.GOMAXPROCS(0))
		if workers > numHeads {
			workers = numHeads
		}
		chunk := (numHeads + workers - 1) / workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start := w * chunk
			end := start + chunk
			if end > numHeads {
				end = numHeads
			}
			if start >= end {
				continue
			}
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for h := start; h < end; h++ {
					kvh := h / group
					scores := scoresBuf[h*cache.len : (h+1)*cache.len]
					dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
					qv := q[h*c.HeadDim : (h+1)*c.HeadDim]
					cacheAttentionScores(scores, qv, cache, kvh, c.HeadDim, scale)
					tensor.SoftmaxInPlace(scores)
					weightedCacheValueSum(dst, cache, kvh, c.HeadDim, scores)
				}
			}(start, end)
		}
		wg.Wait()
	} else {
		for h := 0; h < numHeads; h++ {
			kvh := h / group
			scores := scoresBuf[h*cache.len : (h+1)*cache.len]
			dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
			qv := q[h*c.HeadDim : (h+1)*c.HeadDim]
			cacheAttentionScores(scores, qv, cache, kvh, c.HeadDim, scale)
			tensor.SoftmaxInPlace(scores)
			weightedCacheValueSum(dst, cache, kvh, c.HeadDim, scores)
		}
	}
	out := sc.att[:c.HiddenSize]
	rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
	return out, false
}

func (rt *Runtime) attentionWithNorm(x []float32, cache *kvCache, tl *textLayer, sc *layerScratch, hasRoPE bool, ropeCos, ropeSin, residual, normWeight, normOut []float32) ([]float32, bool) {
	c := rt.cfg
	qRows := c.NumAttentionHeads * c.HeadDim
	kvRows := c.NumKeyValueHeads * c.HeadDim
	q := sc.q[:qRows]
	k := sc.k[:kvRows]
	v := sc.v[:kvRows]
	if cache.len == 0 {
		qkvHasRoPE := false
		qkvReady := false
		if rt.backend == "vulkan" {
			qkvReady, qkvHasRoPE = rt.fusedFirstTokenKV(q, k, v, x, tl, qRows, kvRows, c.HiddenSize, c.NumKeyValueHeads, c.HeadDim, hasRoPE, ropeCos, ropeSin)
		}
		if !qkvReady {
			rt.matVecMaybeQuant(k, x, tl.w.k, tl.q8.k, tl.q6.k, tl.q4.k, kvRows, c.HiddenSize)
			rt.matVecMaybeQuant(v, x, tl.w.v, tl.q8.v, tl.q6.v, tl.q4.v, kvRows, c.HiddenSize)
		}
		if hasRoPE && !qkvHasRoPE {
			rt.mropeMaybeVulkan(k, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin)
		}
		cache.append(k, v)
		if rt.vulkanTextFirstTokenValueOutAddRMSNorm(normOut, residual, cache, tl, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
			return nil, true
		}
		out := sc.att[:c.HiddenSize]
		if len(normOut) < qRows || len(residual) < qRows || len(normWeight) < qRows {
			if rt.vulkanTextFirstTokenAttentionOut(out, cache, tl, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
				return out, false
			}
		} else if rt.vulkanTextFirstTokenAttentionOut(out, cache, tl, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
			rt.addRMSNormMaybeVulkan(normOut, residual, out, normWeight, float32(c.RMSNormEps))
			return nil, true
		}
		headOut := sc.headOut[:qRows]
		expandValueHeads(headOut, v, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim)
		if rt.matVecAddRMSNormMaybeVulkan(normOut, residual, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, normWeight, c.HiddenSize, qRows, float32(c.RMSNormEps)) {
			return nil, true
		}
		// CPU fused matvec + AddRMSNorm fallback
		if len(normOut) >= c.HiddenSize && len(residual) >= c.HiddenSize && len(normWeight) >= c.HiddenSize {
			eps := float32(c.RMSNormEps)
			if tl.q8.o != nil {
				tensor.MatVecQ8AddRMSNorm(normOut, residual, headOut, tl.q8.o, normWeight, eps)
				return nil, true
			}
			if tl.q4.o != nil {
				tensor.MatVecQ4AddRMSNormOutOnly(normOut, residual, headOut, tl.q4.o, normWeight, eps)
				return nil, true
			}
			if tl.q6.o != nil {
				tensor.MatVecQ6AddRMSNormOutOnly(normOut, residual, headOut, tl.q6.o, normWeight, eps)
				return nil, true
			}
			if tl.w.o != nil {
				tensor.MatVecAddRMSNorm(normOut, residual, headOut, tl.w.o, c.HiddenSize, qRows, normWeight, eps)
				return nil, true
			}
		}
		rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
		return out, false
	}
	// Try the full chain: QKV+MRoPE + attention+output+AddRMSNorm in one command buffer
	// Try the full chain: QKV+MRoPE + attention+output+AddRMSNorm in one command buffer.
	// F32 path (only when no quantized weights present).
	if hasRoPE && tl.q8.q == nil && tl.q6.q == nil && tl.q4.q == nil &&
		rt.vulkanChainedQKVAttentionOutAddRMSNorm(normOut, residual, x, cache, tl, ropeCos, ropeSin, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
qkvHasRoPE := false
	if hasRoPE {
		qkvHasRoPE = rt.fusedQKVMRoPE(q, k, v, x, tl, qRows, kvRows, c.HiddenSize, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin)
	}
	if !qkvHasRoPE && !rt.fusedQKV(q, k, v, x, tl, qRows, kvRows, c.HiddenSize) {
		rt.matVecMaybeQuant(q, x, tl.w.q, tl.q8.q, tl.q6.q, tl.q4.q, qRows, c.HiddenSize)
		rt.matVecMaybeQuant(k, x, tl.w.k, tl.q8.k, tl.q6.k, tl.q4.k, kvRows, c.HiddenSize)
		rt.matVecMaybeQuant(v, x, tl.w.v, tl.q8.v, tl.q6.v, tl.q4.v, kvRows, c.HiddenSize)
	}
	if hasRoPE && !qkvHasRoPE {
		rt.mropePairMaybeVulkan(q, k, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim, ropeCos, ropeSin)
	}
	cache.append(k, v)

	headOut := sc.headOut[:qRows]
	if tl.q8.o == nil && tl.q6.o == nil && tl.q4.o == nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNorm(normOut, residual, q, cache, tl.w.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q8.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ8(normOut, residual, q, cache, tl.q8.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q6.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ6(normOut, residual, q, cache, tl.q6.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q4.o != nil &&
		rt.vulkanTextCacheAttentionOutAddRMSNormQ4(normOut, residual, q, cache, tl.q4.o, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return nil, true
	}
	if tl.q8.o != nil &&
		rt.vulkanTextCacheAttentionOutQ8(sc.att[:c.HiddenSize], q, cache, tl.q8.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q6.o != nil &&
		rt.vulkanTextCacheAttentionOutQ6(sc.att[:c.HiddenSize], q, cache, tl.q6.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q4.o != nil &&
		rt.vulkanTextCacheAttentionOutQ4(sc.att[:c.HiddenSize], q, cache, tl.q4.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if tl.q8.o == nil && tl.q6.o == nil && tl.q4.o == nil &&
		rt.vulkanTextCacheAttentionOut(sc.att[:c.HiddenSize], q, cache, tl.w.o, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		return sc.att[:c.HiddenSize], false
	}
	if rt.vulkanTextCacheAttention(headOut, q, cache, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {
		if rt.matVecOutAddRMSNormCPU(normOut, residual, headOut, tl, c.HiddenSize, qRows, normWeight, float32(c.RMSNormEps)) {
			return nil, true
		}
		out := sc.att[:c.HiddenSize]
		rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
		return out, false
	}
	if cache.len <= 6 {
		group := c.NumAttentionHeads / c.NumKeyValueHeads
		scale := invSqrt(c.HeadDim)
		for kvh := 0; kvh < c.NumKeyValueHeads; kvh++ {
			for g := 0; g < group; g++ {
				h := kvh*group + g
				dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
				if cache.len == 1 {
					copyCacheValue(dst, cache, kvh, c.HeadDim)
					continue
				}
				cacheAttentionSmall(dst, q[h*c.HeadDim:(h+1)*c.HeadDim], cache, kvh, c.HeadDim, scale)
			}
		}
		if rt.matVecOutAddRMSNormCPU(normOut, residual, headOut, tl, c.HiddenSize, qRows, normWeight, float32(c.RMSNormEps)) {
			return nil, true
		}
		out := sc.att[:c.HiddenSize]
		rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
		return out, false
	}
	group := c.NumAttentionHeads / c.NumKeyValueHeads
	scale := invSqrt(c.HeadDim)
	numHeads := c.NumAttentionHeads
	if cap(sc.scores) < numHeads*cache.len {
		sc.scores = make([]float32, numHeads*cache.len)
	}
	scoresBuf := sc.scores[:numHeads*cache.len]
	if numHeads >= 8 && cache.len >= 64 && runtime.GOMAXPROCS(0) > 1 {
		workers := min(numHeads, runtime.GOMAXPROCS(0))
		if workers > numHeads {
			workers = numHeads
		}
		chunk := (numHeads + workers - 1) / workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			start := w * chunk
			end := start + chunk
			if end > numHeads {
				end = numHeads
			}
			if start >= end {
				continue
			}
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for h := start; h < end; h++ {
					kvh := h / group
					scores := scoresBuf[h*cache.len : (h+1)*cache.len]
					dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
					qv := q[h*c.HeadDim : (h+1)*c.HeadDim]
					cacheAttentionScores(scores, qv, cache, kvh, c.HeadDim, scale)
					tensor.SoftmaxInPlace(scores)
					weightedCacheValueSum(dst, cache, kvh, c.HeadDim, scores)
				}
			}(start, end)
		}
		wg.Wait()
	} else {
		for h := 0; h < numHeads; h++ {
			kvh := h / group
			scores := scoresBuf[h*cache.len : (h+1)*cache.len]
			dst := headOut[h*c.HeadDim : (h+1)*c.HeadDim]
			qv := q[h*c.HeadDim : (h+1)*c.HeadDim]
			cacheAttentionScores(scores, qv, cache, kvh, c.HeadDim, scale)
			tensor.SoftmaxInPlace(scores)
			weightedCacheValueSum(dst, cache, kvh, c.HeadDim, scores)
		}
	}
	out := sc.att[:c.HiddenSize]
	// Try CPU fused matvec + AddRMSNorm when Vulkan path failed
	if len(normOut) >= c.HiddenSize && len(residual) >= c.HiddenSize && len(normWeight) >= c.HiddenSize {
		eps := float32(c.RMSNormEps)
		if tl.q8.o != nil {
			tensor.MatVecQ8AddRMSNorm(normOut, residual, headOut, tl.q8.o, normWeight, eps)
			return nil, true
		}
		if tl.q4.o != nil {
			tensor.MatVecQ4AddRMSNormOutOnly(normOut, residual, headOut, tl.q4.o, normWeight, eps)
			return nil, true
		}
		if tl.q6.o != nil {
			tensor.MatVecQ6AddRMSNormOutOnly(normOut, residual, headOut, tl.q6.o, normWeight, eps)
			return nil, true
		}
		if tl.w.o != nil {
			tensor.MatVecAddRMSNorm(normOut, residual, headOut, tl.w.o, c.HiddenSize, qRows, normWeight, eps)
			return nil, true
		}
	}
	rt.matVecMaybeQuant(out, headOut, tl.w.o, tl.q8.o, tl.q6.o, tl.q4.o, c.HiddenSize, qRows)
	return out, false
}

func (rt *Runtime) vulkanTextCacheAttention(out, q []float32, cache *kvCache, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionF32) || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok {
		return false
	}
	if !textAttentionOnlyWorkReady(cache.len, numHeads, headDim) {
		return false
	}
	if len(out) < qRows || len(q) < qRows || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionF32(out, q, kCache, vCache, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionF32, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOut(out, q []float32, cache *kvCache, w []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutF32) || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, false) {
		return false
	}
	if len(out) < qRows || len(q) < qRows || !f32MatVecWeightsReady(w, qRows, qRows) || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutF32(out, q, kCache, vCache, w, rt.zeroBias(qRows), cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutF32, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutAddRMSNorm(normOut, residual, q []float32, cache *kvCache, w, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutAddRMSNormF32) || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || !f32MatVecWeightsReady(w, qRows, qRows) || len(normWeight) < qRows || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutAddRMSNormF32(normOut, residual, q, kCache, vCache, w, rt.zeroBias(qRows), normWeight, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutAddRMSNormF32, err)
	}
	return false
}

// vulkanChainedQKVAttentionOutAddRMSNorm chains the fused QKV+MRoPE dispatch
// with attention+output+AddRMSNorm into a single command buffer.  The QKV
// output (q/k/v) stays in GPU memory and is fed directly to the attention
// kernel.  The new token's k/v is copied into the GPU KV cache buffer via
// vkCmdCopyBuffer before the attention dispatch.  Only for F32 weights and
// layers where cache.len > 0 (generation path, li > 0).
func (rt *Runtime) vulkanChainedQKVAttentionOutAddRMSNorm(normOut, residual, x []float32, cache *kvCache, tl *textLayer, cosTable, sinTable, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormF32) || tl == nil || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || headDim%2 != 0 {
		return false
	}
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	hidden := len(x)
	if hidden <= 0 || qRows <= 0 || kvRows <= 0 {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(x) < hidden || len(normWeight) < qRows {
		return false
	}
	if !f32MatVecWeightsReady(tl.w.q, qRows, hidden) || !f32MatVecWeightsReady(tl.w.k, kvRows, hidden) || !f32MatVecWeightsReady(tl.w.v, kvRows, hidden) || !f32MatVecWeightsReady(tl.w.o, qRows, qRows) {
		return false
	}
	if !fusedMRoPEShapeOK(make([]float32, qRows), make([]float32, kvRows), numHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	newK := make([]float32, kvRows)
	newV := make([]float32, kvRows)
	if err := backend.VulkanChainedQKVMRoPEAttentionOutAddRMSNormF32(
		normOut, residual, x,
		tl.w.q, tl.w.k, tl.w.v,
		cosTable, sinTable,
		tl.w.o, rt.zeroBias(qRows), normWeight,
		kCache, vCache,
		cache.epoch, cache.len, hidden, numHeads, kvHeads, headDim,
		newK, newV,
	); err == nil {
		cache.append(newK, newV)
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedQKVAttentionOutAddRMSNormF32, err)
	}
	return false
}


// vulkanChainedQKVAttentionOutAddRMSNormQ8 chains fused Q8 QKV+MRoPE with
// attention+output+AddRMSNorm into a single command buffer for Q8-quantised
// models.  The q/k/v stay in GPU memory between QKV and attention.
func (rt *Runtime) vulkanChainedQKVAttentionOutAddRMSNormQ8(normOut, residual, rawInput, ln1Weight []float32, cache *kvCache, tl *textLayer, cosTable, sinTable, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormQ8) || tl == nil || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || headDim%2 != 0 {
		return false
	}
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	hidden := len(rawInput)
	if hidden <= 0 || qRows <= 0 || kvRows <= 0 {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(rawInput) < hidden || len(ln1Weight) < hidden || len(normWeight) < qRows {
		return false
	}
	if tl.q8.q == nil || tl.q8.k == nil || tl.q8.v == nil || tl.q8.o == nil {
		return false
	}
	if !q8MatVecShapeOK(tl.q8.q, qRows, hidden) || !q8MatVecShapeOK(tl.q8.k, kvRows, hidden) || !q8MatVecShapeOK(tl.q8.v, kvRows, hidden) || !q8MatVecShapeOK(tl.q8.o, qRows, qRows) {
		return false
	}
	if !fusedMRoPEShapeOK(make([]float32, qRows), make([]float32, kvRows), numHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	newK := make([]float32, kvRows)
	newV := make([]float32, kvRows)
	if err := backend.VulkanChainedQKVMRoPEAttentionOutAddRMSNormQ8(
		normOut, residual, rawInput, ln1Weight,
		tl.q8.q, tl.q8.k, tl.q8.v,
		cosTable, sinTable,
		tl.q8.o, rt.zeroBias(qRows), normWeight,
		kCache, vCache,
		cache.epoch, cache.len, hidden, numHeads, kvHeads, headDim,
		newK, newV,
	); err == nil {
		cache.append(newK, newV)
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedQKVAttentionOutAddRMSNormQ8, err)
	}
	return false
}

// vulkanChainedSwiGLUDownAddRMSNormQ8 chains SwiGLU(gate,up) + Down + AddRMSNorm
// in a single command buffer with device-local intermediate buffers.
func (rt *Runtime) vulkanChainedSwiGLUDownAddRMSNormQ8(normOut, residual, x []float32, tl *textLayer, normWeight []float32, readResidual bool) bool {
	if !rt.vulkanOpEnabled(vulkanOpChainedSwiGLUDownAddRMSNormQ8) || !rt.vulkanOpEnabled(vulkanOpSwiGLUDownAddRMSNormQ8) || tl == nil {
		return false
	}
	if tl.q8.gate == nil || tl.q8.up == nil || tl.q8.down == nil {
		return false
	}
	c := rt.cfg
	useVulkan := fusedSwiGLUWork(c.IntermediateSize, c.HiddenSize, c.HiddenSize) >= vulkanMatVecMinWork()
	if !useVulkan || c.RMSNormEps != 1e-6 {
		return false
	}
	if !q8SwiGLUDownShapeOK(normOut, x, tl.q8.gate, tl.q8.up, tl.q8.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) {
		return false
	}
	if err := backend.VulkanChainedSwiGLUDownAddRMSNormQ8(normOut, residual, x, tl.q8.gate, tl.q8.up, tl.q8.down, normWeight); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedSwiGLUDownAddRMSNormQ8, err)
	}
	return false
}

// vulkanLayerChainQ8 fuses the attention and MLP chains into a single submit
// with device-local intermediates.  Returns true if the fused path was used.
func (rt *Runtime) vulkanLayerChainQ8(normOut, residual, rawInput, ln1Weight []float32, cache *kvCache, tl *textLayer, cosTable, sinTable, ln2Weight []float32, numHeads, kvHeads, headDim int, nextNormWeight []float32, devInputReady bool, readResidual bool) bool {
	if !rt.vulkanOpEnabled(vulkanOpLayerChainQ8) || tl == nil || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || headDim%2 != 0 {
		return false
	}
	c := rt.cfg
	hidden := c.HiddenSize
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	if hidden <= 0 || qRows <= 0 || kvRows <= 0 {
		return false
	}
	if len(normOut) < hidden || len(residual) < hidden || len(rawInput) < hidden || len(ln1Weight) < hidden || len(ln2Weight) < qRows || len(nextNormWeight) < hidden {
		return false
	}
	if tl.q8.q == nil || tl.q8.k == nil || tl.q8.v == nil || tl.q8.o == nil || tl.q8.gate == nil || tl.q8.up == nil || tl.q8.down == nil {
		return false
	}
	if !q8MatVecShapeOK(tl.q8.q, qRows, hidden) || !q8MatVecShapeOK(tl.q8.k, kvRows, hidden) || !q8MatVecShapeOK(tl.q8.v, kvRows, hidden) || !q8MatVecShapeOK(tl.q8.o, qRows, qRows) {
		return false
	}
	if !fusedMRoPEShapeOK(make([]float32, qRows), make([]float32, kvRows), numHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	if !q8SwiGLUDownShapeOK(normOut, rawInput, tl.q8.gate, tl.q8.up, tl.q8.down, c.IntermediateSize, hidden, hidden) {
		return false
	}
	if c.RMSNormEps != 1e-6 {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	newK := make([]float32, kvRows)
	newV := make([]float32, kvRows)
	zeroBias := rt.zeroBias(qRows)
	if err := backend.VulkanLayerChainQ8Win(
		normOut, residual, rawInput, ln1Weight,
		tl.q8.q, tl.q8.k, tl.q8.v,
		cosTable, sinTable,
		tl.q8.o, zeroBias, ln2Weight,
		kCache, vCache,
		cache.epoch, cache.len, hidden, numHeads, kvHeads, headDim,
		newK, newV,
		tl.q8.gate, tl.q8.up, tl.q8.down,
		nextNormWeight,
		devInputReady,
		readResidual,
	); err == nil {
		cache.append(newK, newV)
		return true
	} else {
		rt.disableVulkanOp(vulkanOpLayerChainQ8, err)
	}
	return false
}

func (rt *Runtime) vulkanTextFirstTokenValueOutAddRMSNorm(normOut, residual []float32, cache *kvCache, tl *textLayer, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if tl == nil || cache == nil || cache.len != 1 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, kvDim, _, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok {
		return false
	}
	if !textFirstTokenValueOutNormWorkReady(qRows, kvDim) {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(cache.k) < kvDim || len(cache.v) < kvDim || len(normWeight) < qRows {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if tl.q8.o != nil {
		if !q8MatVecShapeOK(tl.q8.o, qRows, qRows) || !rt.vulkanOpEnabled(vulkanOpTextFirstTokenValueOutAddRMSNormQ8) {
			return false
		}
		if err := backend.VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, kCache, vCache, tl.q8.o, normWeight, cache.epoch, numHeads, kvHeads, headDim); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpTextFirstTokenValueOutAddRMSNormQ8, err)
		}
		return false
	}
	if tl.q6.o != nil {
		if !q6MatVecShapeOK(tl.q6.o, qRows, qRows) || !rt.vulkanOpEnabled(vulkanOpTextFirstTokenValueOutAddRMSNormQ6) {
			return false
		}
		if err := backend.VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, kCache, vCache, tl.q6.o, normWeight, cache.epoch, numHeads, kvHeads, headDim); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpTextFirstTokenValueOutAddRMSNormQ6, err)
		}
		return false
	}
	if tl.q4.o != nil {
		if !q4MatVecShapeOK(tl.q4.o, qRows, qRows) || !rt.vulkanOpEnabled(vulkanOpTextFirstTokenValueOutAddRMSNormQ4) {
			return false
		}
		if err := backend.VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, kCache, vCache, tl.q4.o, normWeight, cache.epoch, numHeads, kvHeads, headDim); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpTextFirstTokenValueOutAddRMSNormQ4, err)
		}
		return false
	}
	if !rt.vulkanOpEnabled(vulkanOpTextFirstTokenValueOutAddRMSNormF32) || !f32MatVecWeightsReady(tl.w.o, qRows, qRows) {
		return false
	}
	if err := backend.VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, kCache, vCache, tl.w.o, rt.zeroBias(qRows), normWeight, cache.epoch, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextFirstTokenValueOutAddRMSNormF32, err)
	}
	return false
}

func (rt *Runtime) vulkanTextFirstTokenAttentionOut(out []float32, cache *kvCache, tl *textLayer, numHeads, kvHeads, headDim int) bool {
	if tl == nil || cache == nil || cache.len != 1 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows := numHeads * headDim
	if len(out) < qRows {
		return false
	}
	q := rt.zeroBias(qRows)
	if tl.q8.o != nil {
		return rt.vulkanTextCacheAttentionOutQ8(out, q, cache, tl.q8.o, numHeads, kvHeads, headDim)
	}
	if tl.q6.o != nil {
		return rt.vulkanTextCacheAttentionOutQ6(out, q, cache, tl.q6.o, numHeads, kvHeads, headDim)
	}
	if tl.q4.o != nil {
		return rt.vulkanTextCacheAttentionOutQ4(out, q, cache, tl.q4.o, numHeads, kvHeads, headDim)
	}
	return rt.vulkanTextCacheAttentionOut(out, q, cache, tl.w.o, numHeads, kvHeads, headDim)
}

func (rt *Runtime) vulkanTextCacheAttentionOutAddRMSNormQ8(normOut, residual, q []float32, cache *kvCache, w *tensor.Q8Matrix, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutAddRMSNormQ8) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || !q8MatVecShapeOK(w, qRows, qRows) || len(normWeight) < qRows || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutAddRMSNormQ8(normOut, residual, q, kCache, vCache, w, normWeight, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutAddRMSNormQ8, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutAddRMSNormQ6(normOut, residual, q []float32, cache *kvCache, w *tensor.Q6Matrix, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutAddRMSNormQ6) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || !q6MatVecShapeOK(w, qRows, qRows) || len(normWeight) < qRows || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutAddRMSNormQ6(normOut, residual, q, kCache, vCache, w, normWeight, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutAddRMSNormQ6, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutAddRMSNormQ4(normOut, residual, q []float32, cache *kvCache, w *tensor.Q4Matrix, normWeight []float32, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutAddRMSNormQ4) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
		return false
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || !q4MatVecShapeOK(w, qRows, qRows) || len(normWeight) < qRows || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutAddRMSNormQ4(normOut, residual, q, kCache, vCache, w, normWeight, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutAddRMSNormQ4, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutQ8(out, q []float32, cache *kvCache, w *tensor.Q8Matrix, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutQ8) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, false) {
		return false
	}
	if len(out) < qRows || len(q) < qRows || !q8MatVecShapeOK(w, qRows, qRows) || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutQ8(out, q, kCache, vCache, w, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutQ8, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutQ6(out, q []float32, cache *kvCache, w *tensor.Q6Matrix, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutQ6) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, false) {
		return false
	}
	if len(out) < qRows || len(q) < qRows || !q6MatVecShapeOK(w, qRows, qRows) || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutQ6(out, q, kCache, vCache, w, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutQ6, err)
	}
	return false
}

func (rt *Runtime) vulkanTextCacheAttentionOutQ4(out, q []float32, cache *kvCache, w *tensor.Q4Matrix, numHeads, kvHeads, headDim int) bool {
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutQ4) || cache == nil || w == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return false
	}
	qRows, _, cacheElems, ok := checkedTextAttentionDims(cache, numHeads, kvHeads, headDim)
	if !ok || w.Rows != qRows || w.Cols != qRows {
		return false
	}
	if !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, false) {
		return false
	}
	if len(out) < qRows || len(q) < qRows || !q4MatVecShapeOK(w, qRows, qRows) || !textCacheStorageReady(cache, cacheElems) {
		return false
	}
	kCache, vCache := cache.vulkanBufferSlices()
	if err := backend.VulkanTextAttentionOutQ4(out, q, kCache, vCache, w, cache.epoch, cache.len, numHeads, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpTextAttentionOutQ4, err)
	}
	return false
}

func vulkanTextAttentionMinWork() int {
	return cachedVulkanMinWork(&vulkanTextAttentionMinWorkOnce, &vulkanTextAttentionMinWorkValue, vulkanTextAttentionMinWorkEnv, 0)
}

func vulkanMatVecMinWork() int {
	return cachedVulkanMinWork(&vulkanMatVecMinWorkOnce, &vulkanMatVecMinWorkValue, vulkanMatVecMinWorkEnv, defaultVulkanMatVecMinWork)
}

func vulkanVectorMinWork() int {
	return cachedVulkanMinWork(&vulkanVectorMinWorkOnce, &vulkanVectorMinWorkValue, vulkanVectorMinWorkEnv, defaultVulkanVectorMinWork)
}

var (
	vulkanMatVecMinWorkOnce         sync.Once
	vulkanMatVecMinWorkValue        int
	vulkanTextAttentionMinWorkOnce  sync.Once
	vulkanTextAttentionMinWorkValue int
	vulkanVectorMinWorkOnce         sync.Once
	vulkanVectorMinWorkValue        int
	vulkanVisionMinWorkOnce         sync.Once
	vulkanVisionMinWorkValue        int
)

func cachedVulkanMinWork(once *sync.Once, value *int, env string, defaultValue int) int {
	once.Do(func() {
		*value = vulkanMinWorkFromEnv(env, defaultValue)
	})
	return *value
}

func vulkanMinWorkFromEnv(env string, defaultValue int) int {
	raw := os.Getenv(env)
	if raw == "" {
		return defaultValue
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return defaultValue
	}
	return n
}

func (rt *Runtime) zeroBias(n int) []float32 {
	if n <= 0 {
		return nil
	}
	if n <= len(rt.zeroBiasSmall) {
		return rt.zeroBiasSmall[:n]
	}
	rt.zeroBiasMu.Lock()
	defer rt.zeroBiasMu.Unlock()
	if cap(rt.zeroBiasBuf) < n {
		rt.zeroBiasBuf = make([]float32, n)
	}
	rt.zeroBiasBuf = rt.zeroBiasBuf[:n]
	return rt.zeroBiasBuf
}

func copyCacheValue(dst []float32, cache *kvCache, head, dim int) {
	base := head * dim
	copy(dst[:dim], cache.v[base:base+dim])
}

func expandValueHeads(dst, v []float32, numHeads, kvHeads, dim int) {
	group := numHeads / kvHeads
	if group == 1 {
		copy(dst[:numHeads*dim], v[:numHeads*dim])
		return
	}
	for kvh := 0; kvh < kvHeads; kvh++ {
		src := v[kvh*dim : (kvh+1)*dim]
		dstBase := kvh * group * dim
		for g := 0; g < group; g++ {
			copy(dst[dstBase+g*dim:dstBase+(g+1)*dim], src)
		}
	}
}

func cacheAttentionScores(scores, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	if len(scores) == 1 {
		if dim == 128 {
			scores[0] = dotAt128(q, cache.k, headBase) * scale
		} else if dim == 96 {
			scores[0] = dotAt96(q, cache.k, headBase) * scale
		} else if dim == 64 {
			scores[0] = dotAt64(q, cache.k, headBase) * scale
		} else {
			scores[0] = dotAt(q, cache.k, headBase, dim) * scale
		}
		return
	}
	if dim == 128 {
		t := 0
		for ; t+7 < len(scores); t += 8 {
			b0 := t*cache.kvDim + headBase
			b1 := b0 + cache.kvDim
			b2 := b1 + cache.kvDim
			b3 := b2 + cache.kvDim
			b4 := b3 + cache.kvDim
			b5 := b4 + cache.kvDim
			b6 := b5 + cache.kvDim
			b7 := b6 + cache.kvDim
			s0, s1, s2, s3, s4, s5, s6, s7 := dotAt128OctetAt(q, cache.k, b0, b1, b2, b3, b4, b5, b6, b7)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
			scores[t+4] = s4 * scale
			scores[t+5] = s5 * scale
			scores[t+6] = s6 * scale
			scores[t+7] = s7 * scale
		}
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
	if dim == 96 {
		t := 0
		for ; t+7 < len(scores); t += 8 {
			b0 := t*cache.kvDim + headBase
			b1 := b0 + cache.kvDim
			b2 := b1 + cache.kvDim
			b3 := b2 + cache.kvDim
			b4 := b3 + cache.kvDim
			b5 := b4 + cache.kvDim
			b6 := b5 + cache.kvDim
			b7 := b6 + cache.kvDim
			s0, s1, s2, s3, s4, s5, s6, s7 := dotAt96OctetAt(q, cache.k, b0, b1, b2, b3, b4, b5, b6, b7)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
			scores[t+4] = s4 * scale
			scores[t+5] = s5 * scale
			scores[t+6] = s6 * scale
			scores[t+7] = s7 * scale
		}
		for ; t+3 < len(scores); t += 4 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			base2 := base1 + cache.kvDim
			base3 := base2 + cache.kvDim
			s0, s1, s2, s3 := dotAt96QuadAt(q, cache.k, base0, base1, base2, base3)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
		}
		for ; t+1 < len(scores); t += 2 {
			base0 := t*cache.kvDim + headBase
			base1 := base0 + cache.kvDim
			s0, s1 := dotAt96PairAt(q, cache.k, base0, base1)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
		}
		for ; t < len(scores); t++ {
			base := t*cache.kvDim + headBase
			scores[t] = dotAt96(q, cache.k, base) * scale
		}
		return
	}
	if dim == 64 {
		t := 0
		for ; t+7 < len(scores); t += 8 {
			b0 := t*cache.kvDim + headBase
			b1 := b0 + cache.kvDim
			b2 := b1 + cache.kvDim
			b3 := b2 + cache.kvDim
			b4 := b3 + cache.kvDim
			b5 := b4 + cache.kvDim
			b6 := b5 + cache.kvDim
			b7 := b6 + cache.kvDim
			s0, s1, s2, s3, s4, s5, s6, s7 := dotAt64OctetAt(q, cache.k, b0, b1, b2, b3, b4, b5, b6, b7)
			scores[t] = s0 * scale
			scores[t+1] = s1 * scale
			scores[t+2] = s2 * scale
			scores[t+3] = s3 * scale
			scores[t+4] = s4 * scale
			scores[t+5] = s5 * scale
			scores[t+6] = s6 * scale
			scores[t+7] = s7 * scale
		}
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
	t := 0
	for ; t+3 < len(scores); t += 4 {
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		s0, s1, s2, s3 := dotAtQuadAt(q, cache.k, base0, base1, base2, base3, dim)
		scores[t] = s0 * scale
		scores[t+1] = s1 * scale
		scores[t+2] = s2 * scale
		scores[t+3] = s3 * scale
	}
	for ; t+1 < len(scores); t += 2 {
		base0 := t*cache.kvDim + headBase
		base1 := base0 + cache.kvDim
		s0, s1 := dotAtPairAt(q, cache.k, base0, base1, dim)
		scores[t] = s0 * scale
		scores[t+1] = s1 * scale
	}
	for ; t < len(scores); t++ {
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
	case 5:
		cacheAttentionLen5(dst, q, cache, head, dim, scale)
		return true
	case 6:
		cacheAttentionLen6(dst, q, cache, head, dim, scale)
		return true
	default:
		return false
	}
}

func cacheAttentionLen6(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	base2 := base1 + cache.kvDim
	base3 := base2 + cache.kvDim
	base4 := base3 + cache.kvDim
	base5 := base4 + cache.kvDim
	var s0, s1, s2, s3 float32
	if dim == 128 {
		s0, s1, s2, s3 = dotAt128QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 96 {
		s0, s1, s2, s3 = dotAt96QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 64 {
		s0, s1, s2, s3 = dotAt64QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else {
		s0, s1, s2, s3 = dotAtQuadAt(q, cache.k, headBase, base1, base2, base3, dim)
	}
	var s4, s5 float32
	if dim == 128 {
		s4, s5 = dotAt128PairAt(q, cache.k, base4, base5)
	} else if dim == 96 {
		s4, s5 = dotAt96PairAt(q, cache.k, base4, base5)
	} else if dim == 64 {
		s4, s5 = dotAt64PairAt(q, cache.k, base4, base5)
	} else {
		s4, s5 = dotAtPairAt(q, cache.k, base4, base5, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	s3 *= scale
	s4 *= scale
	s5 *= scale
	m := max32Local(max32Local(max32Local(s0, s1), max32Local(s2, s3)), max32Local(s4, s5))
	e0 := tensor.FastExpF32(s0 - m)
	e1 := tensor.FastExpF32(s1 - m)
	e2 := tensor.FastExpF32(s2 - m)
	e3 := tensor.FastExpF32(s3 - m)
	e4 := tensor.FastExpF32(s4 - m)
	e5 := tensor.FastExpF32(s5 - m)
	inv := 1 / ((e0 + e1) + (e2 + e3) + (e4 + e5))
	w0, w1, w2, w3, w4, w5 := e0*inv, e1*inv, e2*inv, e3*inv, e4*inv, e5*inv
	weights := [...]float32{w0, w1, w2, w3, w4, w5}
	if dim == 128 {
		weightedCacheValueSum128(dst, cache, head, weights[:])
		return
	}
	if dim == 96 {
		weightedCacheValueSum96(dst, cache, head, weights[:])
		return
	}
	if dim == 64 {
		weightedCacheValueSum64(dst, cache, head, weights[:])
		return
	}
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	x2 := cache.v[base2 : base2+dim]
	x3 := cache.v[base3 : base3+dim]
	x4 := cache.v[base4 : base4+dim]
	x5 := cache.v[base5 : base5+dim]
	weightedValueSum6(dst, x0, x1, x2, x3, x4, x5, w0, w1, w2, w3, w4, w5, dim)
}

func cacheAttentionLen5(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	base2 := base1 + cache.kvDim
	base3 := base2 + cache.kvDim
	base4 := base3 + cache.kvDim
	var s0, s1, s2, s3 float32
	if dim == 128 {
		s0, s1, s2, s3 = dotAt128QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 96 {
		s0, s1, s2, s3 = dotAt96QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 64 {
		s0, s1, s2, s3 = dotAt64QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else {
		s0, s1, s2, s3 = dotAtQuadAt(q, cache.k, headBase, base1, base2, base3, dim)
	}
	var s4 float32
	if dim == 128 {
		s4 = dotAt128(q, cache.k, base4)
	} else if dim == 96 {
		s4 = dotAt96(q, cache.k, base4)
	} else if dim == 64 {
		s4 = dotAt64(q, cache.k, base4)
	} else {
		s4 = dotAt(q, cache.k, base4, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	s3 *= scale
	s4 *= scale
	m := max32Local(max32Local(s0, s1), max32Local(max32Local(s2, s3), s4))
	e0 := tensor.FastExpF32(s0 - m)
	e1 := tensor.FastExpF32(s1 - m)
	e2 := tensor.FastExpF32(s2 - m)
	e3 := tensor.FastExpF32(s3 - m)
	e4 := tensor.FastExpF32(s4 - m)
	inv := 1 / ((e0 + e1) + (e2 + e3) + e4)
	w0, w1, w2, w3, w4 := e0*inv, e1*inv, e2*inv, e3*inv, e4*inv
	weights := [...]float32{w0, w1, w2, w3, w4}
	if dim == 128 {
		weightedCacheValueSum128(dst, cache, head, weights[:])
		return
	}
	if dim == 96 {
		weightedCacheValueSum96(dst, cache, head, weights[:])
		return
	}
	if dim == 64 {
		weightedCacheValueSum64(dst, cache, head, weights[:])
		return
	}
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	x2 := cache.v[base2 : base2+dim]
	x3 := cache.v[base3 : base3+dim]
	x4 := cache.v[base4 : base4+dim]
	weightedCacheValueSum5(dst, x0, x1, x2, x3, x4, w0, w1, w2, w3, w4, dim)
}

func cacheAttentionLen3(dst, q []float32, cache *kvCache, head, dim int, scale float32) {
	headBase := head * dim
	base1 := cache.kvDim + headBase
	base2 := base1 + cache.kvDim
	var s0, s1 float32
	if dim == 128 {
		s0, s1 = dotAt128PairAt(q, cache.k, headBase, base1)
	} else if dim == 96 {
		s0, s1 = dotAt96PairAt(q, cache.k, headBase, base1)
	} else if dim == 64 {
		s0, s1 = dotAt64PairAt(q, cache.k, headBase, base1)
	} else {
		s0, s1 = dotAtPairAt(q, cache.k, headBase, base1, dim)
	}
	var s2 float32
	if dim == 128 {
		s2 = dotAt128(q, cache.k, base2)
	} else if dim == 96 {
		s2 = dotAt96(q, cache.k, base2)
	} else if dim == 64 {
		s2 = dotAt64(q, cache.k, base2)
	} else {
		s2 = dotAt(q, cache.k, base2, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	m := max32Local(max32Local(s0, s1), s2)
	e0 := tensor.FastExpF32(s0 - m)
	e1 := tensor.FastExpF32(s1 - m)
	e2 := tensor.FastExpF32(s2 - m)
	inv := 1 / (e0 + e1 + e2)
	x0 := cache.v[headBase : headBase+dim]
	x1 := cache.v[base1 : base1+dim]
	x2 := cache.v[base2 : base2+dim]
	w0, w1, w2 := e0*inv, e1*inv, e2*inv
	if dim == 128 {
		weightedValueSum3_128(dst, x0, x1, x2, w0, w1, w2)
		return
	}
	if dim == 96 {
		weightedValueSum3_96(dst, x0, x1, x2, w0, w1, w2)
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
	} else if dim == 96 {
		s0, s1, s2, s3 = dotAt96QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else if dim == 64 {
		s0, s1, s2, s3 = dotAt64QuadAt(q, cache.k, headBase, base1, base2, base3)
	} else {
		s0, s1, s2, s3 = dotAtQuadAt(q, cache.k, headBase, base1, base2, base3, dim)
	}
	s0 *= scale
	s1 *= scale
	s2 *= scale
	s3 *= scale
	m := max32Local(max32Local(s0, s1), max32Local(s2, s3))
	e0 := tensor.FastExpF32(s0 - m)
	e1 := tensor.FastExpF32(s1 - m)
	e2 := tensor.FastExpF32(s2 - m)
	e3 := tensor.FastExpF32(s3 - m)
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
	if dim == 96 {
		weightedValueSum4_96(dst, x0, x1, x2, x3, w0, w1, w2, w3)
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
	} else if dim == 96 {
		s0, s1 = dotAt96PairAt(q, cache.k, headBase, base1)
	} else if dim == 64 {
		s0, s1 = dotAt64PairAt(q, cache.k, headBase, base1)
	} else {
		s0, s1 = dotAtPairAt(q, cache.k, headBase, base1, dim)
	}
	s0 *= scale
	s1 *= scale
	var w0, w1 float32
	if s0 >= s1 {
		e := tensor.FastExpF32(s1 - s0)
		inv := 1 / (1 + e)
		w0 = inv
		w1 = e * inv
	} else {
		e := tensor.FastExpF32(s0 - s1)
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
	if dim == 96 {
		weightedValueSum2_96(dst, x0, x1, w0, w1)
		return
	}
	if dim == 64 {
		weightedValueSum2_64(dst, x0, x1, w0, w1)
		return
	}
	weightedCacheValueSum2(dst, x0, x1, w0, w1, dim)
}

func dotAt(a, b []float32, offset, n int) float32 {
	if n >= 64 {
		return tensor.Dot(a[:n], b[offset:offset+n])
	}
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

func dotAtPairAt(a, b []float32, offset0, offset1, n int) (float32, float32) {
	if n >= 64 {
		return tensor.DotPair(b[offset0:offset0+n], b[offset1:offset1+n], a[:n])
	}
	return dotAt(a, b, offset0, n), dotAt(a, b, offset1, n)
}

func dotAtQuadAt(a, b []float32, offset0, offset1, offset2, offset3, n int) (float32, float32, float32, float32) {
	if n >= 64 {
		return tensor.DotQuad(b[offset0:offset0+n], b[offset1:offset1+n], b[offset2:offset2+n], b[offset3:offset3+n], a[:n])
	}
	return dotAt(a, b, offset0, n), dotAt(a, b, offset1, n), dotAt(a, b, offset2, n), dotAt(a, b, offset3, n)
}

func dotAt128(a, b []float32, offset int) float32 {
	return tensor.Dot(a[:128], b[offset:offset+128])
}

func dotAt96(a, b []float32, offset int) float32 {
	return tensor.Dot(a[:96], b[offset:offset+96])
}

func dotAt128Scalar(a, b []float32, offset int) float32 {
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
	return tensor.Dot(a[:64], b[offset:offset+64])
}

func dotAt64Scalar(a, b []float32, offset int) float32 {
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
	return tensor.DotPair(b[offset0:offset0+128], b[offset1:offset1+128], a[:128])
}

func dotAt96PairAt(a, b []float32, offset0, offset1 int) (float32, float32) {
	return tensor.DotPair(b[offset0:offset0+96], b[offset1:offset1+96], a[:96])
}

func dotAt64PairAt(a, b []float32, offset0, offset1 int) (float32, float32) {
	return tensor.DotPair(b[offset0:offset0+64], b[offset1:offset1+64], a[:64])
}

func dotAt128QuadAt(a, b []float32, offset0, offset1, offset2, offset3 int) (float32, float32, float32, float32) {
	return tensor.DotQuad(b[offset0:offset0+128], b[offset1:offset1+128], b[offset2:offset2+128], b[offset3:offset3+128], a[:128])
}

func dotAt96QuadAt(a, b []float32, offset0, offset1, offset2, offset3 int) (float32, float32, float32, float32) {
	return tensor.DotQuad(b[offset0:offset0+96], b[offset1:offset1+96], b[offset2:offset2+96], b[offset3:offset3+96], a[:96])
}

func dotAt64QuadAt(a, b []float32, offset0, offset1, offset2, offset3 int) (float32, float32, float32, float32) {
	return tensor.DotQuad(b[offset0:offset0+64], b[offset1:offset1+64], b[offset2:offset2+64], b[offset3:offset3+64], a[:64])
}

func dotAt128OctetAt(a, b []float32, off0, off1, off2, off3, off4, off5, off6, off7 int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	return tensor.DotOctet(b[off0:off0+128], b[off1:off1+128], b[off2:off2+128], b[off3:off3+128], b[off4:off4+128], b[off5:off5+128], b[off6:off6+128], b[off7:off7+128], a[:128])
}

func dotAt96OctetAt(a, b []float32, off0, off1, off2, off3, off4, off5, off6, off7 int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	return tensor.DotOctet(b[off0:off0+96], b[off1:off1+96], b[off2:off2+96], b[off3:off3+96], b[off4:off4+96], b[off5:off5+96], b[off6:off6+96], b[off7:off7+96], a[:96])
}

func dotAt64OctetAt(a, b []float32, off0, off1, off2, off3, off4, off5, off6, off7 int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	return tensor.DotOctet(b[off0:off0+64], b[off1:off1+64], b[off2:off2+64], b[off3:off3+64], b[off4:off4+64], b[off5:off5+64], b[off6:off6+64], b[off7:off7+64], a[:64])
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
		if dim == 96 {
			weightedValueSum2_96(dst, x0, x1, a0, a1)
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
		if dim == 96 {
			weightedValueSum3_96(dst, x0, x1, x2, a0, a1, a2)
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
		if dim == 96 {
			weightedValueSum4_96(dst, x0, x1, x2, x3, a0, a1, a2, a3)
			return
		}
		if dim == 64 {
			weightedValueSum4_64(dst, x0, x1, x2, x3, a0, a1, a2, a3)
			return
		}
		weightedCacheValueSum4(dst, x0, x1, x2, x3, a0, a1, a2, a3, dim)
		return
	}
	if len(weights) == 5 {
		a0, a1, a2, a3, a4 := weights[0], weights[1], weights[2], weights[3], weights[4]
		base1 := cache.kvDim + headBase
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		base4 := base3 + cache.kvDim
		x0 := cache.v[headBase : headBase+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		x3 := cache.v[base3 : base3+dim]
		x4 := cache.v[base4 : base4+dim]
		if dim == 128 {
			weightedValueSum5_128(dst, x0, x1, x2, x3, x4, a0, a1, a2, a3, a4)
			return
		}
		if dim == 96 {
			weightedValueSum5_96(dst, x0, x1, x2, x3, x4, a0, a1, a2, a3, a4)
			return
		}
		if dim == 64 {
			weightedValueSum5_64(dst, x0, x1, x2, x3, x4, a0, a1, a2, a3, a4)
			return
		}
		weightedCacheValueSum5(dst, x0, x1, x2, x3, x4, a0, a1, a2, a3, a4, dim)
		return
	}
	if len(weights) == 6 {
		a0, a1, a2, a3, a4, a5 := weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]
		base1 := cache.kvDim + headBase
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		base4 := base3 + cache.kvDim
		base5 := base4 + cache.kvDim
		x0 := cache.v[headBase : headBase+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		x3 := cache.v[base3 : base3+dim]
		x4 := cache.v[base4 : base4+dim]
		x5 := cache.v[base5 : base5+dim]
		if dim == 128 {
			weightedValueSum6_128(dst, x0, x1, x2, x3, x4, x5, a0, a1, a2, a3, a4, a5)
			return
		}
		if dim == 96 {
			weightedValueSum6_96(dst, x0, x1, x2, x3, x4, x5, a0, a1, a2, a3, a4, a5)
			return
		}
		if dim == 64 {
			weightedValueSum6_64(dst, x0, x1, x2, x3, x4, x5, a0, a1, a2, a3, a4, a5)
			return
		}
		weightedValueSum6(dst, x0, x1, x2, x3, x4, x5, a0, a1, a2, a3, a4, a5, dim)
		return
	}
	if dim == 64 {
		weightedCacheValueSum64(dst, cache, head, weights)
		return
	}
	if dim == 96 {
		weightedCacheValueSum96(dst, cache, head, weights)
		return
	}
	if dim == 128 {
		weightedCacheValueSum128(dst, cache, head, weights)
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+dim]
	tensor.ScaleCopy(dst[:dim], x, a)
	t := 1
	base := cache.kvDim + headBase
	for ; t+3 < len(weights); t, base = t+4, base+4*cache.kvDim {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := base
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+dim]
		x1 := cache.v[base1 : base1+dim]
		x2 := cache.v[base2 : base2+dim]
		x3 := cache.v[base3 : base3+dim]
		tensor.WeightedSumAdd4(dst[:dim], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; t+1 < len(weights); t, base = t+2, base+2*cache.kvDim {
		a0, a1 := weights[t], weights[t+1]
		base0 := base
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+dim]
		x1 := cache.v[base1 : base1+dim]
		tensor.WeightedSumAdd2(dst[:dim], x0, x1, a0, a1)
	}
	for ; t < len(weights); t, base = t+1, base+cache.kvDim {
		a := weights[t]
		x := cache.v[base : base+dim]
		tensor.AddScaled(dst[:dim], x, a)
	}
}

func weightedCacheValueSum2(dst, x0, x1 []float32, a0, a1 float32, dim int) {
	tensor.WeightedSum2(dst[:dim], x0, x1, a0, a1)
}

func weightedValueSum2_128(dst, x0, x1 []float32, a0, a1 float32) {
	tensor.WeightedSum2(dst[:128], x0, x1, a0, a1)
}

func weightedValueSum2_96(dst, x0, x1 []float32, a0, a1 float32) {
	tensor.WeightedSum2(dst[:96], x0, x1, a0, a1)
}

func weightedValueSum2_64(dst, x0, x1 []float32, a0, a1 float32) {
	tensor.WeightedSum2(dst[:64], x0, x1, a0, a1)
}

func weightedValueSum3_128(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	tensor.WeightedSum3(dst[:128], x0, x1, x2, a0, a1, a2)
}

func weightedValueSum3_96(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	tensor.WeightedSum3(dst[:96], x0, x1, x2, a0, a1, a2)
}

func weightedValueSum3_64(dst, x0, x1, x2 []float32, a0, a1, a2 float32) {
	tensor.WeightedSum3(dst[:64], x0, x1, x2, a0, a1, a2)
}

func weightedCacheValueSum3(dst, x0, x1, x2 []float32, a0, a1, a2 float32, dim int) {
	tensor.WeightedSum3(dst[:dim], x0, x1, x2, a0, a1, a2)
}

func weightedValueSum4_128(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	tensor.WeightedSum4(dst[:128], x0, x1, x2, x3, a0, a1, a2, a3)
}

func weightedValueSum4_96(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	tensor.WeightedSum4(dst[:96], x0, x1, x2, x3, a0, a1, a2, a3)
}

func weightedValueSum4_64(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32) {
	tensor.WeightedSum4(dst[:64], x0, x1, x2, x3, a0, a1, a2, a3)
}

func weightedCacheValueSum4(dst, x0, x1, x2, x3 []float32, a0, a1, a2, a3 float32, dim int) {
	tensor.WeightedSum4(dst[:dim], x0, x1, x2, x3, a0, a1, a2, a3)
}

func weightedValueSum5_128(dst, x0, x1, x2, x3, x4 []float32, a0, a1, a2, a3, a4 float32) {
	tensor.WeightedSum4(dst[:128], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.ScaleAdd(dst[:128], x4, a4)
}

func weightedValueSum5_96(dst, x0, x1, x2, x3, x4 []float32, a0, a1, a2, a3, a4 float32) {
	tensor.WeightedSum4(dst[:96], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.ScaleAdd(dst[:96], x4, a4)
}

func weightedValueSum5_64(dst, x0, x1, x2, x3, x4 []float32, a0, a1, a2, a3, a4 float32) {
	tensor.WeightedSum4(dst[:64], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.ScaleAdd(dst[:64], x4, a4)
}

func weightedCacheValueSum5(dst, x0, x1, x2, x3, x4 []float32, a0, a1, a2, a3, a4 float32, dim int) {
	tensor.WeightedSum4(dst[:dim], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.ScaleAdd(dst[:dim], x4, a4)
}

func weightedValueSum6(dst, x0, x1, x2, x3, x4, x5 []float32, a0, a1, a2, a3, a4, a5 float32, dim int) {
	tensor.WeightedSum4(dst[:dim], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.WeightedSumAdd2(dst[:dim], x4, x5, a4, a5)
}

func weightedValueSum6_128(dst, x0, x1, x2, x3, x4, x5 []float32, a0, a1, a2, a3, a4, a5 float32) {
	tensor.WeightedSum4(dst[:128], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.WeightedSumAdd2(dst[:128], x4, x5, a4, a5)
}

func weightedValueSum6_96(dst, x0, x1, x2, x3, x4, x5 []float32, a0, a1, a2, a3, a4, a5 float32) {
	tensor.WeightedSum4(dst[:96], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.WeightedSumAdd2(dst[:96], x4, x5, a4, a5)
}

func weightedValueSum6_64(dst, x0, x1, x2, x3, x4, x5 []float32, a0, a1, a2, a3, a4, a5 float32) {
	tensor.WeightedSum4(dst[:64], x0, x1, x2, x3, a0, a1, a2, a3)
	tensor.WeightedSumAdd2(dst[:64], x4, x5, a4, a5)
}

func addWeightedValueSum(dst, x []float32, a float32, dim int) {
	tensor.ScaleAdd(dst[:dim], x, a)
}

func addWeightedValueSum128(dst, x []float32, a float32) {
	tensor.ScaleAdd(dst[:128], x, a)
}

func addWeightedValueSum96(dst, x []float32, a float32) {
	tensor.ScaleAdd(dst[:96], x, a)
}

func addWeightedValueSum64(dst, x []float32, a float32) {
	tensor.ScaleAdd(dst[:64], x, a)
}

func weightedCacheValueSum128(dst []float32, cache *kvCache, head int, weights []float32) {
	headBase := head * 128
	if len(weights) == 1 {
		copy(dst[:128], cache.v[headBase:headBase+128])
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+128]
	tensor.ScaleCopy(dst[:128], x, a)
	t := 1
	base := cache.kvDim + headBase
	for ; t+3 < len(weights); t, base = t+4, base+4*cache.kvDim {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := base
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+128]
		x1 := cache.v[base1 : base1+128]
		x2 := cache.v[base2 : base2+128]
		x3 := cache.v[base3 : base3+128]
		tensor.WeightedSumAdd4(dst[:128], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; t+1 < len(weights); t, base = t+2, base+2*cache.kvDim {
		a0, a1 := weights[t], weights[t+1]
		base0 := base
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+128]
		x1 := cache.v[base1 : base1+128]
		tensor.WeightedSumAdd2(dst[:128], x0, x1, a0, a1)
	}
	for ; t < len(weights); t, base = t+1, base+cache.kvDim {
		a := weights[t]
		x := cache.v[base : base+128]
		tensor.ScaleAdd(dst[:128], x, a)
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
	tensor.ScaleCopy(dst[:64], x, a)
	t := 1
	base := cache.kvDim + headBase
	for ; t+3 < len(weights); t, base = t+4, base+4*cache.kvDim {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := base
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+64]
		x1 := cache.v[base1 : base1+64]
		x2 := cache.v[base2 : base2+64]
		x3 := cache.v[base3 : base3+64]
		tensor.WeightedSumAdd4(dst[:64], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; t+1 < len(weights); t, base = t+2, base+2*cache.kvDim {
		a0, a1 := weights[t], weights[t+1]
		base0 := base
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+64]
		x1 := cache.v[base1 : base1+64]
		tensor.WeightedSumAdd2(dst[:64], x0, x1, a0, a1)
	}
	for ; t < len(weights); t, base = t+1, base+cache.kvDim {
		a := weights[t]
		x := cache.v[base : base+64]
		tensor.ScaleAdd(dst[:64], x, a)
	}
}

func weightedCacheValueSum96(dst []float32, cache *kvCache, head int, weights []float32) {
	headBase := head * 96
	if len(weights) == 1 {
		copy(dst[:96], cache.v[headBase:headBase+96])
		return
	}
	a := weights[0]
	x := cache.v[headBase : headBase+96]
	tensor.ScaleCopy(dst[:96], x, a)
	t := 1
	base := cache.kvDim + headBase
	for ; t+3 < len(weights); t, base = t+4, base+4*cache.kvDim {
		a0, a1, a2, a3 := weights[t], weights[t+1], weights[t+2], weights[t+3]
		base0 := base
		base1 := base0 + cache.kvDim
		base2 := base1 + cache.kvDim
		base3 := base2 + cache.kvDim
		x0 := cache.v[base0 : base0+96]
		x1 := cache.v[base1 : base1+96]
		x2 := cache.v[base2 : base2+96]
		x3 := cache.v[base3 : base3+96]
		tensor.WeightedSumAdd4(dst[:96], x0, x1, x2, x3, a0, a1, a2, a3)
	}
	for ; t+1 < len(weights); t, base = t+2, base+2*cache.kvDim {
		a0, a1 := weights[t], weights[t+1]
		base0 := base
		base1 := base0 + cache.kvDim
		x0 := cache.v[base0 : base0+96]
		x1 := cache.v[base1 : base1+96]
		tensor.WeightedSumAdd2(dst[:96], x0, x1, a0, a1)
	}
	for ; t < len(weights); t, base = t+1, base+cache.kvDim {
		a := weights[t]
		x := cache.v[base : base+96]
		tensor.ScaleAdd(dst[:96], x, a)
	}
}

func (c *kvCache) append(k, v []float32) {
	if c.kvDim == 0 {
		c.kvDim = len(k)
	}
	if c.kvDim > 0 && len(k) == c.kvDim && len(v) == c.kvDim {
		base := c.len * c.kvDim
		next := base + c.kvDim
		if cap(c.k) >= next && cap(c.v) >= next {
			c.k = c.k[:next]
			c.v = c.v[:next]
			copy(c.k[base:next], k)
			copy(c.v[base:next], v)
			c.len++
			return
		}
	}
	c.k = append(c.k, k...)
	c.v = append(c.v, v...)
	c.len++
}

func (c *kvCache) vulkanBufferSlices() ([]float32, []float32) {
	k := c.k
	v := c.v
	if cap(k) > len(k) {
		k = k[:cap(k)]
	}
	if cap(v) > len(v) {
		v = v[:cap(v)]
	}
	return k, v
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
	useVulkan := fusedSwiGLUWork(c.IntermediateSize, c.HiddenSize, c.HiddenSize) >= vulkanMatVecMinWork()
	if tl.q8.gate != nil && tl.q8.up != nil && tl.q8.down != nil {
		if useVulkan && q8SwiGLUDownShapeOK(out, x, tl.q8.gate, tl.q8.up, tl.q8.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ8) {
			if err := backend.VulkanSwiGLUDownQ8(out, x, tl.q8.gate, tl.q8.up, tl.q8.down); err == nil {
				return out
			} else {
				rt.disableVulkanOp(vulkanOpSwiGLUDownQ8, err)
			}
		}
		if rt.swiGLUGateUpQ8MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q8.gate, tl.q8.up, tl.q8.down) {
			rt.matVecMaybeQuant(out, sc.gate[:c.IntermediateSize], nil, tl.q8.down, nil, nil, c.HiddenSize, c.IntermediateSize)
			return out
		}
		tensor.FusedSwiGLUQ8Scratch(out, x, tl.q8.gate, tl.q8.up, tl.q8.down, sc.gate)
		return out
	}
	if tl.q6.gate != nil && tl.q6.up != nil && tl.q6.down != nil {
		if useVulkan && q6SwiGLUDownShapeOK(out, x, tl.q6.gate, tl.q6.up, tl.q6.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ6) {
			if err := backend.VulkanSwiGLUDownQ6(out, x, tl.q6.gate, tl.q6.up, tl.q6.down); err == nil {
				return out
			} else {
				rt.disableVulkanOp(vulkanOpSwiGLUDownQ6, err)
			}
		}
		if rt.swiGLUGateUpQ6MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q6.gate, tl.q6.up, tl.q6.down) {
			rt.matVecMaybeQuant(out, sc.gate[:c.IntermediateSize], nil, nil, tl.q6.down, nil, c.HiddenSize, c.IntermediateSize)
			return out
		}
		tensor.FusedSwiGLUQ6Scratch(out, x, tl.q6.gate, tl.q6.up, tl.q6.down, sc.gate)
		return out
	}
	if tl.q4.gate != nil && tl.q4.up != nil && tl.q4.down != nil {
		if useVulkan && q4SwiGLUDownShapeOK(out, x, tl.q4.gate, tl.q4.up, tl.q4.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ4) {
			if err := backend.VulkanSwiGLUDownQ4(out, x, tl.q4.gate, tl.q4.up, tl.q4.down); err == nil {
				return out
			} else {
				rt.disableVulkanOp(vulkanOpSwiGLUDownQ4, err)
			}
		}
		if rt.swiGLUGateUpQ4MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q4.gate, tl.q4.up, tl.q4.down) {
			rt.matVecMaybeQuant(out, sc.gate[:c.IntermediateSize], nil, nil, nil, tl.q4.down, c.HiddenSize, c.IntermediateSize)
			return out
		}
		tensor.FusedSwiGLUQ4Scratch(out, x, tl.q4.gate, tl.q4.up, tl.q4.down, sc.gate)
		return out
	}
	g := sc.gate[:c.IntermediateSize]
	if useVulkan && rt.vulkanOpEnabled(vulkanOpSwiGLUDownF32) &&
		f32SwiGLUDownShapeOK(out, x, tl.w.gate, tl.w.up, tl.w.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) {
		if err := backend.VulkanSwiGLUDownF32(out, x, tl.w.gate, tl.w.up, tl.w.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize); err == nil {
			return out
		} else {
			rt.disableVulkanOp(vulkanOpSwiGLUDownF32, err)
		}
	}
	if rt.swiGLUGateUpMaybeVulkan(g, x, tl.w.gate, tl.w.up, c.IntermediateSize, c.HiddenSize, c.HiddenSize) {
		rt.matVecMaybeQuant(out, g, tl.w.down, nil, nil, nil, c.HiddenSize, c.IntermediateSize)
		return out
	}
	uScratch := sc.up
	if cap(uScratch) >= c.IntermediateSize {
		uScratch = uScratch[:c.IntermediateSize]
	} else {
		uScratch = nil
	}
	tensor.FusedSwiGLUF32ScratchWithU(out, x, tl.w.gate, tl.w.up, tl.w.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize, g, uScratch)
	return out
}

// mlpDownAddRMSNormCPU fuses the MLP down-projection matvec with AddRMSNorm on CPU.
// This saves one full pass over the hidden-size output vector.
// Returns true if the fused path was used, false if caller should use separate calls.
func (rt *Runtime) mlpDownAddRMSNormCPU(normOut, residual, x []float32, tl *textLayer, sc *layerScratch, normWeight []float32, readResidual bool) bool {
	c := rt.cfg
	if len(normOut) < c.HiddenSize || len(residual) < c.HiddenSize || len(x) < c.HiddenSize {
		return false
	}
	intermediate := c.IntermediateSize
	g := sc.gate[:intermediate]
	eps := float32(c.RMSNormEps)

	// Q8 path
	if tl.q8.gate != nil && tl.q8.up != nil && tl.q8.down != nil {
		// Phase 1: gate+up SwiGLU -> intermediate
		if !rt.swiGLUGateUpQ8MaybeVulkan(g, x, tl.q8.gate, tl.q8.up, tl.q8.down) {
			uScratch := sc.up
			if cap(uScratch) >= intermediate {
				uScratch = uScratch[:intermediate]
			} else {
				uScratch = nil
			}
			tensor.SwiGLUGateUpQ8Scratch(g, x, tl.q8.gate, tl.q8.up, uScratch)
		}
		// Phase 2: fused down matvec + AddRMSNorm
		if readResidual {
			tensor.MatVecQ8AddRMSNorm(normOut, residual, g, tl.q8.down, normWeight, eps)
		} else {
			tensor.MatVecQ8AddRMSNormOutOnly(normOut, residual, g, tl.q8.down, normWeight, eps)
		}
		return true
	}

	// Q4 path
	if tl.q4.gate != nil && tl.q4.up != nil && tl.q4.down != nil {
		if !rt.swiGLUGateUpQ4MaybeVulkan(g, x, tl.q4.gate, tl.q4.up, tl.q4.down) {
			uScratch := sc.up
			if cap(uScratch) >= intermediate {
				uScratch = uScratch[:intermediate]
			} else {
				uScratch = nil
			}
			tensor.SwiGLUGateUpQ4Scratch(g, x, tl.q4.gate, tl.q4.up, uScratch)
		}
		if readResidual {
			tensor.MatVecQ4AddRMSNorm(normOut, residual, g, tl.q4.down, normWeight, eps)
		} else {
			tensor.MatVecQ4AddRMSNormOutOnly(normOut, residual, g, tl.q4.down, normWeight, eps)
		}
		return true
	}

	// Q6 path
	if tl.q6.gate != nil && tl.q6.up != nil && tl.q6.down != nil {
		if !rt.swiGLUGateUpQ6MaybeVulkan(g, x, tl.q6.gate, tl.q6.up, tl.q6.down) {
			uScratch := sc.up
			if cap(uScratch) >= intermediate {
				uScratch = uScratch[:intermediate]
			} else {
				uScratch = nil
			}
			tensor.SwiGLUGateUpQ6Scratch(g, x, tl.q6.gate, tl.q6.up, uScratch)
		}
		if readResidual {
			tensor.MatVecQ6AddRMSNorm(normOut, residual, g, tl.q6.down, normWeight, eps)
		} else {
			tensor.MatVecQ6AddRMSNormOutOnly(normOut, residual, g, tl.q6.down, normWeight, eps)
		}
		return true
	}

	// F32 path
	uScratch := sc.up
	if cap(uScratch) >= intermediate {
		uScratch = uScratch[:intermediate]
	} else {
		uScratch = nil
	}
	// Phase 1: gate+up SwiGLU -> intermediate
	if !rt.swiGLUGateUpMaybeVulkan(g, x, tl.w.gate, tl.w.up, intermediate, c.HiddenSize, c.HiddenSize) {
		tensor.FusedSwiGLUGateUpF32Scratch(g, uScratch, x, tl.w.gate, tl.w.up, intermediate, c.HiddenSize)
	}
	// Phase 2: fused down matvec + AddRMSNorm
	if readResidual {
		tensor.MatVecAddRMSNorm(normOut, residual, g, tl.w.down, c.HiddenSize, intermediate, normWeight, eps)
	} else {
		tensor.MatVecAddRMSNormOutOnly(normOut, residual, g, tl.w.down, c.HiddenSize, intermediate, normWeight, eps)
	}
	return true
}

func (rt *Runtime) swiGLUGateUpMaybeVulkan(out, x, gate, up []float32, rows, cols, outRows int) bool {
	if !f32SwiGLUGateUpShapeOK(out, x, gate, up, rows, cols) {
		return false
	}
	if !swiGLUSplitVulkanWorkReady(rows, cols, outRows) || !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpF32) {
		return false
	}
	if err := backend.VulkanSwiGLUGateUpF32(out, x, gate, up, rows, cols); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpSwiGLUGateUpF32, err)
	}
	return false
}

func (rt *Runtime) swiGLUGateUpQ8MaybeVulkan(out, x []float32, gate, up, down *tensor.Q8Matrix) bool {
	if gate == nil || up == nil || down == nil || !q8SwiGLUGateUpShapeOK(out, x, gate, up, down, gate.Rows, gate.Cols, down.Rows) {
		return false
	}
	if !swiGLUSplitVulkanWorkReady(gate.Rows, gate.Cols, down.Rows) || !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ8) {
		return false
	}
	if err := backend.VulkanSwiGLUGateUpQ8(out, x, gate, up); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpSwiGLUGateUpQ8, err)
	}
	return false
}

func (rt *Runtime) swiGLUGateUpQ6MaybeVulkan(out, x []float32, gate, up, down *tensor.Q6Matrix) bool {
	if gate == nil || up == nil || down == nil || !q6SwiGLUGateUpShapeOK(out, x, gate, up, down, gate.Rows, gate.Cols, down.Rows) {
		return false
	}
	if !swiGLUSplitVulkanWorkReady(gate.Rows, gate.Cols, down.Rows) || !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ6) {
		return false
	}
	if err := backend.VulkanSwiGLUGateUpQ6(out, x, gate, up); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpSwiGLUGateUpQ6, err)
	}
	return false
}

func (rt *Runtime) swiGLUGateUpQ4MaybeVulkan(out, x []float32, gate, up, down *tensor.Q4Matrix) bool {
	if gate == nil || up == nil || down == nil || !q4SwiGLUGateUpShapeOK(out, x, gate, up, down, gate.Rows, gate.Cols, down.Rows) {
		return false
	}
	if !swiGLUSplitVulkanWorkReady(gate.Rows, gate.Cols, down.Rows) || !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ4) {
		return false
	}
	if err := backend.VulkanSwiGLUGateUpQ4(out, x, gate, up); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpSwiGLUGateUpQ4, err)
	}
	return false
}

func (rt *Runtime) mlpAddRMSNormMaybeVulkan(x []float32, tl *textLayer, sc *layerScratch, residual, normWeight, normOut []float32, eps float32, readResidual bool) bool {
	c := rt.cfg
	useVulkan := fusedSwiGLUWork(c.IntermediateSize, c.HiddenSize, c.HiddenSize) >= vulkanMatVecMinWork()
	if eps != 1e-6 || !useVulkan {
		return false
	}
	if len(normOut) < c.HiddenSize || len(residual) < c.HiddenSize || len(x) < c.HiddenSize ||
		len(normWeight) < c.HiddenSize {
		return false
	}
	if tl.q8.gate != nil && tl.q8.up != nil && tl.q8.down != nil {
		// Try device-local chained SwiGLU+Down+AddRMSNorm first
		if rt.vulkanChainedSwiGLUDownAddRMSNormQ8(normOut, residual, x, tl, normWeight, readResidual) {
			return true
		}
		op := vulkanOpSwiGLUDownAddRMSNormOutOnlyQ8
		if readResidual {
			op = vulkanOpSwiGLUDownAddRMSNormQ8
		}
		if q8SwiGLUDownShapeOK(normOut, x, tl.q8.gate, tl.q8.up, tl.q8.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(op) {
			var err error
			if readResidual {
				err = backend.VulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x, tl.q8.gate, tl.q8.up, tl.q8.down, normWeight)
			} else {
				err = backend.VulkanSwiGLUDownAddRMSNormQ8OutOnly(normOut, residual, x, tl.q8.gate, tl.q8.up, tl.q8.down, normWeight)
			}
			if err == nil {
				return true
			}
			rt.disableVulkanOp(op, err)
		}
		if readResidual &&
			rt.matVecAddRMSNormVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, tl.q8.down, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) &&
			rt.swiGLUGateUpQ8MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q8.gate, tl.q8.up, tl.q8.down) &&
			rt.matVecAddRMSNormMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, tl.q8.down, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) {
			return true
		}
		if !readResidual &&
			rt.matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, tl.q8.down, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) &&
			rt.swiGLUGateUpQ8MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q8.gate, tl.q8.up, tl.q8.down) &&
			rt.matVecAddRMSNormOutOnlyMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, tl.q8.down, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) {
			return true
		}
		return false
	}
	if tl.q6.gate != nil && tl.q6.up != nil && tl.q6.down != nil {
		op := vulkanOpSwiGLUDownAddRMSNormOutOnlyQ6
		if readResidual {
			op = vulkanOpSwiGLUDownAddRMSNormQ6
		}
		if q6SwiGLUDownShapeOK(normOut, x, tl.q6.gate, tl.q6.up, tl.q6.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(op) {
			var err error
			if readResidual {
				err = backend.VulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x, tl.q6.gate, tl.q6.up, tl.q6.down, normWeight)
			} else {
				err = backend.VulkanSwiGLUDownAddRMSNormQ6OutOnly(normOut, residual, x, tl.q6.gate, tl.q6.up, tl.q6.down, normWeight)
			}
			if err == nil {
				return true
			}
			rt.disableVulkanOp(op, err)
		}
		if readResidual &&
			rt.matVecAddRMSNormVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, tl.q6.down, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) &&
			rt.swiGLUGateUpQ6MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q6.gate, tl.q6.up, tl.q6.down) &&
			rt.matVecAddRMSNormMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, tl.q6.down, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) {
			return true
		}
		if !readResidual &&
			rt.matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, tl.q6.down, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) &&
			rt.swiGLUGateUpQ6MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q6.gate, tl.q6.up, tl.q6.down) &&
			rt.matVecAddRMSNormOutOnlyMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, tl.q6.down, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) {
			return true
		}
		return false
	}
	if tl.q4.gate != nil && tl.q4.up != nil && tl.q4.down != nil {
		op := vulkanOpSwiGLUDownAddRMSNormOutOnlyQ4
		if readResidual {
			op = vulkanOpSwiGLUDownAddRMSNormQ4
		}
		if q4SwiGLUDownShapeOK(normOut, x, tl.q4.gate, tl.q4.up, tl.q4.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
			rt.vulkanOpEnabled(op) {
			var err error
			if readResidual {
				err = backend.VulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x, tl.q4.gate, tl.q4.up, tl.q4.down, normWeight)
			} else {
				err = backend.VulkanSwiGLUDownAddRMSNormQ4OutOnly(normOut, residual, x, tl.q4.gate, tl.q4.up, tl.q4.down, normWeight)
			}
			if err == nil {
				return true
			}
			rt.disableVulkanOp(op, err)
		}
		if readResidual &&
			rt.matVecAddRMSNormVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, nil, tl.q4.down, normWeight, c.HiddenSize, c.IntermediateSize, eps) &&
			rt.swiGLUGateUpQ4MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q4.gate, tl.q4.up, tl.q4.down) &&
			rt.matVecAddRMSNormMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, nil, tl.q4.down, normWeight, c.HiddenSize, c.IntermediateSize, eps) {
			return true
		}
		if !readResidual &&
			rt.matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, nil, tl.q4.down, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) &&
			rt.swiGLUGateUpQ4MaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.q4.gate, tl.q4.up, tl.q4.down) &&
			rt.matVecAddRMSNormOutOnlyMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], nil, nil, nil, tl.q4.down, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) {
			return true
		}
		return false
	}
	if tl.q6.gate != nil || tl.q6.up != nil || tl.q6.down != nil || tl.q4.gate != nil || tl.q4.up != nil || tl.q4.down != nil {
		return false
	}
	op := vulkanOpSwiGLUDownAddRMSNormOutOnlyF32
	if readResidual {
		op = vulkanOpSwiGLUDownAddRMSNormF32
	}
	if rt.vulkanOpEnabled(op) &&
		f32SwiGLUDownShapeOK(normOut, x, tl.w.gate, tl.w.up, tl.w.down, c.IntermediateSize, c.HiddenSize, c.HiddenSize) {
		var err error
		if readResidual {
			err = backend.VulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, tl.w.gate, tl.w.up, tl.w.down, normWeight, c.IntermediateSize, c.HiddenSize, c.HiddenSize)
		} else {
			err = backend.VulkanSwiGLUDownAddRMSNormF32OutOnly(normOut, residual, x, tl.w.gate, tl.w.up, tl.w.down, normWeight, c.IntermediateSize, c.HiddenSize, c.HiddenSize)
		}
		if err == nil {
			return true
		}
		rt.disableVulkanOp(op, err)
	}
	if readResidual &&
		rt.matVecAddRMSNormVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], tl.w.down, nil, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) &&
		rt.swiGLUGateUpMaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.w.gate, tl.w.up, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
		rt.matVecAddRMSNormMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], tl.w.down, nil, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps) {
		return true
	}
	if !readResidual &&
		rt.matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, sc.gate[:c.IntermediateSize], tl.w.down, nil, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) &&
		rt.swiGLUGateUpMaybeVulkan(sc.gate[:c.IntermediateSize], x, tl.w.gate, tl.w.up, c.IntermediateSize, c.HiddenSize, c.HiddenSize) &&
		rt.matVecAddRMSNormOutOnlyMaybeVulkan(normOut, residual, sc.gate[:c.IntermediateSize], tl.w.down, nil, nil, nil, normWeight, c.HiddenSize, c.IntermediateSize, eps, sc.mlp[:c.HiddenSize]) {
		return true
	}
	return false
}

func fusedSwiGLUWork(rows, cols, outRows int) int {
	gateUp, ok := checkedModelMulInt(rows, cols)
	if !ok || gateUp > maxModelInt()/2 {
		return 0
	}
	gateUp *= 2
	down, ok := checkedModelMulInt(outRows, rows)
	if !ok || gateUp > maxModelInt()-down {
		return 0
	}
	return gateUp + down
}

func swiGLUSplitVulkanWorkReady(rows, cols, outRows int) bool {
	return fusedSwiGLUWork(rows, cols, outRows) >= vulkanMatVecMinWork()
}

func matVecMaybeQ8(out, x, w []float32, q *tensor.Q8Matrix, rows, cols int) {
	if q != nil {
		tensor.MatVecQ8(out, x, q)
		return
	}
	tensor.MatVec(out, x, w, rows, cols)
}

func (rt *Runtime) fusedQKV(q, k, v, x []float32, tl *textLayer, qRows, kvRows, hidden int) bool {
	useVulkan := fusedMatVec3Work(qRows, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if tl.q8.q != nil && tl.q8.k != nil && tl.q8.v != nil && sameColsQ8(hidden, tl.q8.q, tl.q8.k, tl.q8.v) {
		if useVulkan && q8FusedMatVec3ShapeOK(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v, qRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ8) {
			if err := backend.VulkanFusedMatVec3Q8(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVQ8, err)
			}
		}
		if rt.matVec3MaybeVulkan(q, k, v, x, nil, nil, nil, tl.q8.q, tl.q8.k, tl.q8.v, nil, nil, nil, nil, nil, nil, qRows, kvRows, kvRows, hidden) {
			return true
		}
		tensor.FusedMatVec3Q8(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v)
		return true
	}
	if tl.q6.q != nil && tl.q6.k != nil && tl.q6.v != nil && sameColsQ6(hidden, tl.q6.q, tl.q6.k, tl.q6.v) {
		if useVulkan && q6FusedMatVec3ShapeOK(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v, qRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ6) {
			if err := backend.VulkanFusedMatVec3Q6(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVQ6, err)
			}
		}
		if rt.matVec3MaybeVulkan(q, k, v, x, nil, nil, nil, nil, nil, nil, tl.q6.q, tl.q6.k, tl.q6.v, nil, nil, nil, qRows, kvRows, kvRows, hidden) {
			return true
		}
		tensor.FusedMatVec3Q6(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v)
		return true
	}
	if tl.q4.q != nil && tl.q4.k != nil && tl.q4.v != nil && sameColsQ4(hidden, tl.q4.q, tl.q4.k, tl.q4.v) {
		if useVulkan && q4FusedMatVec3ShapeOK(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v, qRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ4) {
			if err := backend.VulkanFusedMatVec3Q4(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVQ4, err)
			}
		}
		if rt.matVec3MaybeVulkan(q, k, v, x, nil, nil, nil, nil, nil, nil, nil, nil, nil, tl.q4.q, tl.q4.k, tl.q4.v, qRows, kvRows, kvRows, hidden) {
			return true
		}
		tensor.FusedMatVec3Q4(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v)
		return true
	}
	if len(tl.w.q) >= qRows*hidden && len(tl.w.k) >= kvRows*hidden && len(tl.w.v) >= kvRows*hidden {
		if useVulkan && f32FusedMatVec3ShapeOK(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVF32) {
			if err := backend.VulkanFusedMatVec3F32(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVF32, err)
			}
		}
		if rt.matVec3MaybeVulkan(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, nil, nil, nil, nil, nil, nil, nil, nil, nil, qRows, kvRows, kvRows, hidden) {
			return true
		}
		tensor.FusedMatVec3(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden)
		return true
	}
	return false
}

func (rt *Runtime) fusedQKVMRoPE(q, k, v, x []float32, tl *textLayer, qRows, kvRows, hidden, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) bool {
	useVulkan := fusedMatVec3Work(qRows, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if tl.q8.q != nil && tl.q8.k != nil && tl.q8.v != nil && sameColsQ8(hidden, tl.q8.q, tl.q8.k, tl.q8.v) {
		if useVulkan && q8FusedMatVec3ShapeOK(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v, qRows, kvRows, kvRows, hidden) &&
			fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ8) {
			if err := backend.VulkanFusedMatVec3MRoPEQ8(q, k, v, x, tl.q8.q, tl.q8.k, tl.q8.v, cosTable, sinTable, qHeads, kvHeads, headDim); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ8, err)
			}
		}
		return false
	}
	if tl.q6.q != nil && tl.q6.k != nil && tl.q6.v != nil && sameColsQ6(hidden, tl.q6.q, tl.q6.k, tl.q6.v) {
		if useVulkan && q6FusedMatVec3ShapeOK(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v, qRows, kvRows, kvRows, hidden) &&
			fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ6) {
			if err := backend.VulkanFusedMatVec3MRoPEQ6(q, k, v, x, tl.q6.q, tl.q6.k, tl.q6.v, cosTable, sinTable, qHeads, kvHeads, headDim); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ6, err)
			}
		}
		return false
	}
	if tl.q4.q != nil && tl.q4.k != nil && tl.q4.v != nil && sameColsQ4(hidden, tl.q4.q, tl.q4.k, tl.q4.v) {
		if useVulkan && q4FusedMatVec3ShapeOK(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v, qRows, kvRows, kvRows, hidden) &&
			fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ4) {
			if err := backend.VulkanFusedMatVec3MRoPEQ4(q, k, v, x, tl.q4.q, tl.q4.k, tl.q4.v, cosTable, sinTable, qHeads, kvHeads, headDim); err == nil {
				return true
			} else {
				rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ4, err)
			}
		}
		return false
	}
	if tl.q8.q != nil || tl.q8.k != nil || tl.q8.v != nil || tl.q6.q != nil || tl.q6.k != nil || tl.q6.v != nil || tl.q4.q != nil || tl.q4.k != nil || tl.q4.v != nil {
		return false
	}
	if len(tl.w.q) < qRows*hidden || len(tl.w.k) < kvRows*hidden || len(tl.w.v) < kvRows*hidden {
		return false
	}
	if useVulkan && f32FusedMatVec3ShapeOK(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden) &&
		fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEF32) {
		if err := backend.VulkanFusedMatVec3MRoPEF32(q, k, v, x, tl.w.q, tl.w.k, tl.w.v, cosTable, sinTable, qRows, kvRows, kvRows, hidden, qHeads, kvHeads, headDim); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpFusedQKVMRoPEF32, err)
		}
	}
	return false
}

// fusedQKVMRoPEWithNorm chains RMSNorm(rawInput, normWeight) with fused QKV+MRoPE
// into a single Vulkan command buffer.  Only for F32 weights.  Returns true on
// success, false to fall back to separate RMSNorm + fusedQKVMRoPE.
func (rt *Runtime) fusedQKVMRoPEWithNorm(q, k, v, rawInput, normWeight []float32, tl *textLayer, qRows, kvRows, hidden, qHeads, kvHeads, headDim int, cosTable, sinTable []float32, eps float32) bool {
	if eps != 1e-6 || !rt.vulkanOpEnabled(vulkanOpChainedRMSNormQKVMRoPEF32) {
		return false
	}
	if tl.q8.q != nil || tl.q6.q != nil || tl.q4.q != nil {
		return false
	}
	useVulkan := fusedMatVec3Work(qRows, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if !useVulkan {
		return false
	}
	if !rmsNormShapeOK(rawInput, rawInput, normWeight) || !f32FusedMatVec3ShapeOK(q, k, v, rawInput, tl.w.q, tl.w.k, tl.w.v, qRows, kvRows, kvRows, hidden) {
		return false
	}
	if !fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	n := hidden
	if n < vulkanVectorMinWork() {
		return false
	}
	if err := backend.VulkanChainedRMSNormFusedQKVMRoPEF32(q, k, v, rawInput, normWeight, tl.w.q, tl.w.k, tl.w.v, cosTable, sinTable, n, qRows, kvRows, kvRows, hidden, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedRMSNormQKVMRoPEF32, err)
	}
	return false
}

// fusedQKVMRoPEWithNormQ8 chains RMSNorm(rawInput, normWeight) with the fused
// Q8 QKV+MRoPE projection into a single Vulkan command buffer.  Only for Q8
// weights.  Returns true on success, false to fall back to separate RMSNorm +
// fusedQKVMRoPE.
func (rt *Runtime) fusedQKVMRoPEWithNormQ8(q, k, v, rawInput, normWeight []float32, tl *textLayer, qRows, kvRows, hidden, qHeads, kvHeads, headDim int, cosTable, sinTable []float32, eps float32) bool {
	if eps != 1e-6 || !rt.vulkanOpEnabled(vulkanOpChainedRMSNormQKVMRoPEQ8) {
		return false
	}
	if tl.q8.q == nil || tl.q8.k == nil || tl.q8.v == nil || !sameColsQ8(hidden, tl.q8.q, tl.q8.k, tl.q8.v) {
		return false
	}
	useVulkan := fusedMatVec3Work(qRows, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if !useVulkan {
		return false
	}
	if !rmsNormShapeOK(rawInput, rawInput, normWeight) || !q8FusedMatVec3ShapeOK(q, k, v, rawInput, tl.q8.q, tl.q8.k, tl.q8.v, qRows, kvRows, kvRows, hidden) {
		return false
	}
	if !fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	n := hidden
	if n < vulkanVectorMinWork() {
		return false
	}
	if err := backend.VulkanChainedRMSNormFusedQKVMRoPEQ8(q, k, v, rawInput, normWeight, cosTable, sinTable, tl.q8.q, tl.q8.k, tl.q8.v, n, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedRMSNormQKVMRoPEQ8, err)
	}
	return false
}

func (rt *Runtime) fusedFirstTokenKV(q, k, v, x []float32, tl *textLayer, qRows, kvRows, hidden, kvHeads, headDim int, hasRoPE bool, cosTable, sinTable []float32) (ready bool, kHasRoPE bool) {
	dummyRows := headDim
	if dummyRows <= 0 {
		return false, false
	}
	haveDummyQ := qRows >= dummyRows && len(q) >= dummyRows
	useVulkan := fusedMatVec3Work(dummyRows, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	kvUseVulkan := fusedMatVec3Work(1, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if tl.q8.k != nil && tl.q8.v != nil && tl.q8.k.Cols == hidden && tl.q8.v.Cols == hidden {
		dummyQ := &tensor.Q8Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q8FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q8.k, tl.q8.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ8) {
			if err := backend.VulkanFusedMatVec2MRoPEQ8(k, v, x, dummyQ, tl.q8.k, tl.q8.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ8, err)
			}
		}
		if kvUseVulkan && q8FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q8.k, tl.q8.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ8) {
			if err := backend.VulkanFusedMatVec2Q8(k, v, x, dummyQ, tl.q8.k, tl.q8.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ8, err)
			}
		}
		if haveDummyQ && tl.q8.q != nil && sameColsQ8(hidden, tl.q8.q, tl.q8.k, tl.q8.v) && q8MatVecMinRowsShapeOK(tl.q8.q, dummyRows, hidden) {
			qHead := *tl.q8.q
			qHead.Rows = dummyRows
			qHead.Data = qHead.Data[:dummyRows*qHead.Cols]
			qHead.Scale = qHead.Scale[:dummyRows]
			if useVulkan && hasRoPE && q8FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q8.k, tl.q8.v, dummyRows, kvRows, kvRows, hidden) &&
				fusedMRoPEShapeOK(q[:dummyRows], k, 1, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ8) {
				if err := backend.VulkanFusedMatVec3MRoPEQ8(q[:dummyRows], k, v, x, &qHead, tl.q8.k, tl.q8.v, cosTable, sinTable, 1, kvHeads, headDim); err == nil {
					return true, true
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ8, err)
				}
			}
			if useVulkan && q8FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q8.k, tl.q8.v, dummyRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ8) {
				if err := backend.VulkanFusedMatVec3Q8(q[:dummyRows], k, v, x, &qHead, tl.q8.k, tl.q8.v); err == nil {
					return true, false
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVQ8, err)
				}
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, tl.q8.k, tl.q8.v, nil, nil, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q8(k, v, x, tl.q8.k, tl.q8.v)
		return true, false
	}
	if tl.q6.k != nil && tl.q6.v != nil && tl.q6.k.Cols == hidden && tl.q6.v.Cols == hidden {
		dummyQ := &tensor.Q6Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q6FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q6.k, tl.q6.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ6) {
			if err := backend.VulkanFusedMatVec2MRoPEQ6(k, v, x, dummyQ, tl.q6.k, tl.q6.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ6, err)
			}
		}
		if kvUseVulkan && q6FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q6.k, tl.q6.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ6) {
			if err := backend.VulkanFusedMatVec2Q6(k, v, x, dummyQ, tl.q6.k, tl.q6.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ6, err)
			}
		}
		if haveDummyQ && tl.q6.q != nil && sameColsQ6(hidden, tl.q6.q, tl.q6.k, tl.q6.v) && q6MatVecMinRowsShapeOK(tl.q6.q, dummyRows, hidden) {
			qHead := *tl.q6.q
			qHead.Rows = dummyRows
			packedCols := tensor.PackedQ6Cols(qHead.Cols)
			qHead.Data = qHead.Data[:dummyRows*packedCols]
			qHead.Scale = qHead.Scale[:dummyRows]
			if useVulkan && hasRoPE && q6FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q6.k, tl.q6.v, dummyRows, kvRows, kvRows, hidden) &&
				fusedMRoPEShapeOK(q[:dummyRows], k, 1, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ6) {
				if err := backend.VulkanFusedMatVec3MRoPEQ6(q[:dummyRows], k, v, x, &qHead, tl.q6.k, tl.q6.v, cosTable, sinTable, 1, kvHeads, headDim); err == nil {
					return true, true
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ6, err)
				}
			}
			if useVulkan && q6FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q6.k, tl.q6.v, dummyRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ6) {
				if err := backend.VulkanFusedMatVec3Q6(q[:dummyRows], k, v, x, &qHead, tl.q6.k, tl.q6.v); err == nil {
					return true, false
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVQ6, err)
				}
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, nil, nil, tl.q6.k, tl.q6.v, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q6(k, v, x, tl.q6.k, tl.q6.v)
		return true, false
	}
	if tl.q4.k != nil && tl.q4.v != nil && tl.q4.k.Cols == hidden && tl.q4.v.Cols == hidden {
		dummyQ := &tensor.Q4Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q4FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ4) {
			if err := backend.VulkanFusedMatVec2MRoPEQ4(k, v, x, dummyQ, tl.q4.k, tl.q4.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ4, err)
			}
		}
		if kvUseVulkan && q4FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ4) {
			if err := backend.VulkanFusedMatVec2Q4(k, v, x, dummyQ, tl.q4.k, tl.q4.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ4, err)
			}
		}
		if haveDummyQ && tl.q4.q != nil && sameColsQ4(hidden, tl.q4.q, tl.q4.k, tl.q4.v) && q4MatVecMinRowsShapeOK(tl.q4.q, dummyRows, hidden) {
			qHead := *tl.q4.q
			qHead.Rows = dummyRows
			qHead.Data = qHead.Data[:dummyRows*((qHead.Cols+1)/2)]
			qHead.Scale = qHead.Scale[:dummyRows]
			if useVulkan && hasRoPE && q4FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q4.k, tl.q4.v, dummyRows, kvRows, kvRows, hidden) &&
				fusedMRoPEShapeOK(q[:dummyRows], k, 1, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ4) {
				if err := backend.VulkanFusedMatVec3MRoPEQ4(q[:dummyRows], k, v, x, &qHead, tl.q4.k, tl.q4.v, cosTable, sinTable, 1, kvHeads, headDim); err == nil {
					return true, true
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVMRoPEQ4, err)
				}
			}
			if useVulkan && q4FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, &qHead, tl.q4.k, tl.q4.v, dummyRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVQ4) {
				if err := backend.VulkanFusedMatVec3Q4(q[:dummyRows], k, v, x, &qHead, tl.q4.k, tl.q4.v); err == nil {
					return true, false
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVQ4, err)
				}
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, nil, nil, nil, nil, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q4(k, v, x, tl.q4.k, tl.q4.v)
		return true, false
	}
	if len(tl.w.k) >= kvRows*hidden && len(tl.w.v) >= kvRows*hidden {
		if kvUseVulkan && hasRoPE && f32FusedMatVec2ShapeOK(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEF32) {
			if err := backend.VulkanFusedMatVec2MRoPEF32(k, v, x, nil, tl.w.k, tl.w.v, cosTable, sinTable, kvRows, kvRows, hidden, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEF32, err)
			}
		}
		if kvUseVulkan && f32FusedMatVec2ShapeOK(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVF32) {
			if err := backend.VulkanFusedMatVec2F32(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVF32, err)
			}
		}
		if haveDummyQ && len(tl.w.q) >= dummyRows*hidden {
			if useVulkan && hasRoPE && f32FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, tl.w.q[:dummyRows*hidden], tl.w.k, tl.w.v, dummyRows, kvRows, kvRows, hidden) &&
				fusedMRoPEShapeOK(q[:dummyRows], k, 1, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEF32) {
				if err := backend.VulkanFusedMatVec3MRoPEF32(q[:dummyRows], k, v, x, tl.w.q[:dummyRows*hidden], tl.w.k, tl.w.v, cosTable, sinTable, dummyRows, kvRows, kvRows, hidden, 1, kvHeads, headDim); err == nil {
					return true, true
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVMRoPEF32, err)
				}
			}
			if useVulkan && f32FusedMatVec3ShapeOK(q[:dummyRows], k, v, x, tl.w.q[:dummyRows*hidden], tl.w.k, tl.w.v, dummyRows, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedQKVF32) {
				if err := backend.VulkanFusedMatVec3F32(q[:dummyRows], k, v, x, tl.w.q[:dummyRows*hidden], tl.w.k, tl.w.v, dummyRows, kvRows, kvRows, hidden); err == nil {
					return true, false
				} else {
					rt.disableVulkanOp(vulkanOpFusedQKVF32, err)
				}
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, tl.w.k, tl.w.v, nil, nil, nil, nil, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden)
		return true, false
	}
	return false, false
}

func (rt *Runtime) fusedKV(k, v, x []float32, tl *textLayer, kvRows, hidden, kvHeads, headDim int, hasRoPE bool, cosTable, sinTable []float32) (ready bool, kHasRoPE bool) {
	kvUseVulkan := fusedMatVec3Work(1, kvRows, kvRows, hidden) >= vulkanMatVecMinWork()
	if tl.q8.k != nil && tl.q8.v != nil && tl.q8.k.Cols == hidden && tl.q8.v.Cols == hidden {
		dummyQ := &tensor.Q8Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q8FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q8.k, tl.q8.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ8) {
			if err := backend.VulkanFusedMatVec2MRoPEQ8(k, v, x, dummyQ, tl.q8.k, tl.q8.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ8, err)
			}
		}
		if kvUseVulkan && q8FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q8.k, tl.q8.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ8) {
			if err := backend.VulkanFusedMatVec2Q8(k, v, x, dummyQ, tl.q8.k, tl.q8.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ8, err)
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, tl.q8.k, tl.q8.v, nil, nil, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q8(k, v, x, tl.q8.k, tl.q8.v)
		return true, false
	}
	if tl.q6.k != nil && tl.q6.v != nil && tl.q6.k.Cols == hidden && tl.q6.v.Cols == hidden {
		dummyQ := &tensor.Q6Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q6FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q6.k, tl.q6.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ6) {
			if err := backend.VulkanFusedMatVec2MRoPEQ6(k, v, x, dummyQ, tl.q6.k, tl.q6.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ6, err)
			}
		}
		if kvUseVulkan && q6FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q6.k, tl.q6.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ6) {
			if err := backend.VulkanFusedMatVec2Q6(k, v, x, dummyQ, tl.q6.k, tl.q6.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ6, err)
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, nil, nil, tl.q6.k, tl.q6.v, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q6(k, v, x, tl.q6.k, tl.q6.v)
		return true, false
	}
	if tl.q4.k != nil && tl.q4.v != nil && tl.q4.k.Cols == hidden && tl.q4.v.Cols == hidden {
		dummyQ := &tensor.Q4Matrix{Cols: hidden}
		if kvUseVulkan && hasRoPE && q4FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ4) {
			if err := backend.VulkanFusedMatVec2MRoPEQ4(k, v, x, dummyQ, tl.q4.k, tl.q4.v, cosTable, sinTable, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEQ4, err)
			}
		}
		if kvUseVulkan && q4FusedMatVec2ShapeOK(k, v, x, dummyQ, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVQ4) {
			if err := backend.VulkanFusedMatVec2Q4(k, v, x, dummyQ, tl.q4.k, tl.q4.v); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVQ4, err)
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, nil, nil, nil, nil, nil, nil, tl.q4.k, tl.q4.v, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2Q4(k, v, x, tl.q4.k, tl.q4.v)
		return true, false
	}
	if len(tl.w.k) >= kvRows*hidden && len(tl.w.v) >= kvRows*hidden {
		if kvUseVulkan && hasRoPE && f32FusedMatVec2ShapeOK(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden) &&
			mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEF32) {
			if err := backend.VulkanFusedMatVec2MRoPEF32(k, v, x, nil, tl.w.k, tl.w.v, cosTable, sinTable, kvRows, kvRows, hidden, kvHeads, headDim); err == nil {
				return true, true
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVMRoPEF32, err)
			}
		}
		if kvUseVulkan && f32FusedMatVec2ShapeOK(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden) && rt.vulkanOpEnabled(vulkanOpFusedKVF32) {
			if err := backend.VulkanFusedMatVec2F32(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden); err == nil {
				return true, false
			} else {
				rt.disableVulkanOp(vulkanOpFusedKVF32, err)
			}
		}
		if rt.matVec2MaybeVulkan(k, v, x, tl.w.k, tl.w.v, nil, nil, nil, nil, nil, nil, kvRows, kvRows, hidden) {
			return true, false
		}
		tensor.FusedMatVec2(k, v, x, tl.w.k, tl.w.v, kvRows, kvRows, hidden)
		return true, false
	}
	return false, false
}

func fusedMatVec3Work(rowsA, rowsB, rowsC, cols int) int {
	rows, ok := checkedModelAddInt(rowsA, rowsB)
	if !ok {
		return 0
	}
	rows, ok = checkedModelAddInt(rows, rowsC)
	if !ok {
		return 0
	}
	work, ok := checkedModelMulInt(rows, cols)
	if !ok {
		return 0
	}
	return work
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

func q8MatVecShapeOK(q *tensor.Q8Matrix, rows, cols int) bool {
	elements, ok := checkedModelMulInt(rows, cols)
	return ok && q != nil && q.Rows == rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func q6MatVecShapeOK(q *tensor.Q6Matrix, rows, cols int) bool {
	packedCols, ok := checkedPackedQ6Cols(cols)
	elements, elemOK := checkedModelMulInt(rows, packedCols)
	return ok && elemOK && q != nil && q.Rows == rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func q4MatVecShapeOK(q *tensor.Q4Matrix, rows, cols int) bool {
	packedCols, ok := checkedPackedQ4Cols(cols)
	elements, elemOK := checkedModelMulInt(rows, packedCols)
	return ok && elemOK && q != nil && q.Rows == rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func q8FusedMatVec3ShapeOK(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix, rowsA, rowsB, rowsC, cols int) bool {
	return q8MatVecShapeOK(a, rowsA, cols) && q8MatVecShapeOK(b, rowsB, cols) && q8MatVecShapeOK(c, rowsC, cols) &&
		len(outA) >= rowsA && len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q6FusedMatVec3ShapeOK(outA, outB, outC, x []float32, a, b, c *tensor.Q6Matrix, rowsA, rowsB, rowsC, cols int) bool {
	return q6MatVecShapeOK(a, rowsA, cols) && q6MatVecShapeOK(b, rowsB, cols) && q6MatVecShapeOK(c, rowsC, cols) &&
		len(outA) >= rowsA && len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q4FusedMatVec3ShapeOK(outA, outB, outC, x []float32, a, b, c *tensor.Q4Matrix, rowsA, rowsB, rowsC, cols int) bool {
	return q4MatVecShapeOK(a, rowsA, cols) && q4MatVecShapeOK(b, rowsB, cols) && q4MatVecShapeOK(c, rowsC, cols) &&
		len(outA) >= rowsA && len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q8MatVecMinRowsShapeOK(q *tensor.Q8Matrix, rows, cols int) bool {
	elements, ok := checkedModelMulInt(rows, cols)
	return ok && q != nil && q.Rows >= rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func q6MatVecMinRowsShapeOK(q *tensor.Q6Matrix, rows, cols int) bool {
	packedCols, ok := checkedPackedQ6Cols(cols)
	elements, elemOK := checkedModelMulInt(rows, packedCols)
	return ok && elemOK && q != nil && q.Rows >= rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func q4MatVecMinRowsShapeOK(q *tensor.Q4Matrix, rows, cols int) bool {
	packedCols, ok := checkedPackedQ4Cols(cols)
	elements, elemOK := checkedModelMulInt(rows, packedCols)
	return ok && elemOK && q != nil && q.Rows >= rows && q.Cols == cols && len(q.Data) >= elements && len(q.Scale) >= rows
}

func f32FusedMatVec3ShapeOK(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) bool {
	return f32MatVecShapeOK(outA, x, wa, rowsA, cols) && f32MatVecShapeOK(outB, x, wb, rowsB, cols) && f32MatVecShapeOK(outC, x, wc, rowsC, cols)
}

func q8FusedMatVec2ShapeOK(outB, outC, x []float32, a, b, c *tensor.Q8Matrix, rowsB, rowsC, cols int) bool {
	return a != nil && a.Cols == cols && q8MatVecShapeOK(b, rowsB, cols) && q8MatVecShapeOK(c, rowsC, cols) &&
		len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q6FusedMatVec2ShapeOK(outB, outC, x []float32, a, b, c *tensor.Q6Matrix, rowsB, rowsC, cols int) bool {
	return a != nil && a.Cols == cols && q6MatVecShapeOK(b, rowsB, cols) && q6MatVecShapeOK(c, rowsC, cols) &&
		len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q4FusedMatVec2ShapeOK(outB, outC, x []float32, a, b, c *tensor.Q4Matrix, rowsB, rowsC, cols int) bool {
	return a != nil && a.Cols == cols && q4MatVecShapeOK(b, rowsB, cols) && q4MatVecShapeOK(c, rowsC, cols) &&
		len(outB) >= rowsB && len(outC) >= rowsC && len(x) >= cols
}

func q8SwiGLUGateUpShapeOK(out, x []float32, gate, up, down *tensor.Q8Matrix, rows, cols, outRows int) bool {
	return q8MatVecShapeOK(gate, rows, cols) && q8MatVecShapeOK(up, rows, cols) &&
		down != nil && down.Rows == outRows && down.Cols == rows && len(out) >= rows && len(x) >= cols
}

func q6SwiGLUGateUpShapeOK(out, x []float32, gate, up, down *tensor.Q6Matrix, rows, cols, outRows int) bool {
	return q6MatVecShapeOK(gate, rows, cols) && q6MatVecShapeOK(up, rows, cols) &&
		down != nil && down.Rows == outRows && down.Cols == rows && len(out) >= rows && len(x) >= cols
}

func q4SwiGLUGateUpShapeOK(out, x []float32, gate, up, down *tensor.Q4Matrix, rows, cols, outRows int) bool {
	return q4MatVecShapeOK(gate, rows, cols) && q4MatVecShapeOK(up, rows, cols) &&
		down != nil && down.Rows == outRows && down.Cols == rows && len(out) >= rows && len(x) >= cols
}

func q8SwiGLUDownShapeOK(out, x []float32, gate, up, down *tensor.Q8Matrix, rows, cols, outRows int) bool {
	return q8SwiGLUGateUpShapeOK(out, x, gate, up, down, rows, cols, outRows) && q8MatVecShapeOK(down, outRows, rows) && len(out) >= outRows
}

func q6SwiGLUDownShapeOK(out, x []float32, gate, up, down *tensor.Q6Matrix, rows, cols, outRows int) bool {
	return q6SwiGLUGateUpShapeOK(out, x, gate, up, down, rows, cols, outRows) && q6MatVecShapeOK(down, outRows, rows) && len(out) >= outRows
}

func q4SwiGLUDownShapeOK(out, x []float32, gate, up, down *tensor.Q4Matrix, rows, cols, outRows int) bool {
	return q4SwiGLUGateUpShapeOK(out, x, gate, up, down, rows, cols, outRows) && q4MatVecShapeOK(down, outRows, rows) && len(out) >= outRows
}

func f32SwiGLUDownShapeOK(out, x, gate, up, down []float32, rows, cols, outRows int) bool {
	return f32SwiGLUGateUpWeightsShapeOK(x, gate, up, rows, cols) &&
		outRows > 0 && len(out) >= outRows && f32MatVecWeightsReady(down, outRows, rows)
}

func f32SwiGLUGateUpShapeOK(out, x, gate, up []float32, rows, cols int) bool {
	return len(out) >= rows && f32SwiGLUGateUpWeightsShapeOK(x, gate, up, rows, cols)
}

func f32SwiGLUGateUpWeightsShapeOK(x, gate, up []float32, rows, cols int) bool {
	return len(x) >= cols && f32MatVecWeightsReady(gate, rows, cols) && f32MatVecWeightsReady(up, rows, cols)
}

func f32FusedMatVec2ShapeOK(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) bool {
	return f32MatVecShapeOK(outB, x, wb, rowsB, cols) && f32MatVecShapeOK(outC, x, wc, rowsC, cols)
}

func fusedMRoPEShapeOK(q, k []float32, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) bool {
	return headDim%2 == 0 && mropeShapeOK(q, qHeads, headDim, cosTable, sinTable) && mropeShapeOK(k, kvHeads, headDim, cosTable, sinTable)
}

func f32MatVecShapeOK(out, x, w []float32, rows, cols int) bool {
	elements, ok := checkedModelMulInt(rows, cols)
	return ok && len(out) >= rows && len(x) >= cols && len(w) >= elements
}

func f32MatVecWeightsReady(w []float32, rows, cols int) bool {
	elements, ok := checkedModelMulInt(rows, cols)
	return ok && len(w) >= elements
}

func checkedTextAttentionDims(cache *kvCache, numHeads, kvHeads, headDim int) (qRows, kvDim, cacheElems int, ok bool) {
	if cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || numHeads%kvHeads != 0 {
		return 0, 0, 0, false
	}
	qRows, ok = checkedModelMulInt(numHeads, headDim)
	if !ok {
		return 0, 0, 0, false
	}
	kvDim, ok = checkedModelMulInt(kvHeads, headDim)
	if !ok || cache.kvDim != kvDim {
		return 0, 0, 0, false
	}
	cacheElems, ok = checkedModelMulInt(cache.len, cache.kvDim)
	if !ok {
		return 0, 0, 0, false
	}
	return qRows, kvDim, cacheElems, true
}

func textCacheStorageReady(cache *kvCache, cacheElems int) bool {
	return cache != nil && cacheElems > 0 && len(cache.k) >= cacheElems && len(cache.v) >= cacheElems
}

func textAttentionOnlyWorkReady(cacheLen, numHeads, headDim int) bool {
	work, ok := checkedModelMulInt(cacheLen, numHeads)
	if ok {
		work, ok = checkedModelMulInt(work, headDim)
	}
	return ok && work >= vulkanTextAttentionMinWork()
}

func textAttentionOutWorkReady(cacheLen, numHeads, headDim, qRows int, includeNorm bool) bool {
	attn, ok := checkedTextAttentionOnlyWork(cacheLen, numHeads, headDim)
	if !ok {
		return false
	}
	proj, ok := checkedModelMulInt(qRows, qRows)
	if !ok {
		return false
	}
	work, ok := checkedModelAddInt(attn, proj)
	if !ok {
		return false
	}
	if includeNorm {
		work, ok = checkedModelAddInt(work, qRows)
		if !ok {
			return false
		}
	}
	return work >= vulkanTextAttentionMinWork()
}

func textFirstTokenValueOutNormWorkReady(qRows, kvDim int) bool {
	proj, ok := checkedModelMulInt(qRows, qRows)
	if !ok {
		return false
	}
	work, ok := checkedModelAddInt(proj, kvDim)
	if ok {
		work, ok = checkedModelAddInt(work, qRows)
	}
	return ok && work >= vulkanTextAttentionMinWork()
}

func checkedTextAttentionOnlyWork(cacheLen, numHeads, headDim int) (int, bool) {
	work, ok := checkedModelMulInt(cacheLen, numHeads)
	if ok {
		work, ok = checkedModelMulInt(work, headDim)
	}
	return work, ok
}

func vulkanMatVecWorkReady(rows, cols int) bool {
	work, ok := checkedModelMulInt(rows, cols)
	return ok && work >= vulkanMatVecMinWork()
}

func groupedVulkanMatVecWorkReady(cols int, rows ...int) bool {
	if cols <= 0 {
		return false
	}
	total := 0
	for _, row := range rows {
		if row <= 0 || total > maxModelInt()-row {
			return false
		}
		total += row
	}
	return vulkanMatVecWorkReady(total, cols)
}

func vulkanMatVecAddNormWorkReady(rows, cols int) bool {
	work, ok := checkedModelMulInt(rows, cols)
	if !ok || work > maxModelInt()-rows {
		return false
	}
	return work+rows >= vulkanMatVecMinWork()
}

func vulkanVectorWorkReady(rows, cols int) bool {
	work, ok := checkedModelMulInt(rows, cols)
	return ok && work >= vulkanVectorMinWork()
}

func groupedVulkanVectorWorkReady(cols int, rows ...int) bool {
	if cols <= 0 {
		return false
	}
	total := 0
	for _, row := range rows {
		if row <= 0 || total > maxModelInt()-row {
			return false
		}
		total += row
	}
	return vulkanVectorWorkReady(total, cols)
}

func checkedPackedQ6Cols(cols int) (int, bool) {
	if cols <= 0 || cols > (maxModelInt()-7)/6 {
		return 0, false
	}
	return tensor.PackedQ6Cols(cols), true
}

func checkedPackedQ4Cols(cols int) (int, bool) {
	if cols <= 0 || cols == maxModelInt() {
		return 0, false
	}
	return (cols + 1) / 2, true
}

func checkedModelMulInt(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > maxModelInt()/b {
		return 0, false
	}
	return a * b, true
}

func checkedModelAddInt(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > maxModelInt()-b {
		return 0, false
	}
	return a + b, true
}

func maxModelInt() int {
	return int(^uint(0) >> 1)
}

func rmsNormShapeOK(out, x, weight []float32) bool {
	n := len(x)
	return n > 0 && len(out) >= n && len(weight) >= n
}

func addRMSNormShapeOK(out, dst, add, weight []float32) bool {
	n := len(dst)
	return n > 0 && len(out) >= n && len(add) >= n && len(weight) >= n
}

func mropeShapeOK(x []float32, heads, dim int, cosTable, sinTable []float32) bool {
	half := dim / 2
	width, ok := checkedModelMulInt(heads, dim)
	return ok && len(x) >= width && len(cosTable) >= half && len(sinTable) >= half
}

func (rt *Runtime) rmsNormMaybeVulkan(out, x, weight []float32, eps float32) {
	if eps == 1e-6 && len(x) >= vulkanVectorMinWork() && rmsNormShapeOK(out, x, weight) && rt.vulkanOpEnabled(vulkanOpRMSNormF32) {
		if err := backend.VulkanRMSNormF32(out, x, weight); err == nil {
			return
		} else {
			rt.disableVulkanOp(vulkanOpRMSNormF32, err)
		}
	}
	tensor.RMSNorm(out, x, weight, eps)
}

// rmsNormMatVecChainedMaybeVulkan chains RMSNorm followed by a F32 matvec into a
// single command buffer submission.  The intermediate (RMSNorm output) stays in
// GPU memory, eliminating one host readback + upload + fence wait.
// Returns true on success; caller falls back to separate calls on false.
func (rt *Runtime) rmsNormMatVecChainedMaybeVulkan(matvecOut, x, normWeight, w []float32, n, rows, cols int, eps float32) bool {
	if eps != 1e-6 || n < vulkanVectorMinWork() || !rt.vulkanOpEnabled(vulkanOpChainedRMSNormMatVecF32) {
		return false
	}
	if !rmsNormShapeOK(x, x, normWeight) || !f32MatVecShapeOK(matvecOut, x, w, rows, cols) {
		return false
	}
	if len(x) < n || len(normWeight) < n || len(w) < rows*cols || len(matvecOut) < rows {
		return false
	}
	if err := backend.VulkanChainedRMSNormMatVecF32(matvecOut, x, normWeight, w, n, rows, cols); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedRMSNormMatVecF32, err)
	}
	return false
}

// rmsNormQKVMRoPEChainedMaybeVulkan chains RMSNorm followed by fused QKV+MRoPE
// into a single command buffer submission.  The intermediate (RMSNorm output)
// stays in GPU memory, eliminating one host readback + upload + fence wait.
// Returns true on success; caller falls back to separate calls on false.
func (rt *Runtime) rmsNormQKVMRoPEChainedMaybeVulkan(q, k, v, x, normWeight, wa, wb, wc, cosTable, sinTable []float32, n, qRows, kvRows, hidden, qHeads, kvHeads, headDim int, eps float32) bool {
	if eps != 1e-6 || n < vulkanVectorMinWork() || !rt.vulkanOpEnabled(vulkanOpChainedRMSNormQKVMRoPEF32) {
		return false
	}
	if !rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEF32) || !rt.vulkanOpEnabled(vulkanOpRMSNormF32) {
		return false
	}
	if !rmsNormShapeOK(x, x, normWeight) || !fusedMRoPEShapeOK(q, k, qHeads, kvHeads, headDim, cosTable, sinTable) {
		return false
	}
	if !f32MatVecShapeOK(q, x, wa, qRows, hidden) || !f32MatVecShapeOK(k, x, wb, kvRows, hidden) || !f32MatVecShapeOK(v, x, wc, kvRows, hidden) {
		return false
	}
	if err := backend.VulkanChainedRMSNormFusedQKVMRoPEF32(q, k, v, x, normWeight, wa, wb, wc, cosTable, sinTable, n, qRows, kvRows, kvRows, hidden, kvHeads, headDim); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpChainedRMSNormQKVMRoPEF32, err)
	}
	return false
}

func (rt *Runtime) addRMSNormMaybeVulkan(out, dst, add, weight []float32, eps float32) {
	if eps == 1e-6 && len(dst) >= vulkanVectorMinWork() && addRMSNormShapeOK(out, dst, add, weight) && rt.vulkanOpEnabled(vulkanOpAddRMSNormF32) {
		if err := backend.VulkanAddRMSNormF32(out, dst, add, weight); err == nil {
			return
		} else {
			rt.disableVulkanOp(vulkanOpAddRMSNormF32, err)
		}
	}
	tensor.AddRMSNorm(out, dst, add, weight, eps)
}

func (rt *Runtime) addRMSNormOutOnlyMaybeVulkan(out, dst, add, weight []float32, eps float32) {
	if eps == 1e-6 && len(dst) >= vulkanVectorMinWork() && addRMSNormShapeOK(out, dst, add, weight) && rt.vulkanOpEnabled(vulkanOpAddRMSNormOutOnlyF32) {
		if err := backend.VulkanAddRMSNormF32OutOnly(out, dst, add, weight); err == nil {
			return
		} else {
			rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32, err)
		}
	}
	tensor.AddRMSNormOutOnly(out, dst, add, weight, eps)
}

func (rt *Runtime) matVecAddRMSNormMaybeVulkan(normOut, residual, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, normWeight []float32, rows, cols int, eps float32) bool {
	if !rt.matVecAddRMSNormVulkanReady(normOut, residual, x, w, q8, q6, q4, normWeight, rows, cols, eps) {
		return false
	}
	if q8 != nil {
		if err := backend.VulkanMatVecAddRMSNormQ8(normOut, residual, x, q8, normWeight); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ8, err)
		}
		return false
	}
	if q6 != nil {
		if err := backend.VulkanMatVecAddRMSNormQ6(normOut, residual, x, q6, normWeight); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ6, err)
		}
		return false
	}
	if q4 != nil {
		if err := backend.VulkanMatVecAddRMSNormQ4(normOut, residual, x, q4, normWeight); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ4, err)
		}
		return false
	}
	if err := backend.VulkanMatVecAddRMSNormF32(normOut, residual, x, w, normWeight, rows, cols); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpMatVecAddRMSNormF32, err)
	}
	return false
}

func (rt *Runtime) matVecAddRMSNormVulkanReady(normOut, residual, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, normWeight []float32, rows, cols int, eps float32) bool {
	if eps != 1e-6 || !vulkanMatVecAddNormWorkReady(rows, cols) {
		return false
	}
	if len(normOut) < rows || len(residual) < rows || len(x) < cols || len(normWeight) < rows {
		return false
	}
	if q8 != nil {
		return q8MatVecShapeOK(q8, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecAddRMSNormQ8)
	}
	if q6 != nil {
		return q6MatVecShapeOK(q6, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecAddRMSNormQ6)
	}
	if q4 != nil {
		return q4MatVecShapeOK(q4, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecAddRMSNormQ4)
	}
	elements, ok := checkedModelMulInt(rows, cols)
	return ok && len(w) >= elements && rt.vulkanOpEnabled(vulkanOpMatVecAddRMSNormF32)
}

func (rt *Runtime) matVecAddRMSNormOutOnlyMaybeVulkan(normOut, residual, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, normWeight []float32, rows, cols int, eps float32, tmp []float32) bool {
	if !rt.matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, x, w, q8, q6, q4, normWeight, rows, cols, eps, tmp) {
		return false
	}
	if !rt.matVecOnlyMaybeVulkan(tmp[:rows], x, w, q8, q6, q4, rows, cols) {
		return false
	}
	if err := backend.VulkanAddRMSNormF32OutOnly(normOut, residual, tmp[:rows], normWeight); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32, err)
	}
	return false
}

func (rt *Runtime) matVecAddRMSNormOutOnlyVulkanReady(normOut, residual, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, normWeight []float32, rows, cols int, eps float32, tmp []float32) bool {
	if eps != 1e-6 || rows <= 0 || cols <= 0 || len(tmp) < rows ||
		len(normOut) < rows || len(residual) < rows || len(x) < cols || len(normWeight) < rows {
		return false
	}
	if rows < vulkanVectorMinWork() || !rt.vulkanOpEnabled(vulkanOpAddRMSNormOutOnlyF32) {
		return false
	}
	return rt.matVecVulkanReady(tmp[:rows], x, w, q8, q6, q4, rows, cols)
}

func (rt *Runtime) matVecOnlyMaybeVulkan(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) bool {
	if !rt.matVecVulkanReady(out, x, w, q8, q6, q4, rows, cols) {
		return false
	}
	return rt.matVecOnlyVulkanDispatch(out, x, w, q8, q6, q4, rows, cols)
}

func (rt *Runtime) matVecOnlyVulkanDispatch(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) bool {
	if q8 != nil {
		if err := backend.VulkanMatVecQ8(out, x, q8); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecQ8, err)
		}
		return false
	}
	if q6 != nil {
		if err := backend.VulkanMatVecQ6(out, x, q6); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecQ6, err)
		}
		return false
	}
	if q4 != nil {
		if err := backend.VulkanMatVecQ4(out, x, q4); err == nil {
			return true
		} else {
			rt.disableVulkanOp(vulkanOpMatVecQ4, err)
		}
		return false
	}
	if err := backend.VulkanMatVecF32(out, x, w, rows, cols); err == nil {
		return true
	} else {
		rt.disableVulkanOp(vulkanOpMatVecF32, err)
	}
	return false
}

func (rt *Runtime) matVecVulkanReady(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) bool {
	if len(out) < rows || len(x) < cols || !vulkanMatVecWorkReady(rows, cols) {
		return false
	}
	return rt.matVecVulkanShapeReady(out, x, w, q8, q6, q4, rows, cols)
}

func (rt *Runtime) matVecVulkanShapeReady(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) bool {
	if rows <= 0 || cols <= 0 || len(out) < rows || len(x) < cols {
		return false
	}
	if q8 != nil {
		return q8MatVecShapeOK(q8, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecQ8)
	}
	if q6 != nil {
		return q6MatVecShapeOK(q6, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecQ6)
	}
	if q4 != nil {
		return q4MatVecShapeOK(q4, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecQ4)
	}
	return f32MatVecShapeOK(out, x, w, rows, cols) && rt.vulkanOpEnabled(vulkanOpMatVecF32)
}

func (rt *Runtime) matVec2MaybeVulkan(outA, outB, x, wa, wb []float32, q8a, q8b *tensor.Q8Matrix, q6a, q6b *tensor.Q6Matrix, q4a, q4b *tensor.Q4Matrix, rowsA, rowsB, cols int) bool {
	if !rt.matVec2VulkanReady(outA, outB, x, wa, wb, q8a, q8b, q6a, q6b, q4a, q4b, rowsA, rowsB, cols) {
		return false
	}
	if !rt.matVecOnlyVulkanDispatch(outA, x, wa, q8a, q6a, q4a, rowsA, cols) {
		return false
	}
	return rt.matVecOnlyVulkanDispatch(outB, x, wb, q8b, q6b, q4b, rowsB, cols)
}

func (rt *Runtime) matVec2VulkanReady(outA, outB, x, wa, wb []float32, q8a, q8b *tensor.Q8Matrix, q6a, q6b *tensor.Q6Matrix, q4a, q4b *tensor.Q4Matrix, rowsA, rowsB, cols int) bool {
	return groupedVulkanMatVecWorkReady(cols, rowsA, rowsB) &&
		rt.matVecVulkanShapeReady(outA, x, wa, q8a, q6a, q4a, rowsA, cols) &&
		rt.matVecVulkanShapeReady(outB, x, wb, q8b, q6b, q4b, rowsB, cols)
}

func (rt *Runtime) matVec3MaybeVulkan(outA, outB, outC, x, wa, wb, wc []float32, q8a, q8b, q8c *tensor.Q8Matrix, q6a, q6b, q6c *tensor.Q6Matrix, q4a, q4b, q4c *tensor.Q4Matrix, rowsA, rowsB, rowsC, cols int) bool {
	if !rt.matVec3VulkanReady(outA, outB, outC, x, wa, wb, wc, q8a, q8b, q8c, q6a, q6b, q6c, q4a, q4b, q4c, rowsA, rowsB, rowsC, cols) {
		return false
	}
	if !rt.matVecOnlyVulkanDispatch(outA, x, wa, q8a, q6a, q4a, rowsA, cols) {
		return false
	}
	if !rt.matVecOnlyVulkanDispatch(outB, x, wb, q8b, q6b, q4b, rowsB, cols) {
		return false
	}
	return rt.matVecOnlyVulkanDispatch(outC, x, wc, q8c, q6c, q4c, rowsC, cols)
}

func (rt *Runtime) matVec3VulkanReady(outA, outB, outC, x, wa, wb, wc []float32, q8a, q8b, q8c *tensor.Q8Matrix, q6a, q6b, q6c *tensor.Q6Matrix, q4a, q4b, q4c *tensor.Q4Matrix, rowsA, rowsB, rowsC, cols int) bool {
	return groupedVulkanMatVecWorkReady(cols, rowsA, rowsB, rowsC) &&
		rt.matVecVulkanShapeReady(outA, x, wa, q8a, q6a, q4a, rowsA, cols) &&
		rt.matVecVulkanShapeReady(outB, x, wb, q8b, q6b, q4b, rowsB, cols) &&
		rt.matVecVulkanShapeReady(outC, x, wc, q8c, q6c, q4c, rowsC, cols)
}

func (rt *Runtime) matVecMaybeQuant(out, x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) {
	if rt.matVecOnlyMaybeVulkan(out, x, w, q8, q6, q4, rows, cols) {
		return
	}
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

func (rt *Runtime) matVecArgmaxMaybeVulkan(x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int) (int, float32, bool) {
	if !vulkanMatVecWorkReady(rows, cols) {
		return 0, 0, false
	}
	if len(x) < cols {
		return 0, 0, false
	}
	if q8 != nil {
		if !q8MatVecShapeOK(q8, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ8) {
			return 0, 0, false
		}
		token, score, err := backend.VulkanMatVecArgmaxQ8(x, q8)
		if err == nil {
			return token, score, true
		}
		rt.disableVulkanOp(vulkanOpMatVecArgmaxQ8, err)
		return 0, 0, false
	}
	if q6 != nil {
		if !q6MatVecShapeOK(q6, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ6) {
			return 0, 0, false
		}
		token, score, err := backend.VulkanMatVecArgmaxQ6(x, q6)
		if err == nil {
			return token, score, true
		}
		rt.disableVulkanOp(vulkanOpMatVecArgmaxQ6, err)
		return 0, 0, false
	}
	if q4 != nil {
		if !q4MatVecShapeOK(q4, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ4) {
			return 0, 0, false
		}
		token, score, err := backend.VulkanMatVecArgmaxQ4(x, q4)
		if err == nil {
			return token, score, true
		}
		rt.disableVulkanOp(vulkanOpMatVecArgmaxQ4, err)
		return 0, 0, false
	}
	elements, ok := checkedModelMulInt(rows, cols)
	if !ok || len(w) < elements || !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxF32) {
		return 0, 0, false
	}
	token, score, err := backend.VulkanMatVecArgmaxF32(x, w, rows, cols)
	if err == nil {
		return token, score, true
	}
	rt.disableVulkanOp(vulkanOpMatVecArgmaxF32, err)
	return 0, 0, false
}

func (rt *Runtime) matVecTopKMaybeVulkan(x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols, topK int, scratch *generationScratch) ([]tokenScore, bool) {
	const maxVulkanMatVecTopK = 64
	if topK <= 1 || topK > maxVulkanMatVecTopK || !vulkanMatVecWorkReady(rows, cols) {
		return nil, false
	}
	if len(x) < cols {
		return nil, false
	}
	convertCandidates := func(gpuCandidates []backend.VulkanTokenScore) ([]tokenScore, bool) {
		candidates := makeTokenScores(len(gpuCandidates), scratch)
		for i, c := range gpuCandidates {
			candidates[i] = tokenScore{id: c.Token, score: c.Score}
		}
		return candidates, true
	}
	if q8 != nil {
		if !q8MatVecShapeOK(q8, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecTopKQ8) {
			return nil, false
		}
		gpuCandidates, err := backend.VulkanMatVecTopKQ8(x, q8, topK)
		if err != nil {
			rt.disableVulkanOp(vulkanOpMatVecTopKQ8, err)
			return nil, false
		}
		return convertCandidates(gpuCandidates)
	}
	if q6 != nil {
		if !q6MatVecShapeOK(q6, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecTopKQ6) {
			return nil, false
		}
		gpuCandidates, err := backend.VulkanMatVecTopKQ6(x, q6, topK)
		if err != nil {
			rt.disableVulkanOp(vulkanOpMatVecTopKQ6, err)
			return nil, false
		}
		return convertCandidates(gpuCandidates)
	}
	if q4 != nil {
		if !q4MatVecShapeOK(q4, rows, cols) || !rt.vulkanOpEnabled(vulkanOpMatVecTopKQ4) {
			return nil, false
		}
		gpuCandidates, err := backend.VulkanMatVecTopKQ4(x, q4, topK)
		if err != nil {
			rt.disableVulkanOp(vulkanOpMatVecTopKQ4, err)
			return nil, false
		}
		return convertCandidates(gpuCandidates)
	}
	elements, ok := checkedModelMulInt(rows, cols)
	if !ok || len(w) < elements || !rt.vulkanOpEnabled(vulkanOpMatVecTopKF32) {
		return nil, false
	}
	gpuCandidates, err := backend.VulkanMatVecTopKF32(x, w, rows, cols, topK)
	if err != nil {
		rt.disableVulkanOp(vulkanOpMatVecTopKF32, err)
		return nil, false
	}
	return convertCandidates(gpuCandidates)
}

func (rt *Runtime) matVecTopKMaybeCPU(x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols, topK int, temp float64, scratch *generationScratch) ([]tokenScore, bool) {
	if temp <= 0 || topK <= 1 || topK >= rows || rows <= 0 || cols <= 0 || len(x) < cols {
		return nil, false
	}
	var scores []tensor.TopKScore
	var work []tensor.TopKScore
	if scratch != nil {
		scores = scratch.topKScores
		work = scratch.topKWork
	}
	switch {
	case q8 != nil:
		scores, work, _ = tensor.MatVecTopKQ8WithWork(scores, work, x, q8, topK)
	case q6 != nil:
		scores, work, _ = tensor.MatVecTopKQ6WithWork(scores, work, x, q6, topK)
	case q4 != nil:
		scores, work, _ = tensor.MatVecTopKQ4WithWork(scores, work, x, q4, topK)
	case f32MatVecWeightsReady(w, rows, cols):
		scores, work, _ = tensor.MatVecTopKWithWork(scores, work, x, w, rows, cols, topK)
	default:
		return nil, false
	}
	if scratch != nil {
		scratch.topKScores = scores
		scratch.topKWork = work
	}
	candidates := makeTokenScores(len(scores), scratch)
	for i, s := range scores {
		candidates[i] = tokenScore{id: s.ID, score: s.Score}
	}
	return candidates, true
}

func (rt *Runtime) vulkanOpEnabled(op vulkanOp) bool {
	bank, bit := vulkanOpBankBit(op)
	return rt.backend == "vulkan" && rt.vulkanDisabledOps[bank].Load()&bit == 0
}

func (rt *Runtime) disableVulkanOp(op vulkanOp, reasons ...any) {
	reason := vulkanDisableReasonText(reasons...)
	if reason != "" {
		rt.vulkanDisabledMu.Lock()
		if rt.vulkanDisabledReasons == nil {
			rt.vulkanDisabledReasons = map[vulkanOp]string{}
		}
		rt.vulkanDisabledReasons[op] = reason
		rt.vulkanDisabledMu.Unlock()
	}
	bank, bit := vulkanOpBankBit(op)
	for {
		old := rt.vulkanDisabledOps[bank].Load()
		next := old | bit
		if old == next || rt.vulkanDisabledOps[bank].CompareAndSwap(old, next) {
			return
		}
	}
}

func vulkanDisableReasonText(reasons ...any) string {
	for _, reason := range reasons {
		switch v := reason.(type) {
		case nil:
		case error:
			if v != nil {
				return v.Error()
			}
		case string:
			if v != "" {
				return v
			}
		default:
			text := fmt.Sprint(v)
			if text != "" && text != "<nil>" {
				return text
			}
		}
	}
	return ""
}

func vulkanOpBankBit(op vulkanOp) (int, uint64) {
	idx := uint(op)
	return int(idx >> 6), uint64(1) << (idx & 63)
}

func (rt *Runtime) applyMRoPE(x []float32, heads, dim int, pos ropePos) {
	half := dim / 2
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	rt.buildMRoPETable(cosTable, sinTable, pos)
	rt.mropeMaybeVulkan(x, heads, dim, cosTable, sinTable)
}

func (rt *Runtime) mropeMaybeVulkan(x []float32, heads, dim int, cosTable, sinTable []float32) {
	if vulkanVectorWorkReady(heads, dim) && mropeShapeOK(x, heads, dim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpMRoPEF32) {
		if err := backend.VulkanMRoPEF32(x, cosTable, sinTable, heads, dim); err == nil {
			return
		} else {
			rt.disableVulkanOp(vulkanOpMRoPEF32, err)
		}
	}
	applyMRoPEWithTable(x, heads, dim, cosTable, sinTable)
}

func (rt *Runtime) mropePairMaybeVulkan(q, k []float32, qHeads, kvHeads, dim int, cosTable, sinTable []float32) {
	if groupedVulkanVectorWorkReady(dim, qHeads, kvHeads) && mropeShapeOK(q, qHeads, dim, cosTable, sinTable) && mropeShapeOK(k, kvHeads, dim, cosTable, sinTable) && rt.vulkanOpEnabled(vulkanOpMRoPEPairF32) {
		if err := backend.VulkanMRoPEPairF32(q, k, cosTable, sinTable, qHeads, kvHeads, dim); err == nil {
			return
		} else {
			rt.disableVulkanOp(vulkanOpMRoPEPairF32, err)
		}
	}
	applyMRoPEWithTable(q, qHeads, dim, cosTable, sinTable)
	applyMRoPEWithTable(k, kvHeads, dim, cosTable, sinTable)
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
	tensor.ApplyMRoPETable(x, cosTable, sinTable, heads, dim)
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

func invSqrt(n int) float32 { return float32(1 / math.Sqrt(float64(n))) }

func pow(x, y float64) float64 {
	return math.Pow(x, y)
}

func sin(x float64) float32 { return float32(math.Sin(x)) }
func cos(x float64) float32 { return float32(math.Cos(x)) }
