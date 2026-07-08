package backend

const (
	VulkanOpMatVec      = "matvec"
	VulkanOpFusedQKV    = "fused_qkv"
	VulkanOpFusedSwiGLU = "fused_swiglu"
)

type VulkanPlan struct {
	Kernel      VulkanKernel      `json:"kernel"`
	Name        string            `json:"name,omitempty"`
	Stage       string            `json:"stage,omitempty"`
	PipelineKey VulkanPipelineKey `json:"pipeline_key"`
	Repeat      int               `json:"repeat,omitempty"`
	Dispatches  int               `json:"dispatches,omitempty"`
	GroupsX     int               `json:"groups_x"`
	GroupsY     int               `json:"groups_y"`
	Rows        int               `json:"rows"`
	Cols        int               `json:"cols"`
	Quant       string            `json:"quant,omitempty"`
	SharedByte  int               `json:"shared_bytes,omitempty"`
	WeightByte  int64             `json:"weight_bytes,omitempty"`
	InputBytes  int64             `json:"input_bytes,omitempty"`
	OutputBytes int64             `json:"output_bytes,omitempty"`
}

type VulkanPlanSummary struct {
	Plans         []VulkanPlan `json:"-"`
	PlanCount     int          `json:"plan_count"`
	PipelineCount int          `json:"pipeline_count"`
	Dispatches    int          `json:"dispatches"`
	WeightBytes   int64        `json:"weight_bytes"`
	InputBytes    int64        `json:"input_bytes"`
	OutputBytes   int64        `json:"output_bytes"`
	SharedBytes   int64        `json:"shared_bytes"`
	TextLayers    int          `json:"text_layers,omitempty"`
	VisionLayers  int          `json:"vision_layers,omitempty"`
	DispatchReady bool         `json:"dispatch_ready"`
}

type VulkanStageSummary struct {
	Name          string `json:"name"`
	PlanCount     int    `json:"plan_count"`
	PipelineCount int    `json:"pipeline_count"`
	Dispatches    int    `json:"dispatches"`
	WeightBytes   int64  `json:"weight_bytes"`
	InputBytes    int64  `json:"input_bytes"`
	OutputBytes   int64  `json:"output_bytes"`
	SharedBytes   int64  `json:"shared_bytes"`
}

type VulkanExecutionGraph struct {
	Stages        []VulkanStageSummary `json:"stages,omitempty"`
	PlanCount     int                  `json:"plan_count"`
	PipelineCount int                  `json:"pipeline_count"`
	Dispatches    int                  `json:"dispatches"`
	WeightBytes   int64                `json:"weight_bytes"`
	InputBytes    int64                `json:"input_bytes"`
	OutputBytes   int64                `json:"output_bytes"`
	SharedBytes   int64                `json:"shared_bytes"`
	DispatchReady bool                 `json:"dispatch_ready"`
}

type VulkanPipelinePlan struct {
	Key               VulkanPipelineKey `json:"key"`
	Stage             string            `json:"stage,omitempty"`
	LayoutIndex       int               `json:"layout_index"`
	ShaderModuleIndex int               `json:"shader_module_index"`
	CacheKeyHash      uint64            `json:"cache_key_hash"`
	PlanRefs          int               `json:"plan_refs"`
	Dispatches        int               `json:"dispatches"`
}

type VulkanPipelineLayoutKey struct {
	PushConstantBytes int    `json:"push_constant_bytes"`
	BindingSignature  uint64 `json:"binding_signature"`
}

type VulkanPipelineLayoutPlan struct {
	Key          VulkanPipelineLayoutKey `json:"key"`
	Bindings     []VulkanBinding         `json:"bindings,omitempty"`
	PipelineRefs int                     `json:"pipeline_refs"`
}

type VulkanShaderModulePlan struct {
	KernelName      string                          `json:"kernel_name"`
	EntryPoint      string                          `json:"entry_point"`
	SourceHash      uint64                          `json:"source_hash"`
	SourceBytes     int                             `json:"source_bytes"`
	LocalSizeX      int                             `json:"local_size_x"`
	TileCols        int                             `json:"tile_cols"`
	Specializations [3]VulkanSpecializationConstant `json:"specializations"`
	PipelineRefs    int                             `json:"pipeline_refs"`
}

type VulkanSpecializationConstant struct {
	ID    int    `json:"id"`
	Name  string `json:"name"`
	Value int    `json:"value"`
}

type VulkanCommand struct {
	Stage            string                  `json:"stage,omitempty"`
	Name             string                  `json:"name,omitempty"`
	PipelineIndex    int                     `json:"pipeline_index"`
	DescriptorSet    VulkanDescriptorSetPlan `json:"descriptor_set"`
	PushConstants    VulkanPushConstants     `json:"push_constants"`
	Rows             int                     `json:"rows"`
	Cols             int                     `json:"cols"`
	GroupsX          int                     `json:"groups_x"`
	GroupsY          int                     `json:"groups_y"`
	Repeat           int                     `json:"repeat,omitempty"`
	Dispatches       int                     `json:"dispatches"`
	Resources        []VulkanResource        `json:"resources,omitempty"`
	DescriptorWrites []VulkanDescriptorWrite `json:"descriptor_writes,omitempty"`
}

type VulkanDescriptorSetPlan struct {
	SetIndex         int `json:"set_index"`
	LayoutIndex      int `json:"layout_index"`
	DescriptorOffset int `json:"descriptor_offset"`
	DescriptorCount  int `json:"descriptor_count"`
}

type VulkanPushConstants struct {
	Rows  uint32 `json:"rows"`
	Cols  uint32 `json:"cols"`
	Bytes int    `json:"bytes"`
}

type VulkanResource struct {
	Binding int    `json:"binding"`
	Name    string `json:"name"`
	Role    string `json:"role"`
	Bytes   int64  `json:"bytes"`
	Access  string `json:"access"`
}

type VulkanDescriptorWrite struct {
	Binding        int    `json:"binding"`
	DescriptorType string `json:"descriptor_type"`
	ResourceName   string `json:"resource_name"`
	OffsetBytes    int64  `json:"offset_bytes"`
	RangeBytes     int64  `json:"range_bytes"`
	Access         string `json:"access"`
}

type VulkanCommandRecord struct {
	Op                   string `json:"op"`
	CommandIndex         int    `json:"command_index"`
	PipelineIndex        int    `json:"pipeline_index"`
	LayoutIndex          int    `json:"layout_index"`
	DescriptorWriteStart int    `json:"descriptor_write_start"`
	DescriptorWriteCount int    `json:"descriptor_write_count"`
	PushConstantBytes    int    `json:"push_constant_bytes"`
	Rows                 int    `json:"rows"`
	Cols                 int    `json:"cols"`
	GroupsX              int    `json:"groups_x"`
	GroupsY              int    `json:"groups_y"`
	Repeat               int    `json:"repeat"`
}

type VulkanDispatchBatch struct {
	BatchIndex          int    `json:"batch_index"`
	Stage               string `json:"stage,omitempty"`
	PipelineIndex       int    `json:"pipeline_index"`
	LayoutIndex         int    `json:"layout_index"`
	CommandStart        int    `json:"command_start"`
	CommandCount        int    `json:"command_count"`
	DescriptorSetStart  int    `json:"descriptor_set_start"`
	DescriptorSetCount  int    `json:"descriptor_set_count"`
	Dispatches          int    `json:"dispatches"`
	PipelineBindCount   int    `json:"pipeline_bind_count"`
	DescriptorBindCount int    `json:"descriptor_bind_count"`
	PushConstantCount   int    `json:"push_constant_count"`
}

type VulkanBufferBarrier struct {
	CommandIndex int    `json:"command_index"`
	Binding      int    `json:"binding"`
	ResourceName string `json:"resource_name"`
	Role         string `json:"role"`
	SrcStage     string `json:"src_stage"`
	DstStage     string `json:"dst_stage"`
	SrcAccess    string `json:"src_access"`
	DstAccess    string `json:"dst_access"`
	Bytes        int64  `json:"bytes"`
}

type VulkanBufferAllocation struct {
	CommandIndex     int    `json:"command_index"`
	Binding          int    `json:"binding"`
	ResourceName     string `json:"resource_name"`
	Role             string `json:"role"`
	Usage            string `json:"usage"`
	MemoryProperties string `json:"memory_properties"`
	Bytes            int64  `json:"bytes"`
	AlignmentBytes   int64  `json:"alignment_bytes"`
	AlignedBytes     int64  `json:"aligned_bytes"`
}

type VulkanBufferTransfer struct {
	CommandIndex int    `json:"command_index"`
	Binding      int    `json:"binding"`
	ResourceName string `json:"resource_name"`
	Role         string `json:"role"`
	Direction    string `json:"direction"`
	Bytes        int64  `json:"bytes"`
}

type VulkanDescriptorPoolPlan struct {
	MaxSets            int `json:"max_sets"`
	StorageBufferCount int `json:"storage_buffer_count"`
}

type VulkanCommandPoolPlan struct {
	QueueFamily        string `json:"queue_family"`
	CommandBufferCount int    `json:"command_buffer_count"`
	ResetMode          string `json:"reset_mode"`
}

type VulkanQueueSubmitPlan struct {
	QueueFamily        string `json:"queue_family"`
	SubmitCount        int    `json:"submit_count"`
	CommandBufferCount int    `json:"command_buffer_count"`
	WaitSemaphoreCount int    `json:"wait_semaphore_count"`
	SignalFence        bool   `json:"signal_fence"`
}

type VulkanTimelinePlan struct {
	InitialValue int64  `json:"initial_value"`
	WaitValue    int64  `json:"wait_value"`
	SignalValue  int64  `json:"signal_value"`
	StageMask    string `json:"stage_mask"`
}

type VulkanFencePlan struct {
	CreateSignaled    bool `json:"create_signaled"`
	ResetBeforeSubmit bool `json:"reset_before_submit"`
	WaitAfterSubmit   bool `json:"wait_after_submit"`
}

type VulkanPipelineCachePlan struct {
	CacheKeyHash uint64 `json:"cache_key_hash"`
	EntryCount   int    `json:"entry_count"`
	CreateCount  int    `json:"create_count"`
	ReuseCount   int    `json:"reuse_count"`
	LayoutRefs   int    `json:"layout_refs"`
	ShaderRefs   int    `json:"shader_refs"`
	Persistable  bool   `json:"persistable"`
}

type VulkanPipelineLifecyclePlan struct {
	Steps        [4]string `json:"steps"`
	CreateCount  int       `json:"create_count"`
	DestroyCount int       `json:"destroy_count"`
}

type VulkanCommandPlan struct {
	Pipelines            []VulkanPipelinePlan        `json:"pipelines,omitempty"`
	ShaderModules        []VulkanShaderModulePlan    `json:"shader_modules,omitempty"`
	Layouts              []VulkanPipelineLayoutPlan  `json:"layouts,omitempty"`
	Commands             []VulkanCommand             `json:"commands,omitempty"`
	DispatchBatches      []VulkanDispatchBatch       `json:"dispatch_batches,omitempty"`
	Records              []VulkanCommandRecord       `json:"records,omitempty"`
	Barriers             []VulkanBufferBarrier       `json:"barriers,omitempty"`
	Allocations          []VulkanBufferAllocation    `json:"allocations,omitempty"`
	Uploads              []VulkanBufferTransfer      `json:"uploads,omitempty"`
	Readbacks            []VulkanBufferTransfer      `json:"readbacks,omitempty"`
	DescriptorPool       VulkanDescriptorPoolPlan    `json:"descriptor_pool"`
	CommandPool          VulkanCommandPoolPlan       `json:"command_pool"`
	QueueSubmit          VulkanQueueSubmitPlan       `json:"queue_submit"`
	Timeline             VulkanTimelinePlan          `json:"timeline"`
	Fence                VulkanFencePlan             `json:"fence"`
	PipelineCache        VulkanPipelineCachePlan     `json:"pipeline_cache"`
	PipelineLifecycle    VulkanPipelineLifecyclePlan `json:"pipeline_lifecycle"`
	PipelineCount        int                         `json:"pipeline_count"`
	ShaderModuleCount    int                         `json:"shader_module_count"`
	LayoutCount          int                         `json:"layout_count"`
	CommandCount         int                         `json:"command_count"`
	DispatchBatchCount   int                         `json:"dispatch_batch_count"`
	RecordCount          int                         `json:"record_count"`
	BarrierCount         int                         `json:"barrier_count"`
	AllocationCount      int                         `json:"allocation_count"`
	UploadCount          int                         `json:"upload_count"`
	ReadbackCount        int                         `json:"readback_count"`
	ResourceCount        int                         `json:"resource_count"`
	DescriptorWriteCount int                         `json:"descriptor_write_count"`
	TotalBufferBytes     int64                       `json:"total_buffer_bytes"`
	UploadBytes          int64                       `json:"upload_bytes"`
	ReadbackBytes        int64                       `json:"readback_bytes"`
	Dispatches           int                         `json:"dispatches"`
	PipelineBindCount    int                         `json:"pipeline_bind_count"`
	DescriptorBindCount  int                         `json:"descriptor_bind_count"`
	PushConstantCount    int                         `json:"push_constant_count"`
	DispatchReady        bool                        `json:"dispatch_ready"`
}

type VulkanModelShape struct {
	Quant               string
	VocabSize           int
	HiddenSize          int
	IntermediateSize    int
	NumAttentionHeads   int
	NumKeyValueHeads    int
	HeadDim             int
	VisionHiddenSize    int
	VisionIntermediate  int
	VisionPatchElements int
	TextLayers          int
	VisionLayers        int
}

type VulkanModelArtifacts struct {
	Plans                []VulkanPlan         `json:"plans,omitempty"`
	OptionalPlans        []VulkanPlan         `json:"optional_plans,omitempty"`
	Summary              VulkanPlanSummary    `json:"summary"`
	ExecutionGraph       VulkanExecutionGraph `json:"execution_graph"`
	PipelinePlan         []VulkanPipelinePlan `json:"pipeline_plan,omitempty"`
	OptionalPipelinePlan []VulkanPipelinePlan `json:"optional_pipeline_plan,omitempty"`
	CommandPlan          VulkanCommandPlan    `json:"command_plan"`
	OptionalCommand      VulkanCommandPlan    `json:"optional_command,omitempty"`
}

func DefaultVulkanCompute() VulkanCompute {
	return VulkanCompute{
		Version:       1,
		Workgroup:     256,
		VectorWidth:   4,
		Kernels:       VulkanKernels(),
		Fusions:       []string{"q_proj+k_proj+v_proj", "gate_proj+up_proj+silu_mul", "qkv_quant_matvec"},
		DispatchReady: false,
	}
}

var vulkanKernelRegistry = []VulkanKernel{
	vulkanKernel("matvec_f32_wg256", VulkanOpMatVec, "f32", vulkanMatVecF32GLSL),
	vulkanKernel("matvec_q8_wg256", VulkanOpMatVec, "q8", vulkanMatVecQ8GLSL),
	vulkanKernel("matvec_q6_wg256", VulkanOpMatVec, "q6", vulkanMatVecQ6GLSL),
	vulkanKernel("matvec_q4_wg256", VulkanOpMatVec, "q4", vulkanMatVecQ4GLSL),
	vulkanKernel("fused_qkv_f32_wg256", VulkanOpFusedQKV, "f32", vulkanFusedQKVF32GLSL),
	vulkanKernel("fused_qkv_q8_wg256", VulkanOpFusedQKV, "q8", vulkanFusedQKVQ8GLSL),
	vulkanKernel("fused_qkv_q6_wg256", VulkanOpFusedQKV, "q6", vulkanFusedQKVQ6GLSL),
	vulkanKernel("fused_qkv_q4_wg256", VulkanOpFusedQKV, "q4", vulkanFusedQKVQ4GLSL),
	vulkanKernel("fused_swiglu_f32_wg256", VulkanOpFusedSwiGLU, "f32", vulkanFusedSwiGLUF32GLSL),
	vulkanKernel("fused_swiglu_q8_wg256", VulkanOpFusedSwiGLU, "q8", vulkanFusedSwiGLUQ8GLSL),
	vulkanKernel("fused_swiglu_q6_wg256", VulkanOpFusedSwiGLU, "q6", vulkanFusedSwiGLUQ6GLSL),
	vulkanKernel("fused_swiglu_q4_wg256", VulkanOpFusedSwiGLU, "q4", vulkanFusedSwiGLUQ4GLSL),
}

func vulkanKernel(name, op, quant, source string) VulkanKernel {
	bindings := vulkanKernelBindings(quant)
	return VulkanKernel{
		Name:              name,
		Op:                op,
		Quant:             quant,
		Workgroup:         256,
		RowsPerWarp:       1,
		TileCols:          256,
		PushConstantBytes: 8,
		BindingSignature:  vulkanBindingSignature(bindings),
		Bindings:          bindings,
		SourceHash:        vulkanSourceHash(source),
		SourceBytes:       len(source),
		Source:            source,
	}
}

func vulkanKernelBindings(quant string) []VulkanBinding {
	switch normalizeVulkanQuant(quant) {
	case "q8":
		return []VulkanBinding{
			{Binding: 0, Name: "x", Role: "input", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 1, Name: "w", Role: "weight", Elem: "int8", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 2, Name: "scale", Role: "scale", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 3, Name: "out", Role: "output", Elem: "float32", Access: "writeonly", DescriptorType: "storage_buffer"},
		}
	case "q6":
		return []VulkanBinding{
			{Binding: 0, Name: "x", Role: "input", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 1, Name: "w", Role: "weight", Elem: "packed_q6", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 2, Name: "scale", Role: "scale", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 3, Name: "out", Role: "output", Elem: "float32", Access: "writeonly", DescriptorType: "storage_buffer"},
		}
	case "q4":
		return []VulkanBinding{
			{Binding: 0, Name: "x", Role: "input", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 1, Name: "w", Role: "weight", Elem: "packed_q4", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 2, Name: "scale", Role: "scale", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 3, Name: "out", Role: "output", Elem: "float32", Access: "writeonly", DescriptorType: "storage_buffer"},
		}
	default:
		return []VulkanBinding{
			{Binding: 0, Name: "x", Role: "input", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 1, Name: "w", Role: "weight", Elem: "float32", Access: "readonly", DescriptorType: "storage_buffer"},
			{Binding: 2, Name: "out", Role: "output", Elem: "float32", Access: "writeonly", DescriptorType: "storage_buffer"},
		}
	}
}

func ValidateVulkanKernelABI(k VulkanKernel) string {
	if k.Name == "" {
		return "missing kernel name"
	}
	if k.Workgroup <= 0 {
		return "invalid workgroup"
	}
	if k.PushConstantBytes != 8 {
		return "invalid push constant size"
	}
	if len(k.Bindings) < 3 {
		return "missing descriptor bindings"
	}
	var input, weight, output bool
	seen := uint64(0)
	for i, b := range k.Bindings {
		if b.Binding != i {
			return "descriptor bindings must be contiguous from zero"
		}
		if b.Name == "" || b.Role == "" || b.Elem == "" || b.Access == "" || b.DescriptorType == "" {
			return "incomplete descriptor binding"
		}
		if b.DescriptorType != "storage_buffer" {
			return "unsupported descriptor type"
		}
		bit := uint64(1) << uint(b.Binding)
		if seen&bit != 0 {
			return "duplicate descriptor binding"
		}
		seen |= bit
		switch b.Role {
		case "input":
			input = true
		case "weight":
			weight = true
		case "output":
			output = true
		}
	}
	if !input || !weight || !output {
		return "descriptor roles must include input, weight, output"
	}
	if sig := vulkanBindingSignature(k.Bindings); k.BindingSignature != 0 && sig != k.BindingSignature {
		return "descriptor binding signature mismatch"
	}
	return ""
}

func ValidateVulkanCommandPlan(p VulkanCommandPlan) string {
	if p.PipelineCount != len(p.Pipelines) {
		return "pipeline count mismatch"
	}
	if p.ShaderModuleCount != len(p.ShaderModules) {
		return "shader module count mismatch"
	}
	if p.LayoutCount != len(p.Layouts) {
		return "layout count mismatch"
	}
	if p.CommandCount != len(p.Commands) {
		return "command count mismatch"
	}
	if p.DispatchBatchCount != len(p.DispatchBatches) {
		return "dispatch batch count mismatch"
	}
	if p.RecordCount != len(p.Records) {
		return "record count mismatch"
	}
	if p.BarrierCount != len(p.Barriers) {
		return "barrier count mismatch"
	}
	if p.AllocationCount != len(p.Allocations) {
		return "allocation count mismatch"
	}
	if p.UploadCount != len(p.Uploads) {
		return "upload count mismatch"
	}
	if p.ReadbackCount != len(p.Readbacks) {
		return "readback count mismatch"
	}
	if p.DescriptorWriteCount == 0 && p.CommandCount != 0 {
		return "missing descriptor writes"
	}
	if p.ResourceCount != p.DescriptorWriteCount || p.ResourceCount != p.BarrierCount || p.ResourceCount != p.AllocationCount {
		return "resource-backed plan count mismatch"
	}
	if p.DescriptorPool.MaxSets != p.CommandCount || p.DescriptorPool.StorageBufferCount != p.DescriptorWriteCount {
		return "descriptor pool mismatch"
	}
	if p.CommandPool.QueueFamily != "compute" || p.QueueSubmit.QueueFamily != "compute" {
		return "non-compute queue family"
	}
	if p.CommandPool.CommandBufferCount != p.QueueSubmit.CommandBufferCount {
		return "command buffer count mismatch"
	}
	if p.QueueSubmit.SubmitCount > 0 && (!p.QueueSubmit.SignalFence || !p.Fence.ResetBeforeSubmit || !p.Fence.WaitAfterSubmit) {
		return "incomplete submit synchronization"
	}
	if p.QueueSubmit.SubmitCount > 0 && p.Timeline.SignalValue < int64(p.QueueSubmit.SubmitCount) {
		return "timeline signal value too small"
	}
	if p.PipelineCache.EntryCount != p.PipelineCount || p.PipelineCache.CreateCount != p.PipelineCount {
		return "pipeline cache count mismatch"
	}
	if p.PipelineLifecycle.CreateCount != p.PipelineCount || p.PipelineLifecycle.DestroyCount != p.PipelineCount {
		return "pipeline lifecycle count mismatch"
	}
	var batchDispatches int
	for i, b := range p.DispatchBatches {
		if b.BatchIndex != i {
			return "dispatch batch index mismatch"
		}
		if b.PipelineIndex < 0 || b.PipelineIndex >= p.PipelineCount {
			return "dispatch batch pipeline index out of range"
		}
		if b.LayoutIndex != p.Pipelines[b.PipelineIndex].LayoutIndex {
			return "dispatch batch layout mismatch"
		}
		if b.CommandStart < 0 || b.CommandCount <= 0 || b.CommandStart+b.CommandCount > p.CommandCount {
			return "dispatch batch command range invalid"
		}
		for j := 0; j < b.CommandCount; j++ {
			cmd := p.Commands[b.CommandStart+j]
			if cmd.PipelineIndex != b.PipelineIndex || cmd.DescriptorSet.SetIndex != b.DescriptorSetStart+j {
				return "dispatch batch command mismatch"
			}
		}
		if b.DescriptorSetCount != b.CommandCount || b.PipelineBindCount != 1 || b.DescriptorBindCount != b.CommandCount || b.PushConstantCount != b.CommandCount {
			return "dispatch batch bind count mismatch"
		}
		batchDispatches += b.Dispatches
	}
	if batchDispatches != p.Dispatches || p.PipelineBindCount != p.DispatchBatchCount || p.DescriptorBindCount != p.CommandCount || p.PushConstantCount != p.CommandCount {
		return "dispatch batch totals mismatch"
	}
	for i, pipe := range p.Pipelines {
		if pipe.LayoutIndex < 0 || pipe.LayoutIndex >= p.LayoutCount {
			return "pipeline layout index out of range"
		}
		if pipe.ShaderModuleIndex < 0 || pipe.ShaderModuleIndex >= p.ShaderModuleCount {
			return "pipeline shader index out of range"
		}
		if pipe.CacheKeyHash == 0 {
			return "missing pipeline cache key"
		}
		if p.ShaderModules[pipe.ShaderModuleIndex].KernelName != pipe.Key.KernelName {
			return "pipeline shader module mismatch"
		}
		if i >= p.PipelineCount {
			return "pipeline index overflow"
		}
		for j := 0; j < i; j++ {
			prev := p.Pipelines[j]
			if prev.Key == pipe.Key && prev.Stage == pipe.Stage {
				return "duplicate pipeline"
			}
			if prev.CacheKeyHash == pipe.CacheKeyHash {
				return "duplicate pipeline cache key"
			}
		}
	}
	var dispatches int
	var totalAligned int64
	var uploadBytes, readbackBytes int64
	resourceIndex := 0
	uploadIndex := 0
	readbackIndex := 0
	for commandIndex, cmd := range p.Commands {
		if cmd.PipelineIndex < 0 || cmd.PipelineIndex >= p.PipelineCount {
			return "command pipeline index out of range"
		}
		layoutIndex := p.Pipelines[cmd.PipelineIndex].LayoutIndex
		if cmd.DescriptorSet.SetIndex != commandIndex || cmd.DescriptorSet.LayoutIndex != layoutIndex || cmd.DescriptorSet.DescriptorCount != len(cmd.DescriptorWrites) {
			return "descriptor set plan mismatch"
		}
		if cmd.PushConstants.Bytes != p.Pipelines[cmd.PipelineIndex].Key.PushConstantBytes || int(cmd.PushConstants.Rows) != cmd.Rows || int(cmd.PushConstants.Cols) != cmd.Cols {
			return "push constant payload mismatch"
		}
		if len(cmd.Resources) != len(cmd.DescriptorWrites) {
			return "command resource descriptor mismatch"
		}
		for i, r := range cmd.Resources {
			w := cmd.DescriptorWrites[i]
			if w.Binding != r.Binding || w.ResourceName != r.Name || w.RangeBytes != r.Bytes || w.Access != r.Access || w.DescriptorType != "storage_buffer" {
				return "descriptor resource mismatch"
			}
			if w.OffsetBytes < 0 || w.RangeBytes <= 0 {
				return "invalid descriptor byte range"
			}
			if resourceIndex >= len(p.Allocations) {
				return "allocation index out of range"
			}
			a := p.Allocations[resourceIndex]
			if a.CommandIndex != commandIndex || a.Binding != r.Binding || a.ResourceName != r.Name || a.Bytes != r.Bytes {
				return "allocation resource mismatch"
			}
			if a.AlignmentBytes <= 0 || a.AlignedBytes < a.Bytes || a.AlignedBytes%a.AlignmentBytes != 0 {
				return "invalid aligned allocation"
			}
			if r.Role == "output" {
				if readbackIndex >= p.ReadbackCount {
					return "readback index out of range"
				}
				t := p.Readbacks[readbackIndex]
				if t.CommandIndex != commandIndex || t.Binding != r.Binding || t.ResourceName != r.Name || t.Role != r.Role || t.Direction != "device_to_host" || t.Bytes != r.Bytes {
					return "readback resource mismatch"
				}
				readbackBytes += t.Bytes
				readbackIndex++
			} else {
				if uploadIndex >= p.UploadCount {
					return "upload index out of range"
				}
				t := p.Uploads[uploadIndex]
				if t.CommandIndex != commandIndex || t.Binding != r.Binding || t.ResourceName != r.Name || t.Role != r.Role || t.Direction != "host_to_device" || t.Bytes != r.Bytes {
					return "upload resource mismatch"
				}
				uploadBytes += t.Bytes
				uploadIndex++
			}
			totalAligned += a.AlignedBytes
			resourceIndex++
		}
		dispatches += cmd.Dispatches
	}
	if resourceIndex != p.ResourceCount {
		return "resource walk count mismatch"
	}
	if uploadIndex != p.UploadCount || readbackIndex != p.ReadbackCount {
		return "transfer walk count mismatch"
	}
	if totalAligned != p.TotalBufferBytes {
		return "total aligned buffer byte mismatch"
	}
	if uploadBytes != p.UploadBytes || readbackBytes != p.ReadbackBytes {
		return "transfer byte mismatch"
	}
	if dispatches != p.Dispatches {
		return "dispatch count mismatch"
	}
	return ""
}

func vulkanBindingSignature(bindings []VulkanBinding) uint64 {
	var sig uint64
	for _, b := range bindings {
		role := vulkanBindingRoleCode(b.Role)
		elem := vulkanBindingElemCode(b.Elem)
		access := vulkanBindingAccessCode(b.Access)
		shift := uint(b.Binding * 8)
		if shift >= 64 {
			continue
		}
		sig |= uint64((role&0x7)|((elem&0x7)<<3)|((access&0x3)<<6)) << shift
	}
	return sig
}

func vulkanBindingRoleCode(role string) uint64 {
	switch role {
	case "input":
		return 1
	case "weight":
		return 2
	case "scale":
		return 3
	case "output":
		return 4
	default:
		return 0
	}
}

func vulkanBindingElemCode(elem string) uint64 {
	switch elem {
	case "float32":
		return 1
	case "int8":
		return 2
	case "packed_q6":
		return 3
	case "packed_q4":
		return 4
	default:
		return 0
	}
}

func vulkanBindingAccessCode(access string) uint64 {
	switch access {
	case "readonly":
		return 1
	case "writeonly":
		return 2
	default:
		return 0
	}
}

func VulkanKernels() []VulkanKernel {
	out := make([]VulkanKernel, len(vulkanKernelRegistry))
	copy(out, vulkanKernelRegistry)
	return out
}

func PlanVulkanKernel(op, quant string, rows, cols int) (VulkanPlan, bool) {
	op = lowerASCII(trimASCIIWhitespace(op))
	quant = normalizeVulkanQuant(quant)
	if rows <= 0 || cols <= 0 || rows > maxUint32Int() || cols > maxUint32Int() {
		return VulkanPlan{}, false
	}
	k, ok := vulkanKernelFor(op, quant)
	if !ok {
		return VulkanPlan{}, false
	}
	weightBytes, ok := estimateVulkanWeightBytesCheckedNormalized(quant, rows, cols)
	if !ok {
		return VulkanPlan{}, false
	}
	outputRows, ok := vulkanPlanOutputRows(op, rows)
	if !ok {
		return VulkanPlan{}, false
	}
	inputBytes, ok := checkedMulInt64(int64(cols), 4)
	if !ok {
		return VulkanPlan{}, false
	}
	outputBytes, ok := checkedMulInt64(int64(outputRows), 4)
	if !ok {
		return VulkanPlan{}, false
	}
	return VulkanPlan{
		Kernel: k,
		PipelineKey: VulkanPipelineKey{
			KernelName:        k.Name,
			Op:                k.Op,
			Quant:             k.Quant,
			Workgroup:         k.Workgroup,
			PushConstantBytes: k.PushConstantBytes,
			BindingSignature:  k.BindingSignature,
		},
		GroupsX:     ceilDiv(outputRows, max(1, k.RowsPerWarp)),
		GroupsY:     1,
		Rows:        rows,
		Cols:        cols,
		Quant:       quant,
		SharedByte:  k.Workgroup * 4,
		WeightByte:  weightBytes,
		InputBytes:  inputBytes,
		OutputBytes: outputBytes,
	}, true
}

func planVulkanKernelNormalized(op, quant string, rows, cols int) (VulkanPlan, bool) {
	if rows <= 0 || cols <= 0 || rows > maxUint32Int() || cols > maxUint32Int() {
		return VulkanPlan{}, false
	}
	k, ok := vulkanKernelFor(op, quant)
	if !ok {
		return VulkanPlan{}, false
	}
	weightBytes, ok := estimateVulkanWeightBytesCheckedNormalized(quant, rows, cols)
	if !ok {
		return VulkanPlan{}, false
	}
	outputRows, ok := vulkanPlanOutputRows(op, rows)
	if !ok {
		return VulkanPlan{}, false
	}
	inputBytes, ok := checkedMulInt64(int64(cols), 4)
	if !ok {
		return VulkanPlan{}, false
	}
	outputBytes, ok := checkedMulInt64(int64(outputRows), 4)
	if !ok {
		return VulkanPlan{}, false
	}
	return VulkanPlan{
		Kernel: k,
		PipelineKey: VulkanPipelineKey{
			KernelName:        k.Name,
			Op:                k.Op,
			Quant:             k.Quant,
			Workgroup:         k.Workgroup,
			PushConstantBytes: k.PushConstantBytes,
			BindingSignature:  k.BindingSignature,
		},
		GroupsX:     ceilDiv(outputRows, max(1, k.RowsPerWarp)),
		GroupsY:     1,
		Rows:        rows,
		Cols:        cols,
		Quant:       quant,
		SharedByte:  k.Workgroup * 4,
		WeightByte:  weightBytes,
		InputBytes:  inputBytes,
		OutputBytes: outputBytes,
	}, true
}

func vulkanPlanOutputRows(op string, rows int) (int, bool) {
	if op != VulkanOpFusedSwiGLU {
		return rows, true
	}
	if rows <= 0 || rows%2 != 0 {
		return 0, false
	}
	return rows / 2, true
}

func vulkanKernelFor(op, quant string) (VulkanKernel, bool) {
	switch op {
	case VulkanOpMatVec:
		switch quant {
		case "f32":
			return vulkanKernelRegistry[0], true
		case "q8":
			return vulkanKernelRegistry[1], true
		case "q6":
			return vulkanKernelRegistry[2], true
		case "q4":
			return vulkanKernelRegistry[3], true
		}
	case VulkanOpFusedQKV:
		switch quant {
		case "f32":
			return vulkanKernelRegistry[4], true
		case "q8":
			return vulkanKernelRegistry[5], true
		case "q6":
			return vulkanKernelRegistry[6], true
		case "q4":
			return vulkanKernelRegistry[7], true
		}
	case VulkanOpFusedSwiGLU:
		switch quant {
		case "f32":
			return vulkanKernelRegistry[8], true
		case "q8":
			return vulkanKernelRegistry[9], true
		case "q6":
			return vulkanKernelRegistry[10], true
		case "q4":
			return vulkanKernelRegistry[11], true
		}
	}
	return VulkanKernel{}, false
}

func VulkanModelPlans(shape VulkanModelShape) []VulkanPlan {
	return VulkanModelPlansInto(nil, shape)
}

func VulkanModelPlansInto(dst []VulkanPlan, shape VulkanModelShape) []VulkanPlan {
	q := normalizeVulkanQuant(shape.Quant)
	if shape.HeadDim == 0 && shape.NumAttentionHeads != 0 {
		shape.HeadDim = shape.HiddenSize / shape.NumAttentionHeads
	}
	if shape.NumKeyValueHeads == 0 {
		shape.NumKeyValueHeads = shape.NumAttentionHeads
	}
	qRows, ok := checkedMulInt(shape.NumAttentionHeads, shape.HeadDim)
	if !ok {
		return dst[:0]
	}
	kvRows, ok := checkedMulInt(shape.NumKeyValueHeads, shape.HeadDim)
	if !ok {
		return dst[:0]
	}
	qkvRows, ok := checkedAddInt(qRows, kvRows)
	if ok {
		qkvRows, ok = checkedAddInt(qkvRows, kvRows)
	}
	if !ok {
		return dst[:0]
	}
	mlpRows, ok := checkedMulInt(shape.IntermediateSize, 2)
	if !ok {
		return dst[:0]
	}
	visionQKVRows, ok := checkedMulInt(shape.VisionHiddenSize, 3)
	if !ok {
		return dst[:0]
	}
	visionMLPRows, ok := checkedMulInt(shape.VisionIntermediate, 2)
	if !ok {
		return dst[:0]
	}
	out := dst[:0]
	if cap(out) < 10 {
		out = make([]VulkanPlan, 0, 10)
	}
	appendPlan := func(name, stage, op, quant string, rows, cols, repeat int) {
		p, ok := planVulkanKernelNormalized(op, quant, rows, cols)
		if !ok {
			return
		}
		p.Name = name
		p.Stage = stage
		if repeat < 1 {
			repeat = 1
		}
		p.Repeat = repeat
		dispatches, ok := checkedDispatches(repeat, p.GroupsX, max(1, p.GroupsY))
		if !ok {
			return
		}
		p.Dispatches = dispatches
		out = append(out, p)
	}
	appendPlan("text.qkv", "text", VulkanOpFusedQKV, q, qkvRows, shape.HiddenSize, shape.TextLayers)
	appendPlan("text.o_proj", "text", VulkanOpMatVec, q, shape.HiddenSize, qRows, shape.TextLayers)
	appendPlan("text.mlp_gate_up", "text", VulkanOpFusedSwiGLU, q, mlpRows, shape.HiddenSize, shape.TextLayers)
	appendPlan("text.mlp_down", "text", VulkanOpMatVec, q, shape.HiddenSize, shape.IntermediateSize, shape.TextLayers)
	appendPlan("text.lm_head", "text", VulkanOpMatVec, q, shape.VocabSize, shape.HiddenSize, 1)
	appendPlan("vision.patch", "vision", VulkanOpMatVec, "f32", shape.VisionHiddenSize, shape.VisionPatchElements, 1)
	appendPlan("vision.qkv", "vision", VulkanOpFusedQKV, "f32", visionQKVRows, shape.VisionHiddenSize, shape.VisionLayers)
	appendPlan("vision.o_proj", "vision", VulkanOpMatVec, "f32", shape.VisionHiddenSize, shape.VisionHiddenSize, shape.VisionLayers)
	appendPlan("vision.mlp_gate_up", "vision", VulkanOpFusedSwiGLU, "f32", visionMLPRows, shape.VisionHiddenSize, shape.VisionLayers)
	appendPlan("vision.mlp_down", "vision", VulkanOpMatVec, "f32", shape.VisionHiddenSize, shape.VisionIntermediate, shape.VisionLayers)
	return out
}

func VulkanModelPlanSummary(shape VulkanModelShape) VulkanPlanSummary {
	plans := VulkanModelPlans(shape)
	return VulkanModelPlanSummaryFromPlans(plans, shape)
}

func VulkanModelPlanSummaryInto(dst []VulkanPlan, shape VulkanModelShape) VulkanPlanSummary {
	plans := VulkanModelPlansInto(dst, shape)
	return VulkanModelPlanSummaryFromPlans(plans, shape)
}

func VulkanModelExecutionGraph(shape VulkanModelShape) VulkanExecutionGraph {
	plans := VulkanModelPlans(shape)
	return VulkanModelExecutionGraphFromPlans(plans)
}

func VulkanModelExecutionGraphInto(planDst []VulkanPlan, stageDst []VulkanStageSummary, shape VulkanModelShape) (VulkanExecutionGraph, []VulkanPlan, []VulkanStageSummary) {
	plans := VulkanModelPlansInto(planDst, shape)
	graph, stages := VulkanModelExecutionGraphFromPlansInto(stageDst, plans)
	return graph, plans, stages
}

func VulkanModelPipelinePlan(shape VulkanModelShape) []VulkanPipelinePlan {
	plans := VulkanModelPlans(shape)
	return VulkanPipelinePlanFromPlans(nil, plans)
}

func VulkanModelPipelinePlanInto(planDst []VulkanPlan, pipeDst []VulkanPipelinePlan, shape VulkanModelShape) ([]VulkanPipelinePlan, []VulkanPlan) {
	plans := VulkanModelPlansInto(planDst, shape)
	return VulkanPipelinePlanFromPlans(pipeDst, plans), plans
}

func VulkanModelArtifactsForShape(shape VulkanModelShape) VulkanModelArtifacts {
	artifacts, _, _, _, _, _, _, _, _, _, _, _, _, _, _ := VulkanModelArtifactsForShapeInto(nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, shape)
	return artifacts
}

func VulkanModelArtifactsForShapeInto(planDst []VulkanPlan, stageDst []VulkanStageSummary, pipeDst []VulkanPipelinePlan, shaderDst []VulkanShaderModulePlan, layoutDst []VulkanPipelineLayoutPlan, cmdDst []VulkanCommand, batchDst []VulkanDispatchBatch, resourceDst []VulkanResource, descriptorDst []VulkanDescriptorWrite, recordDst []VulkanCommandRecord, barrierDst []VulkanBufferBarrier, allocationDst []VulkanBufferAllocation, uploadDst []VulkanBufferTransfer, readbackDst []VulkanBufferTransfer, shape VulkanModelShape) (VulkanModelArtifacts, []VulkanPlan, []VulkanStageSummary, []VulkanPipelinePlan, []VulkanShaderModulePlan, []VulkanPipelineLayoutPlan, []VulkanCommand, []VulkanDispatchBatch, []VulkanResource, []VulkanDescriptorWrite, []VulkanCommandRecord, []VulkanBufferBarrier, []VulkanBufferAllocation, []VulkanBufferTransfer, []VulkanBufferTransfer) {
	plans := VulkanModelPlansInto(planDst, shape)
	pipes := VulkanPipelinePlanFromPlans(pipeDst, plans)
	summary, graph, stages := vulkanModelSummaryGraphFromPlansPipelinesInto(stageDst, plans, pipes, shape)
	command, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks := VulkanCommandPlanFromPlansInto(shaderDst, layoutDst, cmdDst, batchDst, resourceDst, descriptorDst, recordDst, barrierDst, allocationDst, uploadDst, readbackDst, pipes, plans)
	return VulkanModelArtifacts{
		Plans:          plans,
		Summary:        summary,
		ExecutionGraph: graph,
		PipelinePlan:   pipes,
		CommandPlan:    command,
	}, plans, stages, pipes, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks
}

func VulkanPipelinePlanFromPlans(dst []VulkanPipelinePlan, plans []VulkanPlan) []VulkanPipelinePlan {
	out := dst[:0]
	if cap(out) < len(plans) {
		out = make([]VulkanPipelinePlan, 0, len(plans))
	}
	var layoutKeys [16]VulkanPipelineLayoutKey
	layoutCount := 0
	var shaderKeys [16]string
	shaderCount := 0
	for _, p := range plans {
		stage := p.Stage
		if stage == "" {
			stage = vulkanPlanStage(p.Name)
		}
		idx := -1
		for i := range out {
			if out[i].Key == p.PipelineKey && out[i].Stage == stage {
				idx = i
				break
			}
		}
		if idx < 0 {
			layoutIdx := vulkanPipelineLayoutIndex(&layoutKeys, &layoutCount, vulkanPipelineLayoutKey(p.PipelineKey))
			shaderIdx := vulkanShaderModuleIndex(&shaderKeys, &shaderCount, p.PipelineKey.KernelName)
			out = append(out, VulkanPipelinePlan{
				Key:               p.PipelineKey,
				Stage:             stage,
				LayoutIndex:       layoutIdx,
				ShaderModuleIndex: shaderIdx,
				CacheKeyHash:      vulkanPipelineCacheKeyHash(p.PipelineKey, layoutIdx, shaderIdx),
			})
			idx = len(out) - 1
		}
		dispatches := p.Dispatches
		if dispatches == 0 {
			repeat := p.Repeat
			if repeat < 1 {
				repeat = 1
			}
			dispatches = repeat * p.GroupsX * max(1, p.GroupsY)
		}
		out[idx].PlanRefs++
		out[idx].Dispatches += dispatches
	}
	return out
}

func VulkanShaderModulePlanFromPipelines(dst []VulkanShaderModulePlan, pipes []VulkanPipelinePlan) []VulkanShaderModulePlan {
	out := dst[:0]
	if cap(out) < len(pipes) {
		out = make([]VulkanShaderModulePlan, 0, len(pipes))
	}
	for _, p := range pipes {
		idx := -1
		for i := range out {
			if out[i].KernelName == p.Key.KernelName {
				idx = i
				break
			}
		}
		if idx < 0 {
			out = append(out, vulkanShaderModulePlan(p.Key))
			idx = len(out) - 1
		}
		out[idx].PipelineRefs++
	}
	return out
}

func VulkanPipelineLayoutPlanFromPipelines(dst []VulkanPipelineLayoutPlan, pipes []VulkanPipelinePlan) []VulkanPipelineLayoutPlan {
	out := dst[:0]
	if cap(out) < len(pipes) {
		out = make([]VulkanPipelineLayoutPlan, 0, len(pipes))
	}
	for _, p := range pipes {
		key := vulkanPipelineLayoutKey(p.Key)
		idx := -1
		for i := range out {
			if out[i].Key == key {
				idx = i
				break
			}
		}
		if idx < 0 {
			out = append(out, VulkanPipelineLayoutPlan{Key: key, Bindings: vulkanPipelineBindings(p.Key)})
			idx = len(out) - 1
		}
		out[idx].PipelineRefs++
	}
	return out
}

func vulkanPipelineLayoutKey(key VulkanPipelineKey) VulkanPipelineLayoutKey {
	return VulkanPipelineLayoutKey{
		PushConstantBytes: key.PushConstantBytes,
		BindingSignature:  key.BindingSignature,
	}
}

func vulkanPipelineLayoutIndex(keys *[16]VulkanPipelineLayoutKey, count *int, key VulkanPipelineLayoutKey) int {
	for i := 0; i < *count && i < len(keys); i++ {
		if keys[i] == key {
			return i
		}
	}
	idx := *count
	if idx < len(keys) {
		keys[idx] = key
	}
	*count++
	return idx
}

func vulkanShaderModuleIndex(keys *[16]string, count *int, kernelName string) int {
	for i := 0; i < *count && i < len(keys); i++ {
		if keys[i] == kernelName {
			return i
		}
	}
	idx := *count
	if idx < len(keys) {
		keys[idx] = kernelName
	}
	*count++
	return idx
}

func vulkanPipelineBindings(key VulkanPipelineKey) []VulkanBinding {
	k, ok := vulkanKernelByName(key.KernelName)
	if ok && k.BindingSignature == key.BindingSignature {
		return k.Bindings
	}
	return nil
}

func vulkanShaderModulePlan(key VulkanPipelineKey) VulkanShaderModulePlan {
	k, ok := vulkanKernelByName(key.KernelName)
	if !ok {
		return VulkanShaderModulePlan{KernelName: key.KernelName, EntryPoint: "main"}
	}
	return VulkanShaderModulePlan{
		KernelName:  k.Name,
		EntryPoint:  "main",
		SourceHash:  k.SourceHash,
		SourceBytes: k.SourceBytes,
		LocalSizeX:  k.Workgroup,
		TileCols:    k.TileCols,
		Specializations: [3]VulkanSpecializationConstant{
			{ID: 0, Name: "local_size_x", Value: k.Workgroup},
			{ID: 1, Name: "tile_cols", Value: k.TileCols},
			{ID: 2, Name: "quant_bits", Value: vulkanQuantBits(k.Quant)},
		},
	}
}

func vulkanQuantBits(q string) int {
	switch normalizeVulkanQuant(q) {
	case "q8":
		return 8
	case "q6":
		return 6
	case "q4":
		return 4
	default:
		return 32
	}
}

func vulkanKernelByName(name string) (VulkanKernel, bool) {
	switch name {
	case "matvec_f32_wg256":
		return vulkanKernelRegistry[0], true
	case "matvec_q8_wg256":
		return vulkanKernelRegistry[1], true
	case "matvec_q6_wg256":
		return vulkanKernelRegistry[2], true
	case "matvec_q4_wg256":
		return vulkanKernelRegistry[3], true
	case "fused_qkv_f32_wg256":
		return vulkanKernelRegistry[4], true
	case "fused_qkv_q8_wg256":
		return vulkanKernelRegistry[5], true
	case "fused_qkv_q6_wg256":
		return vulkanKernelRegistry[6], true
	case "fused_qkv_q4_wg256":
		return vulkanKernelRegistry[7], true
	case "fused_swiglu_f32_wg256":
		return vulkanKernelRegistry[8], true
	case "fused_swiglu_q8_wg256":
		return vulkanKernelRegistry[9], true
	case "fused_swiglu_q6_wg256":
		return vulkanKernelRegistry[10], true
	case "fused_swiglu_q4_wg256":
		return vulkanKernelRegistry[11], true
	}
	return VulkanKernel{}, false
}

func vulkanSourceHash(s string) uint64 {
	const (
		offset = uint64(14695981039346656037)
		prime  = uint64(1099511628211)
	)
	h := offset
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime
	}
	return h
}

func VulkanModelCommandPlan(shape VulkanModelShape) VulkanCommandPlan {
	plans := VulkanModelPlans(shape)
	pipes := VulkanPipelinePlanFromPlans(nil, plans)
	return VulkanCommandPlanFromPlans(nil, pipes, plans)
}

func VulkanModelCommandPlanInto(planDst []VulkanPlan, pipeDst []VulkanPipelinePlan, shaderDst []VulkanShaderModulePlan, layoutDst []VulkanPipelineLayoutPlan, cmdDst []VulkanCommand, batchDst []VulkanDispatchBatch, resourceDst []VulkanResource, descriptorDst []VulkanDescriptorWrite, recordDst []VulkanCommandRecord, barrierDst []VulkanBufferBarrier, allocationDst []VulkanBufferAllocation, uploadDst []VulkanBufferTransfer, readbackDst []VulkanBufferTransfer, shape VulkanModelShape) (VulkanCommandPlan, []VulkanPlan, []VulkanPipelinePlan, []VulkanShaderModulePlan, []VulkanPipelineLayoutPlan, []VulkanCommand, []VulkanDispatchBatch, []VulkanResource, []VulkanDescriptorWrite, []VulkanCommandRecord, []VulkanBufferBarrier, []VulkanBufferAllocation, []VulkanBufferTransfer, []VulkanBufferTransfer) {
	plans := VulkanModelPlansInto(planDst, shape)
	pipes := VulkanPipelinePlanFromPlans(pipeDst, plans)
	cmdPlan, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks := VulkanCommandPlanFromPlansInto(shaderDst, layoutDst, cmdDst, batchDst, resourceDst, descriptorDst, recordDst, barrierDst, allocationDst, uploadDst, readbackDst, pipes, plans)
	return cmdPlan, plans, pipes, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks
}

func VulkanCommandPlanFromPlans(cmdDst []VulkanCommand, pipes []VulkanPipelinePlan, plans []VulkanPlan) VulkanCommandPlan {
	plan, _, _, _, _, _, _, _, _, _, _, _ := VulkanCommandPlanFromPlansInto(nil, nil, cmdDst, nil, nil, nil, nil, nil, nil, nil, nil, pipes, plans)
	return plan
}

func VulkanCommandPlanFromPlansInto(shaderDst []VulkanShaderModulePlan, layoutDst []VulkanPipelineLayoutPlan, cmdDst []VulkanCommand, batchDst []VulkanDispatchBatch, resourceDst []VulkanResource, descriptorDst []VulkanDescriptorWrite, recordDst []VulkanCommandRecord, barrierDst []VulkanBufferBarrier, allocationDst []VulkanBufferAllocation, uploadDst []VulkanBufferTransfer, readbackDst []VulkanBufferTransfer, pipes []VulkanPipelinePlan, plans []VulkanPlan) (VulkanCommandPlan, []VulkanShaderModulePlan, []VulkanPipelineLayoutPlan, []VulkanCommand, []VulkanDispatchBatch, []VulkanResource, []VulkanDescriptorWrite, []VulkanCommandRecord, []VulkanBufferBarrier, []VulkanBufferAllocation, []VulkanBufferTransfer, []VulkanBufferTransfer) {
	cmds := cmdDst[:0]
	if cap(cmds) < len(plans) {
		cmds = make([]VulkanCommand, 0, len(plans))
	}
	batches := batchDst[:0]
	if cap(batches) < len(plans) {
		batches = make([]VulkanDispatchBatch, 0, len(plans))
	}
	records := recordDst[:0]
	if cap(records) < len(plans)*4 {
		records = make([]VulkanCommandRecord, 0, len(plans)*4)
	}
	resources := resourceDst[:0]
	f32Plans := 0
	for _, p := range plans {
		if p.Quant == "f32" {
			f32Plans++
		}
	}
	needResources := len(plans)*4 - f32Plans
	needUploads := len(plans)*3 - f32Plans
	if cap(resources) < needResources {
		resources = make([]VulkanResource, 0, needResources)
	}
	descriptors := descriptorDst[:0]
	if cap(descriptors) < needResources {
		descriptors = make([]VulkanDescriptorWrite, 0, needResources)
	}
	barriers := barrierDst[:0]
	if cap(barriers) < needResources {
		barriers = make([]VulkanBufferBarrier, 0, needResources)
	}
	allocations := allocationDst[:0]
	if cap(allocations) < needResources {
		allocations = make([]VulkanBufferAllocation, 0, needResources)
	}
	uploads := uploadDst[:0]
	if cap(uploads) < needUploads {
		uploads = make([]VulkanBufferTransfer, 0, needUploads)
	}
	readbacks := readbackDst[:0]
	if cap(readbacks) < len(plans) {
		readbacks = make([]VulkanBufferTransfer, 0, len(plans))
	}
	shaders := VulkanShaderModulePlanFromPipelines(shaderDst, pipes)
	layouts := VulkanPipelineLayoutPlanFromPipelines(layoutDst, pipes)
	var out VulkanCommandPlan
	out.Pipelines = pipes
	out.ShaderModules = shaders
	out.Layouts = layouts
	for _, p := range plans {
		stage := p.Stage
		if stage == "" {
			stage = vulkanPlanStage(p.Name)
		}
		pipeIdx := vulkanPipelineIndex(pipes, p.PipelineKey, stage)
		repeat := p.Repeat
		if repeat < 1 {
			repeat = 1
		}
		dispatches := p.Dispatches
		if dispatches == 0 {
			dispatches = repeat * p.GroupsX * max(1, p.GroupsY)
		}
		cmd := VulkanCommand{
			Stage:         stage,
			Name:          p.Name,
			PipelineIndex: pipeIdx,
			Rows:          p.Rows,
			Cols:          p.Cols,
			GroupsX:       p.GroupsX,
			GroupsY:       p.GroupsY,
			Repeat:        repeat,
			Dispatches:    dispatches,
		}
		start := len(resources)
		resources = vulkanCommandResourcesInto(resources, p)
		cmd.Resources = resources[start:len(resources)]
		descriptorStart := len(descriptors)
		var totalBufferBytes, uploadBytes, readbackBytes int64
		descriptors, barriers, allocations, uploads, readbacks, totalBufferBytes, uploadBytes, readbackBytes = vulkanCommandResourceArtifactsInto(descriptors, barriers, allocations, uploads, readbacks, len(cmds), cmd.Resources)
		cmd.DescriptorWrites = descriptors[descriptorStart:len(descriptors)]
		layoutIdx := -1
		if pipeIdx >= 0 && pipeIdx < len(pipes) {
			layoutIdx = pipes[pipeIdx].LayoutIndex
		}
		cmd.DescriptorSet = VulkanDescriptorSetPlan{
			SetIndex:         len(cmds),
			LayoutIndex:      layoutIdx,
			DescriptorOffset: descriptorStart,
			DescriptorCount:  len(cmd.DescriptorWrites),
		}
		cmd.PushConstants = VulkanPushConstants{Rows: uint32(p.Rows), Cols: uint32(p.Cols), Bytes: p.Kernel.PushConstantBytes}
		cmds = append(cmds, cmd)
		commandIndex := len(cmds) - 1
		records = vulkanCommandRecordsInto(records, commandIndex, cmd)
		out.TotalBufferBytes += totalBufferBytes
		out.UploadBytes += uploadBytes
		out.ReadbackBytes += readbackBytes
		out.Dispatches += dispatches
	}
	out.Commands = cmds
	batches = vulkanDispatchBatchesInto(batches, cmds)
	out.DispatchBatches = batches
	out.Records = records
	out.Barriers = barriers
	out.Allocations = allocations
	out.Uploads = uploads
	out.Readbacks = readbacks
	out.PipelineCount = len(pipes)
	out.ShaderModuleCount = len(shaders)
	out.LayoutCount = len(layouts)
	out.CommandCount = len(cmds)
	out.DispatchBatchCount = len(batches)
	out.RecordCount = len(records)
	out.BarrierCount = len(barriers)
	out.AllocationCount = len(allocations)
	out.UploadCount = len(uploads)
	out.ReadbackCount = len(readbacks)
	out.ResourceCount = len(resources)
	out.DescriptorWriteCount = len(descriptors)
	out.PipelineBindCount = len(batches)
	out.DescriptorBindCount = len(cmds)
	out.PushConstantCount = len(cmds)
	out.DescriptorPool = vulkanDescriptorPoolPlan(out.CommandCount, out.DescriptorWriteCount)
	out.CommandPool = vulkanCommandPoolPlan(out.CommandCount)
	out.QueueSubmit = vulkanQueueSubmitPlan(out.CommandPool.CommandBufferCount)
	out.Timeline = vulkanTimelinePlan(out.QueueSubmit.SubmitCount)
	out.Fence = vulkanFencePlan(out.QueueSubmit.SubmitCount)
	out.PipelineCache = vulkanPipelineCachePlan(pipes)
	out.PipelineLifecycle = vulkanPipelineLifecyclePlan(out.PipelineCount)
	return out, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks
}

func vulkanCommandResourcesInto(dst []VulkanResource, p VulkanPlan) []VulkanResource {
	bindings := p.Kernel.Bindings
	if len(bindings) == 0 {
		return dst
	}
	start := len(dst)
	dst = dst[:start+len(bindings)]
	if len(bindings) == 3 {
		dst[start] = vulkanResourceFromBinding(bindings[0], p.InputBytes)
		dst[start+1] = vulkanResourceFromBinding(bindings[1], p.WeightByte)
		dst[start+2] = vulkanResourceFromBinding(bindings[2], p.OutputBytes)
		return dst
	}
	if len(bindings) == 4 {
		dst[start] = vulkanResourceFromBinding(bindings[0], p.InputBytes)
		dst[start+1] = vulkanResourceFromBinding(bindings[1], p.WeightByte)
		dst[start+2] = vulkanResourceFromBinding(bindings[2], int64(p.Rows)*4)
		dst[start+3] = vulkanResourceFromBinding(bindings[3], p.OutputBytes)
		return dst
	}
	for i, b := range bindings {
		dst[start+i] = vulkanResourceFromBinding(b, vulkanResourceBytes(p, b.Role))
	}
	return dst
}

func vulkanDispatchBatchesInto(dst []VulkanDispatchBatch, cmds []VulkanCommand) []VulkanDispatchBatch {
	out := dst[:0]
	for i, cmd := range cmds {
		layoutIdx := cmd.DescriptorSet.LayoutIndex
		if len(out) > 0 {
			last := &out[len(out)-1]
			if last.PipelineIndex == cmd.PipelineIndex && last.LayoutIndex == layoutIdx && last.CommandStart+last.CommandCount == i {
				last.CommandCount++
				last.DescriptorSetCount++
				last.Dispatches += cmd.Dispatches
				last.DescriptorBindCount++
				last.PushConstantCount++
				continue
			}
		}
		out = append(out, VulkanDispatchBatch{
			BatchIndex:          len(out),
			Stage:               cmd.Stage,
			PipelineIndex:       cmd.PipelineIndex,
			LayoutIndex:         layoutIdx,
			CommandStart:        i,
			CommandCount:        1,
			DescriptorSetStart:  cmd.DescriptorSet.SetIndex,
			DescriptorSetCount:  1,
			Dispatches:          cmd.Dispatches,
			PipelineBindCount:   1,
			DescriptorBindCount: 1,
			PushConstantCount:   1,
		})
	}
	return out
}

func vulkanResourceFromBinding(b VulkanBinding, bytes int64) VulkanResource {
	return VulkanResource{
		Binding: b.Binding,
		Name:    b.Name,
		Role:    b.Role,
		Bytes:   bytes,
		Access:  b.Access,
	}
}

func vulkanCommandResourceArtifactsInto(descriptorDst []VulkanDescriptorWrite, barrierDst []VulkanBufferBarrier, allocationDst []VulkanBufferAllocation, uploadDst []VulkanBufferTransfer, readbackDst []VulkanBufferTransfer, commandIndex int, resources []VulkanResource) ([]VulkanDescriptorWrite, []VulkanBufferBarrier, []VulkanBufferAllocation, []VulkanBufferTransfer, []VulkanBufferTransfer, int64, int64, int64) {
	descriptorStart := len(descriptorDst)
	barrierStart := len(barrierDst)
	allocationStart := len(allocationDst)
	descriptorDst = descriptorDst[:descriptorStart+len(resources)]
	barrierDst = barrierDst[:barrierStart+len(resources)]
	allocationDst = allocationDst[:allocationStart+len(resources)]
	standardBindings := len(resources) == 3 || len(resources) == 4
	lastResource := len(resources) - 1
	var totalBufferBytes, uploadBytes, readbackBytes int64
	for i, r := range resources {
		descriptorDst[descriptorStart+i] = VulkanDescriptorWrite{
			Binding:        r.Binding,
			DescriptorType: "storage_buffer",
			ResourceName:   r.Name,
			OffsetBytes:    0,
			RangeBytes:     r.Bytes,
			Access:         r.Access,
		}
		b := VulkanBufferBarrier{
			CommandIndex: commandIndex,
			Binding:      r.Binding,
			ResourceName: r.Name,
			Role:         r.Role,
			Bytes:        r.Bytes,
		}
		isOutput := r.Role == "output"
		alignmentBytes := vulkanStorageBufferAlignment(r)
		if standardBindings {
			isOutput = i == lastResource
			if i == 1 {
				alignmentBytes = 256
			} else {
				alignmentBytes = 64
			}
		}
		if isOutput {
			b.SrcStage = "compute_shader"
			b.DstStage = "compute_shader"
			b.SrcAccess = "shader_write"
			b.DstAccess = "shader_read"
		} else {
			b.SrcStage = "host"
			b.DstStage = "compute_shader"
			b.SrcAccess = "host_write"
			b.DstAccess = "shader_read"
		}
		barrierDst[barrierStart+i] = b
		a := VulkanBufferAllocation{
			CommandIndex:   commandIndex,
			Binding:        r.Binding,
			ResourceName:   r.Name,
			Role:           r.Role,
			Bytes:          r.Bytes,
			AlignmentBytes: alignmentBytes,
		}
		a.AlignedBytes = alignInt64(r.Bytes, a.AlignmentBytes)
		if isOutput {
			a.Usage = "storage_buffer|transfer_src"
			a.MemoryProperties = "device_local"
			readbackDst = append(readbackDst, VulkanBufferTransfer{
				CommandIndex: commandIndex,
				Binding:      r.Binding,
				ResourceName: r.Name,
				Role:         r.Role,
				Direction:    "device_to_host",
				Bytes:        r.Bytes,
			})
			readbackBytes += r.Bytes
		} else {
			a.Usage = "storage_buffer|transfer_dst"
			a.MemoryProperties = "host_visible|device_local"
			uploadDst = append(uploadDst, VulkanBufferTransfer{
				CommandIndex: commandIndex,
				Binding:      r.Binding,
				ResourceName: r.Name,
				Role:         r.Role,
				Direction:    "host_to_device",
				Bytes:        r.Bytes,
			})
			uploadBytes += r.Bytes
		}
		allocationDst[allocationStart+i] = a
		totalBufferBytes += a.AlignedBytes
	}
	return descriptorDst, barrierDst, allocationDst, uploadDst, readbackDst, totalBufferBytes, uploadBytes, readbackBytes
}

func vulkanCommandRecordsInto(dst []VulkanCommandRecord, commandIndex int, cmd VulkanCommand) []VulkanCommandRecord {
	layoutIdx := cmd.DescriptorSet.LayoutIndex
	start := len(dst)
	dst = dst[:start+4]
	dst[start] = VulkanCommandRecord{
		Op:            "bind_pipeline",
		CommandIndex:  commandIndex,
		PipelineIndex: cmd.PipelineIndex,
		LayoutIndex:   layoutIdx,
	}
	dst[start+1] = VulkanCommandRecord{
		Op:                   "bind_descriptor_set",
		CommandIndex:         commandIndex,
		PipelineIndex:        cmd.PipelineIndex,
		LayoutIndex:          layoutIdx,
		DescriptorWriteStart: cmd.DescriptorSet.DescriptorOffset,
		DescriptorWriteCount: cmd.DescriptorSet.DescriptorCount,
	}
	dst[start+2] = VulkanCommandRecord{
		Op:                "push_constants",
		CommandIndex:      commandIndex,
		LayoutIndex:       layoutIdx,
		PushConstantBytes: cmd.PushConstants.Bytes,
		Rows:              int(cmd.PushConstants.Rows),
		Cols:              int(cmd.PushConstants.Cols),
	}
	dst[start+3] = VulkanCommandRecord{
		Op:           "dispatch",
		CommandIndex: commandIndex,
		GroupsX:      cmd.GroupsX,
		GroupsY:      cmd.GroupsY,
		Repeat:       cmd.Repeat,
	}
	return dst
}

func vulkanStorageBufferAlignment(r VulkanResource) int64 {
	switch r.Role {
	case "weight":
		return 256
	case "scale", "input", "output":
		return 64
	default:
		return 16
	}
}

func alignInt64(v, alignment int64) int64 {
	if alignment <= 1 || v <= 0 {
		return v
	}
	rem := v % alignment
	if rem == 0 {
		return v
	}
	return v + alignment - rem
}

func vulkanDescriptorPoolPlan(commandCount, descriptorWriteCount int) VulkanDescriptorPoolPlan {
	return VulkanDescriptorPoolPlan{
		MaxSets:            commandCount,
		StorageBufferCount: descriptorWriteCount,
	}
}

func vulkanCommandPoolPlan(commandCount int) VulkanCommandPoolPlan {
	bufferCount := 0
	if commandCount > 0 {
		bufferCount = 1
	}
	return VulkanCommandPoolPlan{
		QueueFamily:        "compute",
		CommandBufferCount: bufferCount,
		ResetMode:          "reset_command_buffer",
	}
}

func vulkanQueueSubmitPlan(commandBufferCount int) VulkanQueueSubmitPlan {
	submitCount := 0
	if commandBufferCount > 0 {
		submitCount = 1
	}
	return VulkanQueueSubmitPlan{
		QueueFamily:        "compute",
		SubmitCount:        submitCount,
		CommandBufferCount: commandBufferCount,
		WaitSemaphoreCount: 0,
		SignalFence:        submitCount > 0,
	}
}

func vulkanTimelinePlan(submitCount int) VulkanTimelinePlan {
	if submitCount <= 0 {
		return VulkanTimelinePlan{StageMask: "compute_shader"}
	}
	return VulkanTimelinePlan{
		InitialValue: 0,
		WaitValue:    0,
		SignalValue:  int64(submitCount),
		StageMask:    "compute_shader",
	}
}

func vulkanFencePlan(submitCount int) VulkanFencePlan {
	return VulkanFencePlan{
		CreateSignaled:    false,
		ResetBeforeSubmit: submitCount > 0,
		WaitAfterSubmit:   submitCount > 0,
	}
}

func vulkanPipelineCachePlan(pipes []VulkanPipelinePlan) VulkanPipelineCachePlan {
	var refs int
	var hash uint64 = 14695981039346656037
	for _, p := range pipes {
		refs += p.PlanRefs
		hash ^= p.CacheKeyHash + 0x9e3779b97f4a7c15 + (hash << 6) + (hash >> 2)
	}
	return VulkanPipelineCachePlan{
		CacheKeyHash: hash,
		EntryCount:   len(pipes),
		CreateCount:  len(pipes),
		ReuseCount:   max(0, refs-len(pipes)),
		LayoutRefs:   refs,
		ShaderRefs:   refs,
		Persistable:  len(pipes) > 0,
	}
}

func vulkanPipelineCacheKeyHash(key VulkanPipelineKey, layoutIdx, shaderIdx int) uint64 {
	hash := key.BindingSignature
	hash ^= uint64(key.PushConstantBytes) * 0x9e3779b185ebca87
	hash ^= uint64(key.Workgroup) * 0xc2b2ae3d27d4eb4f
	hash ^= uint64(layoutIdx+1) * 0x165667b19e3779f9
	hash ^= uint64(shaderIdx+1) * 0x85ebca77c2b2ae63
	hash ^= hash >> 33
	hash *= 0xff51afd7ed558ccd
	hash ^= hash >> 33
	hash *= 0xc4ceb9fe1a85ec53
	hash ^= hash >> 33
	return hash
}

func vulkanPipelineLifecyclePlan(pipelineCount int) VulkanPipelineLifecyclePlan {
	return VulkanPipelineLifecyclePlan{
		Steps:        [4]string{"load_pipeline_cache", "create_compute_pipelines", "record_command_buffer", "destroy_pipelines_after_cache_save"},
		CreateCount:  pipelineCount,
		DestroyCount: pipelineCount,
	}
}

func vulkanMixUint64(hash, v uint64) uint64 {
	const prime = uint64(1099511628211)
	for i := 0; i < 8; i++ {
		hash ^= uint64(byte(v))
		hash *= prime
		v >>= 8
	}
	return hash
}

func vulkanResourceBytes(p VulkanPlan, role string) int64 {
	switch role {
	case "input":
		return p.InputBytes
	case "weight":
		return p.WeightByte
	case "scale":
		return int64(p.Rows) * 4
	case "output":
		return p.OutputBytes
	default:
		return 0
	}
}

func vulkanPipelineIndex(pipes []VulkanPipelinePlan, key VulkanPipelineKey, stage string) int {
	if len(pipes) == 6 {
		idx := -1
		switch stage {
		case "text":
			switch key.KernelName {
			case "fused_qkv_f32_wg256", "fused_qkv_q8_wg256", "fused_qkv_q6_wg256", "fused_qkv_q4_wg256":
				idx = 0
			case "matvec_f32_wg256", "matvec_q8_wg256", "matvec_q6_wg256", "matvec_q4_wg256":
				idx = 1
			case "fused_swiglu_f32_wg256", "fused_swiglu_q8_wg256", "fused_swiglu_q6_wg256", "fused_swiglu_q4_wg256":
				idx = 2
			}
		case "vision":
			switch key.KernelName {
			case "matvec_f32_wg256":
				idx = 3
			case "fused_qkv_f32_wg256":
				idx = 4
			case "fused_swiglu_f32_wg256":
				idx = 5
			}
		}
		if idx >= 0 && pipes[idx].Key == key && pipes[idx].Stage == stage {
			return idx
		}
	}
	for i := range pipes {
		if pipes[i].Key == key && pipes[i].Stage == stage {
			return i
		}
	}
	return -1
}

func VulkanModelExecutionGraphFromPlans(plans []VulkanPlan) VulkanExecutionGraph {
	graph, _ := VulkanModelExecutionGraphFromPlansInto(nil, plans)
	return graph
}

func VulkanModelExecutionGraphFromPlansInto(stageDst []VulkanStageSummary, plans []VulkanPlan) (VulkanExecutionGraph, []VulkanStageSummary) {
	var graph VulkanExecutionGraph
	stages := stageDst[:0]
	if cap(stages) < 2 {
		stages = make([]VulkanStageSummary, 0, 2)
	}
	textIdx, visionIdx := -1, -1
	var allKeys, textKeys, visionKeys [16]VulkanPipelineKey
	allPipelineCount, textPipelineCount, visionPipelineCount := 0, 0, 0
	hasOtherStage := false
	for _, p := range plans {
		stageName := p.Stage
		if stageName == "" {
			stageName = vulkanPlanStage(p.Name)
		}
		addUniqueVulkanPipelineKey(&allKeys, &allPipelineCount, p.PipelineKey)
		idx := -1
		switch stageName {
		case "text":
			addUniqueVulkanPipelineKey(&textKeys, &textPipelineCount, p.PipelineKey)
			idx = textIdx
			if idx < 0 {
				stages = append(stages, VulkanStageSummary{Name: stageName})
				idx = len(stages) - 1
				textIdx = idx
			}
		case "vision":
			addUniqueVulkanPipelineKey(&visionKeys, &visionPipelineCount, p.PipelineKey)
			idx = visionIdx
			if idx < 0 {
				stages = append(stages, VulkanStageSummary{Name: stageName})
				idx = len(stages) - 1
				visionIdx = idx
			}
		default:
			hasOtherStage = true
			stages = append(stages, VulkanStageSummary{Name: stageName})
			idx = len(stages) - 1
		}
		addVulkanPlanToStage(&stages[idx], p)
	}
	graph.Stages = stages
	graph.PipelineCount = allPipelineCount
	for i := range stages {
		switch stages[i].Name {
		case "text":
			stages[i].PipelineCount = textPipelineCount
		case "vision":
			stages[i].PipelineCount = visionPipelineCount
		default:
			if hasOtherStage {
				stages[i].PipelineCount = countUniqueVulkanPipelineKeys(plans, stages[i].Name)
			}
		}
	}
	for _, st := range stages {
		graph.PlanCount += st.PlanCount
		graph.Dispatches += st.Dispatches
		graph.WeightBytes += st.WeightBytes
		graph.InputBytes += st.InputBytes
		graph.OutputBytes += st.OutputBytes
		graph.SharedBytes += st.SharedBytes
	}
	return graph, stages
}

func addUniqueVulkanPipelineKey(keys *[16]VulkanPipelineKey, count *int, key VulkanPipelineKey) {
	for i := 0; i < *count; i++ {
		if keys[i] == key {
			return
		}
	}
	if *count < len(keys) {
		keys[*count] = key
	}
	*count++
}

func VulkanModelPlanSummaryFromPlans(plans []VulkanPlan, shape VulkanModelShape) VulkanPlanSummary {
	var st VulkanPlanSummary
	st.Plans = plans
	st.PlanCount = len(plans)
	st.TextLayers = shape.TextLayers
	st.VisionLayers = shape.VisionLayers
	var keys [16]VulkanPipelineKey
	pipelineCount := 0
	for _, p := range plans {
		addUniqueVulkanPipelineKey(&keys, &pipelineCount, p.PipelineKey)
		repeat := p.Repeat
		if repeat < 1 {
			repeat = 1
		}
		dispatches := p.Dispatches
		if dispatches == 0 {
			dispatches = repeat * p.GroupsX * max(1, p.GroupsY)
		}
		st.Dispatches += dispatches
		st.WeightBytes += p.WeightByte * int64(repeat)
		st.InputBytes += p.InputBytes * int64(repeat)
		st.OutputBytes += p.OutputBytes * int64(repeat)
		st.SharedBytes += int64(p.SharedByte) * int64(dispatches)
	}
	st.PipelineCount = pipelineCount
	return st
}

func vulkanModelSummaryGraphFromPlansPipelinesInto(stageDst []VulkanStageSummary, plans []VulkanPlan, pipes []VulkanPipelinePlan, shape VulkanModelShape) (VulkanPlanSummary, VulkanExecutionGraph, []VulkanStageSummary) {
	var summary VulkanPlanSummary
	var graph VulkanExecutionGraph
	stages := stageDst[:0]
	if cap(stages) < 2 {
		stages = make([]VulkanStageSummary, 0, 2)
	}
	textIdx, visionIdx := -1, -1
	for _, p := range plans {
		stageName := p.Stage
		if stageName == "" {
			stageName = vulkanPlanStage(p.Name)
		}
		idx := -1
		switch stageName {
		case "text":
			idx = textIdx
			if idx < 0 {
				stages = append(stages, VulkanStageSummary{Name: stageName})
				idx = len(stages) - 1
				textIdx = idx
			}
		case "vision":
			idx = visionIdx
			if idx < 0 {
				stages = append(stages, VulkanStageSummary{Name: stageName})
				idx = len(stages) - 1
				visionIdx = idx
			}
		default:
			stages = append(stages, VulkanStageSummary{Name: stageName})
			idx = len(stages) - 1
		}
		addVulkanPlanToStage(&stages[idx], p)
	}
	for _, p := range pipes {
		for i := range stages {
			if stages[i].Name == p.Stage {
				stages[i].PipelineCount++
				break
			}
		}
	}
	for _, st := range stages {
		graph.PlanCount += st.PlanCount
		graph.Dispatches += st.Dispatches
		graph.WeightBytes += st.WeightBytes
		graph.InputBytes += st.InputBytes
		graph.OutputBytes += st.OutputBytes
		graph.SharedBytes += st.SharedBytes
	}
	graph.Stages = stages
	graph.PipelineCount = len(pipes)
	summary.Plans = plans
	summary.PlanCount = len(plans)
	summary.PipelineCount = len(pipes)
	summary.Dispatches = graph.Dispatches
	summary.WeightBytes = graph.WeightBytes
	summary.InputBytes = graph.InputBytes
	summary.OutputBytes = graph.OutputBytes
	summary.SharedBytes = graph.SharedBytes
	summary.TextLayers = shape.TextLayers
	summary.VisionLayers = shape.VisionLayers
	return summary, graph, stages
}

func countUniqueVulkanPipelineKeys(plans []VulkanPlan, stage string) int {
	var keys [16]VulkanPipelineKey
	count := 0
	for _, p := range plans {
		pStage := p.Stage
		if pStage == "" {
			pStage = vulkanPlanStage(p.Name)
		}
		if stage != "" && pStage != stage {
			continue
		}
		seen := false
		for i := 0; i < count && i < len(keys); i++ {
			if keys[i] == p.PipelineKey {
				seen = true
				break
			}
		}
		if !seen {
			if count < len(keys) {
				keys[count] = p.PipelineKey
			}
			count++
		}
	}
	return count
}

func addVulkanPlanToStage(st *VulkanStageSummary, p VulkanPlan) {
	repeat := p.Repeat
	if repeat < 1 {
		repeat = 1
	}
	dispatches := p.Dispatches
	if dispatches == 0 {
		dispatches = repeat * p.GroupsX * max(1, p.GroupsY)
	}
	st.PlanCount++
	st.Dispatches += dispatches
	st.WeightBytes += p.WeightByte * int64(repeat)
	st.InputBytes += p.InputBytes * int64(repeat)
	st.OutputBytes += p.OutputBytes * int64(repeat)
	st.SharedBytes += int64(p.SharedByte) * int64(dispatches)
}

func vulkanPlanStage(name string) string {
	for i := 0; i < len(name); i++ {
		if name[i] == '.' {
			if i == 0 {
				return "unknown"
			}
			return name[:i]
		}
	}
	if name == "" {
		return "unknown"
	}
	return name
}

func normalizeVulkanQuant(q string) string {
	q = lowerASCII(trimASCIIWhitespace(q))
	switch q {
	case "", "none", "float32":
		return "f32"
	case "q8_0":
		return "q8"
	case "q6_k":
		return "q6"
	case "q4_0", "q4_k":
		return "q4"
	default:
		return q
	}
}

func ceilDiv(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func estimateVulkanWeightBytes(quant string, rows, cols int) int64 {
	n, _ := estimateVulkanWeightBytesChecked(quant, rows, cols)
	return n
}

func estimateVulkanWeightBytesChecked(quant string, rows, cols int) (int64, bool) {
	return estimateVulkanWeightBytesCheckedNormalized(normalizeVulkanQuant(quant), rows, cols)
}

func estimateVulkanWeightBytesCheckedNormalized(quant string, rows, cols int) (int64, bool) {
	if rows <= 0 || cols <= 0 {
		return 0, true
	}
	elements, ok := checkedMulInt64(int64(rows), int64(cols))
	if !ok {
		return 0, false
	}
	scale, ok := checkedMulInt64(int64(rows), 4)
	if !ok {
		return 0, false
	}
	switch quant {
	case "q8":
		return checkedAddInt64(elements, scale)
	case "q6":
		bits, ok := checkedMulInt64(elements, 6)
		if !ok {
			return 0, false
		}
		bits, ok = checkedAddInt64(bits, 7)
		if !ok {
			return 0, false
		}
		return checkedAddInt64(bits/8, scale)
	case "q4":
		packed, ok := checkedAddInt64(elements, 1)
		if !ok {
			return 0, false
		}
		return checkedAddInt64(packed/2, scale)
	default:
		return checkedMulInt64(elements, 4)
	}
}

func checkedDispatches(repeat, groupsX, groupsY int) (int, bool) {
	n, ok := checkedMulInt(repeat, groupsX)
	if !ok {
		return 0, false
	}
	return checkedMulInt(n, groupsY)
}

func checkedMulInt(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > maxInt()/a {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt(a, b int) (int, bool) {
	if a < 0 || b < 0 || a > maxInt()-b {
		return 0, false
	}
	return a + b, true
}

func checkedMulInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > maxInt64()/a {
		return 0, false
	}
	return a * b, true
}

func checkedAddInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || a > maxInt64()-b {
		return 0, false
	}
	return a + b, true
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

func maxInt64() int64 {
	return int64(^uint64(0) >> 1)
}

func maxUint32Int() int {
	if ^uint(0)>>32 == 0 {
		return maxInt()
	}
	return int(^uint32(0))
}

const vulkanMatVecF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[row * pc.cols + c] * x[c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  if (lid == 0) outv[row] = scratch[0];
}`

const vulkanMatVecQ8GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { int w[]; };
layout(set=0,binding=2) readonly buffer S { float scale[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) sum += float(w[row * pc.cols + c]) * x[c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanMatVecQ6GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { uint packed[]; };
layout(set=0,binding=2) readonly buffer S { float scale[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q6(uint bit) {
  uint word = packed[bit >> 5];
  uint shift = bit & 31u;
  uint v = (word >> shift) & 63u;
  return float(int(v) - 32);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint rowBits = row * pc.cols * 6u;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) sum += q6(rowBits + c * 6u) * x[c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanMatVecQ4GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { uint packed[]; };
layout(set=0,binding=2) readonly buffer S { float scale[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q4(uint idx) {
  uint word = packed[idx >> 3];
  uint shift = (idx & 7u) * 4u;
  uint v = (word >> shift) & 15u;
  return float(int(v) - 8);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint rowBase = row * pc.cols;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) sum += q4(rowBase + c) * x[c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanFusedQKVF32GLSL = vulkanMatVecF32GLSL
const vulkanFusedQKVQ8GLSL = vulkanMatVecQ8GLSL
const vulkanFusedQKVQ6GLSL = vulkanMatVecQ6GLSL
const vulkanFusedQKVQ4GLSL = vulkanMatVecQ4GLSL
const vulkanFusedSwiGLUF32GLSL = vulkanMatVecF32GLSL
const vulkanFusedSwiGLUQ8GLSL = vulkanMatVecQ8GLSL
const vulkanFusedSwiGLUQ6GLSL = vulkanMatVecQ6GLSL
const vulkanFusedSwiGLUQ4GLSL = vulkanMatVecQ4GLSL
