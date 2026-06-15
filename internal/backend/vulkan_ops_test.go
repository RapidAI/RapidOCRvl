package backend

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDefaultVulkanComputeRegistersOptimizedKernels(t *testing.T) {
	c := DefaultVulkanCompute()
	if c.Version == 0 || c.Workgroup != 256 || c.VectorWidth != 4 {
		t.Fatalf("compute=%+v", c)
	}
	if len(c.Kernels) < 8 {
		t.Fatalf("kernels=%+v", c.Kernels)
	}
	want := map[string]bool{
		VulkanOpMatVec + "/f32":      false,
		VulkanOpMatVec + "/q8":       false,
		VulkanOpMatVec + "/q6":       false,
		VulkanOpMatVec + "/q4":       false,
		VulkanOpFusedQKV + "/f32":    false,
		VulkanOpFusedQKV + "/q8":     false,
		VulkanOpFusedQKV + "/q6":     false,
		VulkanOpFusedQKV + "/q4":     false,
		VulkanOpFusedSwiGLU + "/f32": false,
		VulkanOpFusedSwiGLU + "/q8":  false,
		VulkanOpFusedSwiGLU + "/q6":  false,
		VulkanOpFusedSwiGLU + "/q4":  false,
	}
	for _, k := range c.Kernels {
		if k.Source == "" {
			t.Fatalf("kernel missing source: %+v", k)
		}
		if reason := ValidateVulkanKernelABI(k); reason != "" {
			t.Fatalf("invalid kernel ABI %s: %s", k.Name, reason)
		}
		key := k.Op + "/" + k.Quant
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, ok := range want {
		if !ok {
			t.Fatalf("missing kernel %s", key)
		}
	}
}

func TestPlanVulkanKernel(t *testing.T) {
	plan, ok := PlanVulkanKernel(" matvec ", "Q4_0", 513, 1024)
	if !ok {
		t.Fatal("expected q4 plan")
	}
	if plan.Kernel.Name != "matvec_q4_wg256" || plan.GroupsX != 513 || plan.GroupsY != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.SharedByte != 1024 {
		t.Fatalf("shared bytes=%d", plan.SharedByte)
	}
	if plan.InputBytes != 1024*4 || plan.OutputBytes != 513*4 {
		t.Fatalf("io bytes in=%d out=%d", plan.InputBytes, plan.OutputBytes)
	}
	if plan.Kernel.PushConstantBytes != 8 || len(plan.Kernel.Bindings) != 4 {
		t.Fatalf("kernel ABI=%+v", plan.Kernel)
	}
	if plan.Kernel.Bindings[1].Elem != "packed_q4" || plan.Kernel.Bindings[3].Role != "output" {
		t.Fatalf("bindings=%+v", plan.Kernel.Bindings)
	}
	if plan.PipelineKey.KernelName != "matvec_q4_wg256" || plan.PipelineKey.BindingSignature != plan.Kernel.BindingSignature {
		t.Fatalf("pipeline key=%+v kernel=%+v", plan.PipelineKey, plan.Kernel)
	}
	wantBytes := int64((513*1024+1)/2 + 513*4)
	if plan.WeightByte != wantBytes {
		t.Fatalf("weight bytes=%d want %d", plan.WeightByte, wantBytes)
	}
}

func TestVulkanPipelineKeyReuse(t *testing.T) {
	a, ok := PlanVulkanKernel(VulkanOpMatVec, "q4", 512, 1024)
	if !ok {
		t.Fatal("missing q4 plan")
	}
	b, ok := PlanVulkanKernel(VulkanOpMatVec, "q4", 4096, 2048)
	if !ok {
		t.Fatal("missing q4 plan")
	}
	if a.PipelineKey != b.PipelineKey {
		t.Fatalf("same kernel should reuse pipeline key: %+v != %+v", a.PipelineKey, b.PipelineKey)
	}
	c, ok := PlanVulkanKernel(VulkanOpMatVec, "q8", 4096, 2048)
	if !ok {
		t.Fatal("missing q8 plan")
	}
	if a.PipelineKey == c.PipelineKey {
		t.Fatalf("different quant should not share pipeline key: %+v", a.PipelineKey)
	}
}

func TestVulkanMixUint64UsesAllBytes(t *testing.T) {
	const offset = uint64(14695981039346656037)
	low := vulkanMixUint64(offset, 0x00000000000000ff)
	high := vulkanMixUint64(offset, 0xff00000000000000)
	if low == 0 || high == 0 || low == high {
		t.Fatalf("hash low=%d high=%d", low, high)
	}
	if got := vulkanMixUint64(offset, 0xff00000000000000); got != high {
		t.Fatalf("hash not deterministic: got=%d want=%d", got, high)
	}
}

func TestValidateVulkanKernelABIRejectsBrokenBinding(t *testing.T) {
	k := vulkanKernel("bad", VulkanOpMatVec, "q4", vulkanMatVecQ4GLSL)
	k.Bindings[1].Binding = 3
	if reason := ValidateVulkanKernelABI(k); reason == "" {
		t.Fatal("expected broken binding rejection")
	}
}

func TestPlanVulkanKernelRejectsInvalidShape(t *testing.T) {
	if _, ok := PlanVulkanKernel(VulkanOpMatVec, "q8", 0, 1024); ok {
		t.Fatal("expected invalid shape rejection")
	}
	if _, ok := PlanVulkanKernel("unknown", "q8", 1, 1024); ok {
		t.Fatal("expected unknown op rejection")
	}
}

func TestVulkanComputeJSONOmitsShaderSources(t *testing.T) {
	raw, err := json.Marshal(DefaultVulkanCompute())
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	if strings.Contains(body, "#version 450") || strings.Contains(body, "layout(") {
		t.Fatalf("shader source leaked into JSON: %s", body)
	}
	if !strings.Contains(body, "matvec_q8_wg256") {
		t.Fatalf("missing kernel metadata: %s", body)
	}
}

func TestVulkanModelPlans(t *testing.T) {
	plans := VulkanModelPlans(VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	})
	if len(plans) != 10 {
		t.Fatalf("plans=%+v", plans)
	}
	seen := map[string]VulkanPlan{}
	for _, p := range plans {
		seen[p.Name] = p
		if p.GroupsX <= 0 || p.SharedByte != 1024 {
			t.Fatalf("bad plan %+v", p)
		}
	}
	if seen["text.qkv"].Rows != 3072 || seen["text.qkv"].Cols != 2048 || seen["text.qkv"].Quant != "q4" {
		t.Fatalf("text.qkv=%+v", seen["text.qkv"])
	}
	if seen["text.qkv"].Repeat != 24 || seen["text.qkv"].Dispatches != 3072*24 {
		t.Fatalf("text.qkv dispatch=%+v", seen["text.qkv"])
	}
	if seen["vision.patch"].Quant != "f32" || seen["vision.patch"].WeightByte != int64(1024*588*4) {
		t.Fatalf("vision.patch=%+v", seen["vision.patch"])
	}
}

func TestVulkanModelPlanSummary(t *testing.T) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	st := VulkanModelPlanSummary(shape)
	if st.PlanCount != 10 || len(st.Plans) != 10 {
		t.Fatalf("summary=%+v", st)
	}
	if st.PipelineCount != 6 {
		t.Fatalf("pipeline count=%d want 6", st.PipelineCount)
	}
	if st.TextLayers != 24 || st.VisionLayers != 24 {
		t.Fatalf("layers text=%d vision=%d", st.TextLayers, st.VisionLayers)
	}
	if st.Dispatches <= 0 || st.WeightBytes <= 0 || st.InputBytes <= 0 || st.OutputBytes <= 0 || st.SharedBytes <= 0 {
		t.Fatalf("summary=%+v", st)
	}
	if st.DispatchReady {
		t.Fatal("dispatch should not be ready before command submission exists")
	}
	graph := VulkanModelExecutionGraph(shape)
	if len(graph.Stages) != 2 || graph.Stages[0].Name != "text" || graph.Stages[1].Name != "vision" {
		t.Fatalf("graph=%+v", graph)
	}
	if graph.PlanCount != st.PlanCount || graph.PipelineCount != st.PipelineCount || graph.Dispatches != st.Dispatches || graph.WeightBytes != st.WeightBytes || graph.InputBytes != st.InputBytes || graph.OutputBytes != st.OutputBytes || graph.SharedBytes != st.SharedBytes {
		t.Fatalf("graph=%+v summary=%+v", graph, st)
	}
	if graph.Stages[0].PlanCount != 5 || graph.Stages[0].PipelineCount != 3 || graph.Stages[1].PlanCount != 5 || graph.Stages[1].PipelineCount != 3 {
		t.Fatalf("stages=%+v", graph.Stages)
	}
	pipes := VulkanModelPipelinePlan(shape)
	if len(pipes) != 6 {
		t.Fatalf("pipes=%+v", pipes)
	}
	var pipeDispatches, pipeRefs, textPipes, visionPipes int
	for _, p := range pipes {
		pipeDispatches += p.Dispatches
		pipeRefs += p.PlanRefs
		if p.LayoutIndex < 0 || p.LayoutIndex > 1 {
			t.Fatalf("bad pipe layout index %+v", p)
		}
		if p.ShaderModuleIndex < 0 || p.ShaderModuleIndex >= 6 {
			t.Fatalf("bad pipe shader index %+v", p)
		}
		if p.CacheKeyHash == 0 {
			t.Fatalf("missing pipe cache key %+v", p)
		}
		switch p.Stage {
		case "text":
			textPipes++
		case "vision":
			visionPipes++
		default:
			t.Fatalf("bad pipe stage %+v", p)
		}
	}
	if pipeDispatches != graph.Dispatches || pipeRefs != graph.PlanCount || textPipes != 3 || visionPipes != 3 {
		t.Fatalf("pipes=%+v graph=%+v", pipes, graph)
	}
	cmdPlan := VulkanModelCommandPlan(shape)
	if reason := ValidateVulkanCommandPlan(cmdPlan); reason != "" {
		t.Fatalf("invalid command plan: %s", reason)
	}
	if cmdPlan.PipelineCount != len(pipes) || cmdPlan.CommandCount != 10 || len(cmdPlan.Commands) != 10 {
		t.Fatalf("command plan=%+v pipes=%+v", cmdPlan, pipes)
	}
	if cmdPlan.DispatchBatchCount != len(cmdPlan.DispatchBatches) || cmdPlan.DispatchBatchCount != 9 {
		t.Fatalf("dispatch batches=%+v", cmdPlan.DispatchBatches)
	}
	if cmdPlan.PipelineBindCount != cmdPlan.DispatchBatchCount || cmdPlan.DescriptorBindCount != cmdPlan.CommandCount || cmdPlan.PushConstantCount != cmdPlan.CommandCount {
		t.Fatalf("bind counts plan=%+v", cmdPlan)
	}
	if cmdPlan.RecordCount != 40 || len(cmdPlan.Records) != 40 {
		t.Fatalf("records=%+v", cmdPlan.Records)
	}
	if cmdPlan.BarrierCount != 35 || len(cmdPlan.Barriers) != 35 {
		t.Fatalf("barriers=%+v", cmdPlan.Barriers)
	}
	if cmdPlan.AllocationCount != 35 || len(cmdPlan.Allocations) != 35 {
		t.Fatalf("allocations=%+v", cmdPlan.Allocations)
	}
	if cmdPlan.UploadCount != 25 || len(cmdPlan.Uploads) != 25 {
		t.Fatalf("uploads=%+v", cmdPlan.Uploads)
	}
	if cmdPlan.ReadbackCount != 10 || len(cmdPlan.Readbacks) != 10 {
		t.Fatalf("readbacks=%+v", cmdPlan.Readbacks)
	}
	if cmdPlan.LayoutCount != 2 || len(cmdPlan.Layouts) != 2 {
		t.Fatalf("layouts=%+v", cmdPlan.Layouts)
	}
	if cmdPlan.ShaderModuleCount != 6 || len(cmdPlan.ShaderModules) != 6 {
		t.Fatalf("shader modules=%+v", cmdPlan.ShaderModules)
	}
	for _, sm := range cmdPlan.ShaderModules {
		if sm.KernelName == "" || sm.EntryPoint != "main" || sm.SourceHash == 0 || sm.SourceBytes == 0 || sm.LocalSizeX != 256 || sm.TileCols != 256 || sm.PipelineRefs != 1 {
			t.Fatalf("shader module=%+v", sm)
		}
		if sm.Specializations[0].Name != "local_size_x" || sm.Specializations[0].Value != 256 || sm.Specializations[1].Name != "tile_cols" || sm.Specializations[1].Value != 256 {
			t.Fatalf("shader specializations=%+v", sm)
		}
		if strings.Contains(sm.KernelName, "_q4_") && sm.Specializations[2].Value != 4 {
			t.Fatalf("q4 specialization=%+v", sm)
		}
		if strings.Contains(sm.KernelName, "_f32_") && sm.Specializations[2].Value != 32 {
			t.Fatalf("f32 specialization=%+v", sm)
		}
	}
	if cmdPlan.Layouts[0].PipelineRefs != 3 || cmdPlan.Layouts[1].PipelineRefs != 3 {
		t.Fatalf("layout refs=%+v", cmdPlan.Layouts)
	}
	if len(cmdPlan.Layouts[0].Bindings) != 4 || len(cmdPlan.Layouts[1].Bindings) != 3 {
		t.Fatalf("layout bindings=%+v", cmdPlan.Layouts)
	}
	if cmdPlan.Layouts[0].Bindings[0].DescriptorType != "storage_buffer" || cmdPlan.Layouts[1].Bindings[2].DescriptorType != "storage_buffer" {
		t.Fatalf("layout descriptor types=%+v", cmdPlan.Layouts)
	}
	if cmdPlan.ResourceCount != 35 || cmdPlan.DescriptorWriteCount != 35 {
		t.Fatalf("command plan resource/write counts=%+v", cmdPlan)
	}
	if cmdPlan.DescriptorPool.MaxSets != cmdPlan.CommandCount || cmdPlan.DescriptorPool.StorageBufferCount != cmdPlan.DescriptorWriteCount {
		t.Fatalf("descriptor pool=%+v plan=%+v", cmdPlan.DescriptorPool, cmdPlan)
	}
	if cmdPlan.CommandPool.QueueFamily != "compute" || cmdPlan.CommandPool.CommandBufferCount != 1 || cmdPlan.CommandPool.ResetMode != "reset_command_buffer" {
		t.Fatalf("command pool=%+v", cmdPlan.CommandPool)
	}
	if cmdPlan.QueueSubmit.QueueFamily != "compute" || cmdPlan.QueueSubmit.SubmitCount != 1 || cmdPlan.QueueSubmit.CommandBufferCount != 1 || !cmdPlan.QueueSubmit.SignalFence {
		t.Fatalf("queue submit=%+v", cmdPlan.QueueSubmit)
	}
	if cmdPlan.Timeline.InitialValue != 0 || cmdPlan.Timeline.WaitValue != 0 || cmdPlan.Timeline.SignalValue != 1 || cmdPlan.Timeline.StageMask != "compute_shader" {
		t.Fatalf("timeline=%+v", cmdPlan.Timeline)
	}
	if cmdPlan.Fence.CreateSignaled || !cmdPlan.Fence.ResetBeforeSubmit || !cmdPlan.Fence.WaitAfterSubmit {
		t.Fatalf("fence=%+v", cmdPlan.Fence)
	}
	if cmdPlan.PipelineCache.EntryCount != cmdPlan.PipelineCount || cmdPlan.PipelineCache.CreateCount != cmdPlan.PipelineCount || cmdPlan.PipelineCache.CacheKeyHash == 0 || !cmdPlan.PipelineCache.Persistable {
		t.Fatalf("pipeline cache=%+v plan=%+v", cmdPlan.PipelineCache, cmdPlan)
	}
	if cmdPlan.PipelineCache.LayoutRefs != graph.PlanCount || cmdPlan.PipelineCache.ShaderRefs != graph.PlanCount || cmdPlan.PipelineCache.ReuseCount != graph.PlanCount-cmdPlan.PipelineCount {
		t.Fatalf("pipeline cache refs=%+v graph=%+v", cmdPlan.PipelineCache, graph)
	}
	if cmdPlan.PipelineLifecycle.CreateCount != cmdPlan.PipelineCount || cmdPlan.PipelineLifecycle.DestroyCount != cmdPlan.PipelineCount {
		t.Fatalf("pipeline lifecycle=%+v", cmdPlan.PipelineLifecycle)
	}
	if cmdPlan.PipelineLifecycle.Steps[0] != "load_pipeline_cache" || cmdPlan.PipelineLifecycle.Steps[1] != "create_compute_pipelines" || cmdPlan.PipelineLifecycle.Steps[3] == "" {
		t.Fatalf("pipeline lifecycle steps=%+v", cmdPlan.PipelineLifecycle)
	}
	if cmdPlan.TotalBufferBytes == 0 {
		t.Fatalf("total buffer bytes=%d", cmdPlan.TotalBufferBytes)
	}
	var rawAllocBytes, alignedAllocBytes int64
	for _, a := range cmdPlan.Allocations {
		rawAllocBytes += a.Bytes
		alignedAllocBytes += a.AlignedBytes
	}
	if cmdPlan.TotalBufferBytes != alignedAllocBytes || cmdPlan.TotalBufferBytes < rawAllocBytes {
		t.Fatalf("allocation byte accounting total=%d raw=%d aligned=%d", cmdPlan.TotalBufferBytes, rawAllocBytes, alignedAllocBytes)
	}
	if cmdPlan.UploadBytes == 0 || cmdPlan.ReadbackBytes == 0 {
		t.Fatalf("transfer bytes upload=%d readback=%d", cmdPlan.UploadBytes, cmdPlan.ReadbackBytes)
	}
	if cmdPlan.Dispatches != graph.Dispatches || cmdPlan.DispatchReady {
		t.Fatalf("command plan dispatch=%+v graph=%+v", cmdPlan, graph)
	}
	for _, cmd := range cmdPlan.Commands {
		if cmd.PipelineIndex < 0 || cmd.PipelineIndex >= len(cmdPlan.Pipelines) {
			t.Fatalf("bad command pipeline index %+v", cmd)
		}
		if cmd.GroupsX <= 0 || cmd.GroupsY <= 0 || cmd.Dispatches <= 0 {
			t.Fatalf("bad command grid %+v", cmd)
		}
	}
	modelPlans := VulkanModelPlans(shape)
	q4Plan := modelPlans[0]
	q4Cmd := cmdPlan.Commands[0]
	if len(q4Cmd.Resources) != 4 {
		t.Fatalf("q4 resources=%+v", q4Cmd.Resources)
	}
	if q4Cmd.Resources[0].Name != "x" || q4Cmd.Resources[0].Bytes != q4Plan.InputBytes || q4Cmd.Resources[0].Access != "readonly" {
		t.Fatalf("q4 input resource=%+v", q4Cmd.Resources[0])
	}
	if q4Cmd.Resources[1].Name != "w" || q4Cmd.Resources[1].Bytes != q4Plan.WeightByte || q4Cmd.Resources[1].Access != "readonly" {
		t.Fatalf("q4 weight resource=%+v", q4Cmd.Resources[1])
	}
	if q4Cmd.Resources[2].Name != "scale" || q4Cmd.Resources[2].Bytes != int64(q4Plan.Rows)*4 || q4Cmd.Resources[2].Access != "readonly" {
		t.Fatalf("q4 scale resource=%+v", q4Cmd.Resources[2])
	}
	if q4Cmd.Resources[3].Name != "out" || q4Cmd.Resources[3].Bytes != q4Plan.OutputBytes || q4Cmd.Resources[3].Access != "writeonly" {
		t.Fatalf("q4 output resource=%+v", q4Cmd.Resources[3])
	}
	if q4Cmd.DescriptorSet.SetIndex != 0 || q4Cmd.DescriptorSet.LayoutIndex != cmdPlan.Pipelines[q4Cmd.PipelineIndex].LayoutIndex || q4Cmd.DescriptorSet.DescriptorOffset != 0 || q4Cmd.DescriptorSet.DescriptorCount != 4 {
		t.Fatalf("q4 descriptor set=%+v", q4Cmd.DescriptorSet)
	}
	if q4Cmd.PushConstants.Rows != uint32(q4Plan.Rows) || q4Cmd.PushConstants.Cols != uint32(q4Plan.Cols) || q4Cmd.PushConstants.Bytes != 8 {
		t.Fatalf("q4 push constants=%+v plan=%+v", q4Cmd.PushConstants, q4Plan)
	}
	firstBatch := cmdPlan.DispatchBatches[0]
	if firstBatch.BatchIndex != 0 || firstBatch.CommandStart != 0 || firstBatch.CommandCount != 1 || firstBatch.PipelineIndex != q4Cmd.PipelineIndex || firstBatch.LayoutIndex != q4Cmd.DescriptorSet.LayoutIndex || firstBatch.Dispatches != q4Cmd.Dispatches {
		t.Fatalf("first dispatch batch=%+v cmd=%+v", firstBatch, q4Cmd)
	}
	firstRecords := cmdPlan.Records[:4]
	if firstRecords[0].Op != "bind_pipeline" || firstRecords[1].Op != "bind_descriptor_set" || firstRecords[2].Op != "push_constants" || firstRecords[3].Op != "dispatch" {
		t.Fatalf("first command records=%+v", firstRecords)
	}
	if firstRecords[0].PipelineIndex != q4Cmd.PipelineIndex || firstRecords[0].LayoutIndex != cmdPlan.Pipelines[q4Cmd.PipelineIndex].LayoutIndex {
		t.Fatalf("bind pipeline record=%+v cmd=%+v", firstRecords[0], q4Cmd)
	}
	if firstRecords[1].DescriptorWriteStart != 0 || firstRecords[1].DescriptorWriteCount != 4 {
		t.Fatalf("descriptor bind record=%+v", firstRecords[1])
	}
	if firstRecords[1].DescriptorWriteStart != q4Cmd.DescriptorSet.DescriptorOffset || firstRecords[1].DescriptorWriteCount != q4Cmd.DescriptorSet.DescriptorCount {
		t.Fatalf("descriptor record=%+v set=%+v", firstRecords[1], q4Cmd.DescriptorSet)
	}
	if firstRecords[2].PushConstantBytes != 8 || firstRecords[2].Rows != q4Plan.Rows || firstRecords[2].Cols != q4Plan.Cols {
		t.Fatalf("push constants record=%+v plan=%+v", firstRecords[2], q4Plan)
	}
	if firstRecords[3].GroupsX != q4Cmd.GroupsX || firstRecords[3].GroupsY != q4Cmd.GroupsY || firstRecords[3].Repeat != q4Cmd.Repeat {
		t.Fatalf("dispatch record=%+v cmd=%+v", firstRecords[3], q4Cmd)
	}
	firstBarriers := cmdPlan.Barriers[:4]
	if firstBarriers[0].CommandIndex != 0 || firstBarriers[0].ResourceName != "x" || firstBarriers[0].SrcStage != "host" || firstBarriers[0].DstAccess != "shader_read" {
		t.Fatalf("input barrier=%+v", firstBarriers[0])
	}
	if firstBarriers[1].ResourceName != "w" || firstBarriers[1].SrcAccess != "host_write" || firstBarriers[1].Bytes != q4Plan.WeightByte {
		t.Fatalf("weight barrier=%+v", firstBarriers[1])
	}
	if firstBarriers[3].ResourceName != "out" || firstBarriers[3].SrcStage != "compute_shader" || firstBarriers[3].SrcAccess != "shader_write" || firstBarriers[3].DstAccess != "shader_read" {
		t.Fatalf("output barrier=%+v", firstBarriers[3])
	}
	firstAllocations := cmdPlan.Allocations[:4]
	if firstAllocations[0].ResourceName != "x" || firstAllocations[0].Usage != "storage_buffer|transfer_dst" || firstAllocations[0].MemoryProperties != "host_visible|device_local" {
		t.Fatalf("input allocation=%+v", firstAllocations[0])
	}
	if firstAllocations[1].ResourceName != "w" || firstAllocations[1].Bytes != q4Plan.WeightByte {
		t.Fatalf("weight allocation=%+v", firstAllocations[1])
	}
	if firstAllocations[0].AlignmentBytes != 64 || firstAllocations[0].AlignedBytes < firstAllocations[0].Bytes || firstAllocations[0].AlignedBytes%64 != 0 {
		t.Fatalf("input aligned allocation=%+v", firstAllocations[0])
	}
	if firstAllocations[1].AlignmentBytes != 256 || firstAllocations[1].AlignedBytes < firstAllocations[1].Bytes || firstAllocations[1].AlignedBytes%256 != 0 {
		t.Fatalf("weight aligned allocation=%+v", firstAllocations[1])
	}
	if firstAllocations[3].ResourceName != "out" || firstAllocations[3].Usage != "storage_buffer|transfer_src" || firstAllocations[3].MemoryProperties != "device_local" {
		t.Fatalf("output allocation=%+v", firstAllocations[3])
	}
	if firstAllocations[3].AlignmentBytes != 64 || firstAllocations[3].AlignedBytes < firstAllocations[3].Bytes || firstAllocations[3].AlignedBytes%64 != 0 {
		t.Fatalf("output aligned allocation=%+v", firstAllocations[3])
	}
	if cmdPlan.Uploads[0].CommandIndex != 0 || cmdPlan.Uploads[0].ResourceName != "x" || cmdPlan.Uploads[0].Direction != "host_to_device" {
		t.Fatalf("first upload=%+v", cmdPlan.Uploads[0])
	}
	if cmdPlan.Uploads[2].ResourceName != "scale" || cmdPlan.Uploads[2].Bytes != int64(q4Plan.Rows)*4 {
		t.Fatalf("scale upload=%+v", cmdPlan.Uploads[2])
	}
	if cmdPlan.Readbacks[0].CommandIndex != 0 || cmdPlan.Readbacks[0].ResourceName != "out" || cmdPlan.Readbacks[0].Direction != "device_to_host" || cmdPlan.Readbacks[0].Bytes != q4Plan.OutputBytes {
		t.Fatalf("first readback=%+v", cmdPlan.Readbacks[0])
	}
	if len(q4Cmd.DescriptorWrites) != len(q4Cmd.Resources) {
		t.Fatalf("q4 descriptor writes=%+v resources=%+v", q4Cmd.DescriptorWrites, q4Cmd.Resources)
	}
	for i, w := range q4Cmd.DescriptorWrites {
		r := q4Cmd.Resources[i]
		if w.Binding != r.Binding || w.ResourceName != r.Name || w.RangeBytes != r.Bytes || w.Access != r.Access || w.DescriptorType != "storage_buffer" {
			t.Fatalf("q4 descriptor write=%+v resource=%+v", w, r)
		}
		if w.OffsetBytes != 0 {
			t.Fatalf("q4 descriptor offset=%+v", w)
		}
	}
	f32Cmd := cmdPlan.Commands[5]
	if len(f32Cmd.Resources) != 3 {
		t.Fatalf("f32 resources=%+v", f32Cmd.Resources)
	}
	if f32Cmd.Resources[0].Name != "x" || f32Cmd.Resources[1].Name != "w" || f32Cmd.Resources[2].Name != "out" {
		t.Fatalf("f32 resource order=%+v", f32Cmd.Resources)
	}
	if len(f32Cmd.DescriptorWrites) != 3 || f32Cmd.DescriptorWrites[2].Access != "writeonly" {
		t.Fatalf("f32 descriptor writes=%+v", f32Cmd.DescriptorWrites)
	}
	broken := cmdPlan
	broken.PipelineCount++
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected pipeline count mismatch")
	}
	broken = cmdPlan
	broken.Commands = append([]VulkanCommand(nil), cmdPlan.Commands...)
	broken.Commands[0].PipelineIndex = len(cmdPlan.Pipelines)
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad command pipeline index")
	}
	broken = cmdPlan
	broken.Timeline.SignalValue = 0
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad timeline signal")
	}
	broken = cmdPlan
	broken.Allocations = append([]VulkanBufferAllocation(nil), cmdPlan.Allocations...)
	broken.Allocations[0].AlignedBytes = broken.Allocations[0].Bytes - 1
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad aligned allocation")
	}
	broken = cmdPlan
	broken.Commands = append([]VulkanCommand(nil), cmdPlan.Commands...)
	broken.Commands[0].PushConstants.Rows++
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad push constants")
	}
	broken = cmdPlan
	broken.Commands = append([]VulkanCommand(nil), cmdPlan.Commands...)
	broken.Commands[0].DescriptorSet.LayoutIndex++
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad descriptor set")
	}
	broken = cmdPlan
	broken.DispatchBatches = append([]VulkanDispatchBatch(nil), cmdPlan.DispatchBatches...)
	broken.DispatchBatches[0].PipelineIndex = len(cmdPlan.Pipelines)
	if reason := ValidateVulkanCommandPlan(broken); reason == "" {
		t.Fatal("expected bad dispatch batch")
	}
}

func BenchmarkPlanVulkanKernelQ4(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, ok := PlanVulkanKernel(VulkanOpMatVec, "q4", 4096, 2048); !ok {
			b.Fatal("missing q4 plan")
		}
	}
}

func BenchmarkVulkanModelPlansInto(b *testing.B) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	dst := make([]VulkanPlan, 0, 10)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		dst = VulkanModelPlansInto(dst, shape)
		if len(dst) != 10 {
			b.Fatal(len(dst))
		}
	}
}

func BenchmarkVulkanModelPlanSummaryInto(b *testing.B) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	dst := make([]VulkanPlan, 0, 10)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		st := VulkanModelPlanSummaryInto(dst, shape)
		dst = st.Plans
		if st.PlanCount != 10 || st.Dispatches == 0 {
			b.Fatal(st)
		}
	}
}

func BenchmarkVulkanModelExecutionGraphInto(b *testing.B) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	dst := make([]VulkanPlan, 0, 10)
	stages := make([]VulkanStageSummary, 0, 2)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		graph, plans, nextStages := VulkanModelExecutionGraphInto(dst, stages, shape)
		dst, stages = plans, nextStages
		if len(graph.Stages) != 2 || graph.Dispatches == 0 {
			b.Fatal(graph)
		}
	}
}

func BenchmarkVulkanModelPipelinePlanInto(b *testing.B) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	plans := make([]VulkanPlan, 0, 10)
	pipes := make([]VulkanPipelinePlan, 0, 6)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var nextPlans []VulkanPlan
		pipes, nextPlans = VulkanModelPipelinePlanInto(plans, pipes, shape)
		plans = nextPlans
		if len(pipes) != 6 {
			b.Fatal(pipes)
		}
	}
}

func BenchmarkVulkanModelCommandPlanInto(b *testing.B) {
	shape := VulkanModelShape{
		Quant:               "q4",
		VocabSize:           32000,
		HiddenSize:          2048,
		IntermediateSize:    8192,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    4,
		HeadDim:             128,
		VisionHiddenSize:    1024,
		VisionIntermediate:  4096,
		VisionPatchElements: 588,
		TextLayers:          24,
		VisionLayers:        24,
	}
	plans := make([]VulkanPlan, 0, 10)
	pipes := make([]VulkanPipelinePlan, 0, 6)
	shaders := make([]VulkanShaderModulePlan, 0, 6)
	layouts := make([]VulkanPipelineLayoutPlan, 0, 2)
	cmds := make([]VulkanCommand, 0, 10)
	batches := make([]VulkanDispatchBatch, 0, 10)
	resources := make([]VulkanResource, 0, 35)
	descriptors := make([]VulkanDescriptorWrite, 0, 35)
	records := make([]VulkanCommandRecord, 0, 40)
	barriers := make([]VulkanBufferBarrier, 0, 35)
	allocations := make([]VulkanBufferAllocation, 0, 35)
	uploads := make([]VulkanBufferTransfer, 0, 25)
	readbacks := make([]VulkanBufferTransfer, 0, 10)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var nextPipes []VulkanPipelinePlan
		var nextPlans []VulkanPlan
		var nextShaders []VulkanShaderModulePlan
		var nextLayouts []VulkanPipelineLayoutPlan
		var nextCmds []VulkanCommand
		var nextBatches []VulkanDispatchBatch
		var nextResources []VulkanResource
		var nextDescriptors []VulkanDescriptorWrite
		var nextRecords []VulkanCommandRecord
		var nextBarriers []VulkanBufferBarrier
		var nextAllocations []VulkanBufferAllocation
		var nextUploads []VulkanBufferTransfer
		var nextReadbacks []VulkanBufferTransfer
		cmdPlan, nextPlans, nextPipes, nextShaders, nextLayouts, nextCmds, nextBatches, nextResources, nextDescriptors, nextRecords, nextBarriers, nextAllocations, nextUploads, nextReadbacks := VulkanModelCommandPlanInto(plans, pipes, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks, shape)
		plans, pipes, shaders, layouts, cmds, batches, resources, descriptors, records, barriers, allocations, uploads, readbacks = nextPlans, nextPipes, nextShaders, nextLayouts, nextCmds, nextBatches, nextResources, nextDescriptors, nextRecords, nextBarriers, nextAllocations, nextUploads, nextReadbacks
		if cmdPlan.CommandCount != 10 || cmdPlan.DispatchBatchCount != 9 || cmdPlan.PipelineCount != 6 || cmdPlan.ShaderModuleCount != 6 || cmdPlan.LayoutCount != 2 || cmdPlan.RecordCount != 40 || cmdPlan.BarrierCount != 35 || cmdPlan.AllocationCount != 35 || cmdPlan.UploadCount != 25 || cmdPlan.ReadbackCount != 10 {
			b.Fatal(cmdPlan)
		}
	}
}
