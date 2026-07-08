//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanVisionAttentionF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanVisionAttentionOutF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanVisionRoPEPairF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanVisionQKVF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanVisionAttentionF32RunnerCache struct {
	once   sync.Once
	runner *vulkanVisionAttentionF32WinRunner
	err    error
}

func VulkanVisionAttentionF32(out, q, k, v [][]float32, tokens, heads, headDim int) error {
	if tokens == 0 {
		return nil
	}
	if tokens <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision attention shape tokens=%d heads=%d headDim=%d", tokens, heads, headDim)
	}
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	if len(out) < tokens || len(q) < tokens || len(k) < tokens || len(v) < tokens {
		return fmt.Errorf("invalid Vulkan vision attention rows out=%d q=%d k=%d v=%d tokens=%d", len(out), len(q), len(k), len(v), tokens)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(q[i]) < hidden || len(k[i]) < hidden || len(v[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision attention row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, q, k, v, tokens, heads, headDim)
}

func VulkanVisionAttentionOutF32(out, q, k, v [][]float32, w, bias []float32, tokens, heads, headDim int) error {
	if tokens == 0 {
		return nil
	}
	if tokens <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision attention+out shape tokens=%d heads=%d headDim=%d", tokens, heads, headDim)
	}
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention+out")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	if len(out) < tokens || len(q) < tokens || len(k) < tokens || len(v) < tokens || len(w) < dims.wLen || len(bias) < hidden {
		return fmt.Errorf("invalid Vulkan vision attention+out buffers out=%d q=%d k=%d v=%d w=%d bias=%d tokens=%d hidden=%d",
			len(out), len(q), len(k), len(v), len(w), len(bias), tokens, hidden)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(q[i]) < hidden || len(k[i]) < hidden || len(v[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision attention+out row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runOut(out, q, k, v, w, bias, tokens, heads, headDim)
}

func VulkanVisionRoPEAttentionOutF32(out, q, k, v [][]float32, w, bias, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	tokens := len(q)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision rope+attention+out shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d", tokens, gridH, gridW, heads, headDim)
	}
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope+attention+out")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	quarter := headDim / 4
	if quarter <= 0 || len(out) < tokens || len(k) < tokens || len(v) < tokens || len(w) < dims.wLen || len(bias) < hidden ||
		len(cosH) < dims.hTableLen || len(sinH) < dims.hTableLen || len(cosW) < dims.wTableLen || len(sinW) < dims.wTableLen {
		return fmt.Errorf("invalid Vulkan vision rope+attention+out buffers out=%d q=%d k=%d v=%d w=%d bias=%d cosH=%d sinH=%d cosW=%d sinW=%d hidden=%d quarter=%d",
			len(out), len(q), len(k), len(v), len(w), len(bias), len(cosH), len(sinH), len(cosW), len(sinW), hidden, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(q[i]) < hidden || len(k[i]) < hidden || len(v[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision rope+attention+out row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runRoPEOut(out, q, k, v, w, bias, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
}

func VulkanVisionQKVRoPEAttentionOutF32(out, x [][]float32, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim, hidden int) error {
	tokens := len(x)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 || hidden <= 0 {
		return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d hidden=%d", tokens, gridH, gridW, heads, headDim, hidden)
	}
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, hidden, gridH, gridW, "Vulkan vision qkv+rope+attention+out")
	if err != nil {
		return err
	}
	quarter := headDim / 4
	if quarter <= 0 || len(out) < tokens || len(qw) < dims.wLen || len(qb) < hidden || len(kw) < dims.wLen || len(kb) < hidden || len(vw) < dims.wLen || len(vb) < hidden ||
		len(ow) < dims.wLen || len(ob) < hidden || len(cosH) < dims.hTableLen || len(sinH) < dims.hTableLen || len(cosW) < dims.wTableLen || len(sinW) < dims.wTableLen {
		return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out buffers out=%d x=%d qw=%d qb=%d kw=%d kb=%d vw=%d vb=%d ow=%d ob=%d cosH=%d sinH=%d cosW=%d sinW=%d hidden=%d quarter=%d",
			len(out), len(x), len(qw), len(qb), len(kw), len(kb), len(vw), len(vb), len(ow), len(ob), len(cosH), len(sinH), len(cosW), len(sinW), hidden, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(x[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out row %d out=%d x=%d hidden=%d", i, len(out[i]), len(x[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runQKVRoPEOut(out, x, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim, hidden)
}

func VulkanVisionRoPEPairF32(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	tokens := len(q)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision rope pair shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d", tokens, gridH, gridW, heads, headDim)
	}
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope pair")
	if err != nil {
		return err
	}
	quarter := headDim / 4
	hidden := dims.hidden
	if quarter <= 0 || len(k) < tokens || len(cosH) < dims.hTableLen || len(sinH) < dims.hTableLen || len(cosW) < dims.wTableLen || len(sinW) < dims.wTableLen {
		return fmt.Errorf("invalid Vulkan vision rope pair buffers q=%d k=%d cosH=%d sinH=%d cosW=%d sinW=%d gridH=%d gridW=%d quarter=%d",
			len(q), len(k), len(cosH), len(sinH), len(cosW), len(sinW), gridH, gridW, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(q[i]) < hidden || len(k[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision rope pair row %d q=%d k=%d hidden=%d", i, len(q[i]), len(k[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runRoPEPair(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
}

func getVulkanVisionAttentionF32RunnerWindows() (*vulkanVisionAttentionF32WinRunner, error) {
	vulkanVisionAttentionF32RunnerCache.once.Do(func() {
		vulkanVisionAttentionF32RunnerCache.runner, vulkanVisionAttentionF32RunnerCache.err = newVulkanVisionAttentionF32WinRunner()
	})
	return vulkanVisionAttentionF32RunnerCache.runner, vulkanVisionAttentionF32RunnerCache.err
}

type vulkanVisionAttentionF32WinRunner struct {
	vk              *vulkanWin
	instance        uintptr
	device          uintptr
	queue           uintptr
	queueFamily     uint32
	memProps        vkPhysicalDeviceMemoryProperties
	setLayout       uintptr
	descriptorPool  uintptr
	descriptorSet   uintptr
	pipelineLayout  uintptr
	pipeline        uintptr
	projPipeline    uintptr
	ropePipeline    uintptr
	qkvPipeline     uintptr
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	qBuf            vkHostBufferWin
	kBuf            vkHostBufferWin
	vBuf            vkHostBufferWin
	outBuf          vkHostBufferWin
	finalBuf        vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [18]vulkanDescriptorBindingWin
	commandRecorded bool
	commandKind     int
	commandTokens   int
	commandHeads    int
	commandHeadDim  int
	commandHidden   int
	sharedDevice    bool
	mu              sync.Mutex
}

const (
	vulkanVisionAttentionCommandOnly       = 1
	vulkanVisionAttentionCommandOut        = 2
	vulkanVisionAttentionCommandRoPE       = 3
	vulkanVisionAttentionCommandRoPEOut    = 4
	vulkanVisionAttentionCommandQKVRoPEOut = 5
)

type vulkanVisionAttentionDimsWin struct {
	hidden    int
	bufLen    int
	wLen      int
	hTableLen int
	wTableLen int
	gridLen   int
}

func checkedVisionAttentionDimsWin(tokens, heads, headDim, hidden, gridH, gridW int, label string) (vulkanVisionAttentionDimsWin, error) {
	computedHidden, ok := checkedMulInt(heads, headDim)
	if !ok {
		return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s hidden length overflows: heads=%d headDim=%d", label, heads, headDim)
	}
	if hidden == 0 {
		hidden = computedHidden
	} else if hidden != computedHidden {
		return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s hidden mismatch: hidden=%d heads=%d headDim=%d", label, hidden, heads, headDim)
	}
	bufLen, ok := checkedMulInt(tokens, hidden)
	if !ok {
		return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s token buffer length overflows: tokens=%d hidden=%d", label, tokens, hidden)
	}
	wLen, ok := checkedMulInt(hidden, hidden)
	if !ok {
		return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s projection weight length overflows: hidden=%d", label, hidden)
	}
	dims := vulkanVisionAttentionDimsWin{hidden: hidden, bufLen: bufLen, wLen: wLen}
	if gridH > 0 || gridW > 0 {
		gridLen, ok := checkedMulInt(gridH, gridW)
		if !ok {
			return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s grid length overflows: gridH=%d gridW=%d", label, gridH, gridW)
		}
		quarter := headDim / 4
		hTableLen, ok := checkedMulInt(gridH, quarter)
		if !ok {
			return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s h rope table length overflows: gridH=%d quarter=%d", label, gridH, quarter)
		}
		wTableLen, ok := checkedMulInt(gridW, quarter)
		if !ok {
			return vulkanVisionAttentionDimsWin{}, fmt.Errorf("%s w rope table length overflows: gridW=%d quarter=%d", label, gridW, quarter)
		}
		dims.hTableLen = hTableLen
		dims.wTableLen = wTableLen
		dims.gridLen = gridLen
	}
	return dims, nil
}

func newVulkanVisionAttentionF32WinRunner() (*vulkanVisionAttentionF32WinRunner, error) {
	spv, err := vulkanVisionAttentionF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	projSPV, err := vulkanVisionAttentionOutF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	ropeSPV, err := vulkanVisionRoPEPairF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	qkvSPV, err := vulkanVisionQKVF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	instance := ctx.instance
	queueFamily := ctx.queueFamily
	entryName := append([]byte("main"), 0)
	r := &vulkanVisionAttentionF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	bindings := make([]vkDescriptorSetLayoutBinding, 18)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 20}
	plci := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.pipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	projSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(projSPV) * 4), PCode: uintptr(unsafe.Pointer(&projSPV[0]))}
	var projShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&projSMCI)), 0, uintptr(unsafe.Pointer(&projShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule projection: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, projShader, 0)
	projStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: projShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	projCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: projStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&projCPCI)), 0, uintptr(unsafe.Pointer(&r.projPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines projection: %d", int32(res))
	}
	ropeSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(ropeSPV) * 4), PCode: uintptr(unsafe.Pointer(&ropeSPV[0]))}
	var ropeShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&ropeSMCI)), 0, uintptr(unsafe.Pointer(&ropeShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule rope: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, ropeShader, 0)
	ropeStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: ropeShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	ropeCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: ropeStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&ropeCPCI)), 0, uintptr(unsafe.Pointer(&r.ropePipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines rope: %d", int32(res))
	}
	qkvSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(qkvSPV) * 4), PCode: uintptr(unsafe.Pointer(&qkvSPV[0]))}
	var qkvShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&qkvSMCI)), 0, uintptr(unsafe.Pointer(&qkvShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule qkv: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, qkvShader, 0)
	qkvStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: qkvShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	qkvCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: qkvStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&qkvCPCI)), 0, uintptr(unsafe.Pointer(&r.qkvPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines qkv: %d", int32(res))
	}
	cpci2 := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, r.device, uintptr(unsafe.Pointer(&cpci2)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{SType: vkStructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vkCommandBufferLevelPrimary, CommandBufferCount: 1}
	if res := vk.call(vk.allocateCommandBuffers, r.device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, r.device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanVisionAttentionF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.projPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.projPipeline, 0)
	}
	if r.ropePipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.ropePipeline, 0)
	}
	if r.qkvPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.qkvPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.qBuf)
	r.vk.destroyBuffer(r.device, r.kBuf)
	r.vk.destroyBuffer(r.device, r.vBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.finalBuf)
	for _, b := range r.weightBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.biasBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	if r.descriptorPool != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorPool, r.device, r.descriptorPool, 0)
	}
	if r.pipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.pipelineLayout, 0)
	}
	if r.setLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.setLayout, 0)
	}
	if !r.sharedDevice {
		if r.device != 0 {
			r.vk.callVoid(r.vk.destroyDevice, r.device, 0)
		}
		if r.instance != 0 {
			r.vk.callVoid(r.vk.destroyInstance, r.instance, 0)
		}
	}
}

func (r *vulkanVisionAttentionF32WinRunner) runOut(out, q, k, v [][]float32, w, bias []float32, tokens, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention out runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrWin(dims.bufLen, "Vulkan vision attention out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(dims.wLen, "Vulkan vision attention out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrWin(hidden, "Vulkan vision attention out runner bias")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.qBuf, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.kBuf, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.vBuf, v, tokens, hidden); err != nil {
		return err
	}
	bufInfos := [7]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: bufBytes},
		{Buffer: r.kBuf.buffer, Range: bufBytes},
		{Buffer: r.vBuf.buffer, Range: bufBytes},
		{Buffer: r.outBuf.buffer, Range: bufBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: bufBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionCommandOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != hidden {
		if err := r.recordOutCommand(tokens, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readRowsPrefixInto(r.device, r.finalBuf, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32WinRunner) runRoPEOut(out, q, k, v [][]float32, w, bias, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(q)
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope out runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrWin(dims.bufLen, "Vulkan vision rope out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(dims.wLen, "Vulkan vision rope out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrWin(hidden, "Vulkan vision rope out runner bias")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrWin(dims.hTableLen, "Vulkan vision rope out runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrWin(dims.wTableLen, "Vulkan vision rope out runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	cosHBuf, err := r.cachedBuffer(cosH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinHBuf, err := r.cachedBuffer(sinH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	cosWBuf, err := r.cachedBuffer(cosW[:dims.wTableLen], wTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.qBuf, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.kBuf, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.vBuf, v, tokens, hidden); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: bufBytes},
		{Buffer: r.kBuf.buffer, Range: bufBytes},
		{Buffer: r.vBuf.buffer, Range: bufBytes},
		{Buffer: r.outBuf.buffer, Range: bufBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: bufBytes},
		{Buffer: cosHBuf.buffer, Range: hTableBytes},
		{Buffer: sinHBuf.buffer, Range: hTableBytes},
		{Buffer: cosWBuf.buffer, Range: wTableBytes},
		{Buffer: sinWBuf.buffer, Range: wTableBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionCommandRoPEOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readRowsPrefixInto(r.device, r.finalBuf, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32WinRunner) runQKVRoPEOut(out, x [][]float32, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim, hidden int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(x)
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, hidden, gridH, gridW, "Vulkan vision qkv rope out runner")
	if err != nil {
		return err
	}
	bufBytes, err := checkedFloat32ByteLenErrWin(dims.bufLen, "Vulkan vision qkv rope out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(dims.wLen, "Vulkan vision qkv rope out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrWin(hidden, "Vulkan vision qkv rope out runner bias")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrWin(dims.hTableLen, "Vulkan vision qkv rope out runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrWin(dims.wTableLen, "Vulkan vision qkv rope out runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	qwBuf, err := r.cachedBuffer(qw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	qbBuf, err := r.cachedBuffer(qb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	kwBuf, err := r.cachedBuffer(kw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	kbBuf, err := r.cachedBuffer(kb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	vwBuf, err := r.cachedBuffer(vw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	vbBuf, err := r.cachedBuffer(vb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	owBuf, err := r.cachedBuffer(ow[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	obBuf, err := r.cachedBuffer(ob[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	cosHBuf, err := r.cachedBuffer(cosH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinHBuf, err := r.cachedBuffer(sinH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	cosWBuf, err := r.cachedBuffer(cosW[:dims.wTableLen], wTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, x, tokens, hidden); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: bufBytes},
		{Buffer: r.kBuf.buffer, Range: bufBytes},
		{Buffer: r.vBuf.buffer, Range: bufBytes},
		{Buffer: r.outBuf.buffer, Range: bufBytes},
		{Buffer: owBuf.buffer, Range: wBytes},
		{Buffer: obBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: bufBytes},
		{Buffer: cosHBuf.buffer, Range: hTableBytes},
		{Buffer: sinHBuf.buffer, Range: hTableBytes},
		{Buffer: cosWBuf.buffer, Range: wTableBytes},
		{Buffer: sinWBuf.buffer, Range: wTableBytes},
		{Buffer: r.xBuf.buffer, Range: bufBytes},
		{Buffer: qwBuf.buffer, Range: wBytes},
		{Buffer: qbBuf.buffer, Range: biasBytes},
		{Buffer: kwBuf.buffer, Range: wBytes},
		{Buffer: kbBuf.buffer, Range: biasBytes},
		{Buffer: vwBuf.buffer, Range: wBytes},
		{Buffer: vbBuf.buffer, Range: biasBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionCommandQKVRoPEOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordQKVRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readRowsPrefixInto(r.device, r.finalBuf, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32WinRunner) recordQKVRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	var pc16 [16]byte
	var pc20 [20]byte
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.qkvPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	binary.LittleEndian.PutUint32(pc20[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc20[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[12:16], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[16:20], uint32(hidden))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc20)), uintptr(unsafe.Pointer(&pc20[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hidden*3), uintptr(tokens), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.ropePipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc16[12:16], uint32(headDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc16)), uintptr(unsafe.Pointer(&pc16[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc16[12:16], uint32(hidden))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc16)), uintptr(unsafe.Pointer(&pc16[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc16[12:16], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc16)), uintptr(unsafe.Pointer(&pc16[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hidden), uintptr(tokens), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision qkv rope out command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.commandKind = vulkanVisionAttentionCommandQKVRoPEOut
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = gridLen
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) recordRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [16]byte
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.ropePipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hidden), uintptr(tokens), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision rope out command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.commandKind = vulkanVisionAttentionCommandRoPEOut
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = gridLen
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) recordOutCommand(tokens, heads, headDim, hidden int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hidden), uintptr(tokens), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanVisionAttentionCommandOut
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = hidden
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) run(out, q, k, v [][]float32, tokens, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrWin(dims.bufLen, "Vulkan vision attention runner buffer")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.qBuf, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.kBuf, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.vBuf, v, tokens, hidden); err != nil {
		return err
	}
	bufInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: bufBytes},
		{Buffer: r.kBuf.buffer, Range: bufBytes},
		{Buffer: r.vBuf.buffer, Range: bufBytes},
		{Buffer: r.outBuf.buffer, Range: bufBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionCommandOnly || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != hidden {
		if err := r.recordAttentionCommand(tokens, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readRowsPrefixInto(r.device, r.outBuf, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32WinRunner) recordAttentionCommand(tokens, heads, headDim, hidden int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanVisionAttentionCommandOnly
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = hidden
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) runRoPEPair(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(q)
	dims, err := checkedVisionAttentionDimsWin(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrWin(dims.bufLen, "Vulkan vision rope runner buffer")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrWin(dims.hTableLen, "Vulkan vision rope runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrWin(dims.wTableLen, "Vulkan vision rope runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, hTableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, hTableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, wTableBytes); err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.qBuf, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.kBuf, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.vBuf, cosH[:dims.hTableLen]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.outBuf, sinH[:dims.hTableLen]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.finalBuf, cosW[:dims.wTableLen]); err != nil {
		return err
	}
	bufInfos := [6]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: bufBytes},
		{Buffer: r.kBuf.buffer, Range: bufBytes},
		{Buffer: r.vBuf.buffer, Range: hTableBytes},
		{Buffer: r.outBuf.buffer, Range: hTableBytes},
		{Buffer: r.finalBuf.buffer, Range: wTableBytes},
		{Buffer: sinWBuf.buffer, Range: wTableBytes},
	}
	bindings := [...]uint32{0, 1, 7, 8, 9, 10}
	updateVulkanDescriptorBindingsWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bindings[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionCommandRoPE || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordRoPEPairCommand(tokens, gridH, gridW, heads, headDim); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readRowsPrefixInto(r.device, r.qBuf, q, tokens, hidden); err != nil {
		return err
	}
	return r.vk.readRowsPrefixInto(r.device, r.kBuf, k, tokens, hidden)
}

func (r *vulkanVisionAttentionF32WinRunner) recordRoPEPairCommand(tokens, gridH, gridW, heads, headDim int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.ropePipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(tokens), uintptr(heads), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision rope command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.commandKind = vulkanVisionAttentionCommandRoPE
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = gridLen
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
	if buf.buffer != 0 && buf.size >= size {
		return nil
	}
	if buf.buffer != 0 || buf.memory != 0 {
		r.vk.destroyBuffer(r.device, *buf)
		*buf = vkHostBufferWin{}
	}
	next, err := r.vk.newHostBuffer(r.device, r.memProps, size)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanVisionAttentionF32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func vulkanVisionAttentionF32ShaderCodeWindows() ([]uint32, error) {
	vulkanVisionAttentionF32SPV.once.Do(func() {
		vulkanVisionAttentionF32SPV.code, vulkanVisionAttentionF32SPV.err = compileVulkanGLSLWindows(vulkanVisionAttentionF32GLSL)
	})
	return vulkanVisionAttentionF32SPV.code, vulkanVisionAttentionF32SPV.err
}

func vulkanVisionAttentionOutF32ShaderCodeWindows() ([]uint32, error) {
	vulkanVisionAttentionOutF32SPV.once.Do(func() {
		vulkanVisionAttentionOutF32SPV.code, vulkanVisionAttentionOutF32SPV.err = compileVulkanGLSLWindows(vulkanVisionAttentionOutF32GLSL)
	})
	return vulkanVisionAttentionOutF32SPV.code, vulkanVisionAttentionOutF32SPV.err
}

func vulkanVisionRoPEPairF32ShaderCodeWindows() ([]uint32, error) {
	vulkanVisionRoPEPairF32SPV.once.Do(func() {
		vulkanVisionRoPEPairF32SPV.code, vulkanVisionRoPEPairF32SPV.err = compileVulkanGLSLWindows(vulkanVisionRoPEPairF32GLSL)
	})
	return vulkanVisionRoPEPairF32SPV.code, vulkanVisionRoPEPairF32SPV.err
}

func vulkanVisionQKVF32ShaderCodeWindows() ([]uint32, error) {
	vulkanVisionQKVF32SPV.once.Do(func() {
		vulkanVisionQKVF32SPV.code, vulkanVisionQKVF32SPV.err = compileVulkanGLSLWindows(vulkanVisionQKVF32GLSL)
	})
	return vulkanVisionQKVF32SPV.code, vulkanVisionQKVF32SPV.err
}

const vulkanVisionAttentionF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint tokens; uint heads; uint headDim; uint hidden; } pc;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float maxScore;
shared float denom;
shared float weight;
void main() {
  uint token = gl_WorkGroupID.x;
  uint head = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint headBase = head * pc.headDim;
  uint qBase = token * pc.hidden + headBase;
  float scale = inversesqrt(float(pc.headDim));
  if (lid == 0) maxScore = -3.4028234663852886e38;
  barrier();
  for (uint key = 0; key < pc.tokens; key++) {
    uint kBase = key * pc.hidden + headBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) maxScore = max(maxScore, scratch[0] * scale);
    barrier();
  }
  float acc = 0.0;
  if (lid == 0) denom = 0.0;
  barrier();
  for (uint key = 0; key < pc.tokens; key++) {
    uint kBase = key * pc.hidden + headBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) {
      weight = exp(scratch[0] * scale - maxScore);
      denom += weight;
    }
    barrier();
    if (lid < pc.headDim) acc += weight * v[key * pc.hidden + headBase + lid];
    barrier();
  }
  if (lid < pc.headDim) outv[token * pc.hidden + headBase + lid] = acc / denom;
}`

const vulkanVisionAttentionOutF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { float w[]; };
layout(set=0,binding=5) readonly buffer B { float bias[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] + bias[row];
}`

const vulkanVisionRoPEPairF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint gridH; uint gridW; uint heads; uint headDim; } pc;
layout(set=0,binding=0) buffer Q { float q[]; };
layout(set=0,binding=1) buffer K { float k[]; };
layout(set=0,binding=7) readonly buffer CH { float cosH[]; };
layout(set=0,binding=8) readonly buffer SH { float sinH[]; };
layout(set=0,binding=9) readonly buffer CW { float cosW[]; };
layout(set=0,binding=10) readonly buffer SW { float sinW[]; };
void rotatePair(uint base, uint aOff, uint bOff, float cs, float sn) {
  float qa = q[base + aOff];
  float qb = q[base + bOff];
  float ka = k[base + aOff];
  float kb = k[base + bOff];
  q[base + aOff] = qa * cs - qb * sn;
  q[base + bOff] = qb * cs + qa * sn;
  k[base + aOff] = ka * cs - kb * sn;
  k[base + bOff] = kb * cs + ka * sn;
}
void main() {
  uint token = gl_WorkGroupID.x;
  uint head = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint halfDim = pc.headDim >> 1;
  uint quarter = halfDim >> 1;
  if (quarter == 0u || pc.gridH == 0u || pc.gridW == 0u) {
    return;
  }
  uint period = pc.gridH * pc.gridW;
  uint pos = token - (token / period) * period;
  uint hy = pos / pc.gridW;
  uint wx = pos - hy * pc.gridW;
  uint hidden = pc.heads * pc.headDim;
  uint base = token * hidden + head * pc.headDim;
  for (uint i = lid; i < quarter; i += 256u) {
    uint hIdx = hy * quarter + i;
    rotatePair(base, i, quarter + i, cosH[hIdx], sinH[hIdx]);
    uint wIdx = wx * quarter + i;
    rotatePair(base, halfDim + i, halfDim + quarter + i, cosW[wIdx], sinW[wIdx]);
  }
}`

const vulkanVisionQKVF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rowsA; uint rowsB; uint rowsC; uint cols; } pc;
layout(set=0,binding=0) writeonly buffer OA { float oa[]; };
layout(set=0,binding=1) writeonly buffer OB { float ob[]; };
layout(set=0,binding=2) writeonly buffer OC { float oc[]; };
layout(set=0,binding=11) readonly buffer X { float x[]; };
layout(set=0,binding=12) readonly buffer WA { float wa[]; };
layout(set=0,binding=13) readonly buffer BA { float ba[]; };
layout(set=0,binding=14) readonly buffer WB { float wb[]; };
layout(set=0,binding=15) readonly buffer BB { float bb[]; };
layout(set=0,binding=16) readonly buffer WC { float wc[]; };
layout(set=0,binding=17) readonly buffer BC { float bc[]; };
shared float scratch[256];
void main() {
  uint globalRow = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint row = globalRow;
  uint segment = 0;
  if (row >= pc.rowsA) {
    row -= pc.rowsA;
    segment = 1;
    if (row >= pc.rowsB) {
      row -= pc.rowsB;
      segment = 2;
    }
  }
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    float xv = x[xBase + c];
    if (segment == 0) sum += wa[wBase + c] * xv;
    else if (segment == 1) sum += wb[wBase + c] * xv;
    else sum += wc[wBase + c] * xv;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    if (segment == 0) oa[batch * pc.rowsA + row] = scratch[0] + ba[row];
    else if (segment == 1) ob[batch * pc.rowsB + row] = scratch[0] + bb[row];
    else oc[batch * pc.rowsC + row] = scratch[0] + bc[row];
  }
}`
