//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanMatRowsBiasF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatRowsBiasAddRowsF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatRowsBiasF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsBiasF32WinRunner
	err    error
}

func VulkanMatRowsBiasF32(out, xs [][]float32, w, bias []float32, rows, cols int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias shape batches=%d rows=%d cols=%d", batches, rows, cols)
	}
	dims, err := checkedMatRowsBiasDimsWin(batches, 0, rows, cols, "Vulkan matrows+bias")
	if err != nil {
		return err
	}
	if len(w) < dims.wLen || len(bias) < rows || len(out) < batches {
		return fmt.Errorf("invalid Vulkan matrows+bias buffers out=%d xs=%d w=%d bias=%d rows=%d cols=%d", len(out), len(xs), len(w), len(bias), rows, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias row %d out=%d x=%d rows=%d cols=%d", i, len(out[i]), len(xs[i]), rows, cols)
		}
	}
	runner, err := getVulkanMatRowsBiasF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, xs, w[:dims.wLen], bias[:rows], rows, cols)
}

func VulkanMatRowsBiasAddRowsF32(out, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) error {
	batches := len(xs)
	addRows := len(add)
	if batches == 0 {
		return nil
	}
	if rows <= 0 || cols <= 0 || addRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias+addrows shape batches=%d addRows=%d rows=%d cols=%d", batches, addRows, rows, cols)
	}
	dims, err := checkedMatRowsBiasDimsWin(batches, addRows, rows, cols, "Vulkan matrows+bias+addrows")
	if err != nil {
		return err
	}
	if len(w) < dims.wLen || len(bias) < rows || len(out) < batches {
		return fmt.Errorf("invalid Vulkan matrows+bias+addrows buffers out=%d xs=%d w=%d bias=%d add=%d rows=%d cols=%d", len(out), len(xs), len(w), len(bias), len(add), rows, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias+addrows row %d out=%d x=%d rows=%d cols=%d", i, len(out[i]), len(xs[i]), rows, cols)
		}
	}
	for i := 0; i < addRows; i++ {
		if len(add[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias+addrows add row %d add=%d rows=%d", i, len(add[i]), rows)
		}
	}
	runner, err := getVulkanMatRowsBiasF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runAddRows(out, xs, w[:dims.wLen], bias[:rows], add, rows, cols)
}

func getVulkanMatRowsBiasF32RunnerWindows() (*vulkanMatRowsBiasF32WinRunner, error) {
	vulkanMatRowsBiasF32RunnerCache.once.Do(func() {
		vulkanMatRowsBiasF32RunnerCache.runner, vulkanMatRowsBiasF32RunnerCache.err = newVulkanMatRowsBiasF32WinRunner()
	})
	return vulkanMatRowsBiasF32RunnerCache.runner, vulkanMatRowsBiasF32RunnerCache.err
}

type vulkanMatRowsBiasF32WinRunner struct {
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
	addPipeline     uintptr
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	outBuf          vkHostBufferWin
	addBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [5]vulkanDescriptorBindingWin
	commandRecorded bool
	commandBatches  int
	commandRows     int
	commandCols     int
	commandAddRows  int
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanMatRowsBiasF32WinRunner() (*vulkanMatRowsBiasF32WinRunner, error) {
	spv, err := vulkanMatRowsBiasF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	addSPV, err := vulkanMatRowsBiasAddRowsF32ShaderCodeWindows()
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
	r := &vulkanMatRowsBiasF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 5)
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
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 16}
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
	addSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(addSPV) * 4), PCode: uintptr(unsafe.Pointer(&addSPV[0]))}
	var addShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&addSMCI)), 0, uintptr(unsafe.Pointer(&addShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule addrows: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, addShader, 0)
	addStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: addShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	addCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: addStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&addCPCI)), 0, uintptr(unsafe.Pointer(&r.addPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines addrows: %d", int32(res))
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

func (r *vulkanMatRowsBiasF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.addPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.addPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.addBuf)
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

func (r *vulkanMatRowsBiasF32WinRunner) run(out, xs [][]float32, w, bias []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	dims, err := checkedMatRowsBiasDimsWin(batches, 0, rows, cols, "Vulkan matrows+bias runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(dims.xLen, "Vulkan matrows+bias runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(dims.wLen, "Vulkan matrows+bias runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan matrows+bias runner bias")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows+bias runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.weightBuffer(bias[:rows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, xs, batches, cols); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandBatches != batches || r.commandRows != rows || r.commandCols != cols || r.commandAddRows != 0 {
		if err := r.recordCommand(batches, rows, cols); err != nil {
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
	return r.vk.readRowsPrefixInto(r.device, r.outBuf, out, batches, rows)
}

func (r *vulkanMatRowsBiasF32WinRunner) runAddRows(out, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	addRows := len(add)
	dims, err := checkedMatRowsBiasDimsWin(batches, addRows, rows, cols, "Vulkan matrows+bias+addrows runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(dims.xLen, "Vulkan matrows+bias+addrows runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(dims.wLen, "Vulkan matrows+bias+addrows runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan matrows+bias+addrows runner bias")
	if err != nil {
		return err
	}
	addBytes, err := checkedFloat32ByteLenErrWin(dims.addLen, "Vulkan matrows+bias+addrows runner add")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows+bias+addrows runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, addBytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.weightBuffer(bias[:rows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, xs, batches, cols); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.addBuf, add, addRows, rows); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: r.addBuf.buffer, Range: addBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandBatches != batches || r.commandRows != rows || r.commandCols != cols || r.commandAddRows != addRows {
		if err := r.recordAddRowsCommand(batches, rows, cols, addRows); err != nil {
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
	return r.vk.readRowsPrefixInto(r.device, r.outBuf, out, batches, rows)
}

func (r *vulkanMatRowsBiasF32WinRunner) recordCommand(batches, rows, cols int) error {
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
	var pc [12]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), uintptr(batches), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandRows = rows
	r.commandCols = cols
	r.commandAddRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBiasF32WinRunner) recordAddRowsCommand(batches, rows, cols, addRows int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.addPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(addRows))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), uintptr(batches), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandRows = rows
	r.commandCols = cols
	r.commandAddRows = addRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBiasF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanMatRowsBiasF32WinRunner) weightBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func vulkanMatRowsBiasF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatRowsBiasF32SPV.once.Do(func() {
		vulkanMatRowsBiasF32SPV.code, vulkanMatRowsBiasF32SPV.err = compileVulkanGLSLWindows(vulkanMatRowsBiasF32GLSL)
	})
	return vulkanMatRowsBiasF32SPV.code, vulkanMatRowsBiasF32SPV.err
}

func vulkanMatRowsBiasAddRowsF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatRowsBiasAddRowsF32SPV.once.Do(func() {
		vulkanMatRowsBiasAddRowsF32SPV.code, vulkanMatRowsBiasAddRowsF32SPV.err = compileVulkanGLSLWindows(vulkanMatRowsBiasAddRowsF32GLSL)
	})
	return vulkanMatRowsBiasAddRowsF32SPV.code, vulkanMatRowsBiasAddRowsF32SPV.err
}

const vulkanMatRowsBiasF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * x[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] + bias[row];
}`

const vulkanMatRowsBiasAddRowsF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint addRows; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
layout(set=0,binding=4) readonly buffer A { float add[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * x[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    uint addRow = batch % pc.addRows;
    outv[batch * pc.rows + row] = scratch[0] + bias[row] + add[addRow * pc.rows + row];
  }
}`
