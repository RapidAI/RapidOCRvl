//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanProjectImageF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanProjectImageF32RunnerCache struct {
	once   sync.Once
	runner *vulkanProjectImageF32WinRunner
	err    error
}

func VulkanProjectImageF32(out, x [][]float32, normW, normB, w1, b1, w2, b2 []float32, gridT, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	if gridT <= 0 || gridH < 2 || gridW < 2 || gridH%2 != 0 || gridW%2 != 0 || visionDim <= 0 || hiddenRows <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan project image shape gridT=%d gridH=%d gridW=%d visionDim=%d hiddenRows=%d outRows=%d", gridT, gridH, gridW, visionDim, hiddenRows, outRows)
	}
	dims, err := checkedProjectImageDimsWin(gridT, gridH, gridW, visionDim, hiddenRows, outRows, "Vulkan project image")
	if err != nil {
		return err
	}
	if len(out) < dims.batches || len(x) < dims.tokens || len(normW) < visionDim || len(normB) < visionDim || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows {
		return fmt.Errorf("invalid Vulkan project image buffers out=%d x=%d normW=%d normB=%d w1=%d b1=%d w2=%d b2=%d tokens=%d batches=%d visionDim=%d hiddenRows=%d outRows=%d",
			len(out), len(x), len(normW), len(normB), len(w1), len(b1), len(w2), len(b2), dims.tokens, dims.batches, visionDim, hiddenRows, outRows)
	}
	for i := 0; i < dims.tokens; i++ {
		if len(x[i]) < visionDim {
			return fmt.Errorf("invalid Vulkan project image x row %d len=%d visionDim=%d", i, len(x[i]), visionDim)
		}
	}
	for i := 0; i < dims.batches; i++ {
		if len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan project image out row %d len=%d outRows=%d", i, len(out[i]), outRows)
		}
	}
	runner, err := getVulkanProjectImageF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, x[:dims.tokens], normW[:visionDim], normB[:visionDim], w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], dims.batches, gridH, gridW, visionDim, hiddenRows, outRows, eps)
}

type vulkanProjectImageDimsWin struct {
	tokens  int
	batches int
	cols    int
	w1Len   int
	w2Len   int
}

func checkedProjectImageDimsWin(gridT, gridH, gridW, visionDim, hiddenRows, outRows int, label string) (vulkanProjectImageDimsWin, error) {
	gridHW, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s token count overflows: gridH=%d gridW=%d", label, gridH, gridW)
	}
	tokens, ok := checkedMulInt(gridT, gridHW)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s token count overflows: gridT=%d gridH=%d gridW=%d", label, gridT, gridH, gridW)
	}
	blocksHW, ok := checkedMulInt(gridH/2, gridW/2)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s batch count overflows: gridH=%d gridW=%d", label, gridH, gridW)
	}
	batches, ok := checkedMulInt(gridT, blocksHW)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s batch count overflows: gridT=%d gridH=%d gridW=%d", label, gridT, gridH, gridW)
	}
	cols, ok := checkedMulInt(visionDim, 4)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s merged cols overflow: visionDim=%d", label, visionDim)
	}
	w1Len, ok := checkedMulInt(hiddenRows, cols)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s w1 length overflows: hiddenRows=%d cols=%d", label, hiddenRows, cols)
	}
	w2Len, ok := checkedMulInt(outRows, hiddenRows)
	if !ok {
		return vulkanProjectImageDimsWin{}, fmt.Errorf("%s w2 length overflows: outRows=%d hiddenRows=%d", label, outRows, hiddenRows)
	}
	return vulkanProjectImageDimsWin{tokens: tokens, batches: batches, cols: cols, w1Len: w1Len, w2Len: w2Len}, nil
}

func getVulkanProjectImageF32RunnerWindows() (*vulkanProjectImageF32WinRunner, error) {
	vulkanProjectImageF32RunnerCache.once.Do(func() {
		vulkanProjectImageF32RunnerCache.runner, vulkanProjectImageF32RunnerCache.err = newVulkanProjectImageF32WinRunner()
	})
	return vulkanProjectImageF32RunnerCache.runner, vulkanProjectImageF32RunnerCache.err
}

type vulkanProjectImageF32WinRunner struct {
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
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	mergedBuf       vkHostBufferWin
	hiddenBuf       vkHostBufferWin
	outBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [10]vulkanDescriptorBindingWin
	commandRecorded bool
	commandBatches  int
	commandGridH    int
	commandGridW    int
	commandVision   int
	commandHidden   int
	commandOutRows  int
	commandEps      uint32
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanProjectImageF32WinRunner() (*vulkanProjectImageF32WinRunner, error) {
	spv, err := vulkanProjectImageF32ShaderCodeWindows()
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
	r := &vulkanProjectImageF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 10)
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
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 32}
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

func (r *vulkanProjectImageF32WinRunner) run(out, x [][]float32, normW, normB, w1, b1, w2, b2 []float32, batches, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(x)
	dims, err := checkedProjectImageDimsWin(1, gridH, gridW, visionDim, hiddenRows, outRows, "Vulkan project image runner")
	if err != nil {
		return err
	}
	if dims.tokens == 0 || dims.batches == 0 || tokens%dims.tokens != 0 || batches%dims.batches != 0 || tokens/dims.tokens != batches/dims.batches {
		return fmt.Errorf("invalid Vulkan project image runner tokens=%d batches=%d gridH=%d gridW=%d", tokens, batches, gridH, gridW)
	}
	cols := dims.cols
	xElems, ok := checkedMulInt(tokens, visionDim)
	if !ok {
		return fmt.Errorf("Vulkan project image runner x length overflows: tokens=%d visionDim=%d", tokens, visionDim)
	}
	mergedElems, ok := checkedMulInt(batches, cols)
	if !ok {
		return fmt.Errorf("Vulkan project image runner merged length overflows: batches=%d cols=%d", batches, cols)
	}
	hiddenElems, ok := checkedMulInt(batches, hiddenRows)
	if !ok {
		return fmt.Errorf("Vulkan project image runner hidden length overflows: batches=%d hiddenRows=%d", batches, hiddenRows)
	}
	outElems, ok := checkedMulInt(batches, outRows)
	if !ok {
		return fmt.Errorf("Vulkan project image runner output length overflows: batches=%d outRows=%d", batches, outRows)
	}
	xBytes, err := checkedFloat32ByteLenErrWin(xElems, "Vulkan project image runner x")
	if err != nil {
		return err
	}
	paramBytes, err := checkedFloat32ByteLenErrWin(visionDim, "Vulkan project image runner norm param")
	if err != nil {
		return err
	}
	mergedBytes, err := checkedFloat32ByteLenErrWin(mergedElems, "Vulkan project image runner merged")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrWin(hiddenElems, "Vulkan project image runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outElems, "Vulkan project image runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrWin(dims.w1Len, "Vulkan project image runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrWin(hiddenRows, "Vulkan project image runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrWin(dims.w2Len, "Vulkan project image runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan project image runner b2")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.mergedBuf, mergedBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	normWBuf, err := r.cachedBuffer(normW, paramBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	normBBuf, err := r.cachedBuffer(normB, paramBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w1Buf, err := r.cachedBuffer(w1, w1Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b1Buf, err := r.cachedBuffer(b1, b1Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w2Buf, err := r.cachedBuffer(w2, w2Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b2Buf, err := r.cachedBuffer(b2, b2Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, x, tokens, visionDim); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: normWBuf.buffer, Range: paramBytes},
		{Buffer: normBBuf.buffer, Range: paramBytes},
		{Buffer: w1Buf.buffer, Range: w1Bytes},
		{Buffer: b1Buf.buffer, Range: b1Bytes},
		{Buffer: w2Buf.buffer, Range: w2Bytes},
		{Buffer: b2Buf.buffer, Range: b2Bytes},
		{Buffer: r.mergedBuf.buffer, Range: mergedBytes},
		{Buffer: r.hiddenBuf.buffer, Range: hiddenBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandBatches != batches || r.commandGridH != gridH || r.commandGridW != gridW || r.commandVision != visionDim || r.commandHidden != hiddenRows || r.commandOutRows != outRows || r.commandEps != epsBits {
		if err := r.recordCommand(batches, gridH, gridW, visionDim, hiddenRows, outRows, eps); err != nil {
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
	return r.vk.readRowsPrefixInto(r.device, r.outBuf, out, batches, outRows)
}

func (r *vulkanProjectImageF32WinRunner) recordCommand(batches, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
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
	var pc [32]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(visionDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[20:24], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[24:28], 0)
	binary.LittleEndian.PutUint32(pc[28:32], math.Float32bits(eps))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 4, uintptr(batches), 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(pc[24:28], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hiddenRows), uintptr(batches), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(pc[24:28], 2)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), uintptr(batches), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandGridH = gridH
	r.commandGridW = gridW
	r.commandVision = visionDim
	r.commandHidden = hiddenRows
	r.commandOutRows = outRows
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanProjectImageF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanProjectImageF32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func (r *vulkanProjectImageF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.mergedBuf)
	r.vk.destroyBuffer(r.device, r.hiddenBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
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

func vulkanProjectImageF32ShaderCodeWindows() ([]uint32, error) {
	vulkanProjectImageF32SPV.once.Do(func() {
		vulkanProjectImageF32SPV.code, vulkanProjectImageF32SPV.err = compileVulkanGLSLWindows(vulkanProjectImageF32GLSL)
	})
	return vulkanProjectImageF32SPV.code, vulkanProjectImageF32SPV.err
}

const vulkanProjectImageF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint gridH; uint gridW; uint vd; uint hiddenRows; uint outRows; uint stage; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer NW { float normW[]; };
layout(set=0,binding=2) readonly buffer NB { float normB[]; };
layout(set=0,binding=3) readonly buffer W1 { float w1[]; };
layout(set=0,binding=4) readonly buffer B1 { float b1[]; };
layout(set=0,binding=5) readonly buffer W2 { float w2[]; };
layout(set=0,binding=6) readonly buffer B2 { float b2[]; };
layout(set=0,binding=7) buffer M { float merged[]; };
layout(set=0,binding=8) buffer H { float hidden[]; };
layout(set=0,binding=9) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
uint patch_base(uint batch, uint patchIdx) {
  uint blocksW = pc.gridW >> 1;
  uint blocksPerT = (pc.gridH >> 1) * blocksW;
  uint t = batch / blocksPerT;
  uint local = batch - t * blocksPerT;
  uint by = local / blocksW;
  uint bx = local - by * blocksW;
  uint base = t * pc.gridH * pc.gridW + by * 2u * pc.gridW + bx * 2u;
  if (patchIdx == 1u) return base + 1u;
  if (patchIdx == 2u) return base + pc.gridW;
  if (patchIdx == 3u) return base + pc.gridW + 1u;
  return base;
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  if (pc.stage == 0u) {
    uint patchIdx = row;
    uint srcBase = patch_base(batch, patchIdx) * pc.vd;
    float sum = 0.0;
    for (uint c = lid; c < pc.vd; c += 256u) sum += x[srcBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mean = scratch[0] / float(pc.vd);
    barrier();
    float varSum = 0.0;
    for (uint c = lid; c < pc.vd; c += 256u) {
      float d = x[srcBase + c] - mean;
      varSum += d * d;
    }
    scratch[lid] = varSum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    float scale = inversesqrt(scratch[0] / float(pc.vd) + pc.eps);
    uint dstBase = batch * pc.vd * 4u + patchIdx * pc.vd;
    for (uint c = lid; c < pc.vd; c += 256u) {
      float v = x[srcBase + c];
      merged[dstBase + c] = (v - mean) * scale * normW[c] + normB[c];
    }
    return;
  }
  float sum = 0.0;
  if (pc.stage == 1u) {
    uint cols = pc.vd * 4u;
    uint mBase = batch * cols;
    uint wBase = row * cols;
    for (uint c = lid; c < cols; c += 256u) sum += w1[wBase + c] * merged[mBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256u) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) outv[batch * pc.outRows + row] = scratch[0] + b2[row];
  }
}`
