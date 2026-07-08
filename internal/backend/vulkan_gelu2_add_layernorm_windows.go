//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanMatRowsGELU2AddLayerNormF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatRowsGELU2AddLayerNormF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsGELU2AddLayerNormF32WinRunner
	err    error
}

func VulkanMatRowsGELU2AddLayerNormF32(out, x, residual [][]float32, w1, b1, w2, b2, normW, normB []float32, hiddenRows, cols, outRows int, eps float32) error {
	batches := len(x)
	if batches == 0 {
		return nil
	}
	if hiddenRows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm shape batches=%d hiddenRows=%d cols=%d outRows=%d", batches, hiddenRows, cols, outRows)
	}
	dims, err := checkedMatRowsGELU2DimsWin(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2+add layernorm")
	if err != nil {
		return err
	}
	if len(out) < batches || len(residual) < batches || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows || len(normW) < outRows || len(normB) < outRows {
		return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm buffers out=%d x=%d residual=%d w1=%d b1=%d w2=%d b2=%d normW=%d normB=%d hiddenRows=%d cols=%d outRows=%d",
			len(out), len(x), len(residual), len(w1), len(b1), len(w2), len(b2), len(normW), len(normB), hiddenRows, cols, outRows)
	}
	for i := 0; i < batches; i++ {
		if len(x[i]) < cols || len(residual[i]) < outRows || len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm row %d out=%d x=%d residual=%d cols=%d outRows=%d", i, len(out[i]), len(x[i]), len(residual[i]), cols, outRows)
		}
	}
	runner, err := getVulkanMatRowsGELU2AddLayerNormF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, x, residual, w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], normW[:outRows], normB[:outRows], hiddenRows, cols, outRows, eps)
}

func getVulkanMatRowsGELU2AddLayerNormF32RunnerWindows() (*vulkanMatRowsGELU2AddLayerNormF32WinRunner, error) {
	vulkanMatRowsGELU2AddLayerNormF32RunnerCache.once.Do(func() {
		vulkanMatRowsGELU2AddLayerNormF32RunnerCache.runner, vulkanMatRowsGELU2AddLayerNormF32RunnerCache.err = newVulkanMatRowsGELU2AddLayerNormF32WinRunner()
	})
	return vulkanMatRowsGELU2AddLayerNormF32RunnerCache.runner, vulkanMatRowsGELU2AddLayerNormF32RunnerCache.err
}

type vulkanMatRowsGELU2AddLayerNormF32WinRunner struct {
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
	residualBuf     vkHostBufferWin
	hiddenBuf       vkHostBufferWin
	mlpBuf          vkHostBufferWin
	outBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [11]vulkanDescriptorBindingWin
	commandRecorded bool
	commandBatches  int
	commandHidden   int
	commandCols     int
	commandOutRows  int
	commandEps      uint32
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanMatRowsGELU2AddLayerNormF32WinRunner() (*vulkanMatRowsGELU2AddLayerNormF32WinRunner, error) {
	spv, err := vulkanMatRowsGELU2AddLayerNormF32ShaderCodeWindows()
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
	r := &vulkanMatRowsGELU2AddLayerNormF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 11)
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
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 24}
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

func (r *vulkanMatRowsGELU2AddLayerNormF32WinRunner) run(out, x, residual [][]float32, w1, b1, w2, b2, normW, normB []float32, hiddenRows, cols, outRows int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(x)
	dims, err := checkedMatRowsGELU2DimsWin(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2+add layernorm runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(dims.xLen, "Vulkan matrows gelu2+add layernorm runner x")
	if err != nil {
		return err
	}
	residualBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows gelu2+add layernorm runner residual")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrWin(dims.hiddenLen, "Vulkan matrows gelu2+add layernorm runner hidden")
	if err != nil {
		return err
	}
	mlpBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows gelu2+add layernorm runner mlp")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows gelu2+add layernorm runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrWin(dims.w1Len, "Vulkan matrows gelu2+add layernorm runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrWin(hiddenRows, "Vulkan matrows gelu2+add layernorm runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrWin(dims.w2Len, "Vulkan matrows gelu2+add layernorm runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan matrows gelu2+add layernorm runner b2")
	if err != nil {
		return err
	}
	normBytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan matrows gelu2+add layernorm runner norm")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, residualBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.mlpBuf, mlpBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
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
	normWBuf, err := r.cachedBuffer(normW, normBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	normBBuf, err := r.cachedBuffer(normB, normBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, x, batches, cols); err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.residualBuf, residual, batches, outRows); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: r.residualBuf.buffer, Range: residualBytes},
		{Buffer: w1Buf.buffer, Range: w1Bytes},
		{Buffer: b1Buf.buffer, Range: b1Bytes},
		{Buffer: w2Buf.buffer, Range: w2Bytes},
		{Buffer: b2Buf.buffer, Range: b2Bytes},
		{Buffer: normWBuf.buffer, Range: normBytes},
		{Buffer: normBBuf.buffer, Range: normBytes},
		{Buffer: r.hiddenBuf.buffer, Range: hiddenBytes},
		{Buffer: r.mlpBuf.buffer, Range: mlpBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandBatches != batches || r.commandHidden != hiddenRows || r.commandCols != cols || r.commandOutRows != outRows || r.commandEps != epsBits {
		if err := r.recordCommand(batches, hiddenRows, cols, outRows, eps); err != nil {
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

func (r *vulkanMatRowsGELU2AddLayerNormF32WinRunner) recordCommand(batches, hiddenRows, cols, outRows int, eps float32) error {
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
	var pc [24]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	binary.LittleEndian.PutUint32(pc[20:24], math.Float32bits(eps))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hiddenRows), uintptr(batches), 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(pc[16:20], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), uintptr(batches), 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(pc[16:20], 2)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(batches), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandHidden = hiddenRows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsGELU2AddLayerNormF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanMatRowsGELU2AddLayerNormF32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func (r *vulkanMatRowsGELU2AddLayerNormF32WinRunner) destroy() {
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
	r.vk.destroyBuffer(r.device, r.residualBuf)
	r.vk.destroyBuffer(r.device, r.hiddenBuf)
	r.vk.destroyBuffer(r.device, r.mlpBuf)
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

func vulkanMatRowsGELU2AddLayerNormF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatRowsGELU2AddLayerNormF32SPV.once.Do(func() {
		vulkanMatRowsGELU2AddLayerNormF32SPV.code, vulkanMatRowsGELU2AddLayerNormF32SPV.err = compileVulkanGLSLWindows(vulkanMatRowsGELU2AddLayerNormF32GLSL)
	})
	return vulkanMatRowsGELU2AddLayerNormF32SPV.code, vulkanMatRowsGELU2AddLayerNormF32SPV.err
}

const vulkanMatRowsGELU2AddLayerNormF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint hiddenRows; uint cols; uint outRows; uint stage; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer R { float residual[]; };
layout(set=0,binding=2) readonly buffer W1 { float w1[]; };
layout(set=0,binding=3) readonly buffer B1 { float b1[]; };
layout(set=0,binding=4) readonly buffer W2 { float w2[]; };
layout(set=0,binding=5) readonly buffer B2 { float b2[]; };
layout(set=0,binding=6) readonly buffer NW { float normW[]; };
layout(set=0,binding=7) readonly buffer NB { float normB[]; };
layout(set=0,binding=8) buffer H { float hidden[]; };
layout(set=0,binding=9) buffer M { float mlp[]; };
layout(set=0,binding=10) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  if (pc.stage == 0u) {
    uint xBase = batch * pc.cols;
    uint wBase = row * pc.cols;
    for (uint c = lid; c < pc.cols; c += 256u) sum += w1[wBase + c] * x[xBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else if (pc.stage == 1u) {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256u) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mlp[batch * pc.outRows + row] = scratch[0] + b2[row];
  } else {
    uint base = row * pc.outRows;
    for (uint c = lid; c < pc.outRows; c += 256u) sum += residual[base + c] + mlp[base + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mean = scratch[0] / float(pc.outRows);
    barrier();
    float varSum = 0.0;
    for (uint c = lid; c < pc.outRows; c += 256u) {
      float v = residual[base + c] + mlp[base + c];
      float d = v - mean;
      varSum += d * d;
    }
    scratch[lid] = varSum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    float scale = inversesqrt(scratch[0] / float(pc.outRows) + pc.eps);
    for (uint c = lid; c < pc.outRows; c += 256u) {
      float v = residual[base + c] + mlp[base + c];
      outv[base + c] = (v - mean) * scale * normW[c] + normB[c];
    }
  }
}`
