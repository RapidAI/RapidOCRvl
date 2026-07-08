//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanMatRowsGELU2F32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatRowsGELU2F32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsGELU2F32WinRunner
	err    error
}

func VulkanMatRowsGELU2F32(out, xs [][]float32, w1, b1, w2, b2 []float32, hiddenRows, cols, outRows int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if hiddenRows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows gelu2 shape batches=%d hiddenRows=%d cols=%d outRows=%d", batches, hiddenRows, cols, outRows)
	}
	dims, err := checkedMatRowsGELU2DimsWin(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2")
	if err != nil {
		return err
	}
	if len(out) < batches || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows {
		return fmt.Errorf("invalid Vulkan matrows gelu2 buffers out=%d xs=%d w1=%d b1=%d w2=%d b2=%d hiddenRows=%d cols=%d outRows=%d",
			len(out), len(xs), len(w1), len(b1), len(w2), len(b2), hiddenRows, cols, outRows)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan matrows gelu2 row %d out=%d x=%d cols=%d outRows=%d", i, len(out[i]), len(xs[i]), cols, outRows)
		}
	}
	runner, err := getVulkanMatRowsGELU2F32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, xs, w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], hiddenRows, cols, outRows)
}

func getVulkanMatRowsGELU2F32RunnerWindows() (*vulkanMatRowsGELU2F32WinRunner, error) {
	vulkanMatRowsGELU2F32RunnerCache.once.Do(func() {
		vulkanMatRowsGELU2F32RunnerCache.runner, vulkanMatRowsGELU2F32RunnerCache.err = newVulkanMatRowsGELU2F32WinRunner()
	})
	return vulkanMatRowsGELU2F32RunnerCache.runner, vulkanMatRowsGELU2F32RunnerCache.err
}

type vulkanMatRowsGELU2F32WinRunner struct {
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
	hiddenBuf       vkHostBufferWin
	outBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [7]vulkanDescriptorBindingWin
	commandRecorded bool
	commandBatches  int
	commandHidden   int
	commandCols     int
	commandOutRows  int
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanMatRowsGELU2F32WinRunner() (*vulkanMatRowsGELU2F32WinRunner, error) {
	spv, err := vulkanMatRowsGELU2F32ShaderCodeWindows()
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
	r := &vulkanMatRowsGELU2F32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 7)
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

func (r *vulkanMatRowsGELU2F32WinRunner) destroy() {
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

func (r *vulkanMatRowsGELU2F32WinRunner) run(out, xs [][]float32, w1, b1, w2, b2 []float32, hiddenRows, cols, outRows int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	dims, err := checkedMatRowsGELU2DimsWin(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2 runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(dims.xLen, "Vulkan matrows gelu2 runner x")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrWin(dims.hiddenLen, "Vulkan matrows gelu2 runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(dims.outLen, "Vulkan matrows gelu2 runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrWin(dims.w1Len, "Vulkan matrows gelu2 runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrWin(hiddenRows, "Vulkan matrows gelu2 runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrWin(dims.w2Len, "Vulkan matrows gelu2 runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan matrows gelu2 runner b2")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	w1Buf, err := r.cachedBuffer(w1, w1Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b1Buf, err := r.cachedBuffer(b1[:hiddenRows], b1Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w2Buf, err := r.cachedBuffer(w2, w2Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b2Buf, err := r.cachedBuffer(b2[:outRows], b2Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, xs, batches, cols); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: w1Buf.buffer, Range: w1Bytes},
		{Buffer: b1Buf.buffer, Range: b1Bytes},
		{Buffer: w2Buf.buffer, Range: w2Bytes},
		{Buffer: b2Buf.buffer, Range: b2Bytes},
		{Buffer: r.hiddenBuf.buffer, Range: hiddenBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandBatches != batches || r.commandHidden != hiddenRows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(batches, hiddenRows, cols, outRows); err != nil {
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

func (r *vulkanMatRowsGELU2F32WinRunner) recordCommand(batches, hiddenRows, cols, outRows int) error {
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
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(hiddenRows), uintptr(batches), 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(pc[16:20], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), uintptr(batches), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandHidden = hiddenRows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsGELU2F32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanMatRowsGELU2F32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func vulkanMatRowsGELU2F32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatRowsGELU2F32SPV.once.Do(func() {
		vulkanMatRowsGELU2F32SPV.code, vulkanMatRowsGELU2F32SPV.err = compileVulkanGLSLWindows(vulkanMatRowsGELU2F32GLSL)
	})
	return vulkanMatRowsGELU2F32SPV.code, vulkanMatRowsGELU2F32SPV.err
}

const vulkanMatRowsGELU2F32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint hiddenRows; uint cols; uint outRows; uint stage; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W1 { float w1[]; };
layout(set=0,binding=2) readonly buffer B1 { float b1[]; };
layout(set=0,binding=3) readonly buffer W2 { float w2[]; };
layout(set=0,binding=4) readonly buffer B2 { float b2[]; };
layout(set=0,binding=5) buffer H { float hidden[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  if (pc.stage == 0) {
    uint xBase = batch * pc.cols;
    uint wBase = row * pc.cols;
    for (uint c = lid; c < pc.cols; c += 256) sum += w1[wBase + c] * x[xBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) outv[batch * pc.outRows + row] = scratch[0] + b2[row];
  }
}`
