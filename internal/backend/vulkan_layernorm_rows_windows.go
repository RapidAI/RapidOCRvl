//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanLayerNormRowsF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

type vulkanLayerNormRowsF32WinRunner struct {
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
	addBuf          vkHostBufferWin
	outBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [5]vulkanDescriptorBindingWin
	commandRecorded bool
	commandRows     int
	commandCols     int
	commandMode     int
	commandEps      uint32
	sharedDevice    bool
	mu              sync.Mutex
}

const (
	vulkanLayerNormRowsModePlain = iota
	vulkanLayerNormRowsModeAdd
)

var vulkanLayerNormRowsF32RunnerCache struct {
	once   sync.Once
	runner *vulkanLayerNormRowsF32WinRunner
	err    error
}

func VulkanLayerNormRowsF32(out, x [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return vulkanLayerNormRowsF32(out, x, nil, weight, bias, rows, cols, vulkanLayerNormRowsModePlain, eps)
}

func VulkanAddThenLayerNormRowsF32(out, x, add [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return vulkanLayerNormRowsF32(out, x, add, weight, bias, rows, cols, vulkanLayerNormRowsModeAdd, eps)
}

func vulkanLayerNormRowsF32(out, x, add [][]float32, weight, bias []float32, rows, cols, mode int, eps float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan layernorm rows shape rows=%d cols=%d", rows, cols)
	}
	if len(out) < rows || len(x) < rows || len(weight) < cols || len(bias) < cols {
		return fmt.Errorf("invalid Vulkan layernorm rows buffers out=%d x=%d weight=%d bias=%d rows=%d cols=%d", len(out), len(x), len(weight), len(bias), rows, cols)
	}
	if mode == vulkanLayerNormRowsModeAdd && len(add) < rows {
		return fmt.Errorf("invalid Vulkan add+layernorm rows add=%d rows=%d", len(add), rows)
	}
	for i := 0; i < rows; i++ {
		if len(out[i]) < cols || len(x[i]) < cols || (mode == vulkanLayerNormRowsModeAdd && len(add[i]) < cols) {
			addLen := 0
			if mode == vulkanLayerNormRowsModeAdd && i < len(add) {
				addLen = len(add[i])
			}
			return fmt.Errorf("invalid Vulkan layernorm row %d out=%d x=%d add=%d cols=%d", i, len(out[i]), len(x[i]), addLen, cols)
		}
	}
	runner, err := getVulkanLayerNormRowsF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, x, add, weight[:cols], bias[:cols], rows, cols, mode, eps)
}

func getVulkanLayerNormRowsF32RunnerWindows() (*vulkanLayerNormRowsF32WinRunner, error) {
	vulkanLayerNormRowsF32RunnerCache.once.Do(func() {
		vulkanLayerNormRowsF32RunnerCache.runner, vulkanLayerNormRowsF32RunnerCache.err = newVulkanLayerNormRowsF32WinRunner()
	})
	return vulkanLayerNormRowsF32RunnerCache.runner, vulkanLayerNormRowsF32RunnerCache.err
}

func newVulkanLayerNormRowsF32WinRunner() (*vulkanLayerNormRowsF32WinRunner, error) {
	spv, err := vulkanLayerNormRowsF32ShaderCodeWindows()
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
	r := &vulkanLayerNormRowsF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
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

func (r *vulkanLayerNormRowsF32WinRunner) run(out, x, add [][]float32, weight, bias []float32, rows, cols, mode int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	bufLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return fmt.Errorf("Vulkan layernorm rows runner buffer length overflows: rows=%d cols=%d", rows, cols)
	}
	bufBytes, err := checkedFloat32ByteLenErrWin(bufLen, "Vulkan layernorm rows runner buffer")
	if err != nil {
		return err
	}
	paramBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan layernorm rows runner params")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	weightBuf, err := cachedFloat32BufferWin(r.vk, r.device, r.memProps, weight[:cols], paramBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := cachedFloat32BufferWin(r.vk, r.device, r.memProps, bias[:cols], paramBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, x, rows, cols); err != nil {
		return err
	}
	if mode == vulkanLayerNormRowsModeAdd {
		if err := r.vk.writeRowsPrefix(r.device, r.addBuf, add, rows, cols); err != nil {
			return err
		}
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: bufBytes},
		{Buffer: r.addBuf.buffer, Range: bufBytes},
		{Buffer: weightBuf.buffer, Range: paramBytes},
		{Buffer: biasBuf.buffer, Range: paramBytes},
		{Buffer: r.outBuf.buffer, Range: bufBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandRows != rows || r.commandCols != cols || r.commandMode != mode || r.commandEps != epsBits {
		if err := r.recordCommand(rows, cols, mode, eps); err != nil {
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
	return r.vk.readRowsPrefixInto(r.device, r.outBuf, out, rows, cols)
}

func (r *vulkanLayerNormRowsF32WinRunner) recordCommand(rows, cols, mode int, eps float32) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(mode))
	binary.LittleEndian.PutUint32(pc[12:16], math.Float32bits(eps))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandMode = mode
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanLayerNormRowsF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanLayerNormRowsF32WinRunner) destroy() {
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
	r.vk.destroyBuffer(r.device, r.addBuf)
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

func vulkanLayerNormRowsF32ShaderCodeWindows() ([]uint32, error) {
	vulkanLayerNormRowsF32SPV.once.Do(func() {
		vulkanLayerNormRowsF32SPV.code, vulkanLayerNormRowsF32SPV.err = compileVulkanGLSLWindows(vulkanLayerNormRowsF32GLSL)
	})
	return vulkanLayerNormRowsF32SPV.code, vulkanLayerNormRowsF32SPV.err
}

const vulkanLayerNormRowsF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; uint mode; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer A { float addv[]; };
layout(set=0,binding=2) readonly buffer W { float weight[]; };
layout(set=0,binding=3) readonly buffer B { float bias[]; };
layout(set=0,binding=4) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint base = row * pc.cols;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    sum += v;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0u) mean = scratch[0] / float(pc.cols);
  barrier();
  float varSum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    float d = v - mean;
    varSum += d * d;
  }
  scratch[lid] = varSum;
  barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  float scale = inversesqrt(scratch[0] / float(pc.cols) + pc.eps);
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    outv[base + c] = (v - mean) * scale * weight[c] + bias[c];
  }
}`
