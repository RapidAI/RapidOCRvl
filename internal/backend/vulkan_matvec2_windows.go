//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanFusedMatVec2F32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanFusedMatVec2F32RunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec2F32WinRunner
	err    error
}

func VulkanFusedMatVec2F32(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) error {
	if rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan fused matvec2 shape rowsB=%d rowsC=%d cols=%d", rowsB, rowsC, cols)
	}
	if len(outB) < rowsB || len(outC) < rowsC || len(x) < cols || len(wb) < rowsB*cols || len(wc) < rowsC*cols {
		return fmt.Errorf("invalid Vulkan fused matvec2 buffers outB=%d outC=%d x=%d wb=%d wc=%d rowsB=%d rowsC=%d cols=%d",
			len(outB), len(outC), len(x), len(wb), len(wc), rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec2F32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(outB, outC, x, wb, wc, rowsB, rowsC, cols)
}

func VulkanFusedMatVec2MRoPEF32(outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsB, rowsC, cols, kvHeads, headDim int) error {
	if rowsB <= 0 || rowsC <= 0 || cols <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope shape rowsB=%d rowsC=%d cols=%d kvHeads=%d headDim=%d", rowsB, rowsC, cols, kvHeads, headDim)
	}
	if rowsB != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope rows rowsB=%d want=%d", rowsB, kvHeads*headDim)
	}
	half := headDim / 2
	if len(outB) < rowsB || len(outC) < rowsC || len(x) < cols || len(wb) < rowsB*cols || len(wc) < rowsC*cols || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope buffers outB=%d outC=%d x=%d wa=%d wb=%d wc=%d cos=%d sin=%d rowsB=%d rowsC=%d cols=%d",
			len(outB), len(outC), len(x), len(wa), len(wb), len(wc), len(cosTable), len(sinTable), rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec3MRoPEF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run2MRoPE(outB, outC, x, wb, wc, cosTable, sinTable, rowsB, rowsC, cols, kvHeads, headDim)
}

func getVulkanFusedMatVec2F32RunnerWindows() (*vulkanFusedMatVec2F32WinRunner, error) {
	vulkanFusedMatVec2F32RunnerCache.once.Do(func() {
		vulkanFusedMatVec2F32RunnerCache.runner, vulkanFusedMatVec2F32RunnerCache.err = newVulkanFusedMatVec2F32WinRunner()
	})
	return vulkanFusedMatVec2F32RunnerCache.runner, vulkanFusedMatVec2F32RunnerCache.err
}

type vulkanFusedMatVec2F32WinRunner struct {
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
	outBBuf         vkHostBufferWin
	outCBuf         vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [5]vulkanDescriptorBindingWin
	commandRecorded bool
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanFusedMatVec2F32WinRunner() (*vulkanFusedMatVec2F32WinRunner, error) {
	spv, err := vulkanFusedMatVec2F32ShaderCodeWindows()
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
	r := &vulkanFusedMatVec2F32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
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
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 12}
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

func (r *vulkanFusedMatVec2F32WinRunner) destroy() {
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
	r.vk.destroyBuffer(r.device, r.outBBuf)
	r.vk.destroyBuffer(r.device, r.outCBuf)
	for _, b := range r.weightBuffers {
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

func (r *vulkanFusedMatVec2F32WinRunner) run(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	wbLen, err := checkedMatVecF32WeightLenWin(rowsB, cols, "Vulkan f32 fused matvec2 runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenWin(rowsC, cols, "Vulkan f32 fused matvec2 runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 fused matvec2 runner x")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrWin(wbLen, "Vulkan f32 fused matvec2 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrWin(wcLen, "Vulkan f32 fused matvec2 runner wc")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan f32 fused matvec2 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan f32 fused matvec2 runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	wbBuf, err := r.cachedBuffer(wb[:rowsB*cols], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.cachedBuffer(wc[:rowsC*cols], wcBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outCBuf.buffer, Range: outCBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(rowsB, rowsC, cols); err != nil {
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
	if err := r.vk.readFloat32Into(r.device, r.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.outCBuf, outC[:rowsC])
}

func (r *vulkanFusedMatVec2F32WinRunner) recordCommand(rowsB, rowsC, cols int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rowsB+rowsC), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec2F32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanFusedMatVec2F32WinRunner) cachedBuffer(data []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, r.weightBuffers)
}

func vulkanFusedMatVec2F32ShaderCodeWindows() ([]uint32, error) {
	vulkanFusedMatVec2F32SPV.once.Do(func() {
		vulkanFusedMatVec2F32SPV.code, vulkanFusedMatVec2F32SPV.err = compileVulkanGLSLWindows(vulkanFusedMatVec2F32GLSL)
	})
	return vulkanFusedMatVec2F32SPV.code, vulkanFusedMatVec2F32SPV.err
}

const vulkanFusedMatVec2F32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rowsB; uint rowsC; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer WB { float wb[]; };
layout(set=0,binding=2) readonly buffer WC { float wc[]; };
layout(set=0,binding=3) writeonly buffer OB { float outB[]; };
layout(set=0,binding=4) writeonly buffer OC { float outC[]; };
shared float scratch[256];
void main() {
  uint globalRow = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  bool isC = globalRow >= pc.rowsB;
  uint row = isC ? globalRow - pc.rowsB : globalRow;
  float sum = 0.0;
  uint base = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    float xv = x[c];
    sum += (isC ? wc[base + c] : wb[base + c]) * xv;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    if (isC) outC[row] = scratch[0];
    else outB[row] = scratch[0];
  }
}`
