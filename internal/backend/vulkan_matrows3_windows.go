//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanMatRowsBias3F32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatRowsBias3F32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsBias3F32WinRunner
	err    error
}

func VulkanMatRowsBias3F32(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias3 shape batches=%d rowsA=%d rowsB=%d rowsC=%d cols=%d", batches, rowsA, rowsB, rowsC, cols)
	}
	dims, err := checkedMatRowsBias3DimsWin(batches, rowsA, rowsB, rowsC, cols, "Vulkan matrows+bias3")
	if err != nil {
		return err
	}
	if len(outA) < batches || len(outB) < batches || len(outC) < batches ||
		len(wa) < dims.waLen || len(ba) < rowsA ||
		len(wb) < dims.wbLen || len(bb) < rowsB ||
		len(wc) < dims.wcLen || len(bc) < rowsC {
		return fmt.Errorf("invalid Vulkan matrows+bias3 buffers batches=%d outA=%d outB=%d outC=%d wa=%d ba=%d wb=%d bb=%d wc=%d bc=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			batches, len(outA), len(outB), len(outC), len(wa), len(ba), len(wb), len(bb), len(wc), len(bc), rowsA, rowsB, rowsC, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(outA[i]) < rowsA || len(outB[i]) < rowsB || len(outC[i]) < rowsC {
			return fmt.Errorf("invalid Vulkan matrows+bias3 row %d x=%d outA=%d outB=%d outC=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
				i, len(xs[i]), len(outA[i]), len(outB[i]), len(outC[i]), rowsA, rowsB, rowsC, cols)
		}
	}
	runner, err := getVulkanMatRowsBias3F32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, xs, wa[:dims.waLen], ba[:rowsA], wb[:dims.wbLen], bb[:rowsB], wc[:dims.wcLen], bc[:rowsC], rowsA, rowsB, rowsC, cols)
}

func getVulkanMatRowsBias3F32RunnerWindows() (*vulkanMatRowsBias3F32WinRunner, error) {
	vulkanMatRowsBias3F32RunnerCache.once.Do(func() {
		vulkanMatRowsBias3F32RunnerCache.runner, vulkanMatRowsBias3F32RunnerCache.err = newVulkanMatRowsBias3F32WinRunner()
	})
	return vulkanMatRowsBias3F32RunnerCache.runner, vulkanMatRowsBias3F32RunnerCache.err
}

type vulkanMatRowsBias3F32WinRunner struct {
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
	outABuf         vkHostBufferWin
	outBBuf         vkHostBufferWin
	outCBuf         vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [10]vulkanDescriptorBindingWin
	commandRecorded bool
	commandBatches  int
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	sharedDevice    bool
	mu              sync.Mutex
}

func newVulkanMatRowsBias3F32WinRunner() (*vulkanMatRowsBias3F32WinRunner, error) {
	spv, err := vulkanMatRowsBias3F32ShaderCodeWindows()
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
	r := &vulkanMatRowsBias3F32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), biasBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
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

func (r *vulkanMatRowsBias3F32WinRunner) destroy() {
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
	r.vk.destroyBuffer(r.device, r.outABuf)
	r.vk.destroyBuffer(r.device, r.outBBuf)
	r.vk.destroyBuffer(r.device, r.outCBuf)
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

func (r *vulkanMatRowsBias3F32WinRunner) run(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	dims, err := checkedMatRowsBias3DimsWin(batches, rowsA, rowsB, rowsC, cols, "Vulkan matrows+bias3 runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(dims.xLen, "Vulkan matrows+bias3 runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrWin(dims.waLen, "Vulkan matrows+bias3 runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrWin(dims.wbLen, "Vulkan matrows+bias3 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrWin(dims.wcLen, "Vulkan matrows+bias3 runner wc")
	if err != nil {
		return err
	}
	baBytes, err := checkedFloat32ByteLenErrWin(rowsA, "Vulkan matrows+bias3 runner ba")
	if err != nil {
		return err
	}
	bbBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan matrows+bias3 runner bb")
	if err != nil {
		return err
	}
	bcBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan matrows+bias3 runner bc")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrWin(dims.outALen, "Vulkan matrows+bias3 runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrWin(dims.outBLen, "Vulkan matrows+bias3 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrWin(dims.outCLen, "Vulkan matrows+bias3 runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	waBuf, err := r.cachedBuffer(wa[:dims.waLen], waBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	baBuf, err := r.cachedBuffer(ba[:rowsA], baBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	wbBuf, err := r.cachedBuffer(wb[:dims.wbLen], wbBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	bbBuf, err := r.cachedBuffer(bb[:rowsB], bbBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	wcBuf, err := r.cachedBuffer(wc[:dims.wcLen], wcBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	bcBuf, err := r.cachedBuffer(bc[:rowsC], bcBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeRowsPrefix(r.device, r.xBuf, xs, batches, cols); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: waBuf.buffer, Range: waBytes},
		{Buffer: baBuf.buffer, Range: baBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: bbBuf.buffer, Range: bbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: bcBuf.buffer, Range: bcBytes},
		{Buffer: r.outABuf.buffer, Range: outABytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outCBuf.buffer, Range: outCBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandBatches != batches || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(batches, rowsA, rowsB, rowsC, cols); err != nil {
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
	if err := r.vk.readRowsPrefixInto(r.device, r.outABuf, outA, batches, rowsA); err != nil {
		return err
	}
	if err := r.vk.readRowsPrefixInto(r.device, r.outBBuf, outB, batches, rowsB); err != nil {
		return err
	}
	if err := r.vk.readRowsPrefixInto(r.device, r.outCBuf, outC, batches, rowsC); err != nil {
		return err
	}
	return nil
}

func (r *vulkanMatRowsBias3F32WinRunner) recordCommand(batches, rowsA, rowsB, rowsC, cols int) error {
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
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsA))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(cols))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rowsA+rowsB+rowsC), uintptr(batches), 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandBatches = batches
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBias3F32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanMatRowsBias3F32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func vulkanMatRowsBias3F32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatRowsBias3F32SPV.once.Do(func() {
		vulkanMatRowsBias3F32SPV.code, vulkanMatRowsBias3F32SPV.err = compileVulkanGLSLWindows(vulkanMatRowsBias3F32GLSL)
	})
	return vulkanMatRowsBias3F32SPV.code, vulkanMatRowsBias3F32SPV.err
}

const vulkanMatRowsBias3F32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rowsA; uint rowsB; uint rowsC; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer WA { float wa[]; };
layout(set=0,binding=2) readonly buffer BA { float ba[]; };
layout(set=0,binding=3) readonly buffer WB { float wb[]; };
layout(set=0,binding=4) readonly buffer BB { float bb[]; };
layout(set=0,binding=5) readonly buffer WC { float wc[]; };
layout(set=0,binding=6) readonly buffer BC { float bc[]; };
layout(set=0,binding=7) writeonly buffer OA { float oa[]; };
layout(set=0,binding=8) writeonly buffer OB { float ob[]; };
layout(set=0,binding=9) writeonly buffer OC { float oc[]; };
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
