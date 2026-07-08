//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

var vulkanSwiGLUGateUpF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanSwiGLUGateUpF32RunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUGateUpF32WinRunner
	err    error
}

func VulkanSwiGLUGateUpF32(out, x, gate, up []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu gate/up shape rows=%d cols=%d", rows, cols)
	}
	if len(out) < rows || len(x) < cols || len(gate) < rows*cols || len(up) < rows*cols {
		return fmt.Errorf("invalid Vulkan swiglu gate/up buffers out=%d x=%d gate=%d up=%d rows=%d cols=%d", len(out), len(x), len(gate), len(up), rows, cols)
	}
	runner, err := getVulkanSwiGLUGateUpF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, x, gate, up, rows, cols)
}

func VulkanSwiGLUDownF32(out, x, gate, up, down []float32, rows, cols, outRows int) error {
	if rows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu/down shape rows=%d cols=%d outRows=%d", rows, cols, outRows)
	}
	if len(out) < outRows || len(x) < cols || len(gate) < rows*cols || len(up) < rows*cols || len(down) < outRows*rows {
		return fmt.Errorf("invalid Vulkan swiglu/down buffers out=%d x=%d gate=%d up=%d down=%d rows=%d cols=%d outRows=%d", len(out), len(x), len(gate), len(up), len(down), rows, cols, outRows)
	}
	runner, err := getVulkanSwiGLUGateUpF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runDown(out, x, gate, up, down, rows, cols, outRows)
}

func VulkanMatVecAddRMSNormF32(normOut, residual, x, w, normWeight []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matvec+add+rmsnorm shape rows=%d cols=%d", rows, cols)
	}
	if len(normOut) < rows || len(residual) < rows || len(x) < cols || len(w) < rows*cols || len(normWeight) < rows {
		return fmt.Errorf("invalid Vulkan matvec+add+rmsnorm buffers normOut=%d residual=%d x=%d w=%d normWeight=%d rows=%d cols=%d",
			len(normOut), len(residual), len(x), len(w), len(normWeight), rows, cols)
	}
	runner, err := getVulkanSwiGLUGateUpF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runMatVecAddRMSNorm(normOut, residual, x, w, normWeight, rows, cols)
}

func VulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows, true)
}

func VulkanSwiGLUDownAddRMSNormF32OutOnly(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows, false)
}

func vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int, readResidual bool) error {
	if rows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu/down+add+rmsnorm shape rows=%d cols=%d outRows=%d", rows, cols, outRows)
	}
	if len(normOut) < outRows || len(residual) < outRows || len(x) < cols || len(gate) < rows*cols || len(up) < rows*cols || len(down) < outRows*rows || len(normWeight) < outRows {
		return fmt.Errorf("invalid Vulkan swiglu/down+add+rmsnorm buffers normOut=%d residual=%d x=%d gate=%d up=%d down=%d normWeight=%d rows=%d cols=%d outRows=%d",
			len(normOut), len(residual), len(x), len(gate), len(up), len(down), len(normWeight), rows, cols, outRows)
	}
	runner, err := getVulkanSwiGLUGateUpF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runDownAddRMSNorm(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows, readResidual)
}

func getVulkanSwiGLUGateUpF32RunnerWindows() (*vulkanSwiGLUGateUpF32WinRunner, error) {
	vulkanSwiGLUGateUpF32RunnerCache.once.Do(func() {
		vulkanSwiGLUGateUpF32RunnerCache.runner, vulkanSwiGLUGateUpF32RunnerCache.err = newVulkanSwiGLUGateUpF32WinRunner()
	})
	return vulkanSwiGLUGateUpF32RunnerCache.runner, vulkanSwiGLUGateUpF32RunnerCache.err
}

type vulkanSwiGLUGateUpF32WinRunner struct {
	vk                  *vulkanWin
	instance            uintptr
	device              uintptr
	queue               uintptr
	queueFamily         uint32
	memProps            vkPhysicalDeviceMemoryProperties
	setLayout           uintptr
	downSetLayout       uintptr
	normSetLayout       uintptr
	descriptorPool      uintptr
	descriptorSet       uintptr
	downDescriptorSet   uintptr
	normDescriptorSet   uintptr
	pipelineLayout      uintptr
	pipeline            uintptr
	downPipelineLayout  uintptr
	downPipeline        uintptr
	normPipelineLayout  uintptr
	normPipeline        uintptr
	commandPool         uintptr
	commandBuffer       uintptr
	fence               uintptr
	xBuf                vkHostBufferWin
	outBuf              vkHostBufferWin
	finalOutBuf         vkHostBufferWin
	residualBuf         vkHostBufferWin
	normBuf             vkHostBufferWin
	weightBuffers       map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache     [4]vulkanDescriptorBindingWin
	downDescriptorCache [3]vulkanDescriptorBindingWin
	normDescriptorCache [4]vulkanDescriptorBindingWin
	commandRecorded     bool
	commandKind         int
	commandRows         int
	commandCols         int
	commandOutRows      int
	sharedDevice        bool
	mu                  sync.Mutex
}

const (
	vulkanSwiGLUF32CommandGateUp   = 1
	vulkanSwiGLUF32CommandDown     = 2
	vulkanSwiGLUF32CommandDownNorm = 3
	vulkanSwiGLUF32CommandMatNorm  = 4
)

func newVulkanSwiGLUGateUpF32WinRunner() (*vulkanSwiGLUGateUpF32WinRunner, error) {
	spv, err := vulkanSwiGLUGateUpF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	downSPV, err := vulkanMatVecF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanAddRMSNormF32ShaderCodeWindows()
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
	r := &vulkanSwiGLUGateUpF32WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 4)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}

	downBindings := make([]vkDescriptorSetLayoutBinding, 3)
	for i := range downBindings {
		downBindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	downDSL := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(downBindings)), PBindings: uintptr(unsafe.Pointer(&downBindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&downDSL)), 0, uintptr(unsafe.Pointer(&r.downSetLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout down: %d", int32(res))
	}

	normBindings := make([]vkDescriptorSetLayoutBinding, 4)
	for i := range normBindings {
		normBindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	normDSL := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(normBindings)), PBindings: uintptr(unsafe.Pointer(&normBindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&normDSL)), 0, uintptr(unsafe.Pointer(&r.normSetLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout norm: %d", int32(res))
	}

	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 11}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 3, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	setLayouts := []uintptr{r.setLayout, r.downSetLayout, r.normSetLayout}
	descSets := make([]uintptr, 3)
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 3, PSetLayouts: uintptr(unsafe.Pointer(&setLayouts[0]))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&descSets[0]))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	r.descriptorSet = descSets[0]
	r.downDescriptorSet = descSets[1]
	r.normDescriptorSet = descSets[2]

	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 8}
	plci := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	downPLCI := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.downSetLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&downPLCI)), 0, uintptr(unsafe.Pointer(&r.downPipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout down: %d", int32(res))
	}
	normPLCI := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.normSetLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&normPLCI)), 0, uintptr(unsafe.Pointer(&r.normPipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout norm: %d", int32(res))
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
	downSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(downSPV) * 4), PCode: uintptr(unsafe.Pointer(&downSPV[0]))}
	var downShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&downSMCI)), 0, uintptr(unsafe.Pointer(&downShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule down: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, downShader, 0)
	downStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: downShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	downCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: downStage, Layout: r.downPipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&downCPCI)), 0, uintptr(unsafe.Pointer(&r.downPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines down: %d", int32(res))
	}
	normSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(normSPV) * 4), PCode: uintptr(unsafe.Pointer(&normSPV[0]))}
	var normShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&normSMCI)), 0, uintptr(unsafe.Pointer(&normShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule norm: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, normShader, 0)
	normStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: normShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	normCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.normPipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&normCPCI)), 0, uintptr(unsafe.Pointer(&r.normPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines norm: %d", int32(res))
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

func (r *vulkanSwiGLUGateUpF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.downPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.downPipeline, 0)
	}
	if r.normPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.normPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.finalOutBuf)
	r.vk.destroyBuffer(r.device, r.residualBuf)
	r.vk.destroyBuffer(r.device, r.normBuf)
	for _, b := range r.weightBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	if r.descriptorPool != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorPool, r.device, r.descriptorPool, 0)
	}
	if r.pipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.pipelineLayout, 0)
	}
	if r.downPipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.downPipelineLayout, 0)
	}
	if r.normPipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.normPipelineLayout, 0)
	}
	if r.setLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.setLayout, 0)
	}
	if r.downSetLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.downSetLayout, 0)
	}
	if r.normSetLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.normSetLayout, 0)
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

func (r *vulkanSwiGLUGateUpF32WinRunner) run(out, x, gate, up []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan swiglu gate/up runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan swiglu gate/up runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan swiglu gate/up runner weight")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan swiglu gate/up runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:rows*cols], wBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:rows*cols], wBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}

	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: wBytes},
		{Buffer: upBuf.buffer, Range: wBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUF32CommandGateUp || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordGateUpCommand(rows, cols); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.outBuf, out[:rows])
}

func (r *vulkanSwiGLUGateUpF32WinRunner) recordGateUpCommand(rows, cols int) error {
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
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUF32CommandGateUp
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUGateUpF32WinRunner) runDown(out, x, gate, up, down []float32, rows, cols, outRows int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	gateLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan swiglu/down runner gate")
	if err != nil {
		return err
	}
	downLen, err := checkedMatVecF32WeightLenWin(outRows, rows, "Vulkan swiglu/down runner down")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan swiglu/down runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan swiglu/down runner inter")
	if err != nil {
		return err
	}
	gateBytes, err := checkedFloat32ByteLenErrWin(gateLen, "Vulkan swiglu/down runner gate")
	if err != nil {
		return err
	}
	downBytes, err := checkedFloat32ByteLenErrWin(downLen, "Vulkan swiglu/down runner down")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan swiglu/down runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalOutBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:rows*cols], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:rows*cols], gateBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.weightBuffer(down[:outRows*rows], downBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}

	swiInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: gateBytes},
		{Buffer: upBuf.buffer, Range: gateBytes},
		{Buffer: r.outBuf.buffer, Range: interBytes},
	}
	downInfos := [3]vkDescriptorBufferInfo{
		{Buffer: r.outBuf.buffer, Range: interBytes},
		{Buffer: downBuf.buffer, Range: downBytes},
		{Buffer: r.finalOutBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUF32CommandDown || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordDownCommand(rows, cols, outRows); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.finalOutBuf, out[:outRows])
}

func (r *vulkanSwiGLUGateUpF32WinRunner) runMatVecAddRMSNorm(normOut, residual, x, w, normWeight []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan swiglu gate/up runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan swiglu gate/up runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan swiglu gate/up runner weight")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan swiglu gate/up runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalOutBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, outBytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(w[:rows*cols], wBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.weightBuffer(normWeight[:rows], outBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:rows]); err != nil {
		return err
	}

	downInfos := [3]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: r.finalOutBuf.buffer, Range: outBytes},
	}
	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.residualBuf.buffer, Range: outBytes},
		{Buffer: r.finalOutBuf.buffer, Range: outBytes},
		{Buffer: normWeightBuf.buffer, Range: outBytes},
		{Buffer: r.normBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.normDescriptorSet, r.normDescriptorCache[:], normInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUF32CommandMatNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != rows {
		if err := r.recordMatVecAddRMSNormCommand(rows, cols); err != nil {
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
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:rows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:rows])
}

func (r *vulkanSwiGLUGateUpF32WinRunner) runDownAddRMSNorm(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int, readResidual bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	gateLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan swiglu/down runner gate")
	if err != nil {
		return err
	}
	downLen, err := checkedMatVecF32WeightLenWin(outRows, rows, "Vulkan swiglu/down runner down")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan swiglu/down runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan swiglu/down runner inter")
	if err != nil {
		return err
	}
	gateBytes, err := checkedFloat32ByteLenErrWin(gateLen, "Vulkan swiglu/down runner gate")
	if err != nil {
		return err
	}
	downBytes, err := checkedFloat32ByteLenErrWin(downLen, "Vulkan swiglu/down runner down")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan swiglu/down runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalOutBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:rows*cols], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:rows*cols], gateBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.weightBuffer(down[:outRows*rows], downBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.weightBuffer(normWeight[:outRows], outBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:outRows]); err != nil {
		return err
	}

	swiInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: gateBytes},
		{Buffer: upBuf.buffer, Range: gateBytes},
		{Buffer: r.outBuf.buffer, Range: interBytes},
	}
	downInfos := [3]vkDescriptorBufferInfo{
		{Buffer: r.outBuf.buffer, Range: interBytes},
		{Buffer: downBuf.buffer, Range: downBytes},
		{Buffer: r.finalOutBuf.buffer, Range: outBytes},
	}
	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.residualBuf.buffer, Range: outBytes},
		{Buffer: r.finalOutBuf.buffer, Range: outBytes},
		{Buffer: normWeightBuf.buffer, Range: outBytes},
		{Buffer: r.normBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.normDescriptorSet, r.normDescriptorCache[:], normInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUF32CommandDownNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordDownAddRMSNormCommand(rows, cols, outRows); err != nil {
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
	if readResidual {
		if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:outRows]); err != nil {
			return err
		}
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:outRows])
}

func (r *vulkanSwiGLUGateUpF32WinRunner) recordMatVecAddRMSNormCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.normDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUF32CommandMatNorm
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = rows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUGateUpF32WinRunner) recordDownCommand(rows, cols, outRows int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUF32CommandDown
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUGateUpF32WinRunner) recordDownAddRMSNormCommand(rows, cols, outRows int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), 1, 1)

	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.normDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUF32CommandDownNorm
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUGateUpF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanSwiGLUGateUpF32WinRunner) weightBuffer(w []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, w, size, r.weightBuffers)
}

type vulkanCachedFloat32BufferWin struct {
	buffer      vkHostBufferWin
	length      int
	hash        uint64
	fingerprint uint64
}

func hashFloat32ForVulkanCache(values []float32) uint64 {
	h := uint64(1469598103934665603)
	for _, v := range values {
		h ^= uint64(math.Float32bits(v))
		h *= 1099511628211
	}
	return h
}

func fingerprintFloat32ForVulkanCache(values []float32) uint64 {
	n := len(values)
	h := uint64(n) * 1099511628211
	if n == 0 {
		return h
	}
	h = vulkanMixFloat32Fingerprint(h, values[0])
	h = vulkanMixFloat32Fingerprint(h, values[n/2])
	h = vulkanMixFloat32Fingerprint(h, values[n-1])
	if n > 3 {
		h = vulkanMixFloat32Fingerprint(h, values[n/3])
		h = vulkanMixFloat32Fingerprint(h, values[(2*n)/3])
	}
	return h
}

func vulkanMixFloat32Fingerprint(h uint64, v float32) uint64 {
	h ^= uint64(math.Float32bits(v))
	h *= 1099511628211
	return h
}

func vulkanSwiGLUGateUpF32ShaderCodeWindows() ([]uint32, error) {
	vulkanSwiGLUGateUpF32SPV.once.Do(func() {
		vulkanSwiGLUGateUpF32SPV.code, vulkanSwiGLUGateUpF32SPV.err = compileVulkanGLSLWindows(vulkanFusedSwiGLUF32GLSL)
	})
	return vulkanSwiGLUGateUpF32SPV.code, vulkanSwiGLUGateUpF32SPV.err
}
