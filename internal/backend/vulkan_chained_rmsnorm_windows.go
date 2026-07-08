package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// VulkanChainedRMSNormMatVecF32 chains RMSNorm followed by matvec into a
// single command buffer submission.  The intermediate result (RMSNorm output)
// stays in GPU memory — no host readback between the two dispatches.
//
// out = w @ rmsnorm(x, weight)
//
// This eliminates one host readback + one host upload + one fence wait
// compared to calling VulkanRMSNormF32 then VulkanMatVecF32 separately.
func VulkanChainedRMSNormMatVecF32(out, x, normWeight, w []float32, n, rows, cols int) error {
	if n <= 0 || rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid chained rmsnorm+matvec shape n=%d rows=%d cols=%d", n, rows, cols)
	}
	if len(x) < n || len(normWeight) < n || len(w) < rows*cols || len(out) < rows {
		return fmt.Errorf("invalid chained rmsnorm+matvec buffers x=%d normWeight=%d w=%d out=%d n=%d rows=%d cols=%d",
			len(x), len(normWeight), len(w), len(out), n, rows, cols)
	}

	// Get the RMSNorm runner (dispatchOnce=true, 3 descriptors)
	normRunner, err := getVulkanRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}
	// Get the matvec runner (dispatchOnce=false, 3 descriptors)
	matvecRunner, err := getVulkanMatVecF32RunnerWindows()
	if err != nil {
		return err
	}

	// Use the matvec runner's command pool for the batch
	matvecRunner.mu.Lock()
	defer matvecRunner.mu.Unlock()

	vk := matvecRunner.vk
	device := matvecRunner.device

	// We need the RMSNorm runner's pipeline and descriptor set layout.
	// Both runners share the same device/queue via the shared context.
	normRunner.mu.Lock()
	defer normRunner.mu.Unlock()

	// Prepare RMSNorm: write x and weight to normRunner's buffers
	xBytes := uint64(n) * 4
	normWeightBytes := uint64(n) * 4
	normOutBytes := uint64(n) * 4

	if err := normRunner.ensureHostBuffer(&normRunner.xBuf, xBytes); err != nil {
		return err
	}
	if err := normRunner.ensureHostBuffer(&normRunner.outBuf, normOutBytes); err != nil {
		return err
	}
	normWeightBuf, err := normRunner.weightBuffer(normWeight[:n], normWeightBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, normRunner.xBuf, x[:n]); err != nil {
		return err
	}

	// Update normRunner's descriptor set: x, weight, out
	normInfos := [...]vkDescriptorBufferInfo{
		{Buffer: normRunner.xBuf.buffer, Range: xBytes},
		{Buffer: normWeightBuf.buffer, Range: normWeightBytes},
		{Buffer: normRunner.outBuf.buffer, Range: normOutBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, normRunner.descriptorSet, normRunner.descriptorCache[:normRunner.descriptorCount], normInfos[:])

	// Prepare matvec: set up descriptor set with normRunner.outBuf as input
	wLen := rows * cols
	wBytes := uint64(wLen) * 4
	matvecOutBytes := uint64(rows) * 4

	if err := matvecRunner.ensureHostBuffer(&matvecRunner.outBuf, matvecOutBytes); err != nil {
		return err
	}
	wBuf, err := matvecRunner.weightBuffer(w[:wLen], wBytes)
	if err != nil {
		return err
	}

	// Create a temporary descriptor pool for the matvec's descriptor set
	// (since the matvec runner's pool only has 1 set)
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(matvecRunner.descriptorCount)}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var tempPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&tempPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool matvec: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool, 0)

	var ds2 uintptr
	dsai2 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&matvecRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai2)), uintptr(unsafe.Pointer(&ds2))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets matvec: %d", int32(res))
	}

	matvecInfos := [...]vkDescriptorBufferInfo{
		{Buffer: normRunner.outBuf.buffer, Range: normOutBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: matvecRunner.outBuf.buffer, Range: matvecOutBytes},
	}
	var ds2Cache [5]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds2, ds2Cache[:matvecRunner.descriptorCount], matvecInfos[:])

	// Record the command buffer
	if res := vk.call(vk.resetCommandPool, device, matvecRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained: %d", int32(res))
	}
	cmd := matvecRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained: %d", int32(res))
	}

	// Dispatch 1: RMSNorm (using normRunner's pipeline and descriptor set)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, normRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, normRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&normRunner.descriptorSet)), 0, 0)
	// RMSNorm uses push constants: rows=n, cols=1
	var normPC [8]byte
	binary.LittleEndian.PutUint32(normPC[0:4], uint32(n))
	binary.LittleEndian.PutUint32(normPC[4:8], uint32(1))
	vk.callVoid(vk.cmdPushConstants, cmd, normRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(normPC)), uintptr(unsafe.Pointer(&normPC[0])))
	// RMSNorm uses dispatchOnce=true, so groupsX=1
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// Barrier: RMSNorm writes must be visible to matvec reads
	vk.computeBarrier(cmd)

	// Dispatch 2: MatVec (using matvecRunner's pipeline and ds2)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, matvecRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, matvecRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds2)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdPushConstants, cmd, matvecRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained: %d", int32(res))
	}

	// Single submit + wait for the entire chain
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&matvecRunner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences chained: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, matvecRunner.queue, 1, uintptr(unsafe.Pointer(&submit)), matvecRunner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit chained: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&matvecRunner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences chained: %d", int32(res))
	}

	// Read back only the final output (matvec output)
	return vk.readFloat32Into(device, matvecRunner.outBuf, out[:rows])
}