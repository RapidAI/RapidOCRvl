package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// VulkanChainedMatVec2F32 chains two matvec operations into a single
// command buffer submission.  The output of op1 (out1 = w1 @ x) becomes
// the input of op2 (out2 = w2 @ out1), with the intermediate result
// staying in GPU memory (host-visible buffer, but no host round-trip
// between the two dispatches).
//
// This eliminates one host readback + one host upload + one fence wait
// compared to calling VulkanMatVecF32 twice.
func VulkanChainedMatVec2F32(out2, x, w1, w2 []float32, rows1, cols1, rows2 int) error {
	if rows1 <= 0 || cols1 <= 0 || rows2 <= 0 {
		return fmt.Errorf("invalid chained matvec2 shape rows1=%d cols1=%d rows2=%d", rows1, cols1, rows2)
	}
	if len(x) < cols1 || len(w1) < rows1*cols1 || len(w2) < rows2*rows1 || len(out2) < rows2 {
		return fmt.Errorf("invalid chained matvec2 buffers x=%d w1=%d w2=%d out2=%d rows1=%d cols1=%d rows2=%d",
			len(x), len(w1), len(w2), len(out2), rows1, cols1, rows2)
	}

	// Get the shared runner (matvec F32 uses the same runner type for all ops)
	runner, err := getVulkanMatVecF32RunnerWindows()
	if err != nil {
		return err
	}

	runner.mu.Lock()
	defer runner.mu.Unlock()

	// Use a secondary command buffer from the runner's own pool
	vk := runner.vk
	device := runner.device

	// Reset and begin a fresh command buffer
	if res := vk.call(vk.resetCommandPool, device, runner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained: %d", int32(res))
	}
	cmd := runner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained: %d", int32(res))
	}

	// Op 1: out1 = w1 @ x
	// We need a temporary buffer for the intermediate result (out1).
	// Reuse the runner's outBuf for op1's output, then use it as input for op2.
	// But the runner's descriptor set is shared, so we need to be careful.

	// Prepare op1: write x to xBuf, set up descriptor set
	if err := runner.prepareMatVecF32(x, w1, rows1, cols1); err != nil {
		return err
	}
	// Record op1 dispatch
	if err := runner.dispatchInto(cmd, rows1, cols1); err != nil {
		return err
	}

	// Insert a compute barrier: op1 writes must be visible to op2 reads
	vk.computeBarrier(cmd)

	// Op 2: out2 = w2 @ out1
	// We need to set up the descriptor set to use out1 (runner.outBuf) as input.
	// But prepareMatVecF32 would overwrite xBuf. Instead, we manually set up
	// the descriptor set to point outBuf -> input, w2 -> weight, outBuf2 -> output.
	// We need a separate output buffer for op2. Use addBuf as the second output.
	out1Bytes := uint64(rows1) * 4
	w2Len := rows2 * rows1
	w2Bytes := uint64(w2Len) * 4
	out2Bytes := uint64(rows2) * 4

	if err := runner.ensureHostBuffer(&runner.addBuf, out2Bytes); err != nil {
		return err
	}
	w2Buf, err := runner.weightBuffer(w2[:w2Len], w2Bytes)
	if err != nil {
		return err
	}

	// We need a SEPARATE descriptor set for op2 because Vulkan descriptor set
	// updates are visible to all recorded commands that reference the same set.
	// If we reuse the same set, both dispatches would see op2's bindings.
	// Create a temporary descriptor pool for op2's set.
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(runner.descriptorCount)}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:        vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:      1,
		PoolSizeCount: 1,
		PPoolSizes:   uintptr(unsafe.Pointer(&poolSize2)),
	}
	var tempPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&tempPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool op2: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool, 0)

	var ds2 uintptr
	dsai2 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&runner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai2)), uintptr(unsafe.Pointer(&ds2))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets op2: %d", int32(res))
	}

	// Update op2's descriptor set: input=outBuf (op1 output), weight=w2Buf, output=addBuf
	bufInfos2 := [...]vkDescriptorBufferInfo{
		{Buffer: runner.outBuf.buffer, Range: out1Bytes},
		{Buffer: w2Buf.buffer, Range: w2Bytes},
		{Buffer: runner.addBuf.buffer, Range: out2Bytes},
	}
	var ds2Cache [5]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds2, ds2Cache[:runner.descriptorCount], bufInfos2[:])

	// Record op2 dispatch with the separate descriptor set
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, runner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, runner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds2)), 0, 0)
	var pc2 [8]byte
	binary.LittleEndian.PutUint32(pc2[0:4], uint32(rows2))
	binary.LittleEndian.PutUint32(pc2[4:8], uint32(rows1))
	vk.callVoid(vk.cmdPushConstants, cmd, runner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc2)), uintptr(unsafe.Pointer(&pc2[0])))
	groupsX2 := rows2
	if runner.dispatchOnce {
		groupsX2 = 1
	}
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groupsX2), 1, 1)

	// End recording
	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained: %d", int32(res))
	}

	// Submit and wait (single fence for the entire chain)
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&runner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences chained: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, runner.queue, 1, uintptr(unsafe.Pointer(&submit)), runner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit chained: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&runner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences chained: %d", int32(res))
	}

	// Read back only the final output (op2's output)
	return vk.readFloat32Into(device, runner.addBuf, out2[:rows2])
}