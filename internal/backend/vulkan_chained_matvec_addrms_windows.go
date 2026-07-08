package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// VulkanChainedMatVecAddRMSNormMatVecF32 chains matvec1 -> AddRMSNorm -> matvec2
// into a single command buffer submission.  The intermediate results
// (matvec1 output, AddRMSNorm output) stay in GPU memory.
//
// out2 = w2 @ rmsnorm(residual + w1 @ x, normWeight)
//
// This eliminates two host readbacks + uploads + fence waits compared to
// calling the three operations separately.
func VulkanChainedMatVecAddRMSNormMatVecF32(out2, x, residual, w1, normWeight, w2 []float32, rows1, cols1, rows2, cols2 int) error {
	if rows1 <= 0 || cols1 <= 0 || rows2 <= 0 || cols2 <= 0 {
		return fmt.Errorf("invalid chained matvec+addrmsnorm+matvec shape rows1=%d cols1=%d rows2=%d cols2=%d", rows1, cols1, rows2, cols2)
	}
	if cols2 != rows1 {
		return fmt.Errorf("chained shape mismatch: cols2=%d != rows1=%d", cols2, rows1)
	}
	if len(x) < cols1 || len(residual) < rows1 || len(w1) < rows1*cols1 || len(normWeight) < rows1 || len(w2) < rows2*rows1 || len(out2) < rows2 {
		return fmt.Errorf("invalid chained matvec+addrmsnorm+matvec buffers")
	}

	// Get the matvec runner (dispatchOnce=false, 3 descriptors)
	matvecRunner, err := getVulkanMatVecF32RunnerWindows()
	if err != nil {
		return err
	}
	// Get the AddRMSNorm runner (dispatchOnce=true, 4 descriptors)
	addNormRunner, err := getVulkanAddRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}

	matvecRunner.mu.Lock()
	defer matvecRunner.mu.Unlock()
	addNormRunner.mu.Lock()
	defer addNormRunner.mu.Unlock()

	vk := matvecRunner.vk
	device := matvecRunner.device

	xBytes := uint64(cols1) * 4
	w1Bytes := uint64(rows1 * cols1) * 4
	out1Bytes := uint64(rows1) * 4
	normWeightBytes := uint64(rows1) * 4
	addBytes := uint64(rows1) * 4 // residual bytes
	outNormBytes := uint64(rows1) * 4
	w2Bytes := uint64(rows2 * rows1) * 4
	out2Bytes := uint64(rows2) * 4

	// --- Prepare matvec1: out1 = w1 @ x ---
	if err := matvecRunner.ensureHostBuffer(&matvecRunner.xBuf, xBytes); err != nil {
		return err
	}
	if err := matvecRunner.ensureHostBuffer(&matvecRunner.outBuf, out1Bytes); err != nil {
		return err
	}
	w1Buf, err := matvecRunner.weightBuffer(w1[:rows1*cols1], w1Bytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, matvecRunner.xBuf, x[:cols1]); err != nil {
		return err
	}
	mv1Infos := [...]vkDescriptorBufferInfo{
		{Buffer: matvecRunner.xBuf.buffer, Range: xBytes},
		{Buffer: w1Buf.buffer, Range: w1Bytes},
		{Buffer: matvecRunner.outBuf.buffer, Range: out1Bytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, matvecRunner.descriptorSet, matvecRunner.descriptorCache[:matvecRunner.descriptorCount], mv1Infos[:])

	// --- Prepare AddRMSNorm: residual2 = residual + out1; out2 = rmsnorm(residual2, normWeight) ---
	// AddRMSNorm descriptor layout: [0]=dst/x (residual, also output), [1]=add (matvec out), [2]=weight, [3]=out
	if err := addNormRunner.ensureHostBuffer(&addNormRunner.xBuf, addBytes); err != nil {
		return err
	}
	if err := addNormRunner.ensureHostBuffer(&addNormRunner.addBuf, addBytes); err != nil {
		return err
	}
	if err := addNormRunner.ensureHostBuffer(&addNormRunner.outBuf, outNormBytes); err != nil {
		return err
	}
	normWeightBuf, err := addNormRunner.weightBuffer(normWeight[:rows1], normWeightBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, addNormRunner.xBuf, residual[:rows1]); err != nil {
		return err
	}
	// Create a temporary descriptor pool for AddRMSNorm set (input=add = matvecRunner.outBuf)
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(addNormRunner.descriptorCount)}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var tempPool2 uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&tempPool2))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool addnorm: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool2, 0)

	var ds2 uintptr
	dsai2 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool2,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&addNormRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai2)), uintptr(unsafe.Pointer(&ds2))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets addnorm: %d", int32(res))
	}
	// AddRMSNorm: dst=residual(xBuf), add=matvecRunner.outBuf, weight=normWeightBuf, out=addNormRunner.outBuf
	addNormInfos := [...]vkDescriptorBufferInfo{
		{Buffer: addNormRunner.xBuf.buffer, Range: addBytes},
		{Buffer: matvecRunner.outBuf.buffer, Range: out1Bytes},
		{Buffer: normWeightBuf.buffer, Range: normWeightBytes},
		{Buffer: addNormRunner.outBuf.buffer, Range: outNormBytes},
	}
	var ds2Cache [5]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds2, ds2Cache[:addNormRunner.descriptorCount], addNormInfos[:])

	// --- Prepare matvec2: finalOut = w2 @ normOut ---
	// Need a separate output buffer for matvec2
	if err := matvecRunner.ensureHostBuffer(&matvecRunner.addBuf, out2Bytes); err != nil {
		return err
	}
	w2Buf, err := matvecRunner.weightBuffer(w2[:rows2*rows1], w2Bytes)
	if err != nil {
		return err
	}
	// Create a temporary descriptor pool for matvec2's set
	poolSize3 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(matvecRunner.descriptorCount)}
	dpci3 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize3)),
	}
	var tempPool3 uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci3)), 0, uintptr(unsafe.Pointer(&tempPool3))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool matvec2: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool3, 0)

	var ds3 uintptr
	dsai3 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool3,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&matvecRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai3)), uintptr(unsafe.Pointer(&ds3))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets matvec2: %d", int32(res))
	}
	// matvec2: input=addNormRunner.outBuf (norm output), weight=w2Buf, output=matvecRunner.addBuf
	mv2Infos := [...]vkDescriptorBufferInfo{
		{Buffer: addNormRunner.outBuf.buffer, Range: outNormBytes},
		{Buffer: w2Buf.buffer, Range: w2Bytes},
		{Buffer: matvecRunner.addBuf.buffer, Range: out2Bytes},
	}
	var ds3Cache [5]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds3, ds3Cache[:matvecRunner.descriptorCount], mv2Infos[:])

	// --- Record the command buffer ---
	if res := vk.call(vk.resetCommandPool, device, matvecRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained: %d", int32(res))
	}
	cmd := matvecRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained: %d", int32(res))
	}

	// Dispatch 1: matvec1 (using matvecRunner's pipeline and descriptor set)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, matvecRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, matvecRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&matvecRunner.descriptorSet)), 0, 0)
	var pc1 [8]byte
	binary.LittleEndian.PutUint32(pc1[0:4], uint32(rows1))
	binary.LittleEndian.PutUint32(pc1[4:8], uint32(cols1))
	vk.callVoid(vk.cmdPushConstants, cmd, matvecRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc1)), uintptr(unsafe.Pointer(&pc1[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows1), 1, 1)

	// Barrier: matvec1 writes must be visible to AddRMSNorm reads
	vk.computeBarrier(cmd)

	// Dispatch 2: AddRMSNorm (using addNormRunner's pipeline and ds2)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, addNormRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, addNormRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds2)), 0, 0)
	var normPC [8]byte
	binary.LittleEndian.PutUint32(normPC[0:4], uint32(rows1))
	binary.LittleEndian.PutUint32(normPC[4:8], uint32(1))
	vk.callVoid(vk.cmdPushConstants, cmd, addNormRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(normPC)), uintptr(unsafe.Pointer(&normPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// Barrier: AddRMSNorm writes must be visible to matvec2 reads
	vk.computeBarrier(cmd)

	// Dispatch 3: matvec2 (using matvecRunner's pipeline and ds3)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, matvecRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, matvecRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds3)), 0, 0)
	var pc2 [8]byte
	binary.LittleEndian.PutUint32(pc2[0:4], uint32(rows2))
	binary.LittleEndian.PutUint32(pc2[4:8], uint32(rows1))
	vk.callVoid(vk.cmdPushConstants, cmd, matvecRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc2)), uintptr(unsafe.Pointer(&pc2[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows2), 1, 1)

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

	// Read back only the final output (matvec2 output)
	return vk.readFloat32Into(device, matvecRunner.addBuf, out2[:rows2])
}
