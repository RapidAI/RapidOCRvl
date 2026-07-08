package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"paddleocrvl-go/internal/tensor"
	"unsafe"
)

// VulkanChainedRMSNormFusedQKVMRoPEF32 chains RMSNorm followed by a fused
// QKV+MRoPE projection into a single command buffer submission.  The
// intermediate result (RMSNorm output = normalised hidden vector) stays in
// GPU memory — no host readback between the two dispatches.
//
// outA = q, outB = k, outC = v (after RoPE)
// x_in = raw hidden vector, normWeight = ln1 weights
//
// This eliminates one host readback + one host upload + one fence wait
// compared to calling VulkanRMSNormF32 then VulkanFusedMatVec3MRoPEF32.
func VulkanChainedRMSNormFusedQKVMRoPEF32(outA, outB, outC, x, normWeight, wa, wb, wc, cosTable, sinTable []float32, n, rowsA, rowsB, rowsC, cols, kvHeads, headDim int) error {
	if n <= 0 || rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("invalid chained rmsnorm+qkvmrope shape n=%d rowsA=%d rowsB=%d rowsC=%d cols=%d kvHeads=%d headDim=%d", n, rowsA, rowsB, rowsC, cols, kvHeads, headDim)
	}
	half := headDim / 2
	if len(x) < n || len(normWeight) < n || len(wa) < rowsA*cols || len(wb) < rowsB*cols || len(wc) < rowsC*cols ||
		len(cosTable) < half || len(sinTable) < half || len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC {
		return fmt.Errorf("invalid chained rmsnorm+qkvmrope buffers")
	}

	// Get the RMSNorm runner (dispatchOnce=true, 3 descriptors)
	normRunner, err := getVulkanRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}
	// Get the fused QKV+MRoPE runner (9 descriptors, mrope=true)
	qkvRunner, err := getVulkanFusedMatVec3MRoPEF32RunnerWindows()
	if err != nil {
		return err
	}

	// Use the QKV runner's command pool for the batch
	qkvRunner.mu.Lock()
	defer qkvRunner.mu.Unlock()
	normRunner.mu.Lock()
	defer normRunner.mu.Unlock()

	vk := qkvRunner.vk
	device := qkvRunner.device

	// --- Prepare RMSNorm: write x and weight to normRunner buffers ---
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

	// --- Prepare fused QKV+MRoPE: set up with normRunner.outBuf as input ---
	qkvXBytes := uint64(cols) * 4
	waBytes := uint64(rowsA*cols) * 4
	wbBytes := uint64(rowsB*cols) * 4
	wcBytes := uint64(rowsC*cols) * 4
	tableBytes := uint64(half) * 4
	outABytes := uint64(rowsA) * 4
	outBBytes := uint64(rowsB) * 4
	outCBytes := uint64(rowsC) * 4

	if err := qkvRunner.ensureHostBuffer(&qkvRunner.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outABuf, outABytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outCBuf, outCBytes); err != nil {
		return err
	}
	waBuf, err := qkvRunner.weightBuffer(wa[:rowsA*cols], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := qkvRunner.weightBuffer(wb[:rowsB*cols], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := qkvRunner.weightBuffer(wc[:rowsC*cols], wcBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.sinBuf, sinTable[:half]); err != nil {
		return err
	}

	// Create a temporary descriptor pool for the QKV descriptor set
	// (since the QKV runner's pool only has 1 set)
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(qkvRunner.descriptorCount)}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var tempPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&tempPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool qkv: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool, 0)

	var ds2 uintptr
	dsai2 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&qkvRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai2)), uintptr(unsafe.Pointer(&ds2))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets qkv: %d", int32(res))
	}

	// QKV descriptor set: input=normRunner.outBuf (RMSNorm output), weights, cos/sin, outputs
	qkvInfos := [...]vkDescriptorBufferInfo{
		{Buffer: normRunner.outBuf.buffer, Range: qkvXBytes}, // x = norm output
		{Buffer: waBuf.buffer, Range: waBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: qkvRunner.cosBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.sinBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.outABuf.buffer, Range: outABytes},
		{Buffer: qkvRunner.outBBuf.buffer, Range: outBBytes},
		{Buffer: qkvRunner.outCBuf.buffer, Range: outCBytes},
	}
	var ds2Cache [9]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds2, ds2Cache[:qkvRunner.descriptorCount], qkvInfos[:])

	// --- Record the command buffer ---
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained: %d", int32(res))
	}

	// Dispatch 1: RMSNorm (using normRunner's pipeline and descriptor set)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, normRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, normRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&normRunner.descriptorSet)), 0, 0)
	var normPC [8]byte
	binary.LittleEndian.PutUint32(normPC[0:4], uint32(n))
	binary.LittleEndian.PutUint32(normPC[4:8], uint32(1))
	vk.callVoid(vk.cmdPushConstants, cmd, normRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(normPC)), uintptr(unsafe.Pointer(&normPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// Barrier: RMSNorm writes must be visible to QKV reads
	vk.computeBarrier(cmd)

	// Dispatch 2: Fused QKV+MRoPE (using qkvRunner's pipeline and ds2)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, qkvRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, qkvRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds2)), 0, 0)
	packed := headDim | (kvHeads << 16)
	var qkvPC [20]byte
	binary.LittleEndian.PutUint32(qkvPC[0:4], uint32(rowsA))
	binary.LittleEndian.PutUint32(qkvPC[4:8], uint32(rowsB))
	binary.LittleEndian.PutUint32(qkvPC[8:12], uint32(rowsC))
	binary.LittleEndian.PutUint32(qkvPC[12:16], uint32(cols))
	binary.LittleEndian.PutUint32(qkvPC[16:20], uint32(packed))
	vk.callVoid(vk.cmdPushConstants, cmd, qkvRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&qkvPC[0])))
	groups := rowsA/2 + rowsB/2 + rowsC
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groups), 1, 1)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained: %d", int32(res))
	}

	// Single submit + wait for the entire chain
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences chained: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, qkvRunner.queue, 1, uintptr(unsafe.Pointer(&submit)), qkvRunner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit chained: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences chained: %d", int32(res))
	}

	// Read back only the final outputs (QKV after RoPE)
	if err := vk.readFloat32Into(device, qkvRunner.outABuf, outA[:rowsA]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outCBuf, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

// VulkanChainedRMSNormFusedQKVMRoPEQ8 chains RMSNorm (F32) followed by a fused
// Q8 QKV+MRoPE projection into a single command buffer submission.  The
// intermediate RMSNorm output stays in GPU memory (normRunner.outBuf) and is
// fed directly as the input vector to the Q8 fused matvec3+mrope dispatch —
// no host readback between the two dispatches.
//
// outA = q, outB = k, outC = v (after RoPE)
// x_in = raw hidden vector, normWeight = ln1 weights (F32)
// a, b, c = Q8 quantised QKV projection weights
//
// Compared to calling VulkanRMSNormF32 then VulkanFusedMatVec3MRoPEQ8 this
// eliminates one host readback + one host upload + one fence wait.
func VulkanChainedRMSNormFusedQKVMRoPEQ8(outA, outB, outC, x, normWeight, cosTable, sinTable []float32, a, b, c *tensor.Q8Matrix, n, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 chained rmsnorm+qkvmrope matrix")
	}
	cols := a.Cols
	rowsA := a.Rows
	rowsB := b.Rows
	rowsC := c.Rows
	if n <= 0 || rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("invalid chained rmsnorm+qkvmrope q8 shape n=%d rowsA=%d rowsB=%d rowsC=%d cols=%d kvHeads=%d headDim=%d", n, rowsA, rowsB, rowsC, cols, kvHeads, headDim)
	}
	if cols != n {
		return fmt.Errorf("chained q8 shape mismatch: cols=%d != n=%d", cols, n)
	}
	half := headDim / 2
	if len(x) < n || len(normWeight) < n || len(cosTable) < half || len(sinTable) < half || len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC {
		return fmt.Errorf("invalid chained rmsnorm+qkvmrope q8 buffers")
	}

	// RMSNorm runner (dispatchOnce=true, 3 descriptors: x, weight, out)
	normRunner, err := getVulkanRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}
	// Q8 fused QKV+MRoPE runner (12 descriptors, mrope=true)
	qkvRunner, err := getVulkanFusedMatVec3MRoPEQ8RunnerWindows()
	if err != nil {
		return err
	}

	qkvRunner.mu.Lock()
	defer qkvRunner.mu.Unlock()
	normRunner.mu.Lock()
	defer normRunner.mu.Unlock()

	vk := qkvRunner.vk
	device := qkvRunner.device

	// --- Prepare RMSNorm: write x and weight to normRunner buffers ---
	xBytes, err := checkedFloat32ByteLenErrWin(n, "Vulkan q8 chained rmsnorm x")
	if err != nil {
		return err
	}
	normWeightBytes, err := checkedFloat32ByteLenErrWin(n, "Vulkan q8 chained rmsnorm weight")
	if err != nil {
		return err
	}
	normOutBytes := xBytes

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

	normInfos := [...]vkDescriptorBufferInfo{
		{Buffer: normRunner.xBuf.buffer, Range: xBytes},
		{Buffer: normWeightBuf.buffer, Range: normWeightBytes},
		{Buffer: normRunner.outBuf.buffer, Range: normOutBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, normRunner.descriptorSet, normRunner.descriptorCache[:normRunner.descriptorCount], normInfos[:])

	// --- Prepare fused Q8 QKV+MRoPE ---
	aLen, err := checkedQ8DataLenWin(rowsA, cols, "Vulkan q8 chained qkvmrope a")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(rowsB, cols, "Vulkan q8 chained qkvmrope b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(rowsC, cols, "Vulkan q8 chained qkvmrope c")
	if err != nil {
		return err
	}
	aBytes, err := checkedAlignedByteLenErrWin(aLen, 4, "Vulkan q8 chained qkvmrope a data")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 chained qkvmrope b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 chained qkvmrope c data")
	if err != nil {
		return err
	}
	saBytes, err := checkedFloat32ByteLenErrWin(rowsA, "Vulkan q8 chained qkvmrope a scale")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan q8 chained qkvmrope b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan q8 chained qkvmrope c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan q8 chained qkvmrope table")
	if err != nil {
		return err
	}
	outABytes := saBytes
	outBBytes := sbBytes
	outCBytes := scBytes

	if err := qkvRunner.ensureHostBuffer(&qkvRunner.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outABuf, outABytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.outCBuf, outCBytes); err != nil {
		return err
	}
	abuf, err := qkvRunner.int8WeightBuffer(a.Data[:aLen], aBytes)
	if err != nil {
		return err
	}
	bbuf, err := qkvRunner.int8WeightBuffer(b.Data[:bLen], bBytes)
	if err != nil {
		return err
	}
	cbuf, err := qkvRunner.int8WeightBuffer(c.Data[:cLen], cBytes)
	if err != nil {
		return err
	}
	asbuf, err := qkvRunner.floatWeightBuffer(a.Scale[:rowsA], saBytes)
	if err != nil {
		return err
	}
	bsbuf, err := qkvRunner.floatWeightBuffer(b.Scale[:rowsB], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := qkvRunner.floatWeightBuffer(c.Scale[:rowsC], scBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.sinBuf, sinTable[:half]); err != nil {
		return err
	}

	// Temporary descriptor pool/set for the Q8 QKV dispatch (input = norm output)
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(qkvRunner.descriptorCount)}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var tempPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&tempPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool qkv q8: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, tempPool, 0)

	var ds2 uintptr
	dsai2 := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     tempPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&qkvRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai2)), uintptr(unsafe.Pointer(&ds2))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets qkv q8: %d", int32(res))
	}

	// Q8 QKV descriptor set: x=normRunner.outBuf, data/scale weights, cos/sin, outputs
	qkvInfos := [...]vkDescriptorBufferInfo{
		{Buffer: normRunner.outBuf.buffer, Range: xBytes}, // x = RMSNorm output
		{Buffer: abuf.buffer, Range: aBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: cbuf.buffer, Range: cBytes},
		{Buffer: asbuf.buffer, Range: saBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: csbuf.buffer, Range: scBytes},
		{Buffer: qkvRunner.cosBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.sinBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.outABuf.buffer, Range: outABytes},
		{Buffer: qkvRunner.outBBuf.buffer, Range: outBBytes},
		{Buffer: qkvRunner.outCBuf.buffer, Range: outCBytes},
	}
	var ds2Cache [12]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, ds2, ds2Cache[:qkvRunner.descriptorCount], qkvInfos[:])

	// --- Record the command buffer (single submit for the whole chain) ---
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained q8: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained q8: %d", int32(res))
	}

	// Dispatch 1: RMSNorm (normRunner pipeline + descriptor set)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, normRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, normRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&normRunner.descriptorSet)), 0, 0)
	var normPC [8]byte
	binary.LittleEndian.PutUint32(normPC[0:4], uint32(n))
	binary.LittleEndian.PutUint32(normPC[4:8], uint32(1))
	vk.callVoid(vk.cmdPushConstants, cmd, normRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(normPC)), uintptr(unsafe.Pointer(&normPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// Barrier: RMSNorm writes must be visible to Q8 QKV reads
	vk.computeBarrier(cmd)

	// Dispatch 2: Fused Q8 QKV+MRoPE (qkvRunner pipeline + ds2)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, qkvRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, qkvRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&ds2)), 0, 0)
	packed := headDim | (kvHeads << 16)
	var qkvPC [20]byte
	binary.LittleEndian.PutUint32(qkvPC[0:4], uint32(rowsA))
	binary.LittleEndian.PutUint32(qkvPC[4:8], uint32(rowsB))
	binary.LittleEndian.PutUint32(qkvPC[8:12], uint32(rowsC))
	binary.LittleEndian.PutUint32(qkvPC[12:16], uint32(cols))
	binary.LittleEndian.PutUint32(qkvPC[16:20], uint32(packed))
	vk.callVoid(vk.cmdPushConstants, cmd, qkvRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&qkvPC[0])))
	groups := rowsA/2 + rowsB/2 + rowsC
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groups), 1, 1)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained q8: %d", int32(res))
	}

	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences chained q8: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, qkvRunner.queue, 1, uintptr(unsafe.Pointer(&submit)), qkvRunner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit chained q8: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences chained q8: %d", int32(res))
	}

	// Read back only the final QKV outputs (after RoPE)
	if err := vk.readFloat32Into(device, qkvRunner.outABuf, outA[:rowsA]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outCBuf, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}
