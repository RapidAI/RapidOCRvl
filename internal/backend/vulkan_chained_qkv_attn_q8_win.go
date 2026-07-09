//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"paddleocrvl-go/internal/tensor"
	"unsafe"
)

// VulkanChainedQKVMRoPEAttentionOutAddRMSNormQ8 chains fused Q8 QKV+MRoPE
// with text attention+output+AddRMSNorm into a single command buffer.
// The q/k/v outputs from QKV stay in GPU memory and are fed directly
// to the attention kernel.  The new token's k/v is copied into the GPU
// KV cache buffer via vkCmdCopyBuffer before the attention dispatch.
//
// This is the Q8 variant — the most common quantization — and applies
// to ALL layers in the generation path (li > 0).
func VulkanChainedQKVMRoPEAttentionOutAddRMSNormQ8(
	normOut, residual, x []float32,
	a, b, c *tensor.Q8Matrix,
	cosTable, sinTable []float32,
	w *tensor.Q8Matrix, bias, normWeight []float32,
	kCache, vCache []float32,
	cacheEpoch uint64, cacheLen, hidden, numHeads, kvHeads, headDim int,
	outK, outV []float32,
) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 chained qkv+attention matrix")
	}
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	half := headDim / 2
	cols := a.Cols
	if cols != hidden {
		return fmt.Errorf("chained q8 qkv cols mismatch: cols=%d hidden=%d", cols, hidden)
	}
	if a.Rows != qRows || b.Rows != kvRows || c.Rows != kvRows {
		return fmt.Errorf("chained q8 qkv shape mismatch: a=%dx%d qRows=%d b=%dx%d kvRows=%d", a.Rows, a.Cols, qRows, b.Rows, b.Cols, kvRows)
	}
	if len(x) < hidden || len(cosTable) < half || len(sinTable) < half ||
		len(normOut) < qRows || len(residual) < qRows ||
		w == nil || w.Rows != qRows || w.Cols != qRows || len(b.Data) < qRows*qRows || len(w.Scale) < qRows || len(bias) < qRows || len(normWeight) < qRows ||
		len(outK) < kvRows || len(outV) < kvRows {
		return fmt.Errorf("invalid chained q8 qkv+attention buffers")
	}

	qkvRunner, err := getVulkanFusedMatVec3MRoPEQ8RunnerWindows()
	if err != nil {
		return err
	}
	attRunner, err := getVulkanTextAttentionF32RunnerWindows()
	if err != nil {
		return err
	}

	qkvRunner.mu.Lock()
	defer qkvRunner.mu.Unlock()
	attRunner.mu.Lock()
	defer attRunner.mu.Unlock()

	vk := qkvRunner.vk
	device := qkvRunner.device

	// --- Prepare Q8 QKV+MRoPE ---
	xBytes, err := checkedFloat32ByteLenErrWin(hidden, "Vulkan q8 chained qkv x")
	if err != nil {
		return err
	}
	aLen, err := checkedQ8DataLenWin(qRows, hidden, "Vulkan q8 chained qkv a")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(kvRows, hidden, "Vulkan q8 chained qkv b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(kvRows, hidden, "Vulkan q8 chained qkv c")
	if err != nil {
		return err
	}
	aBytes, err := checkedAlignedByteLenErrWin(aLen, 4, "Vulkan q8 chained qkv a data")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 chained qkv b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 chained qkv c data")
	if err != nil {
		return err
	}
	saBytes, err := checkedFloat32ByteLenErrWin(qRows, "Vulkan q8 chained qkv a scale")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(kvRows, "Vulkan q8 chained qkv b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(kvRows, "Vulkan q8 chained qkv c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan q8 chained qkv table")
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
	asbuf, err := qkvRunner.floatWeightBuffer(a.Scale[:qRows], saBytes)
	if err != nil {
		return err
	}
	bsbuf, err := qkvRunner.floatWeightBuffer(b.Scale[:kvRows], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := qkvRunner.floatWeightBuffer(c.Scale[:kvRows], scBytes)
	if err != nil {
		return err
	}
	if err := qkvRunner.ensureHostBuffer(&qkvRunner.xBuf, xBytes); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.xBuf, x[:hidden]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, qkvRunner.sinBuf, sinTable[:half]); err != nil {
		return err
	}

	// Temporary descriptor pool for Q8 QKV descriptor set
	poolSize1 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(qkvRunner.descriptorCount)}
	dpci1 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize1)),
	}
	var qkvPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci1)), 0, uintptr(unsafe.Pointer(&qkvPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool qkv q8: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, qkvPool, 0)

	var qkvDS uintptr
	qkvDSAI := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     qkvPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&qkvRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&qkvDSAI)), uintptr(unsafe.Pointer(&qkvDS))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets qkv q8: %d", int32(res))
	}

	qkvInfos := [12]vkDescriptorBufferInfo{
		{Buffer: qkvRunner.xBuf.buffer, Range: xBytes},
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
	var qkvDSCache [12]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, qkvDS, qkvDSCache[:qkvRunner.descriptorCount], qkvInfos[:])

	// --- Prepare attention+output+AddRMSNorm ---
	if err := attRunner.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvRows); err != nil {
		return err
	}
	kvDimBytes := uint64(kvRows) * 4
	fullCacheBytes := uint64(cacheLen+1) * kvDimBytes
	if err := attRunner.ensureHostBuffer(&attRunner.kBuf, fullCacheBytes); err != nil {
		return err
	}
	if err := attRunner.ensureHostBuffer(&attRunner.vBuf, fullCacheBytes); err != nil {
		return err
	}
	qBytes := uint64(qRows) * 4

	if err := attRunner.ensureHostBuffer(&attRunner.outBuf, qBytes); err != nil {
		return err
	}
	if err := attRunner.ensureHostBuffer(&attRunner.finalBuf, qBytes); err != nil {
		return err
	}
	if err := attRunner.ensureHostBuffer(&attRunner.residualBuf, qBytes); err != nil {
		return err
	}
	if err := attRunner.ensureHostBuffer(&attRunner.normBuf, qBytes); err != nil {
		return err
	}
	// Q8 output projection weight: upload as int8 packed data + scale
	wDataLen, err := checkedQ8DataLenWin(qRows, qRows, "Vulkan q8 chained qkv output w")
	if err != nil {
		return err
	}
	wDataBytes, err := checkedAlignedByteLenErrWin(wDataLen, 4, "Vulkan q8 chained qkv output w data")
	if err != nil {
		return err
	}
	wScaleBytes, err := checkedFloat32ByteLenErrWin(qRows, "Vulkan q8 chained qkv output w scale")
	if err != nil {
		return err
	}
	wDataBuf, err := attRunner.q8DataBuffer(w.Data[:wDataLen], wDataBytes)
	if err != nil {
		return err
	}
	wScaleBuf, err := attRunner.cachedBuffer(w.Scale[:qRows], wScaleBytes, attRunner.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := attRunner.cachedBuffer(normWeight[:qRows], qBytes, attRunner.biasBuffers)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, attRunner.residualBuf, residual[:qRows]); err != nil {
		return err
	}

	// Attention descriptor set: [0]=q (from qkvRunner.outABuf), [1]=k, [2]=v,
	// [3]=outBuf, [4]=w, [5]=bias, [6]=finalBuf, [7]=residualBuf, [8]=normWeight, [9]=normBuf
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 10}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var attPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&attPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool att q8: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, attPool, 0)

	var attDS uintptr
	attDSAI := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     attPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&attRunner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&attDSAI)), uintptr(unsafe.Pointer(&attDS))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets att q8: %d", int32(res))
	}

	attInfos := [10]vkDescriptorBufferInfo{
		{Buffer: qkvRunner.outABuf.buffer, Range: qBytes},
		{Buffer: attRunner.kBuf.buffer, Range: attRunner.kBuf.size},
		{Buffer: attRunner.vBuf.buffer, Range: attRunner.vBuf.size},
		{Buffer: attRunner.outBuf.buffer, Range: qBytes},
		{Buffer: wDataBuf.buffer, Range: wDataBytes},  // binding 4: Q8 weight data
		{Buffer: wScaleBuf.buffer, Range: wScaleBytes}, // binding 5: Q8 weight scale (replaces bias)
		{Buffer: attRunner.finalBuf.buffer, Range: qBytes},
		{Buffer: attRunner.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: attRunner.normBuf.buffer, Range: qBytes},
	}
	var attDSCache [10]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, attDS, attDSCache[:], attInfos[:])

	// --- Record command buffer (single submit for the whole chain) ---
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained q8: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained q8: %d", int32(res))
	}

	// Dispatch 1: Q8 QKV+MRoPE
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, qkvRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, qkvRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&qkvDS)), 0, 0)
	packed := headDim | (kvHeads << 16)
	var qkvPC [20]byte
	binary.LittleEndian.PutUint32(qkvPC[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(qkvPC[4:8], uint32(kvRows))
	binary.LittleEndian.PutUint32(qkvPC[8:12], uint32(kvRows))
	binary.LittleEndian.PutUint32(qkvPC[12:16], uint32(hidden))
	binary.LittleEndian.PutUint32(qkvPC[16:20], uint32(packed))
	vk.callVoid(vk.cmdPushConstants, cmd, qkvRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&qkvPC[0])))
	groups := qRows/2 + kvRows/2 + kvRows
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groups), 1, 1)

	// Barrier: QKV writes visible to k/v cache copy
	vk.computeBarrier(cmd)

	// Copy new k/v from QKV output into KV cache at position cacheLen
	kvCopyBytes := uint64(kvRows) * 4
	kvCopyOffset := uint64(cacheLen) * uint64(kvRows) * 4
	vk.cmdCopyBufferOffsetWin(cmd, qkvRunner.outBBuf.buffer, attRunner.kBuf.buffer, kvCopyOffset, kvCopyBytes)
	vk.cmdCopyBufferOffsetWin(cmd, qkvRunner.outCBuf.buffer, attRunner.vBuf.buffer, kvCopyOffset, kvCopyBytes)

	// Barrier: copy writes visible to attention reads
	vk.computeBarrier(cmd)

	// Dispatch 2: Attention
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, attRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&attDS)), 0, 0)
	var attPC [20]byte
	binary.LittleEndian.PutUint32(attPC[0:4], uint32(cacheLen+1))
	binary.LittleEndian.PutUint32(attPC[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(attPC[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(attPC[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(attPC[16:20], uint32(kvRows))
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)

	// Barrier: attention writes visible to output proj
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	// Dispatch 3: Output projection
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.q8ProjPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], 1)
	binary.LittleEndian.PutUint32(attPC[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[12:16], 0)
	binary.LittleEndian.PutUint32(attPC[16:20], 0)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)

	// Barrier: output proj writes visible to AddRMSNorm
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	// Dispatch 4: AddRMSNorm
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.normPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[4:8], 1)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained q8: %d", int32(res))
	}

	// Single submit + wait
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

	// Read back: residual, normOut, new k/v for host cache update
	if err := vk.readFloat32Into(device, attRunner.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outBBuf, outK[:kvRows]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, qkvRunner.outCBuf, outV[:kvRows]); err != nil {
		return err
	}
	return vk.readFloat32Into(device, attRunner.normBuf, normOut[:qRows])
}