package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"
)

// VulkanChainedQKVMRoPEAttentionOutAddRMSNormF32 chains fused QKV+MRoPE
// with text attention+output+AddRMSNorm into a single command buffer.
// The q/k/v outputs from QKV stay in GPU memory and are fed directly
// to the attention kernel - no host readback between the two operations.
//
// This is the most impactful chain: it applies to ALL layers in the
// generation path (li > 0), not just layer 0.
//
// outK/outV are filled with the new token's k/v (post-RoPE) for host
// cache update.  The k/v are also copied into the GPU KV cache buffer
// at position cacheLen via vkCmdCopyBuffer before the attention dispatch.
func VulkanChainedQKVMRoPEAttentionOutAddRMSNormF32(
	normOut, residual, x []float32,
	wa, wb, wc []float32, // QKV weights (F32)
	cosTable, sinTable []float32,
	w, bias, normWeight []float32, // output proj weight, bias, norm weight
	kCache, vCache []float32,
	cacheEpoch uint64, cacheLen, hidden, numHeads, kvHeads, headDim int,
	outK, outV []float32, // filled with new token's k/v for host cache update
) error {
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	half := headDim / 2
	if len(x) < hidden || len(wa) < qRows*hidden || len(wb) < kvRows*hidden || len(wc) < kvRows*hidden ||
		len(cosTable) < half || len(sinTable) < half || len(normOut) < qRows || len(residual) < qRows ||
		len(w) < qRows*qRows || len(bias) < qRows || len(normWeight) < qRows || len(outK) < kvRows || len(outV) < kvRows {
		return fmt.Errorf("invalid chained qkv+attention buffers")
	}

	qkvRunner, err := getVulkanFusedMatVec3MRoPEF32RunnerWindows()
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

	xBytes := uint64(hidden) * 4
	waBytes := uint64(qRows * hidden) * 4
	wbBytes := uint64(kvRows * hidden) * 4
	wcBytes := uint64(kvRows * hidden) * 4
	tableBytes := uint64(half) * 4
	outABytes := uint64(qRows) * 4
	outBBytes := uint64(kvRows) * 4
	outCBytes := uint64(kvRows) * 4

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
	waBuf, err := qkvRunner.weightBuffer(wa[:qRows*hidden], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := qkvRunner.weightBuffer(wb[:kvRows*hidden], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := qkvRunner.weightBuffer(wc[:kvRows*hidden], wcBytes)
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

	// Temporary descriptor pool for QKV descriptor set
	poolSize1 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(qkvRunner.descriptorCount)}
	dpci1 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize1)),
	}
	var qkvPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci1)), 0, uintptr(unsafe.Pointer(&qkvPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool qkv: %d", int32(res))
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
		return fmt.Errorf("vkAllocateDescriptorSets qkv: %d", int32(res))
	}

	qkvInfos := [9]vkDescriptorBufferInfo{
		{Buffer: qkvRunner.xBuf.buffer, Range: xBytes},
		{Buffer: waBuf.buffer, Range: waBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: qkvRunner.cosBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.sinBuf.buffer, Range: tableBytes},
		{Buffer: qkvRunner.outABuf.buffer, Range: outABytes},
		{Buffer: qkvRunner.outBBuf.buffer, Range: outBBytes},
		{Buffer: qkvRunner.outCBuf.buffer, Range: outCBytes},
	}
	var qkvDSCache [9]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, qkvDS, qkvDSCache[:qkvRunner.descriptorCount], qkvInfos[:])

	// --- Prepare attention+output+AddRMSNorm ---
	// Upload KV cache (positions 0..cacheLen-1).  The kBuf/vBuf are sized for
	// cacheLen+1 tokens so the new token's k/v (from QKV output) can be copied
	// in at position cacheLen.
	if err := attRunner.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvRows); err != nil {
		return err
	}
	// Ensure kBuf/vBuf have room for the new token at position cacheLen
	kvDimBytes := uint64(kvRows) * 4
	fullCacheBytes := uint64(cacheLen+1) * kvDimBytes
	if err := attRunner.ensureHostBuffer(&attRunner.kBuf, fullCacheBytes); err != nil {
		return err
	}
	if err := attRunner.ensureHostBuffer(&attRunner.vBuf, fullCacheBytes); err != nil {
		return err
	}
	qBytes := uint64(qRows) * 4
	wBytes := uint64(qRows * qRows) * 4
	biasBytes := qBytes
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
	wBuf, err := attRunner.cachedBuffer(w[:qRows*qRows], wBytes, attRunner.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := attRunner.cachedBuffer(bias[:qRows], biasBytes, attRunner.biasBuffers)
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

	// Temporary descriptor pool for attention descriptor set
	// [0]=q (from qkvRunner.outABuf!), [1]=k cache, [2]=v cache, [3]=outBuf,
	// [4]=w, [5]=bias, [6]=finalBuf, [7]=residualBuf, [8]=normWeight, [9]=normBuf
	poolSize2 := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 10}
	dpci2 := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize2)),
	}
	var attPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci2)), 0, uintptr(unsafe.Pointer(&attPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool att: %d", int32(res))
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
		return fmt.Errorf("vkAllocateDescriptorSets att: %d", int32(res))
	}

	attInfos := [10]vkDescriptorBufferInfo{
		{Buffer: qkvRunner.outABuf.buffer, Range: qBytes},
		{Buffer: attRunner.kBuf.buffer, Range: attRunner.kBuf.size},
		{Buffer: attRunner.vBuf.buffer, Range: attRunner.vBuf.size},
		{Buffer: attRunner.outBuf.buffer, Range: qBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: attRunner.finalBuf.buffer, Range: qBytes},
		{Buffer: attRunner.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: attRunner.normBuf.buffer, Range: qBytes},
	}
	var attDSCache [10]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, attDS, attDSCache[:], attInfos[:])

	// --- Record command buffer ---
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained: %d", int32(res))
	}

	// Dispatch 1: QKV+MRoPE
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

	// Barrier: QKV writes must be visible to the k/v cache copy
	vk.computeBarrier(cmd)

	// Copy new k/v from QKV output into the KV cache buffer at position cacheLen
	kvCopyBytes := uint64(kvRows) * 4
	kvCopyOffset := uint64(cacheLen) * uint64(kvRows) * 4
	vk.cmdCopyBufferOffsetWin(cmd, qkvRunner.outBBuf.buffer, attRunner.kBuf.buffer, kvCopyOffset, kvCopyBytes)
	vk.cmdCopyBufferOffsetWin(cmd, qkvRunner.outCBuf.buffer, attRunner.vBuf.buffer, kvCopyOffset, kvCopyBytes)

	// Barrier: copy writes must be visible to attention reads
	vk.computeBarrier(cmd)

	// Dispatch 2: Attention (using attRunner pipeline + attDS)
	// cacheLen+1 because the new token's k/v has been appended to the cache
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

	// Barrier: attention writes must be visible to output proj
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	// Dispatch 3: Output projection (using attRunner projPipeline + attDS)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.projPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], 1)
	binary.LittleEndian.PutUint32(attPC[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[12:16], 0)
	binary.LittleEndian.PutUint32(attPC[16:20], 0)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)

	// Barrier: output proj writes must be visible to AddRMSNorm
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	// Dispatch 4: AddRMSNorm (using attRunner normPipeline + attDS)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.normPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[4:8], 1)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained: %d", int32(res))
	}

	// Single submit + wait
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

	// Read back final outputs: residual (updated), normOut, and new k/v
	// for host cache update
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