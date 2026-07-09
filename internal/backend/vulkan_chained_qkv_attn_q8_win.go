//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"paddleocrvl-go/internal/tensor"
	"unsafe"
)

// ensureHostBufferWin allocates or resizes a host-visible buffer to at least size bytes.
func ensureHostBufferWin(vk *vulkanWin, device uintptr, memProps vkPhysicalDeviceMemoryProperties, buf *vkHostBufferWin, size uint64) error {
	if buf.buffer != 0 && buf.size >= size {
		return nil
	}
	if buf.buffer != 0 {
		vk.destroyBuffer(device, *buf)
		*buf = vkHostBufferWin{}
	}
	next, err := vk.newHostBuffer(device, memProps, size)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

// VulkanChainedQKVMRoPEAttentionOutAddRMSNormQ8 chains fused Q8 QKV+MRoPE
// with text attention+output+AddRMSNorm into a single command buffer.
//
// Device-local optimization: all intermediate results (QKV outputs, attention
// output, KV cache) stay in GPU device-local memory.  Only inputs are uploaded
// via staging buffers and only final outputs are read back to host-visible
// memory.  This eliminates per-dispatch host round-trips that occur when
// using host-visible buffers for intermediates.
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

	// --- Compute byte sizes ---
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

	// --- Allocate device-local buffers for intermediates ---
	devOutA, err := vk.newDeviceBuffer(device, qkvRunner.memProps, outABytes)
	if err != nil {
		return fmt.Errorf("device buffer outA: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devOutA)

	devOutB, err := vk.newDeviceBuffer(device, qkvRunner.memProps, outBBytes)
	if err != nil {
		return fmt.Errorf("device buffer outB: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devOutB)

	devOutC, err := vk.newDeviceBuffer(device, qkvRunner.memProps, outCBytes)
	if err != nil {
		return fmt.Errorf("device buffer outC: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devOutC)

	qBytes := uint64(qRows) * 4
	kvDimBytes := uint64(kvRows) * 4
	fullCacheBytes := uint64(cacheLen+1) * kvDimBytes

	devK, err := vk.newDeviceBuffer(device, attRunner.memProps, fullCacheBytes)
	if err != nil {
		return fmt.Errorf("device buffer k: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devK)

	devV, err := vk.newDeviceBuffer(device, attRunner.memProps, fullCacheBytes)
	if err != nil {
		return fmt.Errorf("device buffer v: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devV)

	devOut, err := vk.newDeviceBuffer(device, attRunner.memProps, qBytes)
	if err != nil {
		return fmt.Errorf("device buffer out: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devOut)

	devFinal, err := vk.newDeviceBuffer(device, attRunner.memProps, qBytes)
	if err != nil {
		return fmt.Errorf("device buffer final: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devFinal)

	devResidual, err := vk.newDeviceBuffer(device, attRunner.memProps, qBytes)
	if err != nil {
		return fmt.Errorf("device buffer residual: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devResidual)

	devNorm, err := vk.newDeviceBuffer(device, attRunner.memProps, qBytes)
	if err != nil {
		return fmt.Errorf("device buffer norm: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devNorm)

	// Weight device buffers
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

	devWData, err := vk.newDeviceBuffer(device, attRunner.memProps, wDataBytes)
	if err != nil {
		return fmt.Errorf("device buffer wData: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devWData)

	devWScale, err := vk.newDeviceBuffer(device, attRunner.memProps, wScaleBytes)
	if err != nil {
		return fmt.Errorf("device buffer wScale: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devWScale)

	devNormWeight, err := vk.newDeviceBuffer(device, attRunner.memProps, qBytes)
	if err != nil {
		return fmt.Errorf("device buffer normWeight: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devNormWeight)

	devAData, err := vk.newDeviceBuffer(device, qkvRunner.memProps, aBytes)
	if err != nil {
		return fmt.Errorf("device buffer aData: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devAData)

	devBData, err := vk.newDeviceBuffer(device, qkvRunner.memProps, bBytes)
	if err != nil {
		return fmt.Errorf("device buffer bData: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devBData)

	devCData, err := vk.newDeviceBuffer(device, qkvRunner.memProps, cBytes)
	if err != nil {
		return fmt.Errorf("device buffer cData: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devCData)

	devAScale, err := vk.newDeviceBuffer(device, qkvRunner.memProps, saBytes)
	if err != nil {
		return fmt.Errorf("device buffer aScale: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devAScale)

	devBScale, err := vk.newDeviceBuffer(device, qkvRunner.memProps, sbBytes)
	if err != nil {
		return fmt.Errorf("device buffer bScale: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devBScale)

	devCScale, err := vk.newDeviceBuffer(device, qkvRunner.memProps, scBytes)
	if err != nil {
		return fmt.Errorf("device buffer cScale: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devCScale)

	devX, err := vk.newDeviceBuffer(device, qkvRunner.memProps, xBytes)
	if err != nil {
		return fmt.Errorf("device buffer x: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devX)

	devCos, err := vk.newDeviceBuffer(device, qkvRunner.memProps, tableBytes)
	if err != nil {
		return fmt.Errorf("device buffer cos: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devCos)

	devSin, err := vk.newDeviceBuffer(device, qkvRunner.memProps, tableBytes)
	if err != nil {
		return fmt.Errorf("device buffer sin: %w", err)
	}
	defer vk.destroyDeviceBuffer(device, devSin)

	// --- Staging host buffers ---
	maxFloat32Bytes := xBytes
	if tableBytes > maxFloat32Bytes {
		maxFloat32Bytes = tableBytes
	}
	if qBytes > maxFloat32Bytes {
		maxFloat32Bytes = qBytes
	}
	if saBytes > maxFloat32Bytes {
		maxFloat32Bytes = saBytes
	}
	if sbBytes > maxFloat32Bytes {
		maxFloat32Bytes = sbBytes
	}
	if scBytes > maxFloat32Bytes {
		maxFloat32Bytes = scBytes
	}
	if wScaleBytes > maxFloat32Bytes {
		maxFloat32Bytes = wScaleBytes
	}
	if fullCacheBytes > maxFloat32Bytes {
		maxFloat32Bytes = fullCacheBytes
	}
	var stagingFloat vkHostBufferWin
	defer func() {
		if stagingFloat.buffer != 0 {
			vk.destroyBuffer(device, stagingFloat)
		}
	}()

	maxByteBytes := aBytes
	if bBytes > maxByteBytes {
		maxByteBytes = bBytes
	}
	if cBytes > maxByteBytes {
		maxByteBytes = cBytes
	}
	if wDataBytes > maxByteBytes {
		maxByteBytes = wDataBytes
	}
	var stagingBytes vkHostBufferWin
	defer func() {
		if stagingBytes.buffer != 0 {
			vk.destroyBuffer(device, stagingBytes)
		}
	}()

	// Host-visible readback buffers
	var readbackNorm vkHostBufferWin
	defer func() {
		if readbackNorm.buffer != 0 {
			vk.destroyBuffer(device, readbackNorm)
		}
	}()
	var readbackResidual vkHostBufferWin
	defer func() {
		if readbackResidual.buffer != 0 {
			vk.destroyBuffer(device, readbackResidual)
		}
	}()
	var readbackK vkHostBufferWin
	defer func() {
		if readbackK.buffer != 0 {
			vk.destroyBuffer(device, readbackK)
		}
	}()
	var readbackV vkHostBufferWin
	defer func() {
		if readbackV.buffer != 0 {
			vk.destroyBuffer(device, readbackV)
		}
	}()

	// --- Descriptor pools ---
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
		{Buffer: devX.buffer, Range: xBytes},
		{Buffer: devAData.buffer, Range: aBytes},
		{Buffer: devBData.buffer, Range: bBytes},
		{Buffer: devCData.buffer, Range: cBytes},
		{Buffer: devAScale.buffer, Range: saBytes},
		{Buffer: devBScale.buffer, Range: sbBytes},
		{Buffer: devCScale.buffer, Range: scBytes},
		{Buffer: devCos.buffer, Range: tableBytes},
		{Buffer: devSin.buffer, Range: tableBytes},
		{Buffer: devOutA.buffer, Range: outABytes},
		{Buffer: devOutB.buffer, Range: outBBytes},
		{Buffer: devOutC.buffer, Range: outCBytes},
	}
	var qkvDSCache [12]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, qkvDS, qkvDSCache[:qkvRunner.descriptorCount], qkvInfos[:])

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
		{Buffer: devOutA.buffer, Range: qBytes},
		{Buffer: devK.buffer, Range: fullCacheBytes},
		{Buffer: devV.buffer, Range: fullCacheBytes},
		{Buffer: devOut.buffer, Range: qBytes},
		{Buffer: devWData.buffer, Range: wDataBytes},
		{Buffer: devWScale.buffer, Range: wScaleBytes},
		{Buffer: devFinal.buffer, Range: qBytes},
		{Buffer: devResidual.buffer, Range: qBytes},
		{Buffer: devNormWeight.buffer, Range: qBytes},
		{Buffer: devNorm.buffer, Range: qBytes},
	}
	var attDSCache [10]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, attDS, attDSCache[:], attInfos[:])

	// --- Record command buffer ---
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained q8: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained q8: %d", int32(res))
	}

	// === Upload phase ===
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devX, x[:hidden]); err != nil {
		return fmt.Errorf("upload x: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devCos, cosTable[:half]); err != nil {
		return fmt.Errorf("upload cos: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devSin, sinTable[:half]); err != nil {
		return fmt.Errorf("upload sin: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, &stagingFloat, devResidual, residual[:qRows]); err != nil {
		return fmt.Errorf("upload residual: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, &stagingFloat, devNormWeight, normWeight[:qRows]); err != nil {
		return fmt.Errorf("upload normWeight: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devAScale, a.Scale[:qRows]); err != nil {
		return fmt.Errorf("upload aScale: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devBScale, b.Scale[:kvRows]); err != nil {
		return fmt.Errorf("upload bScale: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, &stagingFloat, devCScale, c.Scale[:kvRows]); err != nil {
		return fmt.Errorf("upload cScale: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, &stagingFloat, devWScale, w.Scale[:qRows]); err != nil {
		return fmt.Errorf("upload wScale: %w", err)
	}
	if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, &stagingBytes, devAData, unsafe.Slice((*byte)(unsafe.Pointer(&a.Data[0])), aLen)); err != nil {
		return fmt.Errorf("upload aData: %w", err)
	}
	if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, &stagingBytes, devBData, unsafe.Slice((*byte)(unsafe.Pointer(&b.Data[0])), bLen)); err != nil {
		return fmt.Errorf("upload bData: %w", err)
	}
	if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, &stagingBytes, devCData, unsafe.Slice((*byte)(unsafe.Pointer(&c.Data[0])), cLen)); err != nil {
		return fmt.Errorf("upload cData: %w", err)
	}
	if err := vk.uploadBytesToDevice(device, cmd, attRunner.memProps, &stagingBytes, devWData, unsafe.Slice((*byte)(unsafe.Pointer(&w.Data[0])), wDataLen)); err != nil {
		return fmt.Errorf("upload wData: %w", err)
	}

	// Upload KV cache
	cacheElems := cacheLen * kvRows
	if cacheElems > 0 {
		if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, &stagingFloat, devK, kCache[:cacheElems]); err != nil {
			return fmt.Errorf("upload kCache: %w", err)
		}
		if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, &stagingFloat, devV, vCache[:cacheElems]); err != nil {
			return fmt.Errorf("upload vCache: %w", err)
		}
	}

	// === Compute phase ===
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

	vk.computeBarrier(cmd)

	kvCopyBytes := uint64(kvRows) * 4
	kvCopyOffset := uint64(cacheLen) * uint64(kvRows) * 4
	vk.cmdCopyBufferOffsetWin(cmd, devOutB.buffer, devK.buffer, kvCopyOffset, kvCopyBytes)
	vk.cmdCopyBufferOffsetWin(cmd, devOutC.buffer, devV.buffer, kvCopyOffset, kvCopyBytes)

	vk.computeBarrier(cmd)

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

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.q8ProjPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], 1)
	binary.LittleEndian.PutUint32(attPC[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[12:16], 0)
	binary.LittleEndian.PutUint32(attPC[16:20], 0)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(20), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)

	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.normPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[4:8], 1)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// === Readback phase ===
	if err := ensureHostBufferWin(vk, device, attRunner.memProps, &readbackNorm, qBytes); err != nil {
		return fmt.Errorf("readback norm buffer: %w", err)
	}
	vk.readbackFromDevice(cmd, devNorm, &readbackNorm, qBytes)

	if err := ensureHostBufferWin(vk, device, attRunner.memProps, &readbackResidual, qBytes); err != nil {
		return fmt.Errorf("readback residual buffer: %w", err)
	}
	vk.readbackFromDevice(cmd, devResidual, &readbackResidual, qBytes)

	if err := ensureHostBufferWin(vk, device, attRunner.memProps, &readbackK, uint64(kvRows)*4); err != nil {
		return fmt.Errorf("readback k buffer: %w", err)
	}
	vk.readbackFromDevice(cmd, devOutB, &readbackK, uint64(kvRows)*4)

	if err := ensureHostBufferWin(vk, device, attRunner.memProps, &readbackV, uint64(kvRows)*4); err != nil {
		return fmt.Errorf("readback v buffer: %w", err)
	}
	vk.readbackFromDevice(cmd, devOutC, &readbackV, uint64(kvRows)*4)

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

	if err := vk.readFloat32Into(device, readbackNorm, normOut[:qRows]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, readbackResidual, residual[:qRows]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, readbackK, outK[:kvRows]); err != nil {
		return err
	}
	return vk.readFloat32Into(device, readbackV, outV[:kvRows])
}
