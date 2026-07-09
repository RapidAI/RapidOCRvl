//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"paddleocrvl-go/internal/tensor"
	"unsafe"
)

// VulkanChainedSwiGLUDownAddRMSNormQ8 chains SwiGLU(gate,up) + Down projection +
// AddRMSNorm into a single command buffer with device-local intermediate buffers.
//
// All intermediate results (SwiGLU output, down projection output) stay in GPU
// device-local memory. Only inputs are uploaded via staging and only final
// outputs (normOut, residual) are read back.
func VulkanChainedSwiGLUDownAddRMSNormQ8(
	normOut, residual, x []float32,
	gate, up, down *tensor.Q8Matrix,
	normWeight []float32,
) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q8 chained swiglu+down+norm matrix")
	}
	rows := gate.Rows   // intermediate size
	cols := gate.Cols   // hidden size
	outRows := down.Rows // hidden size
	if cols != len(x) || rows != up.Rows || outRows != down.Cols || cols != down.Cols {
		return fmt.Errorf("chained q8 swiglu+down shape mismatch: gate=%dx%d up=%dx%d down=%dx%d x=%d",
			gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols, len(x))
	}
	if len(normOut) < outRows || len(residual) < outRows || len(normWeight) < outRows {
		return fmt.Errorf("invalid chained q8 swiglu+down+norm buffers")
	}

	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()

	vk := runner.vk
	device := runner.device

	// --- Compute byte sizes ---
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "chained q8 swiglu x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "chained q8 swiglu inter")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outRows, "chained q8 swiglu out")
	if err != nil {
		return err
	}

	gateLen, err := checkedQ8DataLenWin(gate.Rows, gate.Cols, "chained q8 swiglu gate")
	if err != nil {
		return err
	}
	upLen, err := checkedQ8DataLenWin(up.Rows, up.Cols, "chained q8 swiglu up")
	if err != nil {
		return err
	}
	downLen, err := checkedQ8DataLenWin(down.Rows, down.Cols, "chained q8 swiglu down")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrWin(gateLen, 4, "chained q8 swiglu gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrWin(upLen, 4, "chained q8 swiglu up data")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrWin(downLen, 4, "chained q8 swiglu down data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrWin(gate.Rows, "chained q8 swiglu gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrWin(up.Rows, "chained q8 swiglu up scale")
	if err != nil {
		return err
	}
	downScaleBytes, err := checkedFloat32ByteLenErrWin(down.Rows, "chained q8 swiglu down scale")
	if err != nil {
		return err
	}

	// --- Allocate device-local buffers for intermediates ---
	if err := vk.ensureDeviceBuffer(device, runner.memProps, &runner.devInter, interBytes); err != nil {
		return fmt.Errorf("ensure device buffer inter: %w", err)
	}
	devInter := runner.devInter

	if err := vk.ensureDeviceBuffer(device, runner.memProps, &runner.devOut, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer out: %w", err)
	}
	devOut := runner.devOut

	if err := vk.ensureDeviceBuffer(device, runner.memProps, &runner.devResidual, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer residual: %w", err)
	}
	devResidual := runner.devResidual

	if err := vk.ensureDeviceBuffer(device, runner.memProps, &runner.devNorm, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer norm: %w", err)
	}
	devNorm := runner.devNorm

	// Device-local input buffer (x)
	if err := vk.ensureDeviceBuffer(device, runner.memProps, &runner.devX, xBytes); err != nil {
		return fmt.Errorf("ensure device buffer x: %w", err)
	}
	devX := runner.devX

	// Device-local weight buffers
	devGateData, gateDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, runner.memProps, gate.Data[:gateLen], gateBytes, runner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer gateData: %w", err)
	}

	devUpData, upDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, runner.memProps, up.Data[:upLen], upBytes, runner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer upData: %w", err)
	}

	devDownData, downDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, runner.memProps, down.Data[:downLen], downBytes, runner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer downData: %w", err)
	}

	devGateScale, gateScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, runner.memProps, gate.Scale[:gate.Rows], gateScaleBytes, runner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer gateScale: %w", err)
	}

	devUpScale, upScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, runner.memProps, up.Scale[:up.Rows], upScaleBytes, runner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer upScale: %w", err)
	}

	devDownScale, downScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, runner.memProps, down.Scale[:down.Rows], downScaleBytes, runner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer downScale: %w", err)
	}

	devNormWeight, normWeightUpload, err := vk.getOrCreateDeviceFloat32Weight(device, runner.memProps, normWeight[:outRows], outBytes, runner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer normWeight: %w", err)
	}

	// --- Staging host buffers ---
	maxFloatBytes := xBytes
	if interBytes > maxFloatBytes {
		maxFloatBytes = interBytes
	}
	if outBytes > maxFloatBytes {
		maxFloatBytes = outBytes
	}
	if gateScaleBytes > maxFloatBytes {
		maxFloatBytes = gateScaleBytes
	}
	var stagingFloat vkHostBufferWin
	defer func() {
		if stagingFloat.buffer != 0 {
			vk.destroyBuffer(device, stagingFloat)
		}
	}()

	maxByteBytes := gateBytes
	if upBytes > maxByteBytes {
		maxByteBytes = upBytes
	}
	if downBytes > maxByteBytes {
		maxByteBytes = downBytes
	}
	var stagingBytes vkHostBufferWin
	defer func() {
		if stagingBytes.buffer != 0 {
			vk.destroyBuffer(device, stagingBytes)
		}
	}()

	// Readback buffers
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

	// --- Descriptor sets ---
	// SwiGLU set: 6 descriptors
	swiPoolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 6}
	swiDPCI := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&swiPoolSize)),
	}
	var swiPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&swiDPCI)), 0, uintptr(unsafe.Pointer(&swiPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool swiglu: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, swiPool, 0)

	var swiDS uintptr
	swiDSAI := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     swiPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&runner.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&swiDSAI)), uintptr(unsafe.Pointer(&swiDS))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets swiglu: %d", int32(res))
	}

	swiInfos := [6]vkDescriptorBufferInfo{
		{Buffer: devX.buffer, Range: xBytes},
		{Buffer: devGateData.buffer, Range: gateBytes},
		{Buffer: devUpData.buffer, Range: upBytes},
		{Buffer: devGateScale.buffer, Range: gateScaleBytes},
		{Buffer: devUpScale.buffer, Range: upScaleBytes},
		{Buffer: devInter.buffer, Range: interBytes},
	}
	var swiDSCache [6]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, swiDS, swiDSCache[:], swiInfos[:])

	// Down set: 4 descriptors
	downPoolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 4}
	downDPCI := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&downPoolSize)),
	}
	var downPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&downDPCI)), 0, uintptr(unsafe.Pointer(&downPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool down: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, downPool, 0)

	var downDS uintptr
	downDSAI := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     downPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&runner.downSetLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&downDSAI)), uintptr(unsafe.Pointer(&downDS))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets down: %d", int32(res))
	}

	downInfos := [4]vkDescriptorBufferInfo{
		{Buffer: devInter.buffer, Range: interBytes},
		{Buffer: devDownData.buffer, Range: downBytes},
		{Buffer: devDownScale.buffer, Range: downScaleBytes},
		{Buffer: devOut.buffer, Range: outBytes},
	}
	var downDSCache [4]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, downDS, downDSCache[:], downInfos[:])

	// Norm set: 4 descriptors
	normPoolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 4}
	normDPCI := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&normPoolSize)),
	}
	var normPool uintptr
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&normDPCI)), 0, uintptr(unsafe.Pointer(&normPool))); res != vkSuccess {
		return fmt.Errorf("vkCreateDescriptorPool norm: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyDescriptorPool, device, normPool, 0)

	var normDS uintptr
	normDSAI := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     normPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&runner.normSetLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&normDSAI)), uintptr(unsafe.Pointer(&normDS))); res != vkSuccess {
		return fmt.Errorf("vkAllocateDescriptorSets norm: %d", int32(res))
	}

	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: devResidual.buffer, Range: outBytes},
		{Buffer: devOut.buffer, Range: outBytes},
		{Buffer: devNormWeight.buffer, Range: outBytes},
		{Buffer: devNorm.buffer, Range: outBytes},
	}
	var normDSCache [4]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, normDS, normDSCache[:], normInfos[:])

	// --- Record command buffer ---
	if res := vk.call(vk.resetCommandPool, device, runner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool chained swiglu: %d", int32(res))
	}
	cmd := runner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer chained swiglu: %d", int32(res))
	}

	// === Upload phase ===
	if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devX, x[:cols]); err != nil {
		return fmt.Errorf("upload x: %w", err)
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devResidual, residual[:outRows]); err != nil {
		return fmt.Errorf("upload residual: %w", err)
	}
	if normWeightUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devNormWeight, normWeight[:outRows]); err != nil {
			return fmt.Errorf("upload normWeight: %w", err)
		}
	}
	if gateScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devGateScale, gate.Scale[:gate.Rows]); err != nil {
			return fmt.Errorf("upload gateScale: %w", err)
		}
	}
	if upScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devUpScale, up.Scale[:up.Rows]); err != nil {
			return fmt.Errorf("upload upScale: %w", err)
		}
	}
	if downScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, runner.memProps, &stagingFloat, devDownScale, down.Scale[:down.Rows]); err != nil {
			return fmt.Errorf("upload downScale: %w", err)
		}
	}
	if gateDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, runner.memProps, &stagingBytes, devGateData, unsafe.Slice((*byte)(unsafe.Pointer(&gate.Data[0])), gateLen)); err != nil {
			return fmt.Errorf("upload gateData: %w", err)
		}
	}
	if upDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, runner.memProps, &stagingBytes, devUpData, unsafe.Slice((*byte)(unsafe.Pointer(&up.Data[0])), upLen)); err != nil {
			return fmt.Errorf("upload upData: %w", err)
		}
	}
	if downDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, runner.memProps, &stagingBytes, devDownData, unsafe.Slice((*byte)(unsafe.Pointer(&down.Data[0])), downLen)); err != nil {
			return fmt.Errorf("upload downData: %w", err)
		}
	}

	// === Compute phase ===
	// Dispatch 1: SwiGLU gate+up
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, runner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, runner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&swiDS)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdPushConstants, cmd, runner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	// Barrier: SwiGLU writes visible to down reads
	vk.computeBarrier(cmd)

	// Dispatch 2: Down projection
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, runner.downPipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, runner.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&downDS)), 0, 0)
	vk.callVoid(vk.cmdPushConstants, cmd, runner.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(outRows), 1, 1)

	// Barrier: down writes visible to AddRMSNorm reads
	vk.computeBarrier(cmd)

	// Dispatch 3: AddRMSNorm
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, runner.normPipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, runner.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&normDS)), 0, 0)
	vk.callVoid(vk.cmdPushConstants, cmd, runner.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// === Readback phase ===
	if err := ensureHostBufferWin(vk, device, runner.memProps, &readbackNorm, outBytes); err != nil {
		return fmt.Errorf("readback norm: %w", err)
	}
	vk.readbackFromDevice(cmd, devNorm, &readbackNorm, outBytes)

	if err := ensureHostBufferWin(vk, device, runner.memProps, &readbackResidual, outBytes); err != nil {
		return fmt.Errorf("readback residual: %w", err)
	}
	vk.readbackFromDevice(cmd, devResidual, &readbackResidual, outBytes)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer chained swiglu: %d", int32(res))
	}

	// Single submit + wait
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&runner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences chained swiglu: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, runner.queue, 1, uintptr(unsafe.Pointer(&submit)), runner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit chained swiglu: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&runner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences chained swiglu: %d", int32(res))
	}

	// Read from mapped staging buffers
	if err := vk.readFloat32Into(device, readbackNorm, normOut[:outRows]); err != nil {
		return err
	}
	return vk.readFloat32Into(device, readbackResidual, residual[:outRows])
}
