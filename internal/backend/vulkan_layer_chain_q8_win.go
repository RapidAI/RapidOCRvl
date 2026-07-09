//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"paddleocrvl-go/internal/tensor"
	"unsafe"
)

// VulkanLayerChainQ8Win fuses the Q8 attention chain (RMSNorm+QKV+MRoPE+
// attention+output+AddRMSNorm) and the Q8 MLP chain (SwiGLU+Down+AddRMSNorm)
// into a SINGLE command buffer with a SINGLE submit + fence wait.
//
// All intermediate results stay in GPU device-local memory between the
// attention and MLP dispatches. Only the final outputs (normOut for next
// layer, residual, K/V vectors) are read back to host-visible memory.
// This eliminates one full submit+fence-wait cycle per layer and avoids
// the host round-trip of normOut and residual between the attention and
// MLP chains.
func VulkanLayerChainQ8Win(
	normOut, residual, rawInput, ln1Weight []float32,
	qA, qB, qC *tensor.Q8Matrix,
	cosTable, sinTable []float32,
	attW *tensor.Q8Matrix, bias, ln2Weight []float32,
	kCache, vCache []float32,
	cacheEpoch uint64, cacheLen, hidden, numHeads, kvHeads, headDim int,
	outK, outV []float32,
	mlpGate, mlpUp, mlpDown *tensor.Q8Matrix,
	mlpNormWeight []float32,
) error {
	if qA == nil || qB == nil || qC == nil || attW == nil || mlpGate == nil || mlpUp == nil || mlpDown == nil {
		return fmt.Errorf("nil Vulkan q8 layer chain matrix")
	}
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	half := headDim / 2
	cols := qA.Cols
	if cols != hidden {
		return fmt.Errorf("layer chain q8 qkv cols mismatch: cols=%d hidden=%d", cols, hidden)
	}
	if qA.Rows != qRows || qB.Rows != kvRows || qC.Rows != kvRows {
		return fmt.Errorf("layer chain q8 qkv shape mismatch")
	}
	if len(rawInput) < hidden || len(ln1Weight) < hidden || len(cosTable) < half || len(sinTable) < half ||
		len(normOut) < hidden || len(residual) < hidden ||
		attW.Rows != qRows || attW.Cols != qRows || len(bias) < qRows || len(ln2Weight) < qRows ||
		len(outK) < kvRows || len(outV) < kvRows ||
		mlpGate.Cols != hidden || mlpUp.Cols != hidden || mlpDown.Cols != hidden || mlpDown.Rows != hidden ||
		len(mlpNormWeight) < hidden {
		return fmt.Errorf("invalid layer chain q8 buffers")
	}

	// === Get runners ===
	qkvRunner, err := getVulkanFusedMatVec3MRoPEQ8RunnerWindows()
	if err != nil {
		return err
	}
	attRunner, err := getVulkanTextAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	normRunner, err := getVulkanRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}
	swiRunner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}

	// Lock all runners (attention chain uses qkv+att+norm, MLP uses swi)
	qkvRunner.mu.Lock()
	defer qkvRunner.mu.Unlock()
	attRunner.mu.Lock()
	defer attRunner.mu.Unlock()
	normRunner.mu.Lock()
	defer normRunner.mu.Unlock()
	swiRunner.mu.Lock()
	defer swiRunner.mu.Unlock()

	vk := qkvRunner.vk
	device := qkvRunner.device

	// === Compute byte sizes (attention) ===
	xBytes := uint64(hidden) * 4
	qBytes := uint64(qRows) * 4
	kvDimBytes := uint64(kvRows) * 4
	aLen := uint64(qA.Rows) * uint64(qA.Cols)
	bLen := uint64(qB.Rows) * uint64(qB.Cols)
	cLen := uint64(qC.Rows) * uint64(qC.Cols)
	aBytes := (aLen + 3) &^ 3
	bBytes := (bLen + 3) &^ 3
	cBytes := (cLen + 3) &^ 3
	saBytes := uint64(qA.Rows) * 4
	sbBytes := uint64(qB.Rows) * 4
	scBytes := uint64(qC.Rows) * 4
	tableBytes := uint64(half) * 4
	outABytes := saBytes
	outBBytes := sbBytes
	outCBytes := scBytes
	wDataLen := uint64(attW.Rows) * uint64(attW.Cols)
	wDataBytes := (wDataLen + 3) &^ 3
	wScaleBytes := uint64(attW.Rows) * 4
	fullCacheBytes := uint64(cacheLen+1) * kvDimBytes

	// === Compute byte sizes (MLP) ===
	interBytes := uint64(mlpGate.Rows) * 4
	outBytes := uint64(mlpDown.Rows) * 4
	gateLen := uint64(mlpGate.Rows) * uint64(mlpGate.Cols)
	upLen := uint64(mlpUp.Rows) * uint64(mlpUp.Cols)
	downLen := uint64(mlpDown.Rows) * uint64(mlpDown.Cols)
	gateBytes := (gateLen + 3) &^ 3
	upBytes := (upLen + 3) &^ 3
	downBytes := (downLen + 3) &^ 3
	gateScaleBytes := uint64(mlpGate.Rows) * 4
	upScaleBytes := uint64(mlpUp.Rows) * 4
	downScaleBytes := uint64(mlpDown.Rows) * 4

	// === Device-local buffers (attention) ===
	if err := vk.ensureDeviceBuffer(device, qkvRunner.memProps, &qkvRunner.devOutA, outABytes); err != nil {
		return fmt.Errorf("ensure device buffer outA: %w", err)
	}
	devOutA := qkvRunner.devOutA
	if err := vk.ensureDeviceBuffer(device, qkvRunner.memProps, &qkvRunner.devOutB, outBBytes); err != nil {
		return fmt.Errorf("ensure device buffer outB: %w", err)
	}
	devOutB := qkvRunner.devOutB
	if err := vk.ensureDeviceBuffer(device, qkvRunner.memProps, &qkvRunner.devOutC, outCBytes); err != nil {
		return fmt.Errorf("ensure device buffer outC: %w", err)
	}
	devOutC := qkvRunner.devOutC

	var devK, devV vkDeviceBufferWin
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devOut, qBytes); err != nil {
		return fmt.Errorf("ensure device buffer out: %w", err)
	}
	devOut := attRunner.devOut
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devFinal, qBytes); err != nil {
		return fmt.Errorf("ensure device buffer final: %w", err)
	}
	devFinal := attRunner.devFinal
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devResidual, qBytes); err != nil {
		return fmt.Errorf("ensure device buffer residual: %w", err)
	}
	devResidual := attRunner.devResidual
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devNorm, qBytes); err != nil {
		return fmt.Errorf("ensure device buffer norm: %w", err)
	}
	devNorm := attRunner.devNorm

	// === Device-local weight buffers (attention) ===
	devWData, wDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, attRunner.memProps, attW.Data[:wDataLen], wDataBytes, attRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer wData: %w", err)
	}
	devWScale, wScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, attRunner.memProps, attW.Scale[:qRows], wScaleBytes, attRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer wScale: %w", err)
	}
	devLn2Weight, ln2Upload, err := vk.getOrCreateDeviceFloat32Weight(device, attRunner.memProps, ln2Weight[:qRows], qBytes, attRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer ln2Weight: %w", err)
	}
	devAData, aDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, qkvRunner.memProps, qA.Data[:aLen], aBytes, qkvRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer aData: %w", err)
	}
	devBData, bDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, qkvRunner.memProps, qB.Data[:bLen], bBytes, qkvRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer bData: %w", err)
	}
	devCData, cDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, qkvRunner.memProps, qC.Data[:cLen], cBytes, qkvRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer cData: %w", err)
	}
	devAScale, aScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, qA.Scale[:qRows], saBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer aScale: %w", err)
	}
	devBScale, bScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, qB.Scale[:kvRows], sbBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer bScale: %w", err)
	}
	devCScale, cScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, qC.Scale[:kvRows], scBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer cScale: %w", err)
	}
	if err := vk.ensureDeviceBuffer(device, qkvRunner.memProps, &qkvRunner.devX, xBytes); err != nil {
		return fmt.Errorf("ensure device buffer x: %w", err)
	}
	devX := qkvRunner.devX
	devCos, cosUpload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, cosTable[:half], tableBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer cos: %w", err)
	}
	devSin, sinUpload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, sinTable[:half], tableBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer sin: %w", err)
	}
	devLn1Weight, ln1Upload, err := vk.getOrCreateDeviceFloat32Weight(device, qkvRunner.memProps, ln1Weight[:hidden], xBytes, qkvRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer ln1Weight: %w", err)
	}

	// === Device-local KV cache (incremental upload) ===
	maxCacheLen := cap(kCache) / kvRows
	if cap(vCache)/kvRows > maxCacheLen {
		maxCacheLen = cap(vCache) / kvRows
	}
	fullMaxBytes := uint64(maxCacheLen) * kvDimBytes
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devK, fullMaxBytes); err != nil {
		return fmt.Errorf("ensure device buffer k (max): %w", err)
	}
	devK = attRunner.devK
	if err := vk.ensureDeviceBuffer(device, attRunner.memProps, &attRunner.devV, fullMaxBytes); err != nil {
		return fmt.Errorf("ensure device buffer v (max): %w", err)
	}
	devV = attRunner.devV

	cacheElems := cacheLen * kvRows
	fullUpload := attRunner.devCacheEpoch != cacheEpoch ||
		attRunner.devCacheKVDim != kvRows ||
		attRunner.devCacheMaxLen < maxCacheLen ||
		attRunner.devCacheUploaded > cacheLen ||
		attRunner.devCacheUploaded == 0

	// === Device-local buffers (MLP) ===
	if err := vk.ensureDeviceBuffer(device, swiRunner.memProps, &swiRunner.devInter, interBytes); err != nil {
		return fmt.Errorf("ensure device buffer inter: %w", err)
	}
	devInter := swiRunner.devInter
	if err := vk.ensureDeviceBuffer(device, swiRunner.memProps, &swiRunner.devOut, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer mlp out: %w", err)
	}
	devMlpOut := swiRunner.devOut
	if err := vk.ensureDeviceBuffer(device, swiRunner.memProps, &swiRunner.devResidual, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer mlp residual: %w", err)
	}
	devMlpResidual := swiRunner.devResidual
	if err := vk.ensureDeviceBuffer(device, swiRunner.memProps, &swiRunner.devNorm, outBytes); err != nil {
		return fmt.Errorf("ensure device buffer mlp norm: %w", err)
	}
	devMlpNorm := swiRunner.devNorm

	// MLP device-local weights
	devGateData, gateDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, swiRunner.memProps, mlpGate.Data[:gateLen], gateBytes, swiRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer gateData: %w", err)
	}
	devUpData, upDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, swiRunner.memProps, mlpUp.Data[:upLen], upBytes, swiRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer upData: %w", err)
	}
	devDownData, downDataUpload, err := vk.getOrCreateDeviceInt8Weight(device, swiRunner.memProps, mlpDown.Data[:downLen], downBytes, swiRunner.deviceInt8WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer downData: %w", err)
	}
	devGateScale, gateScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, swiRunner.memProps, mlpGate.Scale[:mlpGate.Rows], gateScaleBytes, swiRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer gateScale: %w", err)
	}
	devUpScale, upScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, swiRunner.memProps, mlpUp.Scale[:mlpUp.Rows], upScaleBytes, swiRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer upScale: %w", err)
	}
	devDownScale, downScaleUpload, err := vk.getOrCreateDeviceFloat32Weight(device, swiRunner.memProps, mlpDown.Scale[:mlpDown.Rows], downScaleBytes, swiRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer downScale: %w", err)
	}
	devMlpNormWeight, mlpNormWeightUpload, err := vk.getOrCreateDeviceFloat32Weight(device, swiRunner.memProps, mlpNormWeight[:hidden], outBytes, swiRunner.deviceFloat32WeightCache)
	if err != nil {
		return fmt.Errorf("cached device buffer mlpNormWeight: %w", err)
	}

	// === Staging buffers ===
	stagingFloat := &qkvRunner.stagingFloat
	stagingBytes := &qkvRunner.stagingBytes

	// === Descriptor sets (reuse persistent) ===
	// QKV descriptor set
	qkvDS := qkvRunner.descriptorSet
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

	// Attention descriptor set
	attDS := attRunner.descriptorSet
	attInfos := [10]vkDescriptorBufferInfo{
		{Buffer: devOutA.buffer, Range: qBytes},
		{Buffer: devK.buffer, Range: fullCacheBytes},
		{Buffer: devV.buffer, Range: fullCacheBytes},
		{Buffer: devOut.buffer, Range: qBytes},
		{Buffer: devWData.buffer, Range: wDataBytes},
		{Buffer: devWScale.buffer, Range: wScaleBytes},
		{Buffer: devFinal.buffer, Range: qBytes},
		{Buffer: devResidual.buffer, Range: qBytes},
		{Buffer: devLn2Weight.buffer, Range: qBytes},
		{Buffer: devNorm.buffer, Range: qBytes},
	}
	var attDSCache [10]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, attDS, attDSCache[:], attInfos[:])

	// RMSNorm descriptor set
	rmsDS := normRunner.descriptorSet
	rmsInfos := [3]vkDescriptorBufferInfo{
		{Buffer: devX.buffer, Range: xBytes},
		{Buffer: devLn1Weight.buffer, Range: xBytes},
		{Buffer: devX.buffer, Range: xBytes},
	}
	var rmsDSCache [3]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, rmsDS, rmsDSCache[:], rmsInfos[:])

	// SwiGLU descriptor set - use attRunner.devNorm as input (device-local, no upload)
	swiDS := swiRunner.descriptorSet
	swiInfos := [6]vkDescriptorBufferInfo{
		{Buffer: devNorm.buffer, Range: xBytes}, // x comes from attention chain's devNorm
		{Buffer: devGateData.buffer, Range: gateBytes},
		{Buffer: devUpData.buffer, Range: upBytes},
		{Buffer: devGateScale.buffer, Range: gateScaleBytes},
		{Buffer: devUpScale.buffer, Range: upScaleBytes},
		{Buffer: devInter.buffer, Range: interBytes},
	}
	var swiDSCache [6]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, swiDS, swiDSCache[:], swiInfos[:])

	// Down descriptor set
	downDS := swiRunner.downDescriptorSet
	downInfos := [4]vkDescriptorBufferInfo{
		{Buffer: devInter.buffer, Range: interBytes},
		{Buffer: devDownData.buffer, Range: downBytes},
		{Buffer: devDownScale.buffer, Range: downScaleBytes},
		{Buffer: devMlpOut.buffer, Range: outBytes},
	}
	var downDSCache [4]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, downDS, downDSCache[:], downInfos[:])

	// MLP norm descriptor set - use attRunner.devResidual as residual input
	normDS := swiRunner.normDescriptorSet
	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: devResidual.buffer, Range: outBytes}, // residual from attention chain
		{Buffer: devMlpOut.buffer, Range: outBytes},
		{Buffer: devMlpNormWeight.buffer, Range: outBytes},
		{Buffer: devMlpNorm.buffer, Range: outBytes},
	}
	var normDSCache [4]vulkanDescriptorBindingWin
	updateVulkanDescriptorBuffersWin(vk, device, normDS, normDSCache[:], normInfos[:])

	// === Record command buffer ===
	if res := vk.call(vk.resetCommandPool, device, qkvRunner.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool layer chain: %d", int32(res))
	}
	cmd := qkvRunner.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer layer chain: %d", int32(res))
	}

	// === Upload phase (attention) ===
	if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devX, rawInput[:hidden]); err != nil {
		return fmt.Errorf("upload rawInput: %w", err)
	}
	if ln1Upload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devLn1Weight, ln1Weight[:hidden]); err != nil {
			return fmt.Errorf("upload ln1Weight: %w", err)
		}
	}
	if cosUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devCos, cosTable[:half]); err != nil {
			return fmt.Errorf("upload cos: %w", err)
		}
	}
	if sinUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devSin, sinTable[:half]); err != nil {
			return fmt.Errorf("upload sin: %w", err)
		}
	}
	if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, stagingFloat, devResidual, residual[:qRows]); err != nil {
		return fmt.Errorf("upload residual: %w", err)
	}
	if ln2Upload {
		if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, stagingFloat, devLn2Weight, ln2Weight[:qRows]); err != nil {
			return fmt.Errorf("upload ln2Weight: %w", err)
		}
	}
	if aScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devAScale, qA.Scale[:qRows]); err != nil {
			return fmt.Errorf("upload aScale: %w", err)
		}
	}
	if bScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devBScale, qB.Scale[:kvRows]); err != nil {
			return fmt.Errorf("upload bScale: %w", err)
		}
	}
	if cScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, qkvRunner.memProps, stagingFloat, devCScale, qC.Scale[:kvRows]); err != nil {
			return fmt.Errorf("upload cScale: %w", err)
		}
	}
	if wScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, stagingFloat, devWScale, attW.Scale[:qRows]); err != nil {
			return fmt.Errorf("upload wScale: %w", err)
		}
	}
	if aDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, stagingBytes, devAData, unsafe.Slice((*byte)(unsafe.Pointer(&qA.Data[0])), aLen)); err != nil {
			return fmt.Errorf("upload aData: %w", err)
		}
	}
	if bDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, stagingBytes, devBData, unsafe.Slice((*byte)(unsafe.Pointer(&qB.Data[0])), bLen)); err != nil {
			return fmt.Errorf("upload bData: %w", err)
		}
	}
	if cDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, qkvRunner.memProps, stagingBytes, devCData, unsafe.Slice((*byte)(unsafe.Pointer(&qC.Data[0])), cLen)); err != nil {
			return fmt.Errorf("upload cData: %w", err)
		}
	}
	if wDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, attRunner.memProps, stagingBytes, devWData, unsafe.Slice((*byte)(unsafe.Pointer(&attW.Data[0])), wDataLen)); err != nil {
			return fmt.Errorf("upload wData: %w", err)
		}
	}

	// KV cache upload (incremental)
	if fullUpload {
		if cacheElems > 0 {
			if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, stagingFloat, devK, kCache[:cacheElems]); err != nil {
				return fmt.Errorf("upload kCache (full): %w", err)
			}
			if err := vk.uploadFloat32ToDevice(device, cmd, attRunner.memProps, stagingFloat, devV, vCache[:cacheElems]); err != nil {
				return fmt.Errorf("upload vCache (full): %w", err)
			}
		}
		attRunner.devCacheUploaded = cacheLen
	} else if cacheLen > attRunner.devCacheUploaded {
		startElem := attRunner.devCacheUploaded * kvRows
		newElems := (cacheLen - attRunner.devCacheUploaded) * kvRows
		if newElems > 0 && startElem+newElems <= cacheElems {
			if err := vk.uploadFloat32Offset(device, cmd, attRunner.memProps, stagingFloat, devK, kCache[startElem:startElem+newElems], uint64(startElem)*4); err != nil {
				return fmt.Errorf("upload kCache (incremental): %w", err)
			}
			if err := vk.uploadFloat32Offset(device, cmd, attRunner.memProps, stagingFloat, devV, vCache[startElem:startElem+newElems], uint64(startElem)*4); err != nil {
				return fmt.Errorf("upload vCache (incremental): %w", err)
			}
		}
		attRunner.devCacheUploaded = cacheLen
	}
	attRunner.devCacheEpoch = cacheEpoch
	attRunner.devCacheKVDim = kvRows
	attRunner.devCacheMaxLen = maxCacheLen

	// === MLP weight uploads ===
	if mlpNormWeightUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, swiRunner.memProps, stagingFloat, devMlpNormWeight, mlpNormWeight[:hidden]); err != nil {
			return fmt.Errorf("upload mlpNormWeight: %w", err)
		}
	}
	if gateScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, swiRunner.memProps, stagingFloat, devGateScale, mlpGate.Scale[:mlpGate.Rows]); err != nil {
			return fmt.Errorf("upload gateScale: %w", err)
		}
	}
	if upScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, swiRunner.memProps, stagingFloat, devUpScale, mlpUp.Scale[:mlpUp.Rows]); err != nil {
			return fmt.Errorf("upload upScale: %w", err)
		}
	}
	if downScaleUpload {
		if err := vk.uploadFloat32ToDevice(device, cmd, swiRunner.memProps, stagingFloat, devDownScale, mlpDown.Scale[:mlpDown.Rows]); err != nil {
			return fmt.Errorf("upload downScale: %w", err)
		}
	}
	if gateDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, swiRunner.memProps, stagingBytes, devGateData, unsafe.Slice((*byte)(unsafe.Pointer(&mlpGate.Data[0])), gateLen)); err != nil {
			return fmt.Errorf("upload gateData: %w", err)
		}
	}
	if upDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, swiRunner.memProps, stagingBytes, devUpData, unsafe.Slice((*byte)(unsafe.Pointer(&mlpUp.Data[0])), upLen)); err != nil {
			return fmt.Errorf("upload upData: %w", err)
		}
	}
	if downDataUpload {
		if err := vk.uploadBytesToDevice(device, cmd, swiRunner.memProps, stagingBytes, devDownData, unsafe.Slice((*byte)(unsafe.Pointer(&mlpDown.Data[0])), downLen)); err != nil {
			return fmt.Errorf("upload downData: %w", err)
		}
	}

	// === Compute phase: Attention ===
	// Dispatch 0: RMSNorm
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, normRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, normRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&rmsDS)), 0, 0)
	var rmsPC [8]byte
	binary.LittleEndian.PutUint32(rmsPC[0:4], uint32(hidden))
	binary.LittleEndian.PutUint32(rmsPC[4:8], 1)
	vk.callVoid(vk.cmdPushConstants, cmd, normRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&rmsPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)
	vk.computeBarrier(cmd)

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
	vk.computeBarrier(cmd)

	// Copy new K/V into device cache
	kvCopyBytes := uint64(kvRows) * 4
	kvCopyOffset := uint64(cacheLen) * uint64(kvRows) * 4
	vk.cmdCopyBufferOffsetWin(cmd, devOutB.buffer, devK.buffer, kvCopyOffset, kvCopyBytes)
	vk.cmdCopyBufferOffsetWin(cmd, devOutC.buffer, devV.buffer, kvCopyOffset, kvCopyBytes)
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
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	// Dispatch 4: AddRMSNorm (attention residual + output -> devNorm, devResidual)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, attRunner.normPipeline)
	binary.LittleEndian.PutUint32(attPC[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(attPC[4:8], 1)
	vk.callVoid(vk.cmdPushConstants, cmd, attRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&attPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)
	// devNorm now has the normalized hidden state (MLP input x)
	// devResidual now has the updated residual (MLP residual input)
	vk.computeBarrier(cmd)

	// === Compute phase: MLP ===
	// Dispatch 5: SwiGLU gate+up (reads devNorm as x, writes devInter)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, swiRunner.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, swiRunner.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&swiDS)), 0, 0)
	var mlpPC [8]byte
	binary.LittleEndian.PutUint32(mlpPC[0:4], uint32(mlpGate.Rows))
	binary.LittleEndian.PutUint32(mlpPC[4:8], uint32(hidden))
	vk.callVoid(vk.cmdPushConstants, cmd, swiRunner.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&mlpPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(mlpGate.Rows), 1, 1)
	vk.computeBarrier(cmd)

	// Dispatch 6: Down projection (reads devInter, writes devMlpOut)
	binary.LittleEndian.PutUint32(mlpPC[0:4], uint32(hidden))
	binary.LittleEndian.PutUint32(mlpPC[4:8], uint32(mlpGate.Rows))
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, swiRunner.downPipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, swiRunner.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&downDS)), 0, 0)
	vk.callVoid(vk.cmdPushConstants, cmd, swiRunner.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&mlpPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(hidden), 1, 1)
	vk.computeBarrier(cmd)

	// Dispatch 7: AddRMSNorm (reads devResidual + devMlpOut, writes devMlpNorm, devMlpResidual)
	binary.LittleEndian.PutUint32(mlpPC[0:4], uint32(hidden))
	binary.LittleEndian.PutUint32(mlpPC[4:8], 1)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, swiRunner.normPipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, swiRunner.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&normDS)), 0, 0)
	vk.callVoid(vk.cmdPushConstants, cmd, swiRunner.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(8), uintptr(unsafe.Pointer(&mlpPC[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)

	// === Readback phase ===
	readbackNorm := &swiRunner.readbackNorm
	readbackResidual := &swiRunner.readbackResidual
	readbackK := &attRunner.readbackK
	readbackV := &attRunner.readbackV

	if err := ensureHostBufferWin(vk, device, swiRunner.memProps, readbackNorm, outBytes); err != nil {
		return fmt.Errorf("readback mlp norm: %w", err)
	}
	vk.readbackFromDevice(cmd, devMlpNorm, readbackNorm, outBytes)

	if err := ensureHostBufferWin(vk, device, swiRunner.memProps, readbackResidual, outBytes); err != nil {
		return fmt.Errorf("readback mlp residual: %w", err)
	}
	vk.readbackFromDevice(cmd, devMlpResidual, readbackResidual, outBytes)

	if err := ensureHostBufferWin(vk, device, attRunner.memProps, readbackK, uint64(kvRows)*4); err != nil {
		return fmt.Errorf("readback k: %w", err)
	}
	vk.readbackFromDevice(cmd, devOutB, readbackK, uint64(kvRows)*4)

	if err := ensureHostBufferWin(vk, device, attRunner.memProps, readbackV, uint64(kvRows)*4); err != nil {
		return fmt.Errorf("readback v: %w", err)
	}
	vk.readbackFromDevice(cmd, devOutC, readbackV, uint64(kvRows)*4)

	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer layer chain: %d", int32(res))
	}

	// Single submit + wait
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences layer chain: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&cmd)),
	}
	if res := vk.call(vk.queueSubmit, qkvRunner.queue, 1, uintptr(unsafe.Pointer(&submit)), qkvRunner.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit layer chain: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&qkvRunner.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences layer chain: %d", int32(res))
	}

	// Read results from mapped host buffers
	if err := vk.readFloat32Into(device, *readbackNorm, normOut[:hidden]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, *readbackResidual, residual[:hidden]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, *readbackK, outK[:kvRows]); err != nil {
		return err
	}
	return vk.readFloat32Into(device, *readbackV, outV[:kvRows])
}