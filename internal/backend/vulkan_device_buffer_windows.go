package backend

import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

// vkDeviceBufferWin is a device-local buffer (GPU memory, not host-visible).
// Used for intermediate results that stay on the GPU between chained dispatches,
// avoiding host round-trips.
type vkDeviceBufferWin struct {
	buffer uintptr
	memory uintptr
	size   uint64
}

const (
	vkMemoryPropertyDeviceLocalBit = 1
	vkBufferUsageTransferSrcBit    = 0x00000001
	vkBufferUsageTransferDstBit    = 0x00000002
)

// vkBufferCopy describes a region for vkCmdCopyBuffer.
type vkBufferCopy struct {
	SrcOffset uint64
	DstOffset uint64
	Size      uint64
}

// newDeviceBuffer allocates a device-local buffer of the given size.
// The buffer is usable as a storage buffer (compute read/write) and as a
// transfer destination (for host->device copies via a staging buffer).
func (v *vulkanWin) newDeviceBuffer(device uintptr, memProps vkPhysicalDeviceMemoryProperties, size uint64) (vkDeviceBufferWin, error) {
	var out vkDeviceBufferWin
	out.size = size
	usage := uint32(vkBufferUsageStorageBufferBit | vkBufferUsageTransferDstBit | vkBufferUsageTransferSrcBit)
	bci := vkBufferCreateInfo{SType: vkStructureTypeBufferCreateInfo, Size: size, Usage: usage, SharingMode: vkSharingModeExclusive}
	if res := v.call(v.createBuffer, device, uintptr(unsafe.Pointer(&bci)), 0, uintptr(unsafe.Pointer(&out.buffer))); res != vkSuccess {
		return out, fmt.Errorf("vkCreateBuffer device: %d", int32(res))
	}
	var req vkMemoryRequirements
	v.callVoid(v.getBufferMemoryReqs, device, out.buffer, uintptr(unsafe.Pointer(&req)))
	memType, ok := findVulkanMemoryTypeWindows(memProps, req.MemoryTypeBits, vkMemoryPropertyDeviceLocalBit)
	if !ok {
		// Fallback: use host-visible memory if no pure device-local type is available.
		memType, ok = findVulkanMemoryTypeWindows(memProps, req.MemoryTypeBits, vkMemoryPropertyHostVisibleBit|vkMemoryPropertyHostCoherentBit)
		if !ok {
			v.destroyDeviceBuffer(device, out)
			return vkDeviceBufferWin{}, fmt.Errorf("no suitable memory type for device buffer")
		}
	}
	mai := vkMemoryAllocateInfo{SType: vkStructureTypeMemoryAllocateInfo, AllocationSize: req.Size, MemoryTypeIndex: memType}
	if res := v.call(v.allocateMemory, device, uintptr(unsafe.Pointer(&mai)), 0, uintptr(unsafe.Pointer(&out.memory))); res != vkSuccess {
		v.destroyDeviceBuffer(device, out)
		return vkDeviceBufferWin{}, fmt.Errorf("vkAllocateMemory device: %d", int32(res))
	}
	if res := v.call(v.bindBufferMemory, device, out.buffer, out.memory, 0); res != vkSuccess {
		v.destroyDeviceBuffer(device, out)
		return vkDeviceBufferWin{}, fmt.Errorf("vkBindBufferMemory device: %d", int32(res))
	}
	return out, nil
}

func (v *vulkanWin) destroyDeviceBuffer(device uintptr, b vkDeviceBufferWin) {
	if b.buffer != 0 {
		v.callVoid(v.destroyBufferProc, device, b.buffer, 0)
	}
	if b.memory != 0 {
		v.callVoid(v.freeMemory, device, b.memory, 0)
	}
}

// deviceBufferPool manages reusable device-local buffers keyed by size.
// Buffers are pooled to avoid repeated allocation/deallocation overhead.
type deviceBufferPoolWin struct {
	mu      sync.Mutex
	vk      *vulkanWin
	device  uintptr
	memProps vkPhysicalDeviceMemoryProperties
	pool    []vkDeviceBufferWin
}

func newDeviceBufferPoolWin(vk *vulkanWin, device uintptr, memProps vkPhysicalDeviceMemoryProperties) *deviceBufferPoolWin {
	return &deviceBufferPoolWin{vk: vk, device: device, memProps: memProps}
}

// get obtains a device buffer of at least `size` bytes.  Reuses a pooled buffer
// if one is large enough, otherwise allocates a new one.
func (p *deviceBufferPoolWin) get(size uint64) (vkDeviceBufferWin, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// Try to find a pooled buffer that is large enough.
	best := -1
	for i, b := range p.pool {
		if b.size >= size {
			if best < 0 || b.size < p.pool[best].size {
				best = i
			}
		}
	}
	if best >= 0 {
		b := p.pool[best]
		p.pool[best] = p.pool[len(p.pool)-1]
		p.pool = p.pool[:len(p.pool)-1]
		return b, nil
	}
	return p.vk.newDeviceBuffer(p.device, p.memProps, size)
}

// put returns a buffer to the pool for future reuse.
func (p *deviceBufferPoolWin) put(b vkDeviceBufferWin) {
	if b.buffer == 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pool = append(p.pool, b)
}

// destroyAll frees all pooled buffers.
func (p *deviceBufferPoolWin) destroyAll() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, b := range p.pool {
		p.vk.destroyDeviceBuffer(p.device, b)
	}
	p.pool = nil
}

// cmdCopyBufferWin records a vkCmdCopyBuffer into the command buffer.
// src must have TRANSFER_SRC usage, dst must have TRANSFER_DST usage.
func (v *vulkanWin) cmdCopyBufferWin(cmd, src, dst uintptr, size uint64) {
	region := vkBufferCopy{SrcOffset: 0, DstOffset: 0, Size: size}
	v.callVoid(v.cmdCopyBuffer, cmd, src, dst, 1, uintptr(unsafe.Pointer(&region)))
}

// cmdCopyBufferOffsetWin copies size bytes from src to dst at the given
// destination offset.  Used to append a single token's k/v into the KV cache
// buffer during chained dispatches.
func (v *vulkanWin) cmdCopyBufferOffsetWin(cmd, src, dst uintptr, dstOffset, size uint64) {
	region := vkBufferCopy{SrcOffset: 0, DstOffset: dstOffset, Size: size}
	v.callVoid(v.cmdCopyBuffer, cmd, src, dst, 1, uintptr(unsafe.Pointer(&region)))
}

// uploadToDevice copies host data to a device-local buffer via a staging
// host-visible buffer and a vkCmdCopyBuffer recorded into the given command
// buffer.  The staging buffer is allocated from the pool and returned for
// later cleanup.
func (v *vulkanWin) uploadToDevice(device, cmd uintptr, hostBuf *vkHostBufferWin, deviceBuf vkDeviceBufferWin, data []float32) error {
	byteLen := uint64(len(data)) * 4
	if byteLen == 0 {
		return nil
	}
	if err := v.writeFloat32(device, *hostBuf, data); err != nil {
				return err
	}
	// Record barrier: host writes must be visible before copy
	hostBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: 0, // implicit host write
		DstAccessMask: vkAccessTransferReadBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageHostBit, vkPipelineStageTransferBit,
		0, 1, uintptr(unsafe.Pointer(&hostBarrier)), 0, 0, 0, 0)
	// Record the copy
	v.cmdCopyBufferWin(cmd, hostBuf.buffer, deviceBuf.buffer, byteLen)
	// Record barrier: transfer writes must be visible before compute reads
	devBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: vkAccessTransferWriteBit,
		DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageTransferBit, vkPipelineStageComputeShaderBit,
		0, 1, uintptr(unsafe.Pointer(&devBarrier)), 0, 0, 0, 0)
	return nil
}


// uploadBytesToDevice copies arbitrary byte data to a device-local buffer via
// a staging host-visible buffer.  Used for int8/byte weight data.
func (v *vulkanWin) uploadBytesToDevice(device, cmd uintptr, memProps vkPhysicalDeviceMemoryProperties, hostBuf *vkHostBufferWin, deviceBuf vkDeviceBufferWin, data []byte) error {
	byteLen := uint64(len(data))
	if byteLen == 0 {
		return nil
	}
	// Ensure staging host buffer is large enough
	if hostBuf.buffer == 0 || hostBuf.size < byteLen {
		if hostBuf.buffer != 0 {
			v.destroyBuffer(device, *hostBuf)
		}
		next, err := v.newHostBuffer(device, memProps, byteLen)
		if err != nil {
			return err
		}
		*hostBuf = next
	}
	// Write bytes into mapped staging buffer
	dst := unsafe.Slice((*byte)(hostBuf.mapped), int(hostBuf.size))
	copy(dst[:len(data)], data)
	// Barrier: host writes visible before transfer read
	hostBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: 0,
		DstAccessMask: vkAccessTransferReadBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageHostBit, vkPipelineStageTransferBit,
		0, 1, uintptr(unsafe.Pointer(&hostBarrier)), 0, 0, 0, 0)
	// Copy
	region := vkBufferCopy{SrcOffset: 0, DstOffset: 0, Size: byteLen}
	v.callVoid(v.cmdCopyBuffer, cmd, hostBuf.buffer, deviceBuf.buffer, 1, uintptr(unsafe.Pointer(&region)))
	// Barrier: transfer writes visible before compute reads
	devBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: vkAccessTransferWriteBit,
		DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageTransferBit, vkPipelineStageComputeShaderBit,
		0, 1, uintptr(unsafe.Pointer(&devBarrier)), 0, 0, 0, 0)
	return nil
}

// uploadFloat32ToDevice copies float32 data to a device-local buffer via staging.
func (v *vulkanWin) uploadFloat32ToDevice(device, cmd uintptr, memProps vkPhysicalDeviceMemoryProperties, hostBuf *vkHostBufferWin, deviceBuf vkDeviceBufferWin, data []float32) error {
	return v.uploadBytesToDevice(device, cmd, memProps, hostBuf, deviceBuf, unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4))
}

// readbackFromDevice copies data from a device-local buffer to a host-visible
// buffer via vkCmdCopyBuffer recorded into the given command buffer.
func (v *vulkanWin) readbackFromDevice(cmd uintptr, deviceBuf vkDeviceBufferWin, hostBuf *vkHostBufferWin, byteLen uint64) {
	// Barrier: compute writes must be visible before transfer reads
	devBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: vkAccessShaderWriteBit,
		DstAccessMask: vkAccessTransferReadBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageComputeShaderBit, vkPipelineStageTransferBit,
		0, 1, uintptr(unsafe.Pointer(&devBarrier)), 0, 0, 0, 0)
	// Record the copy
	v.cmdCopyBufferWin(cmd, deviceBuf.buffer, hostBuf.buffer, byteLen)
	// Barrier: transfer writes must be visible before host reads
	hostBarrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: vkAccessTransferWriteBit,
		DstAccessMask: 0, // implicit host read
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageTransferBit, vkPipelineStageHostBit,
		0, 1, uintptr(unsafe.Pointer(&hostBarrier)), 0, 0, 0, 0)
}

// computeBarrier records a memory barrier ensuring that compute shader writes
// from the previous dispatch are visible to the next dispatch's reads.
func (v *vulkanWin) computeBarrier(cmd uintptr) {
	barrier := vkMemoryBarrier{
		SType:         vkStructureTypeMemoryBarrier,
		SrcAccessMask: vkAccessShaderWriteBit,
		DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit,
	}
	v.callVoid(v.cmdPipelineBarrier, cmd,
		vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit,
		0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
}
// vulkanCommandBatchWin chains multiple compute dispatches into a single
// command buffer with pipeline barriers between them.  Intermediate results
// stay in device-local buffers, eliminating host round-trips between ops.
//
// Usage:
//   batch := newVulkanCommandBatchWin(ctx, pool)
//   batch.begin()
//   batch.uploadInput(hostData)        // host -> device via staging copy
//   batch.dispatch(pipeline1, set1, pc1, groups1)  // op 1 on device buffer
//   batch.barrier()                    // ensure op1 writes visible to op2
//   batch.dispatch(pipeline2, set2, pc2, groups2)  // op 2 reads op1 output
//   batch.readbackOutput(hostBuf)      // device -> host via staging copy
//   batch.end()                        // finish recording
//   batch.submitAndWait()              // one submit + one wait
type vulkanCommandBatchWin struct {
	vk          *vulkanWin
	device      uintptr
	queue       uintptr
	commandPool uintptr
	cmd         uintptr
	fence       uintptr
	pool        *deviceBufferPoolWin

	hostStaging vkHostBufferWin // reusable host-visible staging buffer
	stagingSize uint64
}

// newVulkanCommandBatchWin creates a command batcher that uses its own
// command pool and fence, sharing the device/queue from the shared context.
func newVulkanCommandBatchWin(ctx vulkanSharedContextWin) (*vulkanCommandBatchWin, error) {
	b := &vulkanCommandBatchWin{
		vk:     ctx.vk,
		device: ctx.device,
		queue:  ctx.queue,
		pool:   newDeviceBufferPoolWin(ctx.vk, ctx.device, ctx.memProps),
	}
	vk := ctx.vk
	cpci := vkCommandPoolCreateInfo{
		SType:            vkStructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: ctx.queueFamily,
	}
	if res := vk.call(vk.createCommandPool, ctx.device, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&b.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool batch: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{
		SType:              vkStructureTypeCommandBufferAllocateInfo,
		CommandPool:        b.commandPool,
		Level:              vkCommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	if res := vk.call(vk.allocateCommandBuffers, ctx.device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&b.cmd))); res != vkSuccess {
		vk.callVoid(vk.destroyCommandPool, ctx.device, b.commandPool, 0)
		return nil, fmt.Errorf("vkAllocateCommandBuffers batch: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, ctx.device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&b.fence))); res != vkSuccess {
		vk.callVoid(vk.destroyCommandPool, ctx.device, b.commandPool, 0)
		return nil, fmt.Errorf("vkCreateFence batch: %d", int32(res))
	}
	return b, nil
}

func (b *vulkanCommandBatchWin) destroy() {
	vk := b.vk
	if b.fence != 0 {
		vk.callVoid(vk.destroyFence, b.device, b.fence, 0)
	}
	if b.commandPool != 0 {
		vk.callVoid(vk.destroyCommandPool, b.device, b.commandPool, 0)
	}
	if b.hostStaging.buffer != 0 {
		vk.destroyBuffer(b.device, b.hostStaging)
	}
	b.pool.destroyAll()
}

// begin starts recording a new command buffer.
func (b *vulkanCommandBatchWin) begin() error {
	vk := b.vk
	if res := vk.call(vk.resetCommandPool, b.device, b.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool batch: %d", int32(res))
	}
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo, Flags: vkCommandBufferUsageOneTimeSubmitBit}
	if res := vk.call(vk.beginCommandBuffer, b.cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer batch: %d", int32(res))
	}
	return nil
}

// end finishes recording the command buffer.
func (b *vulkanCommandBatchWin) end() error {
	vk := b.vk
	if res := vk.call(vk.endCommandBuffer, b.cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer batch: %d", int32(res))
	}
	return nil
}

// submitAndWait submits the recorded command buffer and waits for completion.
func (b *vulkanCommandBatchWin) submitAndWait() error {
	vk := b.vk
	if res := vk.call(vk.resetFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences batch: %d", int32(res))
	}
	submit := vkSubmitInfo{
		SType:              vkStructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    uintptr(unsafe.Pointer(&b.cmd)),
	}
	if res := vk.call(vk.queueSubmit, b.queue, 1, uintptr(unsafe.Pointer(&submit)), b.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit batch: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences batch: %d", int32(res))
	}
	return nil
}

// dispatch records a compute dispatch with the given pipeline, descriptor set,
// push constants, and group count.
func (b *vulkanCommandBatchWin) dispatch(pipeline, descriptorSet, pipelineLayout uintptr, pushConstants []byte, groupsX, groupsY, groupsZ uint32) {
	vk := b.vk
	vk.callVoid(vk.cmdBindPipeline, b.cmd, vkPipelineBindPointCompute, pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, b.cmd, vkPipelineBindPointCompute, pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&descriptorSet)), 0, 0)
	if len(pushConstants) > 0 {
		vk.callVoid(vk.cmdPushConstants, b.cmd, pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pushConstants)), uintptr(unsafe.Pointer(&pushConstants[0])))
	}
	vk.callVoid(vk.cmdDispatch, b.cmd, uintptr(groupsX), uintptr(groupsY), uintptr(groupsZ))
}

// barrier records a compute-to-compute memory barrier.
func (b *vulkanCommandBatchWin) barrier() {
	b.vk.computeBarrier(b.cmd)
}

// uploadInput copies host data to a device buffer via a staging buffer.
// The staging host buffer is reused across calls.
func (b *vulkanCommandBatchWin) uploadInput(data []float32, deviceBuf vkDeviceBufferWin) error {
	byteLen := uint64(len(data)) * 4
	if byteLen == 0 {
		return nil
	}
	// Ensure staging host buffer is large enough
	if b.hostStaging.buffer == 0 || b.hostStaging.size < byteLen {
		if b.hostStaging.buffer != 0 {
			b.vk.destroyBuffer(b.device, b.hostStaging)
		}
		staging, err := b.vk.newHostBuffer(b.device, b.pool.memProps, byteLen)
		if err != nil {
			return err
		}
		b.hostStaging = staging
		b.stagingSize = byteLen
	}
	return b.vk.uploadToDevice(b.device, b.cmd, &b.hostStaging, deviceBuf, data)
}

// readbackOutput copies data from a device buffer to the staging host buffer.
// After submitAndWait completes, call readStaging to get the data.
func (b *vulkanCommandBatchWin) readbackOutput(deviceBuf vkDeviceBufferWin, byteLen uint64) error {
	if byteLen == 0 {
		return nil
	}
	// Ensure staging host buffer is large enough
	if b.hostStaging.buffer == 0 || b.hostStaging.size < byteLen {
		if b.hostStaging.buffer != 0 {
			b.vk.destroyBuffer(b.device, b.hostStaging)
		}
		staging, err := b.vk.newHostBuffer(b.device, b.pool.memProps, byteLen)
		if err != nil {
			return err
		}
		b.hostStaging = staging
		b.stagingSize = byteLen
	}
	b.vk.readbackFromDevice(b.cmd, deviceBuf, &b.hostStaging, byteLen)
	return nil
}

// readStaging copies data from the mapped staging buffer into the output slice.
func (b *vulkanCommandBatchWin) readStaging(out []float32) error {
	return b.vk.readFloat32Into(b.device, b.hostStaging, out)
}

// getDeviceBuffer allocates a device buffer from the pool.
func (b *vulkanCommandBatchWin) getDeviceBuffer(size uint64) (vkDeviceBufferWin, error) {
	return b.pool.get(size)
}

// returnDeviceBuffer returns a buffer to the pool.
func (b *vulkanCommandBatchWin) returnDeviceBuffer(buf vkDeviceBufferWin) {
	b.pool.put(buf)
}

// vulkanCachedDeviceInt8BufferWin caches device-local buffers for int8 weight data.
// Weights are uploaded once and reused across layers/tokens.
type vulkanCachedDeviceInt8BufferWin struct {
	buffer      vkDeviceBufferWin
	length      int
	fingerprint uint64
}

// vulkanCachedDeviceFloat32BufferWin caches device-local buffers for float32 weight data.
type vulkanCachedDeviceFloat32BufferWin struct {
	buffer      vkDeviceBufferWin
	length      int
	fingerprint uint64
}

// ensureDeviceInt8Weight allocates or reuses a device-local buffer for int8 weight data.
// On first call (cache miss), uploads via staging into the command buffer.
// On subsequent calls (cache hit), returns the existing device buffer — no upload needed.
func (v *vulkanWin) ensureDeviceInt8Weight(
	device, cmd uintptr,
	memProps vkPhysicalDeviceMemoryProperties,
	data []int8, size uint64,
	cache map[uintptr]vulkanCachedDeviceInt8BufferWin,
	staging *vkHostBufferWin,
) (vkDeviceBufferWin, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintInt8ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil // cache hit — no upload
		}
	}
	// Cache miss: allocate device buffer and upload
	devBuf, err := v.newDeviceBuffer(device, memProps, size)
	if err != nil {
		return vkDeviceBufferWin{}, err
	}
	// Upload via staging into the command buffer
	dataBytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data))
	if err := v.uploadBytesToDevice(device, cmd, memProps, staging, devBuf, dataBytes); err != nil {
		v.destroyDeviceBuffer(device, devBuf)
		return vkDeviceBufferWin{}, err
	}
	cache[key] = vulkanCachedDeviceInt8BufferWin{buffer: devBuf, length: len(data), fingerprint: fingerprint}
	return devBuf, nil
}

// ensureDeviceFloat32Weight allocates or reuses a device-local buffer for float32 weight data.
func (v *vulkanWin) ensureDeviceFloat32Weight(
	device, cmd uintptr,
	memProps vkPhysicalDeviceMemoryProperties,
	data []float32, size uint64,
	cache map[uintptr]vulkanCachedDeviceFloat32BufferWin,
	staging *vkHostBufferWin,
) (vkDeviceBufferWin, error) {
	key := float32SliceKey(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil // cache hit — no upload
		}
	}
	// Cache miss: allocate device buffer and upload
	devBuf, err := v.newDeviceBuffer(device, memProps, size)
	if err != nil {
		return vkDeviceBufferWin{}, err
	}
	if err := v.uploadFloat32ToDevice(device, cmd, memProps, staging, devBuf, data); err != nil {
		v.destroyDeviceBuffer(device, devBuf)
		return vkDeviceBufferWin{}, err
	}
	cache[key] = vulkanCachedDeviceFloat32BufferWin{buffer: devBuf, length: len(data), fingerprint: fingerprint}
	return devBuf, nil
}


// getOrCreateDeviceInt8Weight returns a cached device buffer for int8 weights,
// or allocates a new one. Does NOT upload — caller must upload if needed.
func (v *vulkanWin) getOrCreateDeviceInt8Weight(
	device uintptr,
	memProps vkPhysicalDeviceMemoryProperties,
	data []int8, size uint64,
	cache map[uintptr]vulkanCachedDeviceInt8BufferWin,
) (vkDeviceBufferWin, bool, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintInt8ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, false, nil // cache hit — no upload needed
		}
	}
	// Cache miss: allocate new device buffer
	devBuf, err := v.newDeviceBuffer(device, memProps, size)
	if err != nil {
		return vkDeviceBufferWin{}, false, err
	}
	cache[key] = vulkanCachedDeviceInt8BufferWin{buffer: devBuf, length: len(data), fingerprint: fingerprint}
	return devBuf, true, nil // needs upload
}

// getOrCreateDeviceFloat32Weight returns a cached device buffer for float32 weights.
func (v *vulkanWin) getOrCreateDeviceFloat32Weight(
	device uintptr,
	memProps vkPhysicalDeviceMemoryProperties,
	data []float32, size uint64,
	cache map[uintptr]vulkanCachedDeviceFloat32BufferWin,
) (vkDeviceBufferWin, bool, error) {
	key := float32SliceKey(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, false, nil
		}
	}
	devBuf, err := v.newDeviceBuffer(device, memProps, size)
	if err != nil {
		return vkDeviceBufferWin{}, false, err
	}
	cache[key] = vulkanCachedDeviceFloat32BufferWin{buffer: devBuf, length: len(data), fingerprint: fingerprint}
	return devBuf, true, nil
}
