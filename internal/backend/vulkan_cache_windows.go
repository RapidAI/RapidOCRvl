//go:build windows

package backend

import "unsafe"

func cachedFloat32BufferWin(vk *vulkanWin, device uintptr, memProps vkPhysicalDeviceMemoryProperties, data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	key := float32SliceKey(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil
		}
	}
	if old, ok := cache[key]; ok {
		if old.buffer.size >= size {
			if err := vk.writeFloat32(device, old.buffer, data); err != nil {
				return vkHostBufferWin{}, err
			}
			cache[key] = vulkanCachedFloat32BufferWin{buffer: old.buffer, length: len(data), hash: hashFloat32ForVulkanCache(data), fingerprint: fingerprint}
			return old.buffer, nil
		}
		vk.destroyBuffer(device, old.buffer)
		delete(cache, key)
	}
	buf, err := vk.newHostBuffer(device, memProps, size)
	if err != nil {
		return vkHostBufferWin{}, err
	}
	if err := vk.writeFloat32(device, buf, data); err != nil {
		vk.destroyBuffer(device, buf)
		return vkHostBufferWin{}, err
	}
	cache[key] = vulkanCachedFloat32BufferWin{buffer: buf, length: len(data), hash: hashFloat32ForVulkanCache(data), fingerprint: fingerprint}
	return buf, nil
}

func cachedInt8BufferWin(vk *vulkanWin, device uintptr, memProps vkPhysicalDeviceMemoryProperties, data []int8, size uint64, cache map[uintptr]vulkanCachedInt8BufferWin) (vkHostBufferWin, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintInt8ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil
		}
	}
	if old, ok := cache[key]; ok {
		if old.buffer.size >= size {
			if err := writeInt8Windows(vk, device, old.buffer, data); err != nil {
				return vkHostBufferWin{}, err
			}
			cache[key] = vulkanCachedInt8BufferWin{buffer: old.buffer, length: len(data), hash: hashInt8ForVulkanCache(data), fingerprint: fingerprint}
			return old.buffer, nil
		}
		vk.destroyBuffer(device, old.buffer)
		delete(cache, key)
	}
	buf, err := vk.newHostBuffer(device, memProps, size)
	if err != nil {
		return vkHostBufferWin{}, err
	}
	if err := writeInt8Windows(vk, device, buf, data); err != nil {
		vk.destroyBuffer(device, buf)
		return vkHostBufferWin{}, err
	}
	cache[key] = vulkanCachedInt8BufferWin{buffer: buf, length: len(data), hash: hashInt8ForVulkanCache(data), fingerprint: fingerprint}
	return buf, nil
}

func cachedByteBufferWin(vk *vulkanWin, device uintptr, memProps vkPhysicalDeviceMemoryProperties, data []byte, size uint64, cache map[uintptr]vulkanCachedByteBufferWin) (vkHostBufferWin, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintBytesForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil
		}
	}
	if old, ok := cache[key]; ok {
		if old.buffer.size >= size {
			if err := writeBytesWindows(vk, device, old.buffer, data); err != nil {
				return vkHostBufferWin{}, err
			}
			cache[key] = vulkanCachedByteBufferWin{buffer: old.buffer, length: len(data), hash: hashBytesForVulkanCache(data), fingerprint: fingerprint}
			return old.buffer, nil
		}
		vk.destroyBuffer(device, old.buffer)
		delete(cache, key)
	}
	buf, err := vk.newHostBuffer(device, memProps, size)
	if err != nil {
		return vkHostBufferWin{}, err
	}
	if err := writeBytesWindows(vk, device, buf, data); err != nil {
		vk.destroyBuffer(device, buf)
		return vkHostBufferWin{}, err
	}
	cache[key] = vulkanCachedByteBufferWin{buffer: buf, length: len(data), hash: hashBytesForVulkanCache(data), fingerprint: fingerprint}
	return buf, nil
}
