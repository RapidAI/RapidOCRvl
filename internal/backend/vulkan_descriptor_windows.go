//go:build windows

package backend

import "unsafe"

type vulkanDescriptorBindingWin struct {
	buffer uintptr
	offset uint64
	size   uint64
	valid  bool
}

func updateVulkanDescriptorBuffersWin(vk *vulkanWin, device, descriptorSet uintptr, cache []vulkanDescriptorBindingWin, infos []vkDescriptorBufferInfo) {
	var writes [32]vkWriteDescriptorSet
	changed := 0
	for i := range infos {
		info := infos[i]
		if i < len(cache) {
			cached := &cache[i]
			if cached.valid && cached.buffer == info.Buffer && cached.offset == info.Offset && cached.size == info.Range {
				continue
			}
			*cached = vulkanDescriptorBindingWin{
				buffer: info.Buffer,
				offset: info.Offset,
				size:   info.Range,
				valid:  true,
			}
		}
		writes[changed] = vkWriteDescriptorSet{
			SType:           vkStructureTypeWriteDescriptorSet,
			DstSet:          descriptorSet,
			DstBinding:      uint32(i),
			DescriptorCount: 1,
			DescriptorType:  vkDescriptorTypeStorageBuffer,
			PBufferInfo:     uintptr(unsafe.Pointer(&infos[i])),
		}
		changed++
	}
	if changed == 0 {
		return
	}
	vk.callVoid(vk.updateDescriptorSets, device, uintptr(changed), uintptr(unsafe.Pointer(&writes[0])), 0, 0)
}

func updateVulkanDescriptorBindingsWin(vk *vulkanWin, device, descriptorSet uintptr, cache []vulkanDescriptorBindingWin, bindings []uint32, infos []vkDescriptorBufferInfo) {
	var writes [32]vkWriteDescriptorSet
	changed := 0
	for i := range infos {
		binding := bindings[i]
		info := infos[i]
		if int(binding) < len(cache) {
			cached := &cache[binding]
			if cached.valid && cached.buffer == info.Buffer && cached.offset == info.Offset && cached.size == info.Range {
				continue
			}
			*cached = vulkanDescriptorBindingWin{
				buffer: info.Buffer,
				offset: info.Offset,
				size:   info.Range,
				valid:  true,
			}
		}
		writes[changed] = vkWriteDescriptorSet{
			SType:           vkStructureTypeWriteDescriptorSet,
			DstSet:          descriptorSet,
			DstBinding:      binding,
			DescriptorCount: 1,
			DescriptorType:  vkDescriptorTypeStorageBuffer,
			PBufferInfo:     uintptr(unsafe.Pointer(&infos[i])),
		}
		changed++
	}
	if changed == 0 {
		return
	}
	vk.callVoid(vk.updateDescriptorSets, device, uintptr(changed), uintptr(unsafe.Pointer(&writes[0])), 0, 0)
}
