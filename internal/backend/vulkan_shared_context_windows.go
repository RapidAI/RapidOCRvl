package backend

import (
	"fmt"
	"sync"
	"unsafe"
)

// vulkanSharedContextWin holds a single Vulkan instance, logical device,
// compute queue and memory properties that are reused by all Windows Vulkan
// runners.  Sharing one device avoids creating ~21 separate logical devices
// (each with its own driver-side overhead) and lets the GPU scheduler batch
// work from different runners onto the same queue.
type vulkanSharedContextWin struct {
	vk          *vulkanWin
	instance    uintptr
	device      uintptr
	queue       uintptr
	queueFamily uint32
	memProps    vkPhysicalDeviceMemoryProperties
}

var vulkanSharedCtxWin struct {
	once sync.Once
	ctx  vulkanSharedContextWin
	err  error
}

// getVulkanSharedContextWindows returns a process-wide Vulkan instance/device/queue.
// All runner constructors call this instead of creating their own instance and device.
func getVulkanSharedContextWindows() (vulkanSharedContextWin, error) {
	vulkanSharedCtxWin.once.Do(func() {
		vk := newVulkanWin("vulkan-1.dll")
		if err := vk.load(); err != nil {
			vulkanSharedCtxWin.err = err
			return
		}

		appName := append([]byte("rapidocrvl"), 0)
		engineName := append([]byte("rapidocrvl"), 0)
		app := vkApplicationInfo{
			SType:              vkStructureTypeApplicationInfo,
			PApplicationName:   uintptr(unsafe.Pointer(&appName[0])),
			ApplicationVersion: vkMakeVersion(0, 1, 0),
			PEngineName:        uintptr(unsafe.Pointer(&engineName[0])),
			EngineVersion:      vkMakeVersion(0, 1, 0),
			APIVersion:         vkMakeVersion(1, 1, 0),
		}
		ici := vkInstanceCreateInfo{
			SType:            vkStructureTypeInstanceCreateInfo,
			PApplicationInfo: uintptr(unsafe.Pointer(&app)),
		}
		var instance uintptr
		if res := vk.call(vk.createInstance, uintptr(unsafe.Pointer(&ici)), 0, uintptr(unsafe.Pointer(&instance))); res != vkSuccess {
			vulkanSharedCtxWin.err = fmt.Errorf("vkCreateInstance: %d", int32(res))
			return
		}

		var gpuCount uint32
		if res := vk.call(vk.enumeratePhysicalDevices, instance, uintptr(unsafe.Pointer(&gpuCount)), 0); res != vkSuccess {
			vk.callVoid(vk.destroyInstance, instance, 0)
			vulkanSharedCtxWin.err = fmt.Errorf("vkEnumeratePhysicalDevices count: %d", int32(res))
			return
		}
		if gpuCount == 0 {
			vk.callVoid(vk.destroyInstance, instance, 0)
			vulkanSharedCtxWin.err = fmt.Errorf("no Vulkan physical devices")
			return
		}
		gpus := make([]uintptr, gpuCount)
		if res := vk.call(vk.enumeratePhysicalDevices, instance, uintptr(unsafe.Pointer(&gpuCount)), uintptr(unsafe.Pointer(&gpus[0]))); res != vkSuccess {
			vk.callVoid(vk.destroyInstance, instance, 0)
			vulkanSharedCtxWin.err = fmt.Errorf("vkEnumeratePhysicalDevices: %d", int32(res))
			return
		}
		gpu, queueFamily, memProps, err := vk.selectComputeDevice(gpus)
		if err != nil {
			vk.callVoid(vk.destroyInstance, instance, 0)
			vulkanSharedCtxWin.err = err
			return
		}

		priority := float32(1)
		qci := vkDeviceQueueCreateInfo{
			SType:            vkStructureTypeDeviceQueueCreateInfo,
			QueueFamilyIndex: queueFamily,
			QueueCount:       1,
			PQueuePriorities: uintptr(unsafe.Pointer(&priority)),
		}
		dci := vkDeviceCreateInfo{
			SType:                vkStructureTypeDeviceCreateInfo,
			QueueCreateInfoCount: 1,
			PQueueCreateInfos:    uintptr(unsafe.Pointer(&qci)),
		}
		var device uintptr
		if res := vk.call(vk.createDevice, gpu, uintptr(unsafe.Pointer(&dci)), 0, uintptr(unsafe.Pointer(&device))); res != vkSuccess {
			vk.callVoid(vk.destroyInstance, instance, 0)
			vulkanSharedCtxWin.err = fmt.Errorf("vkCreateDevice: %d", int32(res))
			return
		}
		var queue uintptr
		vk.callVoid(vk.getDeviceQueue, device, uintptr(queueFamily), 0, uintptr(unsafe.Pointer(&queue)))

		vulkanSharedCtxWin.ctx = vulkanSharedContextWin{
			vk:          vk,
			instance:    instance,
			device:      device,
			queue:       queue,
			queueFamily: queueFamily,
			memProps:    memProps,
		}
	})
	return vulkanSharedCtxWin.ctx, vulkanSharedCtxWin.err
}