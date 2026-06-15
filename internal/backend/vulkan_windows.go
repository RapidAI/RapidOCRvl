//go:build windows

package backend

import (
	"fmt"
	"syscall"
	"unsafe"
)

func VulkanInfo() Info {
	dll := syscall.NewLazyDLL("vulkan-1.dll")
	proc := dll.NewProc("vkGetInstanceProcAddr")
	if err := proc.Find(); err != nil {
		return Info{Name: "vulkan", Available: false, Reason: err.Error(), ComputeReason: err.Error()}
	}
	info := Info{Name: "vulkan", Available: true, ComputeReason: "pure Go Vulkan kernels are registered; GPU command submission is not enabled yet, CPU kernels remain active", Drivers: []VulkanDriver{{Library: "vulkan-1.dll"}}}
	if enum := dll.NewProc("vkEnumerateInstanceVersion"); enum.Find() == nil {
		var version uint32
		r1, _, _ := enum.Call(uintptr(unsafe.Pointer(&version)))
		if int32(r1) == 0 && version != 0 {
			info.APIVersion = vulkanVersionString(version)
		}
	}
	return info
}

func vulkanVersionString(v uint32) string {
	major := v >> 22
	minor := (v >> 12) & 0x3ff
	patch := v & 0xfff
	return fmt.Sprintf("%d.%d.%d", major, minor, patch)
}
