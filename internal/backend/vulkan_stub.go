//go:build !windows && !linux

package backend

func VulkanInfo() Info {
	reason := "vulkan loader probing not implemented for this OS yet"
	return Info{Name: "vulkan", Available: false, Reason: reason, ComputeReason: reason}
}
