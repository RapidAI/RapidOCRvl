package backend

import (
	"fmt"
)

type Info struct {
	Name             string         `json:"name"`
	Available        bool           `json:"available"`
	Reason           string         `json:"reason,omitempty"`
	APIVersion       string         `json:"api_version,omitempty"`
	ComputeAvailable bool           `json:"compute_available"`
	ComputeReason    string         `json:"compute_reason,omitempty"`
	Compute          VulkanCompute  `json:"compute,omitempty"`
	Devices          []VulkanDevice `json:"devices,omitempty"`
	Drivers          []VulkanDriver `json:"drivers,omitempty"`
}

type VulkanCompute struct {
	Version       int            `json:"version,omitempty"`
	Workgroup     int            `json:"workgroup,omitempty"`
	VectorWidth   int            `json:"vector_width,omitempty"`
	Kernels       []VulkanKernel `json:"kernels,omitempty"`
	Fusions       []string       `json:"fusions,omitempty"`
	DispatchReady bool           `json:"dispatch_ready"`
}

type VulkanKernel struct {
	Name              string          `json:"name"`
	Op                string          `json:"op"`
	Quant             string          `json:"quant,omitempty"`
	Workgroup         int             `json:"workgroup"`
	RowsPerWarp       int             `json:"rows_per_warp,omitempty"`
	TileCols          int             `json:"tile_cols,omitempty"`
	PushConstantBytes int             `json:"push_constant_bytes,omitempty"`
	BindingSignature  uint64          `json:"binding_signature,omitempty"`
	Bindings          []VulkanBinding `json:"bindings,omitempty"`
	SourceHash        uint64          `json:"source_hash,omitempty"`
	SourceBytes       int             `json:"source_bytes,omitempty"`
	Source            string          `json:"-"`
}

type VulkanBinding struct {
	Binding        int    `json:"binding"`
	Name           string `json:"name"`
	Role           string `json:"role"`
	Elem           string `json:"elem"`
	Access         string `json:"access"`
	DescriptorType string `json:"descriptor_type"`
}

type VulkanPipelineKey struct {
	KernelName        string `json:"kernel_name"`
	Op                string `json:"op"`
	Quant             string `json:"quant,omitempty"`
	Workgroup         int    `json:"workgroup"`
	PushConstantBytes int    `json:"push_constant_bytes"`
	BindingSignature  uint64 `json:"binding_signature"`
}

type VulkanDriver struct {
	Path       string `json:"path"`
	Library    string `json:"library,omitempty"`
	APIVersion string `json:"api_version,omitempty"`
}

type VulkanDevice struct {
	Name       string `json:"name"`
	Type       string `json:"type,omitempty"`
	APIVersion string `json:"api_version,omitempty"`
	Driver     string `json:"driver,omitempty"`
}

type VulkanTokenScore struct {
	Token int     `json:"token"`
	Score float32 `json:"score"`
}

type Selection struct {
	Requested string `json:"requested"`
	Active    string `json:"active"`
	Vulkan    Info   `json:"vulkan"`
}

func Select(requested string) (Selection, error) {
	requested = lowerASCII(trimASCIIWhitespace(requested))
	if requested == "" {
		requested = "auto"
	}
	vk := VulkanInfo()
	vk = normalizeVulkanInfo(vk)
	switch requested {
	case "auto":
		if vk.Available && vk.ComputeAvailable {
			return Selection{Requested: requested, Active: "vulkan", Vulkan: vk}, nil
		}
		return Selection{Requested: requested, Active: "cpu", Vulkan: vk}, nil
	case "cpu":
		return Selection{Requested: requested, Active: "cpu", Vulkan: vk}, nil
	case "vulkan":
		if !vk.Available {
			return Selection{}, fmt.Errorf("vulkan requested but unavailable: %s", vk.Reason)
		}
		if vk.ComputeAvailable {
			return Selection{Requested: requested, Active: "vulkan", Vulkan: vk}, nil
		}
		return Selection{Requested: requested, Active: "cpu", Vulkan: vk}, nil
	default:
		return Selection{}, fmt.Errorf("unknown backend %q; use auto, cpu, or vulkan", requested)
	}
}

func trimASCIIWhitespace(s string) string {
	if len(s) == 0 || (!isASCIIWhitespace(s[0]) && !isASCIIWhitespace(s[len(s)-1])) {
		return s
	}
	start, end := 0, len(s)
	for start < end && isASCIIWhitespace(s[start]) {
		start++
	}
	for end > start && isASCIIWhitespace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isASCIIWhitespace(c byte) bool {
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f'
}

func lowerASCII(s string) string {
	firstUpper := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			firstUpper = i
			break
		}
	}
	if firstUpper < 0 {
		return s
	}
	out := []byte(s)
	for i := firstUpper; i < len(out); i++ {
		c := out[i]
		if 'A' <= c && c <= 'Z' {
			out[i] = c + ('a' - 'A')
		}
	}
	return string(out)
}

func SelectMust(requested string) Selection {
	sel, err := Select(requested)
	if err != nil {
		return Selection{Requested: requested, Active: "unknown", Vulkan: VulkanInfo()}
	}
	return sel
}

func normalizeVulkanInfo(info Info) Info {
	if info.Name == "" {
		info.Name = "vulkan"
	}
	if info.Compute.Version == 0 {
		info.Compute = DefaultVulkanCompute()
	}
	if info.ComputeReason == "" {
		if info.Available {
			info.ComputeReason = "pure Go Vulkan kernels are registered; GPU command submission is not enabled yet, CPU kernels remain active"
		} else {
			info.ComputeReason = "vulkan loader unavailable"
		}
	}
	return info
}
