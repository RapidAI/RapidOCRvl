//go:build linux

package backend

import (
	"debug/elf"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"

	"paddleocrvl-go/internal/jsonutil"
)

const maxVulkanICDManifestBytes = 1 << 20

func VulkanInfo() Info {
	candidates := vulkanLibraryCandidates()
	drivers := vulkanICDDrivers()
	if len(candidates) == 0 {
		return Info{Name: "vulkan", Available: false, Reason: "libvulkan.so not found", ComputeReason: "libvulkan.so not found", Drivers: drivers}
	}
	var lastErr error
	for _, path := range candidates {
		ok, err := elfHasSymbol(path, "vkGetInstanceProcAddr")
		if err != nil {
			lastErr = err
			continue
		}
		if ok {
			info := Info{Name: "vulkan", Available: true, Reason: path, ComputeReason: "pure Go Vulkan kernels are registered; GPU command submission is not enabled yet, CPU kernels remain active", Drivers: drivers}
			info.APIVersion = maxDriverAPIVersion(drivers)
			info.Devices = devicesFromDrivers(drivers)
			return info
		}
		lastErr = errors.New("vkGetInstanceProcAddr not found")
	}
	reason := "libvulkan.so found, vkGetInstanceProcAddr not found"
	if lastErr != nil {
		reason = lastErr.Error()
	}
	return Info{Name: "vulkan", Available: false, Reason: reason, ComputeReason: reason, Drivers: drivers}
}

func vulkanLibraryCandidates() []string {
	patterns := []string{
		"/usr/lib/libvulkan.so*",
		"/usr/lib64/libvulkan.so*",
		"/usr/lib/x86_64-linux-gnu/libvulkan.so*",
		"/usr/lib/aarch64-linux-gnu/libvulkan.so*",
		"/lib/libvulkan.so*",
		"/lib64/libvulkan.so*",
		"/lib/x86_64-linux-gnu/libvulkan.so*",
		"/lib/aarch64-linux-gnu/libvulkan.so*",
	}
	for _, dir := range filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")) {
		if dir != "" {
			patterns = append(patterns, filepath.Join(dir, "libvulkan.so*"))
		}
	}
	if sdk := os.Getenv("VULKAN_SDK"); sdk != "" {
		patterns = append(patterns,
			filepath.Join(sdk, "lib", "libvulkan.so*"),
			filepath.Join(sdk, "lib64", "libvulkan.so*"),
		)
	}
	seen := map[string]bool{}
	var out []string
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		for _, path := range matches {
			if seen[path] {
				continue
			}
			if st, err := os.Stat(path); err == nil && !st.IsDir() {
				seen[path] = true
				out = append(out, path)
			}
		}
	}
	sort.Strings(out)
	return out
}

func vulkanICDDrivers() []VulkanDriver {
	patterns := []string{
		"/usr/share/vulkan/icd.d/*.json",
		"/etc/vulkan/icd.d/*.json",
		"/usr/local/share/vulkan/icd.d/*.json",
	}
	seen := map[string]bool{}
	var out []VulkanDriver
	for _, env := range []string{"VK_ICD_FILENAMES", "VK_DRIVER_FILES"} {
		for _, path := range filepath.SplitList(os.Getenv(env)) {
			if path == "" || seen[path] {
				continue
			}
			seen[path] = true
			drv, ok := readVulkanICD(path)
			if ok {
				out = append(out, drv)
			}
		}
	}
	for _, pattern := range patterns {
		matches, _ := filepath.Glob(pattern)
		sort.Strings(matches)
		for _, path := range matches {
			if seen[path] {
				continue
			}
			seen[path] = true
			drv, ok := readVulkanICD(path)
			if ok {
				out = append(out, drv)
			}
		}
	}
	return out
}

func readVulkanICD(path string) (VulkanDriver, bool) {
	st, err := os.Stat(path)
	if err != nil || st.Size() > maxVulkanICDManifestBytes {
		return VulkanDriver{}, false
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return VulkanDriver{}, false
	}
	if err := jsonutil.RejectDuplicateKeys(raw, path); err != nil {
		return VulkanDriver{}, false
	}
	var doc struct {
		ICD struct {
			LibraryPath string `json:"library_path"`
			APIVersion  string `json:"api_version"`
		} `json:"ICD"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return VulkanDriver{}, false
	}
	return VulkanDriver{Path: path, Library: resolveICDLibraryPath(path, doc.ICD.LibraryPath), APIVersion: doc.ICD.APIVersion}, true
}

func resolveICDLibraryPath(manifestPath, libraryPath string) string {
	if libraryPath == "" || filepath.IsAbs(libraryPath) {
		return libraryPath
	}
	if filepath.Base(libraryPath) == libraryPath {
		return libraryPath
	}
	return filepath.Clean(filepath.Join(filepath.Dir(manifestPath), libraryPath))
}

func maxDriverAPIVersion(drivers []VulkanDriver) string {
	best := ""
	for _, d := range drivers {
		if compareVersion(d.APIVersion, best) > 0 {
			best = d.APIVersion
		}
	}
	return best
}

func devicesFromDrivers(drivers []VulkanDriver) []VulkanDevice {
	out := make([]VulkanDevice, 0, len(drivers))
	for _, d := range drivers {
		name := filepath.Base(d.Library)
		if name == "." || name == string(filepath.Separator) || name == "" {
			name = filepath.Base(d.Path)
		}
		out = append(out, VulkanDevice{Name: name, Type: inferVulkanDriverType(name), APIVersion: d.APIVersion, Driver: d.Library})
	}
	return out
}

func inferVulkanDriverType(name string) string {
	switch {
	case containsAny(name, []string{"nvidia", "radeon", "amd", "intel", "lavapipe", "virtio"}):
		return "compute"
	default:
		return "unknown"
	}
}

func containsAny(s string, needles []string) bool {
	for _, n := range needles {
		if containsFold(s, n) {
			return true
		}
	}
	return false
}

func containsFold(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if equalFoldASCII(s[i:i+len(sub)], sub) {
			return true
		}
	}
	return false
}

func equalFoldASCII(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

func compareVersion(a, b string) int {
	ai, bi := 0, 0
	for ai < len(a) || bi < len(b) {
		av, an := nextVersionPart(a, ai)
		bv, bn := nextVersionPart(b, bi)
		if av != bv {
			if av > bv {
				return 1
			}
			return -1
		}
		ai, bi = an, bn
	}
	return 0
}

func nextVersionPart(s string, i int) (int, int) {
	for i < len(s) && (s[i] < '0' || s[i] > '9') {
		i++
	}
	if i >= len(s) {
		return 0, len(s)
	}
	v := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		v = v*10 + int(s[i]-'0')
		i++
	}
	return v, i
}

func elfHasSymbol(path, name string) (bool, error) {
	f, err := elf.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	if syms, err := f.DynamicSymbols(); err == nil {
		for _, sym := range syms {
			if sym.Name == name {
				return true, nil
			}
		}
	}
	if syms, err := f.Symbols(); err == nil {
		for _, sym := range syms {
			if sym.Name == name {
				return true, nil
			}
		}
	}
	return false, nil
}
