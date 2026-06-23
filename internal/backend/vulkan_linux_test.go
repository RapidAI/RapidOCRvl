//go:build linux

package backend

import (
	"os"
	"path/filepath"
	"testing"
)

func TestVulkanLibraryCandidatesDoesNotPanic(t *testing.T) {
	_ = vulkanLibraryCandidates()
}

func TestVulkanLibraryCandidatesIncludesEnvPaths(t *testing.T) {
	dir := t.TempDir()
	lib := filepath.Join(dir, "libvulkan.so.1")
	if err := os.WriteFile(lib, []byte("not an elf"), 0o644); err != nil {
		t.Fatal(err)
	}
	sdk := filepath.Join(t.TempDir(), "sdk")
	sdkLib := filepath.Join(sdk, "lib")
	if err := os.MkdirAll(sdkLib, 0o755); err != nil {
		t.Fatal(err)
	}
	sdkVulkan := filepath.Join(sdkLib, "libvulkan.so")
	if err := os.WriteFile(sdkVulkan, []byte("not an elf"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LD_LIBRARY_PATH", dir)
	t.Setenv("VULKAN_SDK", sdk)
	got := vulkanLibraryCandidates()
	if !stringSliceContains(got, lib) {
		t.Fatalf("missing LD_LIBRARY_PATH lib in %v", got)
	}
	if !stringSliceContains(got, sdkVulkan) {
		t.Fatalf("missing VULKAN_SDK lib in %v", got)
	}
}

func TestVulkanICDDriversIncludesEnvManifests(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "driver.json")
	body := []byte(`{"file_format_version":"1.0.0","ICD":{"library_path":"libvulkan_env.so","api_version":"1.3.7"}}`)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VK_ICD_FILENAMES", path)
	drivers := vulkanICDDrivers()
	for _, drv := range drivers {
		if drv.Path == path && drv.Library == "libvulkan_env.so" && drv.APIVersion == "1.3.7" {
			return
		}
	}
	t.Fatalf("env ICD missing in %+v", drivers)
}

func TestCompareVersion(t *testing.T) {
	if compareVersion("1.3.280", "1.2.199") <= 0 {
		t.Fatal("expected 1.3.280 > 1.2.199")
	}
	if compareVersion("1.2.0", "1.2") != 0 {
		t.Fatal("expected missing patch to compare equal to 0")
	}
	if compareVersion("", "1.0.0") >= 0 {
		t.Fatal("expected empty version lower than 1.0.0")
	}
}

func stringSliceContains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func TestInferVulkanDriverType(t *testing.T) {
	if got := inferVulkanDriverType("libvulkan_nvidia.so"); got != "compute" {
		t.Fatalf("got %q", got)
	}
	if got := inferVulkanDriverType("unknown.so"); got != "unknown" {
		t.Fatalf("got %q", got)
	}
}

func TestReadVulkanICD(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "driver.json")
	body := []byte(`{"file_format_version":"1.0.0","ICD":{"library_path":"libvulkan_test.so","api_version":"1.3.0"}}`)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	drv, ok := readVulkanICD(path)
	if !ok {
		t.Fatal("expected manifest parse")
	}
	if drv.Library != "libvulkan_test.so" || drv.APIVersion != "1.3.0" || drv.Path != path {
		t.Fatalf("driver=%+v", drv)
	}
}

func TestReadVulkanICDRejectsHugeManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "driver.json")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Truncate(maxVulkanICDManifestBytes + 1); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, ok := readVulkanICD(path); ok {
		t.Fatal("expected huge manifest rejection")
	}
}

func TestReadVulkanICDRejectsDuplicateJSONKeys(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "driver.json")
	body := []byte(`{"ICD":{"library_path":"a.so","library_path":"b.so","api_version":"1.3.0"}}`)
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := readVulkanICD(path); ok {
		t.Fatal("expected duplicate JSON key rejection")
	}
}

func TestResolveICDLibraryPath(t *testing.T) {
	manifest := filepath.Join("tmp", "icd.d", "driver.json")
	if got := resolveICDLibraryPath(manifest, "libvulkan_test.so"); got != "libvulkan_test.so" {
		t.Fatalf("basename resolved to %q", got)
	}
	rel := filepath.Join("drivers", "libvulkan_test.so")
	want := filepath.Clean(filepath.Join("tmp", "icd.d", rel))
	if got := resolveICDLibraryPath(manifest, rel); got != want {
		t.Fatalf("relative resolved to %q want %q", got, want)
	}
	abs := filepath.Join(string(filepath.Separator), "opt", "libvulkan_test.so")
	if got := resolveICDLibraryPath(manifest, abs); got != abs {
		t.Fatalf("absolute resolved to %q", got)
	}
}
