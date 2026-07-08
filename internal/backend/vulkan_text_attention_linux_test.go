//go:build cgo && linux

package backend

import (
	"testing"
	"unsafe"

	vk "github.com/vulkan-go/vulkan"

	"paddleocrvl-go/internal/tensor"
)

func TestVulkanTextAttentionLinuxByteSizeChecks(t *testing.T) {
	if got, ok := checkedFloat32ByteLenLinux(7); !ok || got != vk.DeviceSize(28) {
		t.Fatalf("float32 bytes got=%d ok=%v want=28 true", got, ok)
	}
	if got, ok := textAttentionCacheBufferBytesLinux(make([]float32, 8), 4); !ok || got != vk.DeviceSize(32) {
		t.Fatalf("cache bytes len preference got=%d ok=%v want=32 true", got, ok)
	}
	if got, ok := textAttentionCacheBufferBytesLinux(make([]float32, 2), 4); !ok || got != vk.DeviceSize(16) {
		t.Fatalf("cache bytes min preference got=%d ok=%v want=16 true", got, ok)
	}
	if _, ok := checkedFloat32ByteLenLinux(maxInt()/4 + 1); ok {
		t.Fatal("overflowing float32 byte length should be rejected")
	}
	if _, ok := checkedFloat32ByteLenLinuxSquare(maxInt()/2 + 1); ok {
		t.Fatal("overflowing square float32 byte length should be rejected")
	}
	if _, _, _, err := checkedTextAttentionLinuxDims(maxInt()/2+1, 1, 3); err == nil {
		t.Fatal("overflowing q rows should be rejected")
	}
	if _, _, _, err := checkedTextAttentionLinuxDims(1, maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing kv dim should be rejected")
	}
	if _, err := checkedTextAttentionLinuxCacheElems(maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing cache element count should be rejected")
	}
	if _, ok := textAttentionCacheBufferBytesLinux(nil, maxInt()); ok {
		t.Fatal("overflowing cache byte length should be rejected")
	}
	if dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(3, 5); err != nil || dataLen != 15 || dataBytes != vk.DeviceSize(16) {
		t.Fatalf("q8 data bytes got len=%d bytes=%d err=%v want len=15 bytes=16 nil", dataLen, dataBytes, err)
	}
	q6Packed := tensor.PackedQ6Cols(8)
	if dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesLinux(2, 8); err != nil || dataLen != 2*q6Packed || dataBytes != vk.DeviceSize(alignUpInt(2*q6Packed, 4)) {
		t.Fatalf("q6 data bytes got len=%d bytes=%d err=%v want len=%d bytes=%d nil", dataLen, dataBytes, err, 2*q6Packed, alignUpInt(2*q6Packed, 4))
	}
	if dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesLinux(3, 5); err != nil || dataLen != 9 || dataBytes != vk.DeviceSize(12) {
		t.Fatalf("q4 data bytes got len=%d bytes=%d err=%v want len=9 bytes=12 nil", dataLen, dataBytes, err)
	}
	if _, _, err := checkedTextAttentionQ8DataBytesLinux(maxInt()/2+1, 3); err == nil {
		t.Fatal("overflowing q8 data length should be rejected")
	}
	if _, _, err := checkedTextAttentionQ6DataBytesLinux(1, (maxInt()-7)/6+1); err == nil {
		t.Fatal("overflowing q6 packed cols should be rejected")
	}
	if _, _, err := checkedTextAttentionQ6DataBytesLinux(maxInt()/2+1, 8); err == nil {
		t.Fatal("overflowing q6 data length should be rejected")
	}
	if _, _, err := checkedTextAttentionQ4DataBytesLinux(1, maxInt()); err == nil {
		t.Fatal("overflowing q4 packed cols should be rejected")
	}
	if _, _, err := checkedTextAttentionQ4DataBytesLinux(maxInt()/2+1, 4); err == nil {
		t.Fatal("overflowing q4 data length should be rejected")
	}
}

func TestVulkanTextAttentionLinuxUploadTextCacheAppendsWithinEpoch(t *testing.T) {
	const kvDim = 2
	kCache := []float32{1, 2, 3, 4, 5, 6}
	vCache := []float32{10, 20, 30, 40, 50, 60}
	kMapped := make([]float32, len(kCache))
	vMapped := make([]float32, len(vCache))
	r := &vulkanTextAttentionF32LinuxRunner{
		kBuf: vulkanHostBuffer{buffer: vk.Buffer(1), size: vk.DeviceSize(len(kMapped) * 4), mapped: unsafe.Pointer(&kMapped[0])},
		vBuf: vulkanHostBuffer{buffer: vk.Buffer(2), size: vk.DeviceSize(len(vMapped) * 4), mapped: unsafe.Pointer(&vMapped[0])},
	}

	if err := r.uploadTextCacheLocked(kCache, vCache, 7, 2, kvDim); err != nil {
		t.Fatal(err)
	}
	if got, want := kMapped[:4], []float32{1, 2, 3, 4}; !sameFloat32s(got, want) {
		t.Fatalf("initial key upload=%v want %v", got, want)
	}
	if got, want := vMapped[:4], []float32{10, 20, 30, 40}; !sameFloat32s(got, want) {
		t.Fatalf("initial value upload=%v want %v", got, want)
	}

	kCache[0] = 99
	vCache[0] = 199
	if err := r.uploadTextCacheLocked(kCache, vCache, 7, 3, kvDim); err != nil {
		t.Fatal(err)
	}
	if got, want := kMapped, []float32{1, 2, 3, 4, 5, 6}; !sameFloat32s(got, want) {
		t.Fatalf("incremental key upload=%v want %v", got, want)
	}
	if got, want := vMapped, []float32{10, 20, 30, 40, 50, 60}; !sameFloat32s(got, want) {
		t.Fatalf("incremental value upload=%v want %v", got, want)
	}

	if err := r.uploadTextCacheLocked(kCache, vCache, 8, 3, kvDim); err != nil {
		t.Fatal(err)
	}
	if got, want := kMapped, []float32{99, 2, 3, 4, 5, 6}; !sameFloat32s(got, want) {
		t.Fatalf("epoch refresh key upload=%v want %v", got, want)
	}
	if got, want := vMapped, []float32{199, 20, 30, 40, 50, 60}; !sameFloat32s(got, want) {
		t.Fatalf("epoch refresh value upload=%v want %v", got, want)
	}
}

func sameFloat32s(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
