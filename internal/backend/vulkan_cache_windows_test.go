//go:build windows

package backend

import "testing"

func TestVulkanWindowsCacheHashesDetectContentChanges(t *testing.T) {
	f32A := []float32{1, -2, 3.5}
	f32B := []float32{1, -2, 3.5}
	f32C := []float32{1, -2, 3.75}
	if hashFloat32ForVulkanCache(f32A) != hashFloat32ForVulkanCache(f32B) {
		t.Fatal("same float32 content should hash identically")
	}
	if hashFloat32ForVulkanCache(f32A) == hashFloat32ForVulkanCache(f32C) {
		t.Fatal("changed float32 content should change hash")
	}
	if fingerprintFloat32ForVulkanCache(f32A) != fingerprintFloat32ForVulkanCache(f32B) {
		t.Fatal("same float32 content should fingerprint identically")
	}
	if fingerprintFloat32ForVulkanCache(f32A) == fingerprintFloat32ForVulkanCache(f32C) {
		t.Fatal("changed sampled float32 content should change fingerprint")
	}
	i8A := []int8{1, -2, 3}
	i8B := []int8{1, -2, 3}
	i8C := []int8{1, -2, 4}
	if hashInt8ForVulkanCache(i8A) != hashInt8ForVulkanCache(i8B) {
		t.Fatal("same int8 content should hash identically")
	}
	if hashInt8ForVulkanCache(i8A) == hashInt8ForVulkanCache(i8C) {
		t.Fatal("changed int8 content should change hash")
	}
	if fingerprintInt8ForVulkanCache(i8A) != fingerprintInt8ForVulkanCache(i8B) {
		t.Fatal("same int8 content should fingerprint identically")
	}
	if fingerprintInt8ForVulkanCache(i8A) == fingerprintInt8ForVulkanCache(i8C) {
		t.Fatal("changed sampled int8 content should change fingerprint")
	}
	byteA := []byte{1, 2, 3}
	byteB := []byte{1, 2, 3}
	byteC := []byte{1, 2, 4}
	if hashBytesForVulkanCache(byteA) != hashBytesForVulkanCache(byteB) {
		t.Fatal("same byte content should hash identically")
	}
	if hashBytesForVulkanCache(byteA) == hashBytesForVulkanCache(byteC) {
		t.Fatal("changed byte content should change hash")
	}
	if fingerprintBytesForVulkanCache(byteA) != fingerprintBytesForVulkanCache(byteB) {
		t.Fatal("same byte content should fingerprint identically")
	}
	if fingerprintBytesForVulkanCache(byteA) == fingerprintBytesForVulkanCache(byteC) {
		t.Fatal("changed sampled byte content should change fingerprint")
	}
}
