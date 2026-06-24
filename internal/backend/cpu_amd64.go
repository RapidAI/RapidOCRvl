//go:build amd64

package backend

import "golang.org/x/sys/cpu"

func cpuFeatures() []string {
	features := []string{"amd64", "scalar-go", "unrolled"}
	if cpu.X86.HasSSE2 {
		features = append(features, "sse2")
	}
	if cpu.X86.HasSSE3 {
		features = append(features, "sse3")
	}
	if cpu.X86.HasSSSE3 {
		features = append(features, "ssse3")
	}
	if cpu.X86.HasSSE41 {
		features = append(features, "sse4.1")
	}
	if cpu.X86.HasSSE42 {
		features = append(features, "sse4.2")
	}
	if cpu.X86.HasAVX {
		features = append(features, "avx")
	}
	if cpu.X86.HasAVX2 {
		features = append(features, "avx2")
	}
	if cpu.X86.HasFMA {
		features = append(features, "fma")
	}
	if cpu.X86.HasAVX512F {
		features = append(features, "avx512f")
	}
	return features
}
