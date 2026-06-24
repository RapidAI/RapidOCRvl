//go:build arm64

package backend

import "golang.org/x/sys/cpu"

func cpuFeatures() []string {
	features := []string{"arm64", "scalar-go", "unrolled"}
	if cpu.ARM64.HasASIMD {
		features = append(features, "asimd", "neon")
	}
	if cpu.ARM64.HasASIMDDP {
		features = append(features, "asimddp")
	}
	if cpu.ARM64.HasASIMDFHM {
		features = append(features, "asimdfhm")
	}
	if cpu.ARM64.HasASIMDHP {
		features = append(features, "asimdhp")
	}
	if cpu.ARM64.HasASIMDRDM {
		features = append(features, "asimdrdm")
	}
	if cpu.ARM64.HasATOMICS {
		features = append(features, "atomics")
	}
	if cpu.ARM64.HasSVE {
		features = append(features, "sve")
	}
	if cpu.ARM64.HasSVE2 {
		features = append(features, "sve2")
	}
	return features
}
