//go:build !amd64 && !arm64

package backend

func cpuFeatures() []string {
	return []string{"scalar-go"}
}
