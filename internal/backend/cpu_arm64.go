//go:build arm64

package backend

func cpuFeatures() []string {
	return []string{"neon"}
}
