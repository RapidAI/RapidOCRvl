//go:build amd64

package backend

func cpuFeatures() []string {
	return []string{"amd64", "scalar-go", "unrolled"}
}
