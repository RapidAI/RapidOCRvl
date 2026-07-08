//go:build cgo && linux

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"unsafe"

	"paddleocrvl-go/internal/tensor"

	vk "github.com/vulkan-go/vulkan"
)

var vulkanDispatchProbe struct {
	once sync.Once
	err  error
}

var vulkanMatVecF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanArgmaxF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanArgmaxQuantizedF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanBlockTopKF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanBlockTopKQuantizedF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanRMSNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanAddRMSNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecAddRMSNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecQ8AddRMSNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMRoPEF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMRoPEPairF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3F32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2F32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2MRoPEF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanSwiGLUGateUpF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanVisionAttentionF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanVisionAttentionOutF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanVisionRoPEPairF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanVisionQKVF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionOutF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionOutAddRMSNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextFirstTokenValueOutF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionOutQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionOutQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanTextAttentionOutQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatRowsBiasF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatRowsBiasAddRowsF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatRowsBias3F32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatRowsGELU2F32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatRowsGELU2AddLayerNormF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanProjectImageF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanLayerNormRowsF32SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3Q8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2Q8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2MRoPEQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanSwiGLUGateUpQ8SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanSwiGLUGateUpQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanSwiGLUGateUpQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3Q4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2Q4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3Q6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2Q6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2MRoPEQ4SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanFusedMatVec2MRoPEQ6SPVCache struct {
	once sync.Once
	spv  []uint32
	err  error
}

var vulkanMatVecF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32LinuxRunner
	err    error
}

var vulkanRMSNormF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32LinuxRunner
	err    error
}

var vulkanAddRMSNormF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32LinuxRunner
	err    error
}

var vulkanMRoPEF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32LinuxRunner
	err    error
}

var vulkanMRoPEPairF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32LinuxRunner
	err    error
}

var vulkanFusedMatVec3F32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32LinuxRunner
	err    error
}

var vulkanFusedMatVec2F32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32LinuxRunner
	err    error
}

var vulkanFusedMatVec3MRoPEF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32LinuxRunner
	err    error
}

var vulkanFusedMatVec2MRoPEF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32LinuxRunner
	err    error
}

var vulkanSwiGLUGateUpF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUGateUpF32LinuxRunner
	err    error
}

var vulkanSwiGLUDownF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUDownF32LinuxRunner
	err    error
}

var vulkanVisionAttentionF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanVisionAttentionF32LinuxRunner
	err    error
}

var vulkanTextAttentionF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanTextAttentionF32LinuxRunner
	err    error
}

var vulkanMatVecQ8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecQ8LinuxRunner
	err    error
}

var vulkanFusedMatVec3Q8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8LinuxRunner
	err    error
}

var vulkanFusedMatVec2Q8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8LinuxRunner
	err    error
}

var vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8LinuxRunner
	err    error
}

var vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8LinuxRunner
	err    error
}

var vulkanSwiGLUDownQ8LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUDownQ8LinuxRunner
	err    error
}

var vulkanSwiGLUDownQ4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUDownPackedBytesLinuxRunner
	err    error
}

var vulkanSwiGLUDownQ6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUDownPackedBytesLinuxRunner
	err    error
}

var vulkanMatVecQ4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecPackedBytesLinuxRunner
	err    error
}

var vulkanMatVecQ6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecPackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec3Q4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec2Q4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec3Q6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec2Q6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3PackedBytesLinuxRunner
	err    error
}

var vulkanMatRowsBiasF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsBiasF32LinuxRunner
	err    error
}

var vulkanMatRowsBias3F32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsBias3F32LinuxRunner
	err    error
}

var vulkanMatRowsGELU2F32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsGELU2F32LinuxRunner
	err    error
}

var vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner
	err    error
}

var vulkanProjectImageF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanProjectImageF32LinuxRunner
	err    error
}

var vulkanLayerNormRowsF32LinuxRunnerCache struct {
	once   sync.Once
	runner *vulkanLayerNormRowsF32LinuxRunner
	err    error
}

func VulkanDispatchSmokeTest() error {
	vulkanDispatchProbe.once.Do(func() {
		vulkanDispatchProbe.err = runVulkanDispatchSmokeTest()
	})
	return vulkanDispatchProbe.err
}

func VulkanMatVecF32(out, x, w []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matvec shape rows=%d cols=%d", rows, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matvec")
	if err != nil {
		return err
	}
	if len(out) < rows || len(x) < cols || len(w) < wLen {
		return fmt.Errorf("invalid Vulkan matvec buffers out=%d x=%d w=%d rows=%d cols=%d", len(out), len(x), len(w), rows, cols)
	}
	return runVulkanMatVecF32(out, x, w[:wLen], rows, cols)
}

func VulkanMatVecArgmaxF32(x, w []float32, rows, cols int) (int, float32, error) {
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("invalid Vulkan matvec argmax shape rows=%d cols=%d", rows, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matvec argmax")
	if err != nil {
		return 0, 0, err
	}
	if len(x) < cols || len(w) < wLen {
		return 0, 0, fmt.Errorf("invalid Vulkan matvec argmax buffers x=%d w=%d rows=%d cols=%d", len(x), len(w), rows, cols)
	}
	runner, err := getVulkanMatVecArgmaxF32LinuxRunner()
	if err != nil {
		return 0, 0, err
	}
	return runner.runMatVecArgmax(x, w[:wLen], rows, cols)
}

func VulkanMatVecTopKF32(x, w []float32, rows, cols, k int) ([]VulkanTokenScore, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k shape rows=%d cols=%d", rows, cols)
	}
	if k <= 0 || k > vulkanMatVecTopKMaxK {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k k=%d max=%d", k, vulkanMatVecTopKMaxK)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matvec top-k")
	if err != nil {
		return nil, err
	}
	if len(x) < cols || len(w) < wLen {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k buffers x=%d w=%d rows=%d cols=%d", len(x), len(w), rows, cols)
	}
	runner, err := getVulkanMatVecF32LinuxRunner()
	if err != nil {
		return nil, err
	}
	return runner.runMatVecTopK(x, w[:wLen], rows, cols, k)
}

func VulkanRMSNormF32(out, x, weight []float32) error {
	n := len(x)
	if n <= 0 {
		return fmt.Errorf("invalid Vulkan rmsnorm shape n=%d", n)
	}
	if len(out) < n || len(weight) < n {
		return fmt.Errorf("invalid Vulkan rmsnorm buffers out=%d x=%d weight=%d", len(out), len(x), len(weight))
	}
	runner, err := getVulkanRMSNormF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, weight, n, 1)
}

func VulkanAddRMSNormF32(out, dst, add, weight []float32) error {
	n := len(dst)
	if n <= 0 {
		return fmt.Errorf("invalid Vulkan add rmsnorm shape n=%d", n)
	}
	if len(out) < n || len(add) < n || len(weight) < n {
		return fmt.Errorf("invalid Vulkan add rmsnorm buffers out=%d dst=%d add=%d weight=%d", len(out), len(dst), len(add), len(weight))
	}
	runner, err := getVulkanAddRMSNormF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAdd(out, dst, add, weight, n)
}

func VulkanAddRMSNormF32OutOnly(out, dst, add, weight []float32) error {
	n := len(dst)
	if n <= 0 {
		return fmt.Errorf("invalid Vulkan add rmsnorm shape n=%d", n)
	}
	if len(out) < n || len(add) < n || len(weight) < n {
		return fmt.Errorf("invalid Vulkan add rmsnorm buffers out=%d dst=%d add=%d weight=%d", len(out), len(dst), len(add), len(weight))
	}
	runner, err := getVulkanAddRMSNormF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddOutOnly(out, dst, add, weight, n)
}

func VulkanMRoPEF32(x, cosTable, sinTable []float32, heads, dim int) error {
	if heads <= 0 || dim <= 0 || dim%2 != 0 {
		return fmt.Errorf("invalid Vulkan mrope shape heads=%d dim=%d", heads, dim)
	}
	n, ok := checkedMulInt(heads, dim)
	if !ok {
		return fmt.Errorf("Vulkan mrope length overflows: heads=%d dim=%d", heads, dim)
	}
	half := dim / 2
	if len(x) < n || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan mrope buffers x=%d cos=%d sin=%d heads=%d dim=%d", len(x), len(cosTable), len(sinTable), heads, dim)
	}
	runner, err := getVulkanMRoPEF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPE(x, cosTable, sinTable, heads, dim)
}

func VulkanMRoPEPairF32(q, k, cosTable, sinTable []float32, qHeads, kvHeads, dim int) error {
	if qHeads <= 0 || kvHeads <= 0 || dim <= 0 || dim%2 != 0 || dim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan mrope pair shape qHeads=%d kvHeads=%d dim=%d", qHeads, kvHeads, dim)
	}
	qRows, ok := checkedMulInt(qHeads, dim)
	if !ok {
		return fmt.Errorf("Vulkan mrope pair q length overflows: qHeads=%d dim=%d", qHeads, dim)
	}
	kvRows, ok := checkedMulInt(kvHeads, dim)
	if !ok {
		return fmt.Errorf("Vulkan mrope pair kv length overflows: kvHeads=%d dim=%d", kvHeads, dim)
	}
	half := dim / 2
	if len(q) < qRows || len(k) < kvRows || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan mrope pair buffers q=%d k=%d cos=%d sin=%d qHeads=%d kvHeads=%d dim=%d", len(q), len(k), len(cosTable), len(sinTable), qHeads, kvHeads, dim)
	}
	runner, err := getVulkanMRoPEPairF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPEPair(q, k, cosTable, sinTable, qHeads, kvHeads, dim)
}

func VulkanFusedMatVec3F32(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) error {
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan fused matvec3 shape rowsA=%d rowsB=%d rowsC=%d cols=%d", rowsA, rowsB, rowsC, cols)
	}
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan fused matvec3 wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan fused matvec3 wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan fused matvec3 wc")
	if err != nil {
		return err
	}
	if len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC || len(x) < cols ||
		len(wa) < waLen || len(wb) < wbLen || len(wc) < wcLen {
		return fmt.Errorf("invalid Vulkan fused matvec3 buffers outA=%d outB=%d outC=%d x=%d wa=%d wb=%d wc=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			len(outA), len(outB), len(outC), len(x), len(wa), len(wb), len(wc), rowsA, rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec3F32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, x, wa[:waLen], wb[:wbLen], wc[:wcLen], rowsA, rowsB, rowsC, cols)
}

func VulkanFusedMatVec2F32(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) error {
	if rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan fused matvec2 shape rowsB=%d rowsC=%d cols=%d", rowsB, rowsC, cols)
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan fused matvec2 wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan fused matvec2 wc")
	if err != nil {
		return err
	}
	if len(outB) < rowsB || len(outC) < rowsC || len(x) < cols || len(wb) < wbLen || len(wc) < wcLen {
		return fmt.Errorf("invalid Vulkan fused matvec2 buffers outB=%d outC=%d x=%d wb=%d wc=%d rowsB=%d rowsC=%d cols=%d",
			len(outB), len(outC), len(x), len(wb), len(wc), rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec2F32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2(outB, outC, x, wb[:wbLen], wc[:wcLen], rowsB, rowsC, cols)
}

func VulkanFusedMatVec2MRoPEF32(outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsB, rowsC, cols, kvHeads, headDim int) error {
	if rowsB <= 0 || rowsC <= 0 || cols <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope shape rowsB=%d rowsC=%d cols=%d kvHeads=%d headDim=%d", rowsB, rowsC, cols, kvHeads, headDim)
	}
	wantRowsB, ok := checkedMulInt(kvHeads, headDim)
	if !ok {
		return fmt.Errorf("Vulkan fused matvec2+mrope rowsB overflows: kvHeads=%d headDim=%d", kvHeads, headDim)
	}
	if rowsB != wantRowsB {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope rows rowsB=%d want=%d", rowsB, wantRowsB)
	}
	half := headDim / 2
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan fused matvec2+mrope wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan fused matvec2+mrope wc")
	if err != nil {
		return err
	}
	if len(outB) < rowsB || len(outC) < rowsC || len(x) < cols || len(wb) < wbLen || len(wc) < wcLen || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan fused matvec2+mrope buffers outB=%d outC=%d x=%d wa=%d wb=%d wc=%d cos=%d sin=%d rowsB=%d rowsC=%d cols=%d",
			len(outB), len(outC), len(x), len(wa), len(wb), len(wc), len(cosTable), len(sinTable), rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec2MRoPEF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2MRoPE(outB, outC, x, wb[:wbLen], wc[:wcLen], cosTable[:half], sinTable[:half], rowsB, rowsC, cols, kvHeads, headDim)
}

func VulkanFusedMatVec3MRoPEF32(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim int) error {
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope shape rowsA=%d rowsB=%d rowsC=%d cols=%d qHeads=%d kvHeads=%d headDim=%d", rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim)
	}
	wantRowsA, ok := checkedMulInt(qHeads, headDim)
	if !ok {
		return fmt.Errorf("Vulkan fused matvec3+mrope rowsA overflows: qHeads=%d headDim=%d", qHeads, headDim)
	}
	wantRowsB, ok := checkedMulInt(kvHeads, headDim)
	if !ok {
		return fmt.Errorf("Vulkan fused matvec3+mrope rowsB overflows: kvHeads=%d headDim=%d", kvHeads, headDim)
	}
	if rowsA != wantRowsA || rowsB != wantRowsB {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope rows rowsA=%d rowsB=%d want=%d/%d", rowsA, rowsB, wantRowsA, wantRowsB)
	}
	half := headDim / 2
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan fused matvec3+mrope wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan fused matvec3+mrope wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan fused matvec3+mrope wc")
	if err != nil {
		return err
	}
	if len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC || len(x) < cols ||
		len(wa) < waLen || len(wb) < wbLen || len(wc) < wcLen || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope buffers outA=%d outB=%d outC=%d x=%d wa=%d wb=%d wc=%d cos=%d sin=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			len(outA), len(outB), len(outC), len(x), len(wa), len(wb), len(wc), len(cosTable), len(sinTable), rowsA, rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec3MRoPEF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPE(outA, outB, outC, x, wa[:waLen], wb[:wbLen], wc[:wcLen], cosTable[:half], sinTable[:half], rowsA, rowsB, rowsC, cols, kvHeads, headDim)
}

func VulkanSwiGLUGateUpF32(out, x, gate, up []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu gate/up shape rows=%d cols=%d", rows, cols)
	}
	dims, err := checkedSwiGLUDimsLinux(rows, cols, 0, "Vulkan swiglu gate/up")
	if err != nil {
		return err
	}
	if len(out) < rows || len(x) < cols || len(gate) < dims.gateLen || len(up) < dims.gateLen {
		return fmt.Errorf("invalid Vulkan swiglu gate/up buffers out=%d x=%d gate=%d up=%d rows=%d cols=%d", len(out), len(x), len(gate), len(up), rows, cols)
	}
	runner, err := getVulkanSwiGLUGateUpF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, gate[:dims.gateLen], up[:dims.gateLen], rows, cols)
}

func VulkanSwiGLUDownF32(out, x, gate, up, down []float32, rows, cols, outRows int) error {
	if rows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu/down shape rows=%d cols=%d outRows=%d", rows, cols, outRows)
	}
	dims, err := checkedSwiGLUDimsLinux(rows, cols, outRows, "Vulkan swiglu/down")
	if err != nil {
		return err
	}
	if len(out) < outRows || len(x) < cols || len(gate) < dims.gateLen || len(up) < dims.gateLen || len(down) < dims.downLen {
		return fmt.Errorf("invalid Vulkan swiglu/down buffers out=%d x=%d gate=%d up=%d down=%d rows=%d cols=%d outRows=%d", len(out), len(x), len(gate), len(up), len(down), rows, cols, outRows)
	}
	runner, err := getVulkanSwiGLUDownF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, gate[:dims.gateLen], up[:dims.gateLen], down[:dims.downLen], rows, cols, outRows)
}

func VulkanMatVecAddRMSNormF32(normOut, residual, x, w, normWeight []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matvec+add+rmsnorm shape rows=%d cols=%d", rows, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matvec+add+rmsnorm")
	if err != nil {
		return err
	}
	if len(normOut) < rows || len(residual) < rows || len(x) < cols || len(w) < wLen || len(normWeight) < rows {
		return fmt.Errorf("invalid Vulkan matvec+add+rmsnorm buffers normOut=%d residual=%d x=%d w=%d normWeight=%d rows=%d cols=%d",
			len(normOut), len(residual), len(x), len(w), len(normWeight), rows, cols)
	}
	runner, err := getVulkanMatVecF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMatVecAddRMSNorm(normOut, residual, x, w[:wLen], normWeight[:rows], rows, cols)
}

func VulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows, true)
}

func VulkanSwiGLUDownAddRMSNormF32OutOnly(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int) error {
	return vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows, false)
}

func vulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int, updateResidual bool) error {
	if rows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan swiglu/down+add+rmsnorm shape rows=%d cols=%d outRows=%d", rows, cols, outRows)
	}
	dims, err := checkedSwiGLUDimsLinux(rows, cols, outRows, "Vulkan swiglu/down+add+rmsnorm")
	if err != nil {
		return err
	}
	if len(normOut) < outRows || len(residual) < outRows || len(x) < cols || len(gate) < dims.gateLen || len(up) < dims.gateLen || len(down) < dims.downLen || len(normWeight) < outRows {
		return fmt.Errorf("invalid Vulkan swiglu/down+add+rmsnorm buffers normOut=%d residual=%d x=%d gate=%d up=%d down=%d normWeight=%d rows=%d cols=%d outRows=%d",
			len(normOut), len(residual), len(x), len(gate), len(up), len(down), len(normWeight), rows, cols, outRows)
	}
	runner, err := getVulkanSwiGLUDownF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, gate[:dims.gateLen], up[:dims.gateLen], down[:dims.downLen], normWeight[:outRows], rows, cols, outRows, updateResidual)
}

func VulkanMatRowsBiasF32(out, xs [][]float32, w, bias []float32, rows, cols int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias shape rows=%d cols=%d", rows, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matrows+bias")
	if err != nil {
		return err
	}
	if len(out) < batches || len(w) < wLen || len(bias) < rows {
		return fmt.Errorf("invalid Vulkan matrows+bias buffers out=%d xs=%d w=%d bias=%d rows=%d cols=%d", len(out), len(xs), len(w), len(bias), rows, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias row %d buffers out=%d x=%d rows=%d cols=%d", i, len(out[i]), len(xs[i]), rows, cols)
		}
	}
	runner, err := getVulkanMatRowsBiasF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, xs, w[:wLen], bias[:rows], rows, cols)
}

func VulkanMatRowsBiasAddRowsF32(out, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) error {
	batches := len(xs)
	addRows := len(add)
	if batches == 0 {
		return nil
	}
	if rows <= 0 || cols <= 0 || addRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias+addrows shape batches=%d addRows=%d rows=%d cols=%d", batches, addRows, rows, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matrows+bias+addrows")
	if err != nil {
		return err
	}
	if len(out) < batches || len(w) < wLen || len(bias) < rows {
		return fmt.Errorf("invalid Vulkan matrows+bias+addrows buffers out=%d xs=%d w=%d bias=%d add=%d rows=%d cols=%d", len(out), len(xs), len(w), len(bias), len(add), rows, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias+addrows row %d buffers out=%d x=%d rows=%d cols=%d", i, len(out[i]), len(xs[i]), rows, cols)
		}
	}
	for i := 0; i < addRows; i++ {
		if len(add[i]) < rows {
			return fmt.Errorf("invalid Vulkan matrows+bias+addrows add row %d add=%d rows=%d", i, len(add[i]), rows)
		}
	}
	runner, err := getVulkanMatRowsBiasF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRows(out, xs, w[:wLen], bias[:rows], add, rows, cols)
}

func VulkanMatRowsBias3F32(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matrows+bias3 shape batches=%d rowsA=%d rowsB=%d rowsC=%d cols=%d", batches, rowsA, rowsB, rowsC, cols)
	}
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan matrows+bias3 wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan matrows+bias3 wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan matrows+bias3 wc")
	if err != nil {
		return err
	}
	if len(outA) < batches || len(outB) < batches || len(outC) < batches ||
		len(wa) < waLen || len(ba) < rowsA ||
		len(wb) < wbLen || len(bb) < rowsB ||
		len(wc) < wcLen || len(bc) < rowsC {
		return fmt.Errorf("invalid Vulkan matrows+bias3 buffers batches=%d outA=%d outB=%d outC=%d wa=%d ba=%d wb=%d bb=%d wc=%d bc=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			batches, len(outA), len(outB), len(outC), len(wa), len(ba), len(wb), len(bb), len(wc), len(bc), rowsA, rowsB, rowsC, cols)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(outA[i]) < rowsA || len(outB[i]) < rowsB || len(outC[i]) < rowsC {
			return fmt.Errorf("invalid Vulkan matrows+bias3 row %d x=%d outA=%d outB=%d outC=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
				i, len(xs[i]), len(outA[i]), len(outB[i]), len(outC[i]), rowsA, rowsB, rowsC, cols)
		}
	}
	runner, err := getVulkanMatRowsBias3F32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, xs, wa[:waLen], ba[:rowsA], wb[:wbLen], bb[:rowsB], wc[:wcLen], bc[:rowsC], rowsA, rowsB, rowsC, cols)
}

func VulkanVisionAttentionF32(out, q, k, v [][]float32, tokens, heads, headDim int) error {
	if tokens == 0 {
		return nil
	}
	if tokens <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision attention shape tokens=%d heads=%d headDim=%d", tokens, heads, headDim)
	}
	hidden := heads * headDim
	if len(out) < tokens || len(q) < tokens || len(k) < tokens || len(v) < tokens {
		return fmt.Errorf("invalid Vulkan vision attention rows out=%d q=%d k=%d v=%d tokens=%d", len(out), len(q), len(k), len(v), tokens)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(q[i]) < hidden || len(k[i]) < hidden || len(v[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision attention row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, q, k, v, tokens, heads, headDim)
}

func VulkanVisionAttentionOutF32(out, q, k, v [][]float32, w, bias []float32, tokens, heads, headDim int) error {
	if tokens == 0 {
		return nil
	}
	if tokens <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision attention+out shape tokens=%d heads=%d headDim=%d", tokens, heads, headDim)
	}
	d := heads * headDim
	if len(out) < tokens || len(q) < tokens || len(k) < tokens || len(v) < tokens || len(w) < d*d || len(bias) < d {
		return fmt.Errorf("invalid Vulkan vision attention+out buffers out=%d q=%d k=%d v=%d w=%d bias=%d tokens=%d hidden=%d",
			len(out), len(q), len(k), len(v), len(w), len(bias), tokens, d)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < d || len(q[i]) < d || len(k[i]) < d || len(v[i]) < d {
			return fmt.Errorf("invalid Vulkan vision attention+out row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), d)
		}
	}
	runner, err := getVulkanVisionAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOut(out, q, k, v, w, bias, tokens, heads, headDim)
}

func VulkanVisionRoPEPairF32(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	tokens := len(q)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision rope pair shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d", tokens, gridH, gridW, heads, headDim)
	}
	quarter := headDim / 4
	hidden := heads * headDim
	if quarter <= 0 || len(k) < tokens || len(cosH) < gridH*quarter || len(sinH) < gridH*quarter || len(cosW) < gridW*quarter || len(sinW) < gridW*quarter {
		return fmt.Errorf("invalid Vulkan vision rope pair buffers q=%d k=%d cosH=%d sinH=%d cosW=%d sinW=%d gridH=%d gridW=%d quarter=%d",
			len(q), len(k), len(cosH), len(sinH), len(cosW), len(sinW), gridH, gridW, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(q[i]) < hidden || len(k[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision rope pair row %d q=%d k=%d hidden=%d", i, len(q[i]), len(k[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runRoPEPair(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
}

func VulkanVisionRoPEAttentionOutF32(out, q, k, v [][]float32, w, bias, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	tokens := len(q)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan vision rope+attention+out shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d", tokens, gridH, gridW, heads, headDim)
	}
	hidden := heads * headDim
	quarter := headDim / 4
	if quarter <= 0 || len(out) < tokens || len(k) < tokens || len(v) < tokens || len(w) < hidden*hidden || len(bias) < hidden ||
		len(cosH) < gridH*quarter || len(sinH) < gridH*quarter || len(cosW) < gridW*quarter || len(sinW) < gridW*quarter {
		return fmt.Errorf("invalid Vulkan vision rope+attention+out buffers out=%d q=%d k=%d v=%d w=%d bias=%d cosH=%d sinH=%d cosW=%d sinW=%d hidden=%d quarter=%d",
			len(out), len(q), len(k), len(v), len(w), len(bias), len(cosH), len(sinH), len(cosW), len(sinW), hidden, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(q[i]) < hidden || len(k[i]) < hidden || len(v[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision rope+attention+out row %d out=%d q=%d k=%d v=%d hidden=%d", i, len(out[i]), len(q[i]), len(k[i]), len(v[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runRoPEOut(out, q, k, v, w, bias, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
}

func VulkanVisionQKVRoPEAttentionOutF32(out, x [][]float32, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim, hidden int) error {
	tokens := len(x)
	if tokens == 0 {
		return nil
	}
	if gridH <= 0 || gridW <= 0 || heads <= 0 || headDim <= 0 || headDim > 256 || hidden <= 0 || hidden != heads*headDim {
		return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out shape tokens=%d gridH=%d gridW=%d heads=%d headDim=%d hidden=%d", tokens, gridH, gridW, heads, headDim, hidden)
	}
	quarter := headDim / 4
	if quarter <= 0 || len(out) < tokens || len(qw) < hidden*hidden || len(qb) < hidden || len(kw) < hidden*hidden || len(kb) < hidden || len(vw) < hidden*hidden || len(vb) < hidden ||
		len(ow) < hidden*hidden || len(ob) < hidden || len(cosH) < gridH*quarter || len(sinH) < gridH*quarter || len(cosW) < gridW*quarter || len(sinW) < gridW*quarter {
		return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out buffers out=%d x=%d qw=%d qb=%d kw=%d kb=%d vw=%d vb=%d ow=%d ob=%d cosH=%d sinH=%d cosW=%d sinW=%d hidden=%d quarter=%d",
			len(out), len(x), len(qw), len(qb), len(kw), len(kb), len(vw), len(vb), len(ow), len(ob), len(cosH), len(sinH), len(cosW), len(sinW), hidden, quarter)
	}
	for i := 0; i < tokens; i++ {
		if len(out[i]) < hidden || len(x[i]) < hidden {
			return fmt.Errorf("invalid Vulkan vision qkv+rope+attention+out row %d out=%d x=%d hidden=%d", i, len(out[i]), len(x[i]), hidden)
		}
	}
	runner, err := getVulkanVisionAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runQKVRoPEOut(out, x, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim, hidden)
}

func VulkanMatRowsGELU2F32(out, xs [][]float32, w1, b1, w2, b2 []float32, hiddenRows, cols, outRows int) error {
	batches := len(xs)
	if batches == 0 {
		return nil
	}
	if hiddenRows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows gelu2 shape batches=%d hiddenRows=%d cols=%d outRows=%d", batches, hiddenRows, cols, outRows)
	}
	dims, err := checkedMatRowsGELU2DimsLinux(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2")
	if err != nil {
		return err
	}
	if len(out) < batches || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows {
		return fmt.Errorf("invalid Vulkan matrows gelu2 buffers out=%d xs=%d w1=%d b1=%d w2=%d b2=%d hiddenRows=%d cols=%d outRows=%d",
			len(out), len(xs), len(w1), len(b1), len(w2), len(b2), hiddenRows, cols, outRows)
	}
	for i := 0; i < batches; i++ {
		if len(xs[i]) < cols || len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan matrows gelu2 row %d out=%d x=%d cols=%d outRows=%d", i, len(out[i]), len(xs[i]), cols, outRows)
		}
	}
	runner, err := getVulkanMatRowsGELU2F32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, xs, w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], hiddenRows, cols, outRows)
}

func VulkanMatRowsGELU2AddLayerNormF32(out, x, residual [][]float32, w1, b1, w2, b2, normW, normB []float32, hiddenRows, cols, outRows int, eps float32) error {
	batches := len(x)
	if batches == 0 {
		return nil
	}
	if hiddenRows <= 0 || cols <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm shape batches=%d hiddenRows=%d cols=%d outRows=%d", batches, hiddenRows, cols, outRows)
	}
	dims, err := checkedMatRowsGELU2DimsLinux(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2+add layernorm")
	if err != nil {
		return err
	}
	if len(out) < batches || len(residual) < batches || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows || len(normW) < outRows || len(normB) < outRows {
		return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm buffers out=%d x=%d residual=%d w1=%d b1=%d w2=%d b2=%d normW=%d normB=%d hiddenRows=%d cols=%d outRows=%d",
			len(out), len(x), len(residual), len(w1), len(b1), len(w2), len(b2), len(normW), len(normB), hiddenRows, cols, outRows)
	}
	for i := 0; i < batches; i++ {
		if len(x[i]) < cols || len(residual[i]) < outRows || len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan matrows gelu2+add layernorm row %d out=%d x=%d residual=%d cols=%d outRows=%d", i, len(out[i]), len(x[i]), len(residual[i]), cols, outRows)
		}
	}
	runner, err := getVulkanMatRowsGELU2AddLayerNormF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, residual, w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], normW[:outRows], normB[:outRows], hiddenRows, cols, outRows, eps)
}

func VulkanProjectImageF32(out, x [][]float32, normW, normB, w1, b1, w2, b2 []float32, gridT, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	if gridT <= 0 || gridH < 2 || gridW < 2 || gridH%2 != 0 || gridW%2 != 0 || visionDim <= 0 || hiddenRows <= 0 || outRows <= 0 {
		return fmt.Errorf("invalid Vulkan project image shape gridT=%d gridH=%d gridW=%d visionDim=%d hiddenRows=%d outRows=%d", gridT, gridH, gridW, visionDim, hiddenRows, outRows)
	}
	dims, err := checkedProjectImageDimsLinux(gridT, gridH, gridW, visionDim, hiddenRows, outRows, "Vulkan project image")
	if err != nil {
		return err
	}
	if len(out) < dims.batches || len(x) < dims.tokens || len(normW) < visionDim || len(normB) < visionDim || len(w1) < dims.w1Len || len(b1) < hiddenRows || len(w2) < dims.w2Len || len(b2) < outRows {
		return fmt.Errorf("invalid Vulkan project image buffers out=%d x=%d normW=%d normB=%d w1=%d b1=%d w2=%d b2=%d tokens=%d batches=%d visionDim=%d hiddenRows=%d outRows=%d",
			len(out), len(x), len(normW), len(normB), len(w1), len(b1), len(w2), len(b2), dims.tokens, dims.batches, visionDim, hiddenRows, outRows)
	}
	for i := 0; i < dims.tokens; i++ {
		if len(x[i]) < visionDim {
			return fmt.Errorf("invalid Vulkan project image x row %d len=%d visionDim=%d", i, len(x[i]), visionDim)
		}
	}
	for i := 0; i < dims.batches; i++ {
		if len(out[i]) < outRows {
			return fmt.Errorf("invalid Vulkan project image out row %d len=%d outRows=%d", i, len(out[i]), outRows)
		}
	}
	runner, err := getVulkanProjectImageF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x[:dims.tokens], normW[:visionDim], normB[:visionDim], w1[:dims.w1Len], b1[:hiddenRows], w2[:dims.w2Len], b2[:outRows], dims.batches, gridH, gridW, visionDim, hiddenRows, outRows, eps)
}

func VulkanLayerNormRowsF32(out, x [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return vulkanLayerNormRowsF32Linux(out, x, nil, weight, bias, rows, cols, vulkanLayerNormRowsLinuxModePlain, eps)
}

func VulkanAddThenLayerNormRowsF32(out, x, add [][]float32, weight, bias []float32, rows, cols int, eps float32) error {
	return vulkanLayerNormRowsF32Linux(out, x, add, weight, bias, rows, cols, vulkanLayerNormRowsLinuxModeAdd, eps)
}

func vulkanLayerNormRowsF32Linux(out, x, add [][]float32, weight, bias []float32, rows, cols, mode int, eps float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan layernorm rows shape rows=%d cols=%d", rows, cols)
	}
	if len(out) < rows || len(x) < rows || len(weight) < cols || len(bias) < cols {
		return fmt.Errorf("invalid Vulkan layernorm rows buffers out=%d x=%d weight=%d bias=%d rows=%d cols=%d", len(out), len(x), len(weight), len(bias), rows, cols)
	}
	if mode == vulkanLayerNormRowsLinuxModeAdd && len(add) < rows {
		return fmt.Errorf("invalid Vulkan add+layernorm rows add=%d rows=%d", len(add), rows)
	}
	for i := 0; i < rows; i++ {
		if len(out[i]) < cols || len(x[i]) < cols || (mode == vulkanLayerNormRowsLinuxModeAdd && len(add[i]) < cols) {
			addLen := 0
			if mode == vulkanLayerNormRowsLinuxModeAdd && i < len(add) {
				addLen = len(add[i])
			}
			return fmt.Errorf("invalid Vulkan layernorm row %d out=%d x=%d add=%d cols=%d", i, len(out[i]), len(x[i]), addLen, cols)
		}
	}
	runner, err := getVulkanLayerNormRowsF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, add, weight[:cols], bias[:cols], rows, cols, mode, eps)
}

func VulkanTextAttentionF32(out, q, kCache, vCache []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan text attention shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan text attention head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems {
		return fmt.Errorf("invalid Vulkan text attention buffers out=%d q=%d k=%d v=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, q, kCache, vCache, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutF32(out, q, kCache, vCache, w, bias []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan text attention+out shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan text attention+out head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("Vulkan text attention+out weight length overflows: qRows=%d", qRows)
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w) < weightElems || len(bias) < qRows {
		return fmt.Errorf("invalid Vulkan text attention+out buffers out=%d q=%d k=%d v=%d w=%d bias=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w), len(bias), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOut(out, q, kCache, vCache, w, bias, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutAddRMSNormF32(normOut, residual, q, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan text attention+out+add+rmsnorm shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan text attention+out+add+rmsnorm head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("Vulkan text attention+out+add+rmsnorm weight length overflows: qRows=%d", qRows)
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w) < weightElems || len(bias) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d w=%d bias=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w), len(bias), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutAddRMSNorm(normOut, residual, q, kCache, vCache, w, bias, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	if numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan first-token value+out+add+rmsnorm shape numHeads=%d kvHeads=%d headDim=%d", numHeads, kvHeads, headDim)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("Vulkan first-token value+out+add+rmsnorm weight length overflows: qRows=%d", qRows)
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w) < weightElems || len(bias) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d w=%d bias=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w), len(bias), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runFirstTokenValueOutAddRMSNorm(normOut, residual, kCache, vCache, w, bias, normWeight, cacheEpoch, numHeads, kvHeads, headDim)
}

func VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q8 first-token value+out+add+rmsnorm matrix")
	}
	if numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q8 first-token value+out+add+rmsnorm shape numHeads=%d kvHeads=%d headDim=%d", numHeads, kvHeads, headDim)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q8 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runFirstTokenValueOutQ8AddRMSNorm(normOut, residual, kCache, vCache, w, normWeight, cacheEpoch, numHeads, kvHeads, headDim)
}

func VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q6 first-token value+out+add+rmsnorm matrix")
	}
	if numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q6 first-token value+out+add+rmsnorm shape numHeads=%d kvHeads=%d headDim=%d", numHeads, kvHeads, headDim)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q6 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runFirstTokenValueOutQ6AddRMSNorm(normOut, residual, kCache, vCache, w, normWeight, cacheEpoch, numHeads, kvHeads, headDim)
}

func VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q4 first-token value+out+add+rmsnorm matrix")
	}
	if numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q4 first-token value+out+add+rmsnorm shape numHeads=%d kvHeads=%d headDim=%d", numHeads, kvHeads, headDim)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q4 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runFirstTokenValueOutQ4AddRMSNorm(normOut, residual, kCache, vCache, w, normWeight, cacheEpoch, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutQ8(out, q, kCache, vCache []float32, w *tensor.Q8Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q8 text attention+out matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q8 text attention+out shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q8 text attention+out head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ8(out, q, kCache, vCache, w, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutAddRMSNormQ8(normOut, residual, q, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q8 text attention+out+add+rmsnorm matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ8AddRMSNorm(normOut, residual, q, kCache, vCache, w, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutQ6(out, q, kCache, vCache []float32, w *tensor.Q6Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q6 text attention+out matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q6 text attention+out shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q6 text attention+out head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ6(out, q, kCache, vCache, w, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutAddRMSNormQ6(normOut, residual, q, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q6 text attention+out+add+rmsnorm matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ6AddRMSNorm(normOut, residual, q, kCache, vCache, w, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutQ4(out, q, kCache, vCache []float32, w *tensor.Q4Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q4 text attention+out matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q4 text attention+out shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q4 text attention+out head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ4(out, q, kCache, vCache, w, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextAttentionOutAddRMSNormQ4(normOut, residual, q, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if w == nil {
		return fmt.Errorf("nil Vulkan q4 text attention+out+add+rmsnorm matrix")
	}
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runOutQ4AddRMSNorm(normOut, residual, q, kCache, vCache, w, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanMatVecQ8(out, x []float32, q *tensor.Q8Matrix) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q8 matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q8 matvec shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if len(out) < q.Rows || len(x) < q.Cols || len(q.Data) < q.Rows*q.Cols || len(q.Scale) < q.Rows {
		return fmt.Errorf("invalid Vulkan q8 matvec buffers out=%d x=%d data=%d scale=%d rows=%d cols=%d", len(out), len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, q)
}

func VulkanMatVecArgmaxQ8(x []float32, q *tensor.Q8Matrix) (int, float32, error) {
	if q == nil {
		return 0, 0, fmt.Errorf("nil Vulkan q8 matvec argmax matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return 0, 0, fmt.Errorf("invalid Vulkan q8 matvec argmax shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if len(x) < q.Cols || len(q.Data) < q.Rows*q.Cols || len(q.Scale) < q.Rows {
		return 0, 0, fmt.Errorf("invalid Vulkan q8 matvec argmax buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ8LinuxRunner()
	if err != nil {
		return 0, 0, err
	}
	return runner.runArgmax(x, q)
}

func VulkanMatVecTopKQ8(x []float32, q *tensor.Q8Matrix, k int) ([]VulkanTokenScore, error) {
	if q == nil {
		return nil, fmt.Errorf("nil Vulkan q8 matvec top-k matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return nil, fmt.Errorf("invalid Vulkan q8 matvec top-k shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if k <= 0 || k > vulkanMatVecTopKMaxK {
		return nil, fmt.Errorf("invalid Vulkan q8 matvec top-k k=%d max=%d", k, vulkanMatVecTopKMaxK)
	}
	if len(x) < q.Cols || len(q.Data) < q.Rows*q.Cols || len(q.Scale) < q.Rows {
		return nil, fmt.Errorf("invalid Vulkan q8 matvec top-k buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ8LinuxRunner()
	if err != nil {
		return nil, err
	}
	return runner.runTopK(x, q.Data[:q.Rows*q.Cols], q.Scale[:q.Rows], q.Rows, q.Cols, q.Rows*q.Cols, k)
}

func VulkanMatVecAddRMSNormQ8(normOut, residual, x []float32, q *tensor.Q8Matrix, normWeight []float32) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q8 matvec+add+rmsnorm matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q8 matvec+add+rmsnorm shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if len(normOut) < q.Rows || len(residual) < q.Rows || len(x) < q.Cols || len(q.Data) < q.Rows*q.Cols || len(q.Scale) < q.Rows || len(normWeight) < q.Rows {
		return fmt.Errorf("invalid Vulkan q8 matvec+add+rmsnorm buffers normOut=%d residual=%d x=%d data=%d scale=%d normWeight=%d rows=%d cols=%d",
			len(normOut), len(residual), len(x), len(q.Data), len(q.Scale), len(normWeight), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, q, normWeight)
}

func VulkanFusedMatVec3Q8(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 fused matvec3 matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q8 fused matvec3 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < a.Rows*a.Cols || len(b.Data) < b.Rows*b.Cols || len(c.Data) < c.Rows*c.Cols ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q8 fused matvec3 buffers")
	}
	runner, err := getVulkanFusedMatVec3Q8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, x, a, b, c)
}

func VulkanFusedMatVec2Q8(outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 fused matvec2 matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q8 fused matvec2 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < b.Rows*b.Cols || len(c.Data) < c.Rows*c.Cols ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q8 fused matvec2 buffers")
	}
	runner, err := getVulkanFusedMatVec2Q8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2(outB, outC, x, b, c)
}

func VulkanFusedMatVec2MRoPEQ8(outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 fused matvec2+mrope matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q8 fused matvec2+mrope shape a=%dx%d b=%dx%d c=%dx%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, kvHeads, headDim)
	}
	if b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q8 fused matvec2+mrope rows b=%d want=%d", b.Rows, kvHeads*headDim)
	}
	half := headDim / 2
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < b.Rows*b.Cols || len(c.Data) < c.Rows*c.Cols ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q8 fused matvec2+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec2MRoPEQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2MRoPE(outB, outC, x, b, c, cosTable, sinTable, kvHeads, headDim)
}

func VulkanFusedMatVec3MRoPEQ8(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q8 fused matvec3+mrope matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q8 fused matvec3+mrope shape a=%dx%d b=%dx%d c=%dx%d qHeads=%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, qHeads, kvHeads, headDim)
	}
	if a.Rows != qHeads*headDim || b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q8 fused matvec3+mrope rows a=%d b=%d want=%d/%d", a.Rows, b.Rows, qHeads*headDim, kvHeads*headDim)
	}
	half := headDim / 2
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < a.Rows*a.Cols || len(b.Data) < b.Rows*b.Cols || len(c.Data) < c.Rows*c.Cols ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q8 fused matvec3+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec3MRoPEQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPE(outA, outB, outC, x, a, b, c, cosTable, sinTable, kvHeads, headDim)
}

func VulkanSwiGLUDownQ8(out, x []float32, gate, up, down *tensor.Q8Matrix) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q8 swiglu/down matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	if len(out) < down.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gate.Rows*gate.Cols || len(up.Data) < up.Rows*up.Cols || len(down.Data) < down.Rows*down.Cols ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down buffers")
	}
	runner, err := getVulkanSwiGLUDownQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, gate, up, down)
}

func VulkanSwiGLUGateUpQ8(out, x []float32, gate, up *tensor.Q8Matrix) error {
	if gate == nil || up == nil {
		return fmt.Errorf("nil Vulkan q8 swiglu gate/up matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols {
		return fmt.Errorf("invalid Vulkan q8 swiglu gate/up shape gate=%dx%d up=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols)
	}
	if len(out) < gate.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gate.Rows*gate.Cols || len(up.Data) < up.Rows*up.Cols ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu gate/up buffers")
	}
	runner, err := getVulkanSwiGLUDownQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runGateUp(out, x, gate, up)
}

func VulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x, gate, up, down, normWeight, true)
}

func VulkanSwiGLUDownAddRMSNormQ8OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x, gate, up, down, normWeight, false)
}

func vulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32, updateResidual bool) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q8 swiglu/down+add+rmsnorm matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down+add+rmsnorm shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	if len(normOut) < down.Rows || len(residual) < down.Rows || len(x) < gate.Cols || len(normWeight) < down.Rows ||
		len(gate.Data) < gate.Rows*gate.Cols || len(up.Data) < up.Rows*up.Cols || len(down.Data) < down.Rows*down.Cols ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down+add+rmsnorm buffers")
	}
	runner, err := getVulkanSwiGLUDownQ8LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, gate, up, down, normWeight, updateResidual)
}

func VulkanMatVecQ4(out, x []float32, q *tensor.Q4Matrix) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q4 matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q4 matvec shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ4DataLenLinux(q.Rows, q.Cols, "Vulkan q4 matvec")
	if err != nil {
		return err
	}
	if len(out) < q.Rows || len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return fmt.Errorf("invalid Vulkan q4 matvec buffers out=%d x=%d data=%d scale=%d rows=%d cols=%d", len(out), len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen)
}

func VulkanMatVecArgmaxQ4(x []float32, q *tensor.Q4Matrix) (int, float32, error) {
	if q == nil {
		return 0, 0, fmt.Errorf("nil Vulkan q4 matvec argmax matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return 0, 0, fmt.Errorf("invalid Vulkan q4 matvec argmax shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ4DataLenLinux(q.Rows, q.Cols, "Vulkan q4 matvec argmax")
	if err != nil {
		return 0, 0, err
	}
	if len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return 0, 0, fmt.Errorf("invalid Vulkan q4 matvec argmax buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ4LinuxRunner()
	if err != nil {
		return 0, 0, err
	}
	return runner.runArgmax(x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen)
}

func VulkanMatVecTopKQ4(x []float32, q *tensor.Q4Matrix, k int) ([]VulkanTokenScore, error) {
	if q == nil {
		return nil, fmt.Errorf("nil Vulkan q4 matvec top-k matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return nil, fmt.Errorf("invalid Vulkan q4 matvec top-k shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if k <= 0 || k > vulkanMatVecTopKMaxK {
		return nil, fmt.Errorf("invalid Vulkan q4 matvec top-k k=%d max=%d", k, vulkanMatVecTopKMaxK)
	}
	dataLen, err := checkedPackedQ4DataLenLinux(q.Rows, q.Cols, "Vulkan q4 matvec top-k")
	if err != nil {
		return nil, err
	}
	if len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return nil, fmt.Errorf("invalid Vulkan q4 matvec top-k buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ4LinuxRunner()
	if err != nil {
		return nil, err
	}
	return runner.runTopK(x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen, k)
}

func VulkanMatVecAddRMSNormQ4(normOut, residual, x []float32, q *tensor.Q4Matrix, normWeight []float32) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q4 matvec+add+rmsnorm matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q4 matvec+add+rmsnorm shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ4DataLenLinux(q.Rows, q.Cols, "Vulkan q4 matvec+add+rmsnorm")
	if err != nil {
		return err
	}
	if len(normOut) < q.Rows || len(residual) < q.Rows || len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows || len(normWeight) < q.Rows {
		return fmt.Errorf("invalid Vulkan q4 matvec+add+rmsnorm buffers normOut=%d residual=%d x=%d data=%d scale=%d normWeight=%d rows=%d cols=%d",
			len(normOut), len(residual), len(x), len(q.Data), len(q.Scale), len(normWeight), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, q.Data[:dataLen], q.Scale[:q.Rows], normWeight, q.Rows, q.Cols, dataLen)
}

func VulkanFusedMatVec3Q4(outA, outB, outC, x []float32, a, b, c *tensor.Q4Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q4 fused matvec3 matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q4 fused matvec3 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	ap, err := checkedPackedQ4DataLenLinux(a.Rows, a.Cols, "Vulkan q4 fused matvec3 a")
	if err != nil {
		return err
	}
	bp, err := checkedPackedQ4DataLenLinux(b.Rows, b.Cols, "Vulkan q4 fused matvec3 b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedQ4DataLenLinux(c.Rows, c.Cols, "Vulkan q4 fused matvec3 c")
	if err != nil {
		return err
	}
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < ap || len(b.Data) < bp || len(c.Data) < cp ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q4 fused matvec3 buffers")
	}
	runner, err := getVulkanFusedMatVec3Q4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, x,
		a.Data[:ap], b.Data[:bp], c.Data[:cp],
		a.Scale[:a.Rows], b.Scale[:b.Rows], c.Scale[:c.Rows],
		a.Rows, b.Rows, c.Rows, a.Cols, ap, bp, cp)
}

func VulkanFusedMatVec2Q4(outB, outC, x []float32, a, b, c *tensor.Q4Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q4 fused matvec2 matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q4 fused matvec2 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	packedCols, ok := checkedPackedQ4ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q4 fused matvec2 packed cols overflow: cols=%d", a.Cols)
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q4 fused matvec2 b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q4 fused matvec2 c")
	if err != nil {
		return err
	}
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < bp || len(c.Data) < cp ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q4 fused matvec2 buffers")
	}
	runner, err := getVulkanFusedMatVec2Q4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2(outB, outC, x,
		b.Data[:bp], c.Data[:cp],
		b.Scale[:b.Rows], c.Scale[:c.Rows],
		b.Rows, c.Rows, a.Cols, bp, cp)
}

func VulkanFusedMatVec2MRoPEQ4(outB, outC, x []float32, a, b, c *tensor.Q4Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q4 fused matvec2+mrope matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q4 fused matvec2+mrope shape a=%dx%d b=%dx%d c=%dx%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, kvHeads, headDim)
	}
	if b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q4 fused matvec2+mrope rows b=%d want=%d", b.Rows, kvHeads*headDim)
	}
	half := headDim / 2
	packedCols, ok := checkedPackedQ4ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q4 fused matvec2+mrope packed cols overflow: cols=%d", a.Cols)
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q4 fused matvec2+mrope b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q4 fused matvec2+mrope c")
	if err != nil {
		return err
	}
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < bp || len(c.Data) < cp ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q4 fused matvec2+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec2MRoPEQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2MRoPE(outB, outC, x,
		b.Data[:bp], c.Data[:cp],
		b.Scale[:b.Rows], c.Scale[:c.Rows],
		cosTable, sinTable, b.Rows, c.Rows, a.Cols, bp, cp, kvHeads, headDim)
}

func VulkanFusedMatVec3MRoPEQ4(outA, outB, outC, x []float32, a, b, c *tensor.Q4Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q4 fused matvec3+mrope matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q4 fused matvec3+mrope shape a=%dx%d b=%dx%d c=%dx%d qHeads=%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, qHeads, kvHeads, headDim)
	}
	if a.Rows != qHeads*headDim || b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q4 fused matvec3+mrope rows a=%d b=%d want=%d/%d", a.Rows, b.Rows, qHeads*headDim, kvHeads*headDim)
	}
	half := headDim / 2
	packedCols, ok := checkedPackedQ4ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q4 fused matvec3+mrope packed cols overflow: cols=%d", a.Cols)
	}
	ap, err := checkedPackedRowsLinux(a.Rows, packedCols, "Vulkan q4 fused matvec3+mrope a")
	if err != nil {
		return err
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q4 fused matvec3+mrope b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q4 fused matvec3+mrope c")
	if err != nil {
		return err
	}
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < ap || len(b.Data) < bp || len(c.Data) < cp ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q4 fused matvec3+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec3MRoPEQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPE(outA, outB, outC, x,
		a.Data[:ap], b.Data[:bp], c.Data[:cp],
		a.Scale[:a.Rows], b.Scale[:b.Rows], c.Scale[:c.Rows],
		cosTable, sinTable, a.Rows, b.Rows, c.Rows, a.Cols, ap, bp, cp, kvHeads, headDim)
}

func VulkanSwiGLUDownQ4(out, x []float32, gate, up, down *tensor.Q4Matrix) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q4 swiglu/down matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q4 swiglu/down shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	gateLen, err := checkedPackedQ4DataLenLinux(gate.Rows, gate.Cols, "Vulkan q4 swiglu/down gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ4DataLenLinux(up.Rows, up.Cols, "Vulkan q4 swiglu/down up")
	if err != nil {
		return err
	}
	downLen, err := checkedPackedQ4DataLenLinux(down.Rows, down.Cols, "Vulkan q4 swiglu/down down")
	if err != nil {
		return err
	}
	if len(out) < down.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gateLen || len(up.Data) < upLen || len(down.Data) < downLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q4 swiglu/down buffers")
	}
	runner, err := getVulkanSwiGLUDownQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x,
		gate.Data[:gateLen], up.Data[:upLen], down.Data[:downLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows], down.Scale[:down.Rows],
		gate.Rows, gate.Cols, down.Rows, gateLen, upLen, downLen)
}

func VulkanSwiGLUGateUpQ4(out, x []float32, gate, up *tensor.Q4Matrix) error {
	if gate == nil || up == nil {
		return fmt.Errorf("nil Vulkan q4 swiglu gate/up matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols {
		return fmt.Errorf("invalid Vulkan q4 swiglu gate/up shape gate=%dx%d up=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols)
	}
	gateLen, err := checkedPackedQ4DataLenLinux(gate.Rows, gate.Cols, "Vulkan q4 swiglu gate/up gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ4DataLenLinux(up.Rows, up.Cols, "Vulkan q4 swiglu gate/up up")
	if err != nil {
		return err
	}
	if len(out) < gate.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gateLen || len(up.Data) < upLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows {
		return fmt.Errorf("invalid Vulkan q4 swiglu gate/up buffers")
	}
	runner, err := getVulkanSwiGLUDownQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runGateUp(out, x,
		gate.Data[:gateLen], up.Data[:upLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows],
		gate.Rows, gate.Cols, gateLen, upLen)
}

func VulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x []float32, gate, up, down *tensor.Q4Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x, gate, up, down, normWeight, true)
}

func VulkanSwiGLUDownAddRMSNormQ4OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q4Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x, gate, up, down, normWeight, false)
}

func vulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x []float32, gate, up, down *tensor.Q4Matrix, normWeight []float32, updateResidual bool) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q4 swiglu/down+add+rmsnorm matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q4 swiglu/down+add+rmsnorm shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	gateLen, err := checkedPackedQ4DataLenLinux(gate.Rows, gate.Cols, "Vulkan q4 swiglu/down+add+rmsnorm gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ4DataLenLinux(up.Rows, up.Cols, "Vulkan q4 swiglu/down+add+rmsnorm up")
	if err != nil {
		return err
	}
	downLen, err := checkedPackedQ4DataLenLinux(down.Rows, down.Cols, "Vulkan q4 swiglu/down+add+rmsnorm down")
	if err != nil {
		return err
	}
	if len(normOut) < down.Rows || len(residual) < down.Rows || len(x) < gate.Cols || len(normWeight) < down.Rows ||
		len(gate.Data) < gateLen || len(up.Data) < upLen || len(down.Data) < downLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q4 swiglu/down+add+rmsnorm buffers")
	}
	runner, err := getVulkanSwiGLUDownQ4LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x,
		gate.Data[:gateLen], up.Data[:upLen], down.Data[:downLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows], down.Scale[:down.Rows], normWeight[:down.Rows],
		gate.Rows, gate.Cols, down.Rows, gateLen, upLen, downLen, updateResidual)
}

func VulkanMatVecQ6(out, x []float32, q *tensor.Q6Matrix) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q6 matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q6 matvec shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ6DataLenLinux(q.Rows, q.Cols, "Vulkan q6 matvec")
	if err != nil {
		return err
	}
	if len(out) < q.Rows || len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return fmt.Errorf("invalid Vulkan q6 matvec buffers out=%d x=%d data=%d scale=%d rows=%d cols=%d", len(out), len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen)
}

func VulkanMatVecArgmaxQ6(x []float32, q *tensor.Q6Matrix) (int, float32, error) {
	if q == nil {
		return 0, 0, fmt.Errorf("nil Vulkan q6 matvec argmax matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return 0, 0, fmt.Errorf("invalid Vulkan q6 matvec argmax shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ6DataLenLinux(q.Rows, q.Cols, "Vulkan q6 matvec argmax")
	if err != nil {
		return 0, 0, err
	}
	if len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return 0, 0, fmt.Errorf("invalid Vulkan q6 matvec argmax buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ6LinuxRunner()
	if err != nil {
		return 0, 0, err
	}
	return runner.runArgmax(x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen)
}

func VulkanMatVecTopKQ6(x []float32, q *tensor.Q6Matrix, k int) ([]VulkanTokenScore, error) {
	if q == nil {
		return nil, fmt.Errorf("nil Vulkan q6 matvec top-k matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return nil, fmt.Errorf("invalid Vulkan q6 matvec top-k shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	if k <= 0 || k > vulkanMatVecTopKMaxK {
		return nil, fmt.Errorf("invalid Vulkan q6 matvec top-k k=%d max=%d", k, vulkanMatVecTopKMaxK)
	}
	dataLen, err := checkedPackedQ6DataLenLinux(q.Rows, q.Cols, "Vulkan q6 matvec top-k")
	if err != nil {
		return nil, err
	}
	if len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows {
		return nil, fmt.Errorf("invalid Vulkan q6 matvec top-k buffers x=%d data=%d scale=%d rows=%d cols=%d", len(x), len(q.Data), len(q.Scale), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ6LinuxRunner()
	if err != nil {
		return nil, err
	}
	return runner.runTopK(x, q.Data[:dataLen], q.Scale[:q.Rows], q.Rows, q.Cols, dataLen, k)
}

func VulkanMatVecAddRMSNormQ6(normOut, residual, x []float32, q *tensor.Q6Matrix, normWeight []float32) error {
	if q == nil {
		return fmt.Errorf("nil Vulkan q6 matvec+add+rmsnorm matrix")
	}
	if q.Rows <= 0 || q.Cols <= 0 {
		return fmt.Errorf("invalid Vulkan q6 matvec+add+rmsnorm shape rows=%d cols=%d", q.Rows, q.Cols)
	}
	dataLen, err := checkedPackedQ6DataLenLinux(q.Rows, q.Cols, "Vulkan q6 matvec+add+rmsnorm")
	if err != nil {
		return err
	}
	if len(normOut) < q.Rows || len(residual) < q.Rows || len(x) < q.Cols || len(q.Data) < dataLen || len(q.Scale) < q.Rows || len(normWeight) < q.Rows {
		return fmt.Errorf("invalid Vulkan q6 matvec+add+rmsnorm buffers normOut=%d residual=%d x=%d data=%d scale=%d normWeight=%d rows=%d cols=%d",
			len(normOut), len(residual), len(x), len(q.Data), len(q.Scale), len(normWeight), q.Rows, q.Cols)
	}
	runner, err := getVulkanMatVecQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, q.Data[:dataLen], q.Scale[:q.Rows], normWeight, q.Rows, q.Cols, dataLen)
}

func VulkanFusedMatVec3Q6(outA, outB, outC, x []float32, a, b, c *tensor.Q6Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q6 fused matvec3 matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q6 fused matvec3 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	packedCols, ok := checkedPackedQ6ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q6 fused matvec3 packed cols overflow: cols=%d", a.Cols)
	}
	ap, err := checkedPackedRowsLinux(a.Rows, packedCols, "Vulkan q6 fused matvec3 a")
	if err != nil {
		return err
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q6 fused matvec3 b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q6 fused matvec3 c")
	if err != nil {
		return err
	}
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < ap || len(b.Data) < bp || len(c.Data) < cp ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q6 fused matvec3 buffers")
	}
	runner, err := getVulkanFusedMatVec3Q6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, x,
		a.Data[:ap], b.Data[:bp], c.Data[:cp],
		a.Scale[:a.Rows], b.Scale[:b.Rows], c.Scale[:c.Rows],
		a.Rows, b.Rows, c.Rows, a.Cols, ap, bp, cp)
}

func VulkanFusedMatVec2Q6(outB, outC, x []float32, a, b, c *tensor.Q6Matrix) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q6 fused matvec2 matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols {
		return fmt.Errorf("invalid Vulkan q6 fused matvec2 shape a=%dx%d b=%dx%d c=%dx%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols)
	}
	packedCols, ok := checkedPackedQ6ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q6 fused matvec2 packed cols overflow: cols=%d", a.Cols)
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q6 fused matvec2 b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q6 fused matvec2 c")
	if err != nil {
		return err
	}
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < bp || len(c.Data) < cp ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows {
		return fmt.Errorf("invalid Vulkan q6 fused matvec2 buffers")
	}
	runner, err := getVulkanFusedMatVec2Q6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2(outB, outC, x,
		b.Data[:bp], c.Data[:cp],
		b.Scale[:b.Rows], c.Scale[:c.Rows],
		b.Rows, c.Rows, a.Cols, bp, cp)
}

func VulkanFusedMatVec2MRoPEQ6(outB, outC, x []float32, a, b, c *tensor.Q6Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q6 fused matvec2+mrope matrix")
	}
	if b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q6 fused matvec2+mrope shape a=%dx%d b=%dx%d c=%dx%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, kvHeads, headDim)
	}
	if b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q6 fused matvec2+mrope rows b=%d want=%d", b.Rows, kvHeads*headDim)
	}
	half := headDim / 2
	packedCols, ok := checkedPackedQ6ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q6 fused matvec2+mrope packed cols overflow: cols=%d", a.Cols)
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q6 fused matvec2+mrope b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q6 fused matvec2+mrope c")
	if err != nil {
		return err
	}
	if len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(b.Data) < bp || len(c.Data) < cp ||
		len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q6 fused matvec2+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec2MRoPEQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run2MRoPE(outB, outC, x,
		b.Data[:bp], c.Data[:cp],
		b.Scale[:b.Rows], c.Scale[:c.Rows],
		cosTable, sinTable, b.Rows, c.Rows, a.Cols, bp, cp, kvHeads, headDim)
}

func VulkanFusedMatVec3MRoPEQ6(outA, outB, outC, x []float32, a, b, c *tensor.Q6Matrix, cosTable, sinTable []float32, qHeads, kvHeads, headDim int) error {
	if a == nil || b == nil || c == nil {
		return fmt.Errorf("nil Vulkan q6 fused matvec3+mrope matrix")
	}
	if a.Rows <= 0 || b.Rows <= 0 || c.Rows <= 0 || a.Cols <= 0 || b.Cols != a.Cols || c.Cols != a.Cols || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan q6 fused matvec3+mrope shape a=%dx%d b=%dx%d c=%dx%d qHeads=%d kvHeads=%d headDim=%d", a.Rows, a.Cols, b.Rows, b.Cols, c.Rows, c.Cols, qHeads, kvHeads, headDim)
	}
	if a.Rows != qHeads*headDim || b.Rows != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan q6 fused matvec3+mrope rows a=%d b=%d want=%d/%d", a.Rows, b.Rows, qHeads*headDim, kvHeads*headDim)
	}
	half := headDim / 2
	packedCols, ok := checkedPackedQ6ColsLinux(a.Cols)
	if !ok {
		return fmt.Errorf("Vulkan q6 fused matvec3+mrope packed cols overflow: cols=%d", a.Cols)
	}
	ap, err := checkedPackedRowsLinux(a.Rows, packedCols, "Vulkan q6 fused matvec3+mrope a")
	if err != nil {
		return err
	}
	bp, err := checkedPackedRowsLinux(b.Rows, packedCols, "Vulkan q6 fused matvec3+mrope b")
	if err != nil {
		return err
	}
	cp, err := checkedPackedRowsLinux(c.Rows, packedCols, "Vulkan q6 fused matvec3+mrope c")
	if err != nil {
		return err
	}
	if len(outA) < a.Rows || len(outB) < b.Rows || len(outC) < c.Rows || len(x) < a.Cols ||
		len(a.Data) < ap || len(b.Data) < bp || len(c.Data) < cp ||
		len(a.Scale) < a.Rows || len(b.Scale) < b.Rows || len(c.Scale) < c.Rows ||
		len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan q6 fused matvec3+mrope buffers")
	}
	runner, err := getVulkanFusedMatVec3MRoPEQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runMRoPE(outA, outB, outC, x,
		a.Data[:ap], b.Data[:bp], c.Data[:cp],
		a.Scale[:a.Rows], b.Scale[:b.Rows], c.Scale[:c.Rows],
		cosTable, sinTable, a.Rows, b.Rows, c.Rows, a.Cols, ap, bp, cp, kvHeads, headDim)
}

func VulkanSwiGLUDownQ6(out, x []float32, gate, up, down *tensor.Q6Matrix) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q6 swiglu/down matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q6 swiglu/down shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	gateLen, err := checkedPackedQ6DataLenLinux(gate.Rows, gate.Cols, "Vulkan q6 swiglu/down gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ6DataLenLinux(up.Rows, up.Cols, "Vulkan q6 swiglu/down up")
	if err != nil {
		return err
	}
	downLen, err := checkedPackedQ6DataLenLinux(down.Rows, down.Cols, "Vulkan q6 swiglu/down down")
	if err != nil {
		return err
	}
	if len(out) < down.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gateLen || len(up.Data) < upLen || len(down.Data) < downLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q6 swiglu/down buffers")
	}
	runner, err := getVulkanSwiGLUDownQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x,
		gate.Data[:gateLen], up.Data[:upLen], down.Data[:downLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows], down.Scale[:down.Rows],
		gate.Rows, gate.Cols, down.Rows, gateLen, upLen, downLen)
}

func VulkanSwiGLUGateUpQ6(out, x []float32, gate, up *tensor.Q6Matrix) error {
	if gate == nil || up == nil {
		return fmt.Errorf("nil Vulkan q6 swiglu gate/up matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols {
		return fmt.Errorf("invalid Vulkan q6 swiglu gate/up shape gate=%dx%d up=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols)
	}
	gateLen, err := checkedPackedQ6DataLenLinux(gate.Rows, gate.Cols, "Vulkan q6 swiglu gate/up gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ6DataLenLinux(up.Rows, up.Cols, "Vulkan q6 swiglu gate/up up")
	if err != nil {
		return err
	}
	if len(out) < gate.Rows || len(x) < gate.Cols ||
		len(gate.Data) < gateLen || len(up.Data) < upLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows {
		return fmt.Errorf("invalid Vulkan q6 swiglu gate/up buffers")
	}
	runner, err := getVulkanSwiGLUDownQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runGateUp(out, x,
		gate.Data[:gateLen], up.Data[:upLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows],
		gate.Rows, gate.Cols, gateLen, upLen)
}

func VulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x []float32, gate, up, down *tensor.Q6Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x, gate, up, down, normWeight, true)
}

func VulkanSwiGLUDownAddRMSNormQ6OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q6Matrix, normWeight []float32) error {
	return vulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x, gate, up, down, normWeight, false)
}

func vulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x []float32, gate, up, down *tensor.Q6Matrix, normWeight []float32, updateResidual bool) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q6 swiglu/down+add+rmsnorm matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q6 swiglu/down+add+rmsnorm shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	gateLen, err := checkedPackedQ6DataLenLinux(gate.Rows, gate.Cols, "Vulkan q6 swiglu/down+add+rmsnorm gate")
	if err != nil {
		return err
	}
	upLen, err := checkedPackedQ6DataLenLinux(up.Rows, up.Cols, "Vulkan q6 swiglu/down+add+rmsnorm up")
	if err != nil {
		return err
	}
	downLen, err := checkedPackedQ6DataLenLinux(down.Rows, down.Cols, "Vulkan q6 swiglu/down+add+rmsnorm down")
	if err != nil {
		return err
	}
	if len(normOut) < down.Rows || len(residual) < down.Rows || len(x) < gate.Cols || len(normWeight) < down.Rows ||
		len(gate.Data) < gateLen || len(up.Data) < upLen || len(down.Data) < downLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q6 swiglu/down+add+rmsnorm buffers")
	}
	runner, err := getVulkanSwiGLUDownQ6LinuxRunner()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x,
		gate.Data[:gateLen], up.Data[:upLen], down.Data[:downLen],
		gate.Scale[:gate.Rows], up.Scale[:up.Rows], down.Scale[:down.Rows], normWeight[:down.Rows],
		gate.Rows, gate.Cols, down.Rows, gateLen, upLen, downLen, updateResidual)
}

func runVulkanDispatchSmokeTest() error {
	spv, err := vulkanMatVecF32SPV()
	if err != nil {
		return err
	}
	if err := vk.Init(); err != nil {
		return fmt.Errorf("vulkan init: %w", err)
	}

	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-probe\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{
		SType:            vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: &app,
	}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return fmt.Errorf("vkCreateInstance: %s", res)
	}
	defer vk.DestroyInstance(instance, nil)
	if err := vk.InitInstance(instance); err != nil {
		return fmt.Errorf("vulkan init instance: %w", err)
	}

	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}

	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return fmt.Errorf("vkCreateDevice: %s", res)
	}
	defer vk.DestroyDevice(device, nil)
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)

	x := []float32{1, 2, 3, 4}
	w := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		1, 1, 1, 1,
	}
	out := make([]float32, 3)
	xBuf, err := newVulkanHostBuffer(device, memProps, float32ByteLen(len(x)), vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	defer xBuf.destroy(device)
	wBuf, err := newVulkanHostBuffer(device, memProps, float32ByteLen(len(w)), vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	defer wBuf.destroy(device)
	outBuf, err := newVulkanHostBuffer(device, memProps, float32ByteLen(len(out)), vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	defer outBuf.destroy(device)
	if err := xBuf.writeFloat32(device, x); err != nil {
		return err
	}
	if err := wBuf.writeFloat32(device, w); err != nil {
		return err
	}

	var setLayout vk.DescriptorSetLayout
	bindings := []vk.DescriptorSetLayoutBinding{
		{Binding: 0, DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit},
		{Binding: 1, DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit},
		{Binding: 2, DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit},
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &setLayout); res != vk.Success {
		return fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	defer vk.DestroyDescriptorSetLayout(device, setLayout, nil)

	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	defer vk.DestroyPipelineLayout(device, pipelineLayout, nil)

	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)

	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	defer vk.DestroyPipeline(device, pipelines[0], nil)

	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: 3}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
	}, nil, &descPool); res != vk.Success {
		return fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	defer vk.DestroyDescriptorPool(device, descPool, nil)
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{setLayout},
	}, &descSet); res != vk.Success {
		return fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: xBuf.buffer, Offset: 0, Range: xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: outBuf.buffer, Offset: 0, Range: outBuf.size},
	}
	writes := make([]vk.WriteDescriptorSet, 3)
	for i := range writes {
		writes[i] = vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          descSet,
			DstBinding:      uint32(i),
			DescriptorCount: 1,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			PBufferInfo:     bufferInfos[i : i+1],
		}
	}
	vk.UpdateDescriptorSets(device, uint32(len(writes)), writes[:], 0, nil)

	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &cmdPool); res != vk.Success {
		return fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	defer vk.DestroyCommandPool(device, cmdPool, nil)
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        cmdPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	if res := vk.BeginCommandBuffer(cmds[0], &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmds[0], vk.PipelineBindPointCompute, pipelines[0])
	vk.CmdBindDescriptorSets(cmds[0], vk.PipelineBindPointCompute, pipelineLayout, 0, 1, []vk.DescriptorSet{descSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(len(out)))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(len(x)))
	vk.CmdPushConstants(cmds[0], pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmds[0], uint32(len(out)), 1, 1)
	if res := vk.EndCommandBuffer(cmds[0]); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return fmt.Errorf("vkCreateFence: %s", res)
	}
	defer vk.DestroyFence(device, fence, nil)
	submit := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    cmds,
	}
	if res := vk.QueueSubmit(queue, 1, []vk.SubmitInfo{submit}, fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	got, err := outBuf.readFloat32(device, len(out))
	if err != nil {
		return err
	}
	want := []float32{1, 2, 10}
	for i := range want {
		if absFloat32(got[i]-want[i]) > 1e-4 {
			return fmt.Errorf("vulkan matvec mismatch at %d: got %.6f want %.6f", i, got[i], want[i])
		}
	}
	return nil
}

func runVulkanMatVecF32(out, x, w []float32, rows, cols int) error {
	runner, err := getVulkanMatVecF32LinuxRunner()
	if err != nil {
		return err
	}
	return runner.run(out, x, w, rows, cols)
}

func getVulkanMatVecF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	vulkanMatVecF32LinuxRunnerCache.once.Do(func() {
		vulkanMatVecF32LinuxRunnerCache.runner, vulkanMatVecF32LinuxRunnerCache.err = newVulkanMatVecF32LinuxRunner()
	})
	return vulkanMatVecF32LinuxRunnerCache.runner, vulkanMatVecF32LinuxRunnerCache.err
}

func getVulkanMatVecArgmaxF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return getVulkanMatVecF32LinuxRunner()
}

func getVulkanRMSNormF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	vulkanRMSNormF32LinuxRunnerCache.once.Do(func() {
		vulkanRMSNormF32LinuxRunnerCache.runner, vulkanRMSNormF32LinuxRunnerCache.err = newVulkanRMSNormF32LinuxRunner()
	})
	return vulkanRMSNormF32LinuxRunnerCache.runner, vulkanRMSNormF32LinuxRunnerCache.err
}

func getVulkanAddRMSNormF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	vulkanAddRMSNormF32LinuxRunnerCache.once.Do(func() {
		vulkanAddRMSNormF32LinuxRunnerCache.runner, vulkanAddRMSNormF32LinuxRunnerCache.err = newVulkanAddRMSNormF32LinuxRunner()
	})
	return vulkanAddRMSNormF32LinuxRunnerCache.runner, vulkanAddRMSNormF32LinuxRunnerCache.err
}

func getVulkanMRoPEF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	vulkanMRoPEF32LinuxRunnerCache.once.Do(func() {
		vulkanMRoPEF32LinuxRunnerCache.runner, vulkanMRoPEF32LinuxRunnerCache.err = newVulkanMRoPEF32LinuxRunner()
	})
	return vulkanMRoPEF32LinuxRunnerCache.runner, vulkanMRoPEF32LinuxRunnerCache.err
}

func getVulkanMRoPEPairF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	vulkanMRoPEPairF32LinuxRunnerCache.once.Do(func() {
		vulkanMRoPEPairF32LinuxRunnerCache.runner, vulkanMRoPEPairF32LinuxRunnerCache.err = newVulkanMRoPEPairF32LinuxRunner()
	})
	return vulkanMRoPEPairF32LinuxRunnerCache.runner, vulkanMRoPEPairF32LinuxRunnerCache.err
}

type vulkanMatVecF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	normPipeline    vk.Pipeline
	argmaxPipeline  vk.Pipeline
	topKPipeline    vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	addBuf          vulkanHostBuffer
	outBuf          vulkanHostBuffer
	normBuf         vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	topKReadback    []float32
	topKCandidates  []VulkanTokenScore
	descriptorCache [6]vulkanDescriptorBindingLinux
	descriptorCount int
	commandKind     int
	commandRecorded bool
	commandRows     int
	commandCols     int
	dispatchOnce    bool
	mu              sync.Mutex
}

const (
	vulkanMatVecF32LinuxCommandDefault = iota + 1
	vulkanMatVecF32LinuxCommandAddRMSNorm
	vulkanMatVecF32LinuxCommandArgmax
	vulkanMatVecF32LinuxCommandTopK
)

type vulkanDescriptorBindingLinux struct {
	buffer vk.Buffer
	offset vk.DeviceSize
	size   vk.DeviceSize
	valid  bool
}

func updateVulkanDescriptorBuffersLinux(device vk.Device, descriptorSet vk.DescriptorSet, cache []vulkanDescriptorBindingLinux, infos []vk.DescriptorBufferInfo) {
	var stackWrites [32]vk.WriteDescriptorSet
	writes := stackWrites[:]
	if len(infos) > len(stackWrites) {
		writes = make([]vk.WriteDescriptorSet, len(infos))
	}
	changed := 0
	for i := range infos {
		info := infos[i]
		if i < len(cache) {
			cached := &cache[i]
			if cached.valid && cached.buffer == info.Buffer && cached.offset == info.Offset && cached.size == info.Range {
				continue
			}
			*cached = vulkanDescriptorBindingLinux{
				buffer: info.Buffer,
				offset: info.Offset,
				size:   info.Range,
				valid:  true,
			}
		}
		writes[changed] = vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          descriptorSet,
			DstBinding:      uint32(i),
			DescriptorCount: 1,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			PBufferInfo:     infos[i : i+1],
		}
		changed++
	}
	if changed == 0 {
		return
	}
	vk.UpdateDescriptorSets(device, uint32(changed), writes[:changed], 0, nil)
}

func updateVulkanDescriptorBindingsLinux(device vk.Device, descriptorSet vk.DescriptorSet, cache []vulkanDescriptorBindingLinux, bindings []uint32, infos []vk.DescriptorBufferInfo) {
	var stackWrites [32]vk.WriteDescriptorSet
	writes := stackWrites[:]
	if len(infos) > len(stackWrites) {
		writes = make([]vk.WriteDescriptorSet, len(infos))
	}
	changed := 0
	for i := range infos {
		binding := bindings[i]
		info := infos[i]
		if int(binding) < len(cache) {
			cached := &cache[binding]
			if cached.valid && cached.buffer == info.Buffer && cached.offset == info.Offset && cached.size == info.Range {
				continue
			}
			*cached = vulkanDescriptorBindingLinux{
				buffer: info.Buffer,
				offset: info.Offset,
				size:   info.Range,
				valid:  true,
			}
		}
		writes[changed] = vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          descriptorSet,
			DstBinding:      binding,
			DescriptorCount: 1,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			PBufferInfo:     infos[i : i+1],
		}
		changed++
	}
	if changed == 0 {
		return
	}
	vk.UpdateDescriptorSets(device, uint32(changed), writes[:changed], 0, nil)
}

func newVulkanMatVecF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunnerWithPipelines("rapidocrvl-vulkan-matvec-f32\x00", vulkanMatVecF32SPV, vulkanMatVecAddRMSNormF32SPV, vulkanArgmaxF32SPV, vulkanBlockTopKF32SPV, false, 6)
}

func newVulkanMatVecArgmaxF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunnerWithNorm("rapidocrvl-vulkan-matvec-argmax-f32\x00", vulkanMatVecF32SPV, vulkanArgmaxF32SPV, false, 4)
}

func newVulkanRMSNormF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunner("rapidocrvl-vulkan-rmsnorm-f32\x00", vulkanRMSNormF32SPV, true, 3)
}

func newVulkanAddRMSNormF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunner("rapidocrvl-vulkan-add-rmsnorm-f32\x00", vulkanAddRMSNormF32SPV, true, 4)
}

func newVulkanMRoPEF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunner("rapidocrvl-vulkan-mrope-f32\x00", vulkanMRoPEF32SPV, true, 4)
}

func newVulkanMRoPEPairF32LinuxRunner() (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunner("rapidocrvl-vulkan-mrope-pair-f32\x00", vulkanMRoPEPairF32SPV, true, 4)
}

func newVulkanF32TripleBufferLinuxRunner(appName string, shaderCode func() ([]uint32, error), dispatchOnce bool, descriptorCount int) (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunnerWithNorm(appName, shaderCode, nil, dispatchOnce, descriptorCount)
}

func newVulkanF32TripleBufferLinuxRunnerWithNorm(appName string, shaderCode, normShaderCode func() ([]uint32, error), dispatchOnce bool, descriptorCount int) (*vulkanMatVecF32LinuxRunner, error) {
	return newVulkanF32TripleBufferLinuxRunnerWithPipelines(appName, shaderCode, normShaderCode, nil, nil, dispatchOnce, descriptorCount)
}

func newVulkanF32TripleBufferLinuxRunnerWithPipelines(appName string, shaderCode, normShaderCode, argmaxShaderCode, topKShaderCode func() ([]uint32, error), dispatchOnce bool, descriptorCount int) (*vulkanMatVecF32LinuxRunner, error) {
	spv, err := shaderCode()
	if err != nil {
		return nil, err
	}
	var normSPV []uint32
	if normShaderCode != nil {
		normSPV, err = normShaderCode()
		if err != nil {
			return nil, err
		}
	}
	var argmaxSPV []uint32
	if argmaxShaderCode != nil {
		argmaxSPV, err = argmaxShaderCode()
		if err != nil {
			return nil, err
		}
	}
	var topKSPV []uint32
	if topKShaderCode != nil {
		topKSPV, err = topKShaderCode()
		if err != nil {
			return nil, err
		}
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatVecF32LinuxRunner{instance: instance, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux), dispatchOnce: dispatchOnce, descriptorCount: descriptorCount}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	var setLayout vk.DescriptorSetLayout
	bindings := make([]vk.DescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if normShaderCode != nil {
		var normShader vk.ShaderModule
		if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
			SType:    vk.StructureTypeShaderModuleCreateInfo,
			CodeSize: uint(len(normSPV) * 4),
			PCode:    normSPV,
		}, nil, &normShader); res != vk.Success {
			return nil, fmt.Errorf("vkCreateShaderModule matvec norm: %s", res)
		}
		defer vk.DestroyShaderModule(device, normShader, nil)
		normPipelines := make([]vk.Pipeline, 1)
		normStage := vk.PipelineShaderStageCreateInfo{
			SType:  vk.StructureTypePipelineShaderStageCreateInfo,
			Stage:  vk.ShaderStageComputeBit,
			Module: normShader,
			PName:  "main\x00",
		}
		if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
			SType:  vk.StructureTypeComputePipelineCreateInfo,
			Stage:  normStage,
			Layout: pipelineLayout,
		}}, nil, normPipelines); res != vk.Success {
			return nil, fmt.Errorf("vkCreateComputePipelines matvec norm: %s", res)
		}
		r.normPipeline = normPipelines[0]
	}
	if argmaxShaderCode != nil {
		var argmaxShader vk.ShaderModule
		if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
			SType:    vk.StructureTypeShaderModuleCreateInfo,
			CodeSize: uint(len(argmaxSPV) * 4),
			PCode:    argmaxSPV,
		}, nil, &argmaxShader); res != vk.Success {
			return nil, fmt.Errorf("vkCreateShaderModule matvec argmax: %s", res)
		}
		defer vk.DestroyShaderModule(device, argmaxShader, nil)
		argmaxPipelines := make([]vk.Pipeline, 1)
		argmaxStage := vk.PipelineShaderStageCreateInfo{
			SType:  vk.StructureTypePipelineShaderStageCreateInfo,
			Stage:  vk.ShaderStageComputeBit,
			Module: argmaxShader,
			PName:  "main\x00",
		}
		if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
			SType:  vk.StructureTypeComputePipelineCreateInfo,
			Stage:  argmaxStage,
			Layout: pipelineLayout,
		}}, nil, argmaxPipelines); res != vk.Success {
			return nil, fmt.Errorf("vkCreateComputePipelines matvec argmax: %s", res)
		}
		r.argmaxPipeline = argmaxPipelines[0]
	}
	if topKShaderCode != nil {
		var topKShader vk.ShaderModule
		if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
			SType:    vk.StructureTypeShaderModuleCreateInfo,
			CodeSize: uint(len(topKSPV) * 4),
			PCode:    topKSPV,
		}, nil, &topKShader); res != vk.Success {
			return nil, fmt.Errorf("vkCreateShaderModule matvec top-k: %s", res)
		}
		defer vk.DestroyShaderModule(device, topKShader, nil)
		topKPipelines := make([]vk.Pipeline, 1)
		topKStage := vk.PipelineShaderStageCreateInfo{
			SType:  vk.StructureTypePipelineShaderStageCreateInfo,
			Stage:  vk.ShaderStageComputeBit,
			Module: topKShader,
			PName:  "main\x00",
		}
		if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
			SType:  vk.StructureTypeComputePipelineCreateInfo,
			Stage:  topKStage,
			Layout: pipelineLayout,
		}}, nil, topKPipelines); res != vk.Success {
			return nil, fmt.Errorf("vkCreateComputePipelines matvec top-k: %s", res)
		}
		r.topKPipeline = topKPipelines[0]
	}
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
	}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{setLayout},
	}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        cmdPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanMatVecF32LinuxRunner) run(out, x, w []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xLen := cols
	if r.dispatchOnce {
		xLen = rows
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan f32 matvec runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(xLen, "Vulkan f32 matvec runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan f32 matvec runner weight")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan f32 matvec runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(w[:wLen], wBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:xLen]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandDefault || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(device, out[:rows])
}

func (r *vulkanMatVecF32LinuxRunner) runMatVecArgmax(x, w []float32, rows, cols int) (int, float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.argmaxPipeline == vk.NullPipeline {
		return 0, 0, fmt.Errorf("vulkan matvec argmax pipeline is unavailable")
	}
	device := r.device
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan f32 matvec argmax runner")
	if err != nil {
		return 0, 0, err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 matvec argmax runner x")
	if err != nil {
		return 0, 0, err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan f32 matvec argmax runner weight")
	if err != nil {
		return 0, 0, err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan f32 matvec argmax runner output")
	if err != nil {
		return 0, 0, err
	}
	resultBytes, err := checkedFloat32ByteLenErrLinux(2, "Vulkan f32 matvec argmax runner result")
	if err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.addBuf, resultBytes); err != nil {
		return 0, 0, err
	}
	wBuf, err := r.weightBuffer(w[:wLen], wBytes)
	if err != nil {
		return 0, 0, err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return 0, 0, err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandArgmax || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordMatVecArgmaxCommand(rows, cols); err != nil {
			return 0, 0, err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return 0, 0, fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return 0, 0, fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return 0, 0, fmt.Errorf("vkWaitForFences: %s", res)
	}
	var result [2]float32
	if err := r.addBuf.readFloat32Into(device, result[:]); err != nil {
		return 0, 0, err
	}
	return int(result[1]), result[0], nil
}

func (r *vulkanMatVecF32LinuxRunner) runMatVecTopK(x, w []float32, rows, cols, k int) ([]VulkanTokenScore, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.topKPipeline == vk.NullPipeline {
		return nil, fmt.Errorf("vulkan matvec top-k pipeline is unavailable")
	}
	device := r.device
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan f32 matvec top-k runner")
	if err != nil {
		return nil, err
	}
	candidateFloats, err := checkedMatVecTopKCandidateFloatsLinux(rows, "Vulkan f32 matvec top-k runner")
	if err != nil {
		return nil, err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 matvec top-k runner x")
	if err != nil {
		return nil, err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan f32 matvec top-k runner weight")
	if err != nil {
		return nil, err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan f32 matvec top-k runner output")
	if err != nil {
		return nil, err
	}
	resultBytes, err := checkedFloat32ByteLenErrLinux(candidateFloats, "Vulkan f32 matvec top-k runner result")
	if err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.addBuf, resultBytes); err != nil {
		return nil, err
	}
	wBuf, err := r.weightBuffer(w[:wLen], wBytes)
	if err != nil {
		return nil, err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return nil, err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
	}
	bindings := [...]uint32{0, 1, 2, 3}
	updateVulkanDescriptorBindingsLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bindings[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandTopK || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordMatVecTopKCommand(rows, cols); err != nil {
			return nil, err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return nil, fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return nil, fmt.Errorf("vkWaitForFences: %s", res)
	}
	r.topKReadback = ensureVulkanFloat32Scratch(r.topKReadback, candidateFloats)
	candidateData := r.topKReadback
	if err := r.addBuf.readFloat32Into(device, candidateData); err != nil {
		return nil, err
	}
	r.topKCandidates = selectVulkanTopKCandidatesInto(r.topKCandidates, candidateData, rows, k)
	return r.topKCandidates, nil
}

func (r *vulkanMatVecF32LinuxRunner) runMatVecAddRMSNorm(normOut, residual, x, w, normWeight []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.normPipeline == vk.NullPipeline {
		return fmt.Errorf("vulkan matvec+add+rmsnorm pipeline is unavailable")
	}
	device := r.device
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan f32 matvec add rmsnorm runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 matvec add rmsnorm runner x")
	if err != nil {
		return err
	}
	rowsBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan f32 matvec add rmsnorm runner rows")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan f32 matvec add rmsnorm runner weight")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, rowsBytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(w[:wLen], wBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.weightBuffer(normWeight[:rows], rowsBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.addBuf.writeFloat32(device, residual[:rows]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandAddRMSNorm || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordMatVecAddRMSNormCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.addBuf.readFloat32Into(device, residual[:rows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(device, normOut[:rows])
}

func (r *vulkanMatVecF32LinuxRunner) runAdd(out, dst, add, weight []float32, n int) error {
	return r.runAddMaybeReadDst(out, dst, add, weight, n, true)
}

func (r *vulkanMatVecF32LinuxRunner) runAddOutOnly(out, dst, add, weight []float32, n int) error {
	return r.runAddMaybeReadDst(out, dst, add, weight, n, false)
}

func (r *vulkanMatVecF32LinuxRunner) runAddMaybeReadDst(out, dst, add, weight []float32, n int, readDst bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	bytes := float32ByteLen(n)
	if err := r.ensureHostBuffer(&r.xBuf, bytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, bytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bytes); err != nil {
		return err
	}
	wBuf, err := r.weightBuffer(weight[:n], bytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, dst[:n]); err != nil {
		return err
	}
	if err := r.addBuf.writeFloat32(device, add[:n]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandDefault || r.commandRows != n || r.commandCols != 1 {
		if err := r.recordCommand(n, 1); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if readDst {
		if err := r.xBuf.readFloat32Into(device, dst[:n]); err != nil {
			return err
		}
	}
	return r.outBuf.readFloat32Into(device, out[:n])
}

func (r *vulkanMatVecF32LinuxRunner) runMRoPE(x, cosTable, sinTable []float32, heads, dim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	n := heads * dim
	half := dim / 2
	xBytes := float32ByteLen(n)
	tableBytes := float32ByteLen(half)
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, xBytes); err != nil {
		return err
	}
	sinBuf, err := r.weightBuffer(sinTable[:half], tableBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:n]); err != nil {
		return err
	}
	if err := r.addBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
		{Buffer: sinBuf.buffer, Offset: 0, Range: sinBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandDefault || r.commandRows != n || r.commandCols != dim {
		if err := r.recordCommand(n, dim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.xBuf.readFloat32Into(device, x[:n])
}

func (r *vulkanMatVecF32LinuxRunner) runMRoPEPair(q, k, cosTable, sinTable []float32, qHeads, kvHeads, dim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	qRows := qHeads * dim
	kvRows := kvHeads * dim
	half := dim / 2
	qBytes := float32ByteLen(qRows)
	kBytes := float32ByteLen(kvRows)
	tableBytes := float32ByteLen(half)
	if err := r.ensureHostBuffer(&r.xBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, kBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, tableBytes); err != nil {
		return err
	}
	cosBuf, err := r.weightBuffer(cosTable[:half], tableBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, q[:qRows]); err != nil {
		return err
	}
	if err := r.addBuf.writeFloat32(device, k[:kvRows]); err != nil {
		return err
	}
	if err := r.outBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
		{Buffer: cosBuf.buffer, Offset: 0, Range: cosBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	encodedCols := (kvHeads << 16) | dim
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32LinuxCommandDefault || r.commandRows != qHeads || r.commandCols != encodedCols {
		if err := r.recordCommand(qHeads, encodedCols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.xBuf.readFloat32Into(device, q[:qRows]); err != nil {
		return err
	}
	return r.addBuf.readFloat32Into(device, k[:kvRows])
}

func (r *vulkanMatVecF32LinuxRunner) recordCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	groupsX := rows
	if r.dispatchOnce {
		groupsX = 1
	}
	vk.CmdDispatch(cmd, uint32(groupsX), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32LinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32LinuxRunner) recordMatVecAddRMSNormCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32LinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32LinuxRunner) recordMatVecArgmaxCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.argmaxPipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32LinuxCommandArgmax
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32LinuxRunner) recordMatVecTopKCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.topKPipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	blocks := (rows + 255) / 256
	vk.CmdDispatch(cmd, uint32(blocks), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32LinuxCommandTopK
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatVecF32LinuxRunner) weightBuffer(w []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(w)
	fingerprint := fingerprintFloat32ForVulkanCache(w)
	if cached, ok := r.weightBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(w) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, w); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(w), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.weightBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, w); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(w), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanMatVecF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.argmaxPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.argmaxPipeline, nil)
		}
		if r.topKPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.topKPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.addBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanFusedMatVec3F32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	vulkanFusedMatVec3F32LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec3F32LinuxRunnerCache.runner, vulkanFusedMatVec3F32LinuxRunnerCache.err = newVulkanFusedMatVec3F32LinuxRunner()
	})
	return vulkanFusedMatVec3F32LinuxRunnerCache.runner, vulkanFusedMatVec3F32LinuxRunnerCache.err
}

func getVulkanFusedMatVec2F32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	vulkanFusedMatVec2F32LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec2F32LinuxRunnerCache.runner, vulkanFusedMatVec2F32LinuxRunnerCache.err = newVulkanFusedMatVec2F32LinuxRunner()
	})
	return vulkanFusedMatVec2F32LinuxRunnerCache.runner, vulkanFusedMatVec2F32LinuxRunnerCache.err
}

func getVulkanFusedMatVec3MRoPEF32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	vulkanFusedMatVec3MRoPEF32LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEF32LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEF32LinuxRunnerCache.err = newVulkanFusedMatVec3MRoPEF32LinuxRunner()
	})
	return vulkanFusedMatVec3MRoPEF32LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEF32LinuxRunnerCache.err
}

func getVulkanFusedMatVec2MRoPEF32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	vulkanFusedMatVec2MRoPEF32LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec2MRoPEF32LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEF32LinuxRunnerCache.err = newVulkanFusedMatVec2MRoPEF32LinuxRunner()
	})
	return vulkanFusedMatVec2MRoPEF32LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEF32LinuxRunnerCache.err
}

type vulkanFusedMatVec3F32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outABuf         vulkanHostBuffer
	outBBuf         vulkanHostBuffer
	outCBuf         vulkanHostBuffer
	cosBuf          vulkanHostBuffer
	sinBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [9]vulkanDescriptorBindingLinux
	descriptorCount int
	mrope           bool
	commandRecorded bool
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	commandPacked   int
	mu              sync.Mutex
}

func newVulkanFusedMatVec3F32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	return newVulkanFusedMatVec3F32LinuxRunnerWithShader("rapidocrvl-vulkan-f32-fused-matvec3\x00", vulkanFusedMatVec3F32SPV, 7, 16, false)
}

func newVulkanFusedMatVec2F32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	return newVulkanFusedMatVec3F32LinuxRunnerWithShader("rapidocrvl-vulkan-f32-fused-matvec2\x00", vulkanFusedMatVec2F32SPV, 5, 12, false)
}

func newVulkanFusedMatVec3MRoPEF32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	return newVulkanFusedMatVec3F32LinuxRunnerWithShader("rapidocrvl-vulkan-f32-fused-matvec3-mrope\x00", vulkanFusedMatVec3MRoPEF32SPV, 9, 20, true)
}

func newVulkanFusedMatVec2MRoPEF32LinuxRunner() (*vulkanFusedMatVec3F32LinuxRunner, error) {
	return newVulkanFusedMatVec3F32LinuxRunnerWithShader("rapidocrvl-vulkan-f32-fused-matvec2-mrope\x00", vulkanFusedMatVec2MRoPEF32SPV, 7, 16, true)
}

func newVulkanFusedMatVec3F32LinuxRunnerWithShader(appLabel string, shaderCode func() ([]uint32, error), descriptorCount, pushConstantBytes int, mrope bool) (*vulkanFusedMatVec3F32LinuxRunner, error) {
	spv, err := shaderCode()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appLabel,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanFusedMatVec3F32LinuxRunner{instance: instance, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux), descriptorCount: descriptorCount, mrope: mrope}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: uint32(pushConstantBytes)}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: descPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{setLayout}}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: cmdPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) run(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan f32 fused matvec3 runner wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan f32 fused matvec3 runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan f32 fused matvec3 runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 fused matvec3 runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrLinux(waLen, "Vulkan f32 fused matvec3 runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrLinux(wbLen, "Vulkan f32 fused matvec3 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrLinux(wcLen, "Vulkan f32 fused matvec3 runner wc")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrLinux(rowsA, "Vulkan f32 fused matvec3 runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan f32 fused matvec3 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan f32 fused matvec3 runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	waBuf, err := r.weightBuffer(wa[:waLen], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:wbLen], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:wcLen], wcBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: waBuf.buffer, Offset: 0, Range: waBuf.size},
		{Buffer: wbBuf.buffer, Offset: 0, Range: wbBuf.size},
		{Buffer: wcBuf.buffer, Offset: 0, Range: wcBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, 0); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:rowsA]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) run2(outB, outC, x, wb, wc []float32, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan f32 fused matvec2 runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan f32 fused matvec2 runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 fused matvec2 runner x")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrLinux(wbLen, "Vulkan f32 fused matvec2 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrLinux(wcLen, "Vulkan f32 fused matvec2 runner wc")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan f32 fused matvec2 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan f32 fused matvec2 runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:wbLen], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:wcLen], wcBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wbBuf.buffer, Offset: 0, Range: wbBuf.size},
		{Buffer: wcBuf.buffer, Offset: 0, Range: wcBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand2(rowsB, rowsC, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:rowsC])
}

func (r *vulkanFusedMatVec3F32LinuxRunner) runMRoPE(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	half := headDim / 2
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan f32 fused matvec3 mrope runner wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan f32 fused matvec3 mrope runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan f32 fused matvec3 mrope runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 fused matvec3 mrope runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrLinux(waLen, "Vulkan f32 fused matvec3 mrope runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrLinux(wbLen, "Vulkan f32 fused matvec3 mrope runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrLinux(wcLen, "Vulkan f32 fused matvec3 mrope runner wc")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan f32 fused matvec3 mrope runner table")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrLinux(rowsA, "Vulkan f32 fused matvec3 mrope runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan f32 fused matvec3 mrope runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan f32 fused matvec3 mrope runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	waBuf, err := r.weightBuffer(wa[:waLen], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:wbLen], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:wcLen], wcBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: waBuf.buffer, Offset: 0, Range: waBuf.size},
		{Buffer: wbBuf.buffer, Offset: 0, Range: wbBuf.size},
		{Buffer: wcBuf.buffer, Offset: 0, Range: wcBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:rowsA]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:rowsC]); err != nil {
		return err
	}
	_ = kvHeads
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) run2MRoPE(outB, outC, x, wb, wc, cosTable, sinTable []float32, rowsB, rowsC, cols, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	half := headDim / 2
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan f32 fused matvec2 mrope runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan f32 fused matvec2 mrope runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan f32 fused matvec2 mrope runner x")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrLinux(wbLen, "Vulkan f32 fused matvec2 mrope runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrLinux(wcLen, "Vulkan f32 fused matvec2 mrope runner wc")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan f32 fused matvec2 mrope runner table")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan f32 fused matvec2 mrope runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan f32 fused matvec2 mrope runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:wbLen], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:wcLen], wcBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wbBuf.buffer, Offset: 0, Range: wbBuf.size},
		{Buffer: wcBuf.buffer, Offset: 0, Range: wcBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand2MRoPE(rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:rowsC])
}

func (r *vulkanFusedMatVec3F32LinuxRunner) recordCommand2(rowsB, rowsC, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [12]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) recordCommand2MRoPE(rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(packed))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB/2+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) recordCommand(rowsA, rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsA))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(cols))
	pushBytes := 16
	groups := rowsA + rowsB + rowsC
	if r.mrope {
		binary.LittleEndian.PutUint32(pc[16:20], uint32(packed))
		pushBytes = 20
		groups = rowsA/2 + rowsB/2 + rowsC
	}
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(pushBytes), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(groups), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) weightBuffer(w []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(w)
	fingerprint := fingerprintFloat32ForVulkanCache(w)
	if cached, ok := r.weightBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(w) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, w); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(w), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.weightBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, w); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(w), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanFusedMatVec3F32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outABuf.destroy(r.device)
		r.outBBuf.destroy(r.device)
		r.outCBuf.destroy(r.device)
		r.cosBuf.destroy(r.device)
		r.sinBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanSwiGLUGateUpF32LinuxRunner() (*vulkanSwiGLUGateUpF32LinuxRunner, error) {
	vulkanSwiGLUGateUpF32LinuxRunnerCache.once.Do(func() {
		vulkanSwiGLUGateUpF32LinuxRunnerCache.runner, vulkanSwiGLUGateUpF32LinuxRunnerCache.err = newVulkanSwiGLUGateUpF32LinuxRunner()
	})
	return vulkanSwiGLUGateUpF32LinuxRunnerCache.runner, vulkanSwiGLUGateUpF32LinuxRunnerCache.err
}

type vulkanSwiGLUGateUpF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [4]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandRows     int
	commandCols     int
	mu              sync.Mutex
}

func newVulkanSwiGLUGateUpF32LinuxRunner() (*vulkanSwiGLUGateUpF32LinuxRunner, error) {
	spv, err := vulkanSwiGLUGateUpF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-f32-swiglu-gate-up\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanSwiGLUGateUpF32LinuxRunner{instance: instance, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, 4)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: 4}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: descPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{setLayout}}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: cmdPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanSwiGLUGateUpF32LinuxRunner) run(out, x, gate, up []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	dims, err := checkedSwiGLUDimsLinux(rows, cols, 0, "Vulkan swiglu gate/up runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan swiglu gate/up runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(dims.gateLen, "Vulkan swiglu gate/up runner gate/up")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan swiglu gate/up runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:dims.gateLen], wBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:dims.gateLen], wBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:10], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecPackedLinuxCommandDefault || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(device, out[:rows])
}

func (r *vulkanSwiGLUGateUpF32LinuxRunner) recordCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUGateUpF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanSwiGLUGateUpF32LinuxRunner) weightBuffer(w []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(w)
	fingerprint := fingerprintFloat32ForVulkanCache(w)
	if cached, ok := r.weightBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(w) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, w); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(w), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.weightBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, w); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(w), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanSwiGLUGateUpF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanSwiGLUDownF32LinuxRunner() (*vulkanSwiGLUDownF32LinuxRunner, error) {
	vulkanSwiGLUDownF32LinuxRunnerCache.once.Do(func() {
		vulkanSwiGLUDownF32LinuxRunnerCache.runner, vulkanSwiGLUDownF32LinuxRunnerCache.err = newVulkanSwiGLUDownF32LinuxRunner()
	})
	return vulkanSwiGLUDownF32LinuxRunnerCache.runner, vulkanSwiGLUDownF32LinuxRunnerCache.err
}

type vulkanSwiGLUDownF32LinuxRunner struct {
	instance            vk.Instance
	device              vk.Device
	queue               vk.Queue
	queueFamily         uint32
	memProps            vk.PhysicalDeviceMemoryProperties
	setLayout           vk.DescriptorSetLayout
	downSetLayout       vk.DescriptorSetLayout
	descriptorPool      vk.DescriptorPool
	descriptorSet       vk.DescriptorSet
	downDescriptorSet   vk.DescriptorSet
	pipelineLayout      vk.PipelineLayout
	pipeline            vk.Pipeline
	downPipelineLayout  vk.PipelineLayout
	downPipeline        vk.Pipeline
	normPipeline        vk.Pipeline
	commandPool         vk.CommandPool
	commandBuffer       vk.CommandBuffer
	fence               vk.Fence
	xBuf                vulkanHostBuffer
	hiddenBuf           vulkanHostBuffer
	outBuf              vulkanHostBuffer
	residualBuf         vulkanHostBuffer
	normBuf             vulkanHostBuffer
	weightBuffers       map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache     [4]vulkanDescriptorBindingLinux
	downDescriptorCache [6]vulkanDescriptorBindingLinux
	commandKind         int
	commandRecorded     bool
	commandRows         int
	commandCols         int
	commandOutRows      int
	mu                  sync.Mutex
}

const (
	vulkanSwiGLUDownF32LinuxCommandDefault = iota + 1
	vulkanSwiGLUDownF32LinuxCommandAddRMSNorm
)

func newVulkanSwiGLUDownF32LinuxRunner() (*vulkanSwiGLUDownF32LinuxRunner, error) {
	spv, err := vulkanSwiGLUGateUpF32SPV()
	if err != nil {
		return nil, err
	}
	downSPV, err := vulkanMatVecF32SPV()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanMatVecAddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-f32-swiglu-down\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanSwiGLUDownF32LinuxRunner{instance: instance, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	gateBindings := make([]vk.DescriptorSetLayoutBinding, 4)
	for i := range gateBindings {
		gateBindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(gateBindings)), PBindings: gateBindings}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	downBindings := make([]vk.DescriptorSetLayoutBinding, 6)
	for i := range downBindings {
		downBindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(downBindings)), PBindings: downBindings}, nil, &r.downSetLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout down: %s", res)
	}
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(gateBindings) + len(downBindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 2, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	descSets := make([]vk.DescriptorSet, 2)
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 2, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout, r.downSetLayout}}, descSets); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSets[0]
	r.downDescriptorSet = descSets[1]
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.downSetLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.downPipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout down: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var downShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(downSPV) * 4), PCode: downSPV}, nil, &downShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule down: %s", res)
	}
	defer vk.DestroyShaderModule(device, downShader, nil)
	downPipelines := make([]vk.Pipeline, 1)
	downStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: downShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: downStage, Layout: r.downPipelineLayout}}, nil, downPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines down: %s", res)
	}
	r.downPipeline = downPipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(normSPV) * 4), PCode: normSPV}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule swiglu norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: normShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.downPipelineLayout}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines swiglu norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanSwiGLUDownF32LinuxRunner) run(out, x, gate, up, down []float32, rows, cols, outRows int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedSwiGLUDimsLinux(rows, cols, outRows, "Vulkan swiglu/down runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan swiglu/down runner x")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan swiglu/down runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan swiglu/down runner output")
	if err != nil {
		return err
	}
	gateBytes, err := checkedFloat32ByteLenErrLinux(dims.gateLen, "Vulkan swiglu/down runner gate/up")
	if err != nil {
		return err
	}
	downBytes, err := checkedFloat32ByteLenErrLinux(dims.downLen, "Vulkan swiglu/down runner down")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:dims.gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:dims.gateLen], gateBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.weightBuffer(down[:dims.downLen], downBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	gateInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
	}
	downInfos := [3]vk.DescriptorBufferInfo{
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], gateInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownF32LinuxCommandDefault || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(r.device, out[:outRows])
}

func (r *vulkanSwiGLUDownF32LinuxRunner) runAddRMSNorm(normOut, residual, x, gate, up, down, normWeight []float32, rows, cols, outRows int, updateResidual bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedSwiGLUDimsLinux(rows, cols, outRows, "Vulkan swiglu/down+add+rmsnorm runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan swiglu/down+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan swiglu/down+add+rmsnorm runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan swiglu/down+add+rmsnorm runner output")
	if err != nil {
		return err
	}
	gateBytes, err := checkedFloat32ByteLenErrLinux(dims.gateLen, "Vulkan swiglu/down+add+rmsnorm runner gate/up")
	if err != nil {
		return err
	}
	downBytes, err := checkedFloat32ByteLenErrLinux(dims.downLen, "Vulkan swiglu/down+add+rmsnorm runner down")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.weightBuffer(gate[:dims.gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.weightBuffer(up[:dims.gateLen], gateBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.weightBuffer(down[:dims.downLen], downBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.weightBuffer(normWeight[:outRows], outBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:outRows]); err != nil {
		return err
	}
	gateInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
	}
	downInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], gateInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownF32LinuxCommandAddRMSNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordAddRMSNormCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if updateResidual {
		if err := r.residualBuf.readFloat32Into(r.device, residual[:outRows]); err != nil {
			return err
		}
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:outRows])
}

func (r *vulkanSwiGLUDownF32LinuxRunner) recordCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownF32LinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownF32LinuxRunner) recordAddRMSNormCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownF32LinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanSwiGLUDownF32LinuxRunner) weightBuffer(w []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(w)
	fingerprint := fingerprintFloat32ForVulkanCache(w)
	if cached, ok := r.weightBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(w) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, w); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(w), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.weightBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, w); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.weightBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(w), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanSwiGLUDownF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.downPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.downPipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.hiddenBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.downPipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.downPipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		if r.downSetLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.downSetLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanVisionAttentionF32LinuxRunner() (*vulkanVisionAttentionF32LinuxRunner, error) {
	vulkanVisionAttentionF32LinuxRunnerCache.once.Do(func() {
		vulkanVisionAttentionF32LinuxRunnerCache.runner, vulkanVisionAttentionF32LinuxRunnerCache.err = newVulkanVisionAttentionF32LinuxRunner()
	})
	return vulkanVisionAttentionF32LinuxRunnerCache.runner, vulkanVisionAttentionF32LinuxRunnerCache.err
}

type vulkanVisionAttentionF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	projPipeline    vk.Pipeline
	ropePipeline    vk.Pipeline
	qkvPipeline     vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	qBuf            vulkanHostBuffer
	kBuf            vulkanHostBuffer
	vBuf            vulkanHostBuffer
	outBuf          vulkanHostBuffer
	finalBuf        vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [18]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandKind     int
	commandTokens   int
	commandHeads    int
	commandHeadDim  int
	commandHidden   int
	mu              sync.Mutex
}

const (
	vulkanVisionAttentionLinuxCommandOnly = iota + 1
	vulkanVisionAttentionLinuxCommandOut
	vulkanVisionAttentionLinuxCommandRoPE
	vulkanVisionAttentionLinuxCommandRoPEOut
	vulkanVisionAttentionLinuxCommandQKVRoPEOut
)

func newVulkanVisionAttentionF32LinuxRunner() (*vulkanVisionAttentionF32LinuxRunner, error) {
	spv, err := vulkanVisionAttentionF32SPV()
	if err != nil {
		return nil, err
	}
	projSPV, err := vulkanVisionAttentionOutF32SPV()
	if err != nil {
		return nil, err
	}
	ropeSPV, err := vulkanVisionRoPEPairF32SPV()
	if err != nil {
		return nil, err
	}
	qkvSPV, err := vulkanVisionQKVF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-vision-attention-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanVisionAttentionF32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 18)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: uint32(len(poolSize)), PPoolSizes: poolSize}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 20}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var projShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(projSPV) * 4), PCode: projSPV}, nil, &projShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, projShader, nil)
	projPipelines := make([]vk.Pipeline, 1)
	projStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: projShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: projStage, Layout: r.pipelineLayout}}, nil, projPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines projection: %s", res)
	}
	r.projPipeline = projPipelines[0]
	var ropeShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(ropeSPV) * 4), PCode: ropeSPV}, nil, &ropeShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule vision rope: %s", res)
	}
	defer vk.DestroyShaderModule(device, ropeShader, nil)
	ropePipelines := make([]vk.Pipeline, 1)
	ropeStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: ropeShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: ropeStage, Layout: r.pipelineLayout}}, nil, ropePipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines vision rope: %s", res)
	}
	r.ropePipeline = ropePipelines[0]
	var qkvShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(qkvSPV) * 4), PCode: qkvSPV}, nil, &qkvShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule vision qkv: %s", res)
	}
	defer vk.DestroyShaderModule(device, qkvShader, nil)
	qkvPipelines := make([]vk.Pipeline, 1)
	qkvStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: qkvShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: qkvStage, Layout: r.pipelineLayout}}, nil, qkvPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines vision qkv: %s", res)
	}
	r.qkvPipeline = qkvPipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) run(out, q, k, v [][]float32, tokens, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedVisionAttentionDimsLinux(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrLinux(dims.bufLen, "Vulkan vision attention runner buffer")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.qBuf.writeRowsPrefix(r.device, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.kBuf.writeRowsPrefix(r.device, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vBuf.writeRowsPrefix(r.device, v, tokens, hidden); err != nil {
		return err
	}
	bufferInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionLinuxCommandOnly || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != hidden {
		if err := r.recordAttentionCommand(tokens, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32LinuxRunner) recordAttentionCommand(tokens, heads, headDim, hidden int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandKind = vulkanVisionAttentionLinuxCommandOnly
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = hidden
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) runOut(out, q, k, v [][]float32, w, bias []float32, tokens, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	dims, err := checkedVisionAttentionDimsLinux(tokens, heads, headDim, 0, 0, 0, "Vulkan vision attention out runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrLinux(dims.bufLen, "Vulkan vision attention out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(dims.wLen, "Vulkan vision attention out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrLinux(hidden, "Vulkan vision attention out runner bias")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeRowsPrefix(r.device, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.kBuf.writeRowsPrefix(r.device, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vBuf.writeRowsPrefix(r.device, v, tokens, hidden); err != nil {
		return err
	}
	bufferInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionLinuxCommandOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != hidden {
		if err := r.recordAttentionOutCommand(tokens, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readRowsPrefixInto(r.device, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32LinuxRunner) recordAttentionOutCommand(tokens, heads, headDim, hidden int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(hidden), uint32(tokens), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandKind = vulkanVisionAttentionLinuxCommandOut
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = hidden
	r.commandRecorded = true
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) runRoPEPair(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(q)
	dims, err := checkedVisionAttentionDimsLinux(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrLinux(dims.bufLen, "Vulkan vision rope runner buffer")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrLinux(dims.hTableLen, "Vulkan vision rope runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrLinux(dims.wTableLen, "Vulkan vision rope runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, hTableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, hTableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, wTableBytes); err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeRowsPrefix(r.device, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.kBuf.writeRowsPrefix(r.device, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vBuf.writeFloat32(r.device, cosH[:dims.hTableLen]); err != nil {
		return err
	}
	if err := r.outBuf.writeFloat32(r.device, sinH[:dims.hTableLen]); err != nil {
		return err
	}
	if err := r.finalBuf.writeFloat32(r.device, cosW[:dims.wTableLen]); err != nil {
		return err
	}
	bufferInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: sinWBuf.buffer, Offset: 0, Range: sinWBuf.size},
	}
	bindings := [6]uint32{0, 1, 7, 8, 9, 10}
	updateVulkanDescriptorBindingsLinux(r.device, r.descriptorSet, r.descriptorCache[:], bindings[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionLinuxCommandRoPE || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordRoPEPairCommand(tokens, gridH, gridW, heads, headDim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.qBuf.readRowsPrefixInto(r.device, q, tokens, hidden); err != nil {
		return err
	}
	return r.kBuf.readRowsPrefixInto(r.device, k, tokens, hidden)
}

func (r *vulkanVisionAttentionF32LinuxRunner) runRoPEOut(out, q, k, v [][]float32, w, bias, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(q)
	dims, err := checkedVisionAttentionDimsLinux(tokens, heads, headDim, 0, gridH, gridW, "Vulkan vision rope out runner")
	if err != nil {
		return err
	}
	hidden := dims.hidden
	bufBytes, err := checkedFloat32ByteLenErrLinux(dims.bufLen, "Vulkan vision rope out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(dims.wLen, "Vulkan vision rope out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrLinux(hidden, "Vulkan vision rope out runner bias")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrLinux(dims.hTableLen, "Vulkan vision rope out runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrLinux(dims.wTableLen, "Vulkan vision rope out runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	cosHBuf, err := r.cachedBuffer(cosH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinHBuf, err := r.cachedBuffer(sinH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	cosWBuf, err := r.cachedBuffer(cosW[:dims.wTableLen], wTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeRowsPrefix(r.device, q, tokens, hidden); err != nil {
		return err
	}
	if err := r.kBuf.writeRowsPrefix(r.device, k, tokens, hidden); err != nil {
		return err
	}
	if err := r.vBuf.writeRowsPrefix(r.device, v, tokens, hidden); err != nil {
		return err
	}
	bufferInfos := [11]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: cosHBuf.buffer, Offset: 0, Range: cosHBuf.size},
		{Buffer: sinHBuf.buffer, Offset: 0, Range: sinHBuf.size},
		{Buffer: cosWBuf.buffer, Offset: 0, Range: cosWBuf.size},
		{Buffer: sinWBuf.buffer, Offset: 0, Range: sinWBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionLinuxCommandRoPEOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readRowsPrefixInto(r.device, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32LinuxRunner) runQKVRoPEOut(out, x [][]float32, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim, hidden int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(x)
	dims, err := checkedVisionAttentionDimsLinux(tokens, heads, headDim, hidden, gridH, gridW, "Vulkan vision qkv rope out runner")
	if err != nil {
		return err
	}
	bufBytes, err := checkedFloat32ByteLenErrLinux(dims.bufLen, "Vulkan vision qkv rope out runner buffer")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(dims.wLen, "Vulkan vision qkv rope out runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrLinux(hidden, "Vulkan vision qkv rope out runner bias")
	if err != nil {
		return err
	}
	hTableBytes, err := checkedFloat32ByteLenErrLinux(dims.hTableLen, "Vulkan vision qkv rope out runner h table")
	if err != nil {
		return err
	}
	wTableBytes, err := checkedFloat32ByteLenErrLinux(dims.wTableLen, "Vulkan vision qkv rope out runner w table")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.kBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, bufBytes); err != nil {
		return err
	}
	qwBuf, err := r.cachedBuffer(qw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	qbBuf, err := r.cachedBuffer(qb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	kwBuf, err := r.cachedBuffer(kw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	kbBuf, err := r.cachedBuffer(kb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	vwBuf, err := r.cachedBuffer(vw[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	vbBuf, err := r.cachedBuffer(vb[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	owBuf, err := r.cachedBuffer(ow[:dims.wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	obBuf, err := r.cachedBuffer(ob[:hidden], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	cosHBuf, err := r.cachedBuffer(cosH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinHBuf, err := r.cachedBuffer(sinH[:dims.hTableLen], hTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	cosWBuf, err := r.cachedBuffer(cosW[:dims.wTableLen], wTableBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	sinWBuf, err := r.cachedBuffer(sinW[:dims.wTableLen], wTableBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, x, tokens, hidden); err != nil {
		return err
	}
	bufferInfos := [18]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: owBuf.buffer, Offset: 0, Range: owBuf.size},
		{Buffer: obBuf.buffer, Offset: 0, Range: obBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: cosHBuf.buffer, Offset: 0, Range: cosHBuf.size},
		{Buffer: sinHBuf.buffer, Offset: 0, Range: sinHBuf.size},
		{Buffer: cosWBuf.buffer, Offset: 0, Range: cosWBuf.size},
		{Buffer: sinWBuf.buffer, Offset: 0, Range: sinWBuf.size},
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: qwBuf.buffer, Offset: 0, Range: qwBuf.size},
		{Buffer: qbBuf.buffer, Offset: 0, Range: qbBuf.size},
		{Buffer: kwBuf.buffer, Offset: 0, Range: kwBuf.size},
		{Buffer: kbBuf.buffer, Offset: 0, Range: kbBuf.size},
		{Buffer: vwBuf.buffer, Offset: 0, Range: vwBuf.size},
		{Buffer: vbBuf.buffer, Offset: 0, Range: vbBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanVisionAttentionLinuxCommandQKVRoPEOut || r.commandTokens != tokens || r.commandHeads != heads || r.commandHeadDim != headDim || r.commandHidden != dims.gridLen {
		if err := r.recordQKVRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readRowsPrefixInto(r.device, out, tokens, hidden)
}

func (r *vulkanVisionAttentionF32LinuxRunner) recordRoPEPairCommand(tokens, gridH, gridW, heads, headDim int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.ropePipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision rope command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.rememberVisionCommand(vulkanVisionAttentionLinuxCommandRoPE, tokens, heads, headDim, gridLen)
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) recordRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	var pc [16]byte
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.ropePipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(hidden))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(hidden), uint32(tokens), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision rope out command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.rememberVisionCommand(vulkanVisionAttentionLinuxCommandRoPEOut, tokens, heads, headDim, gridLen)
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) recordQKVRoPEOutCommand(tokens, gridH, gridW, heads, headDim, hidden int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	var pc16 [16]byte
	var pc20 [20]byte
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.qkvPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc20[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc20[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[12:16], uint32(hidden))
	binary.LittleEndian.PutUint32(pc20[16:20], uint32(hidden))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc20)), unsafe.Pointer(&pc20[0]))
	vk.CmdDispatch(cmd, uint32(hidden*3), uint32(tokens), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.ropePipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(gridH))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(gridW))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(heads))
	binary.LittleEndian.PutUint32(pc16[12:16], uint32(headDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc16)), unsafe.Pointer(&pc16[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(heads))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(headDim))
	binary.LittleEndian.PutUint32(pc16[12:16], uint32(hidden))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc16)), unsafe.Pointer(&pc16[0]))
	vk.CmdDispatch(cmd, uint32(tokens), uint32(heads), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc16[0:4], uint32(tokens))
	binary.LittleEndian.PutUint32(pc16[4:8], uint32(hidden))
	binary.LittleEndian.PutUint32(pc16[8:12], uint32(hidden))
	binary.LittleEndian.PutUint32(pc16[12:16], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc16)), unsafe.Pointer(&pc16[0]))
	vk.CmdDispatch(cmd, uint32(hidden), uint32(tokens), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	gridLen, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return fmt.Errorf("Vulkan vision qkv rope out command grid length overflows: gridH=%d gridW=%d", gridH, gridW)
	}
	r.rememberVisionCommand(vulkanVisionAttentionLinuxCommandQKVRoPEOut, tokens, heads, headDim, gridLen)
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) rememberVisionCommand(kind, tokens, heads, headDim, hidden int) {
	r.commandKind = kind
	r.commandTokens = tokens
	r.commandHeads = heads
	r.commandHeadDim = headDim
	r.commandHidden = hidden
	r.commandRecorded = true
}

func (r *vulkanVisionAttentionF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanVisionAttentionF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanVisionAttentionF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.projPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.projPipeline, nil)
		}
		if r.ropePipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.ropePipeline, nil)
		}
		if r.qkvPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.qkvPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.qBuf.destroy(r.device)
		r.kBuf.destroy(r.device)
		r.vBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.finalBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanTextAttentionF32LinuxRunner() (*vulkanTextAttentionF32LinuxRunner, error) {
	vulkanTextAttentionF32LinuxRunnerCache.once.Do(func() {
		vulkanTextAttentionF32LinuxRunnerCache.runner, vulkanTextAttentionF32LinuxRunnerCache.err = newVulkanTextAttentionF32LinuxRunner()
	})
	return vulkanTextAttentionF32LinuxRunnerCache.runner, vulkanTextAttentionF32LinuxRunnerCache.err
}

type vulkanTextAttentionF32LinuxRunner struct {
	instance             vk.Instance
	device               vk.Device
	queue                vk.Queue
	queueFamily          uint32
	memProps             vk.PhysicalDeviceMemoryProperties
	setLayout            vk.DescriptorSetLayout
	descriptorPool       vk.DescriptorPool
	descriptorSet        vk.DescriptorSet
	pipelineLayout       vk.PipelineLayout
	pipeline             vk.Pipeline
	projPipeline         vk.Pipeline
	normPipeline         vk.Pipeline
	firstTokenPipeline   vk.Pipeline
	q8ProjPipeline       vk.Pipeline
	q6ProjPipeline       vk.Pipeline
	q4ProjPipeline       vk.Pipeline
	firstTokenQ8Pipeline vk.Pipeline
	firstTokenQ6Pipeline vk.Pipeline
	firstTokenQ4Pipeline vk.Pipeline
	commandPool          vk.CommandPool
	commandBuffer        vk.CommandBuffer
	fence                vk.Fence
	qBuf                 vulkanHostBuffer
	kBuf                 vulkanHostBuffer
	vBuf                 vulkanHostBuffer
	outBuf               vulkanHostBuffer
	finalBuf             vulkanHostBuffer
	residualBuf          vulkanHostBuffer
	normBuf              vulkanHostBuffer
	weightBuffers        map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers          map[uintptr]vulkanCachedFloat32BufferLinux
	q8DataBuffers        map[uintptr]vulkanCachedInt8BufferLinux
	byteDataBuffers      map[uintptr]vulkanCachedByteBufferLinux
	q8ScaleBuffers       map[uintptr]vulkanCachedFloat32BufferLinux
	kCacheKey            uintptr
	vCacheKey            uintptr
	cacheEpoch           uint64
	cacheUploaded        int
	cacheKVDim           int
	descriptorCache      [10]vulkanDescriptorBindingLinux
	commandRecorded      bool
	commandKind          int
	commandCacheLen      int
	commandNumHeads      int
	commandKVHeads       int
	commandHeadDim       int
	commandQRows         int
	commandKVDim         int
	commandPipeline      vk.Pipeline
	mu                   sync.Mutex
}

const (
	vulkanTextAttentionLinuxCommandOnly = iota + 1
	vulkanTextAttentionLinuxCommandOut
	vulkanTextAttentionLinuxCommandOutNorm
	vulkanTextAttentionLinuxCommandFirstTokenOutNorm
	vulkanTextAttentionLinuxCommandFirstTokenOutNormQ8
	vulkanTextAttentionLinuxCommandFirstTokenOutNormQ6
	vulkanTextAttentionLinuxCommandFirstTokenOutNormQ4
)

func newVulkanTextAttentionF32LinuxRunner() (*vulkanTextAttentionF32LinuxRunner, error) {
	spv, err := vulkanTextAttentionF32SPV()
	if err != nil {
		return nil, err
	}
	projSPV, err := vulkanTextAttentionOutF32SPV()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanTextAttentionOutAddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	firstTokenSPV, err := vulkanTextFirstTokenValueOutF32SPV()
	if err != nil {
		return nil, err
	}
	firstTokenQ8SPV, err := vulkanTextFirstTokenValueOutQ8SPV()
	if err != nil {
		return nil, err
	}
	firstTokenQ6SPV, err := vulkanTextFirstTokenValueOutQ6SPV()
	if err != nil {
		return nil, err
	}
	firstTokenQ4SPV, err := vulkanTextFirstTokenValueOutQ4SPV()
	if err != nil {
		return nil, err
	}
	q8ProjSPV, err := vulkanTextAttentionOutQ8SPV()
	if err != nil {
		return nil, err
	}
	q6ProjSPV, err := vulkanTextAttentionOutQ6SPV()
	if err != nil {
		return nil, err
	}
	q4ProjSPV, err := vulkanTextAttentionOutQ4SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-text-attention-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanTextAttentionF32LinuxRunner{
		instance:        instance,
		weightBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:     make(map[uintptr]vulkanCachedFloat32BufferLinux),
		q8DataBuffers:   make(map[uintptr]vulkanCachedInt8BufferLinux),
		byteDataBuffers: make(map[uintptr]vulkanCachedByteBufferLinux),
		q8ScaleBuffers:  make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 10)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: uint32(len(poolSize)), PPoolSizes: poolSize}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 20}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var projShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(projSPV) * 4), PCode: projSPV}, nil, &projShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, projShader, nil)
	projPipelines := make([]vk.Pipeline, 1)
	projStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: projShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: projStage, Layout: r.pipelineLayout}}, nil, projPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines projection: %s", res)
	}
	r.projPipeline = projPipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(normSPV) * 4), PCode: normSPV}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule attention norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: normShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.pipelineLayout}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines attention norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	var firstTokenShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(firstTokenSPV) * 4), PCode: firstTokenSPV}, nil, &firstTokenShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule first-token projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, firstTokenShader, nil)
	firstTokenPipelines := make([]vk.Pipeline, 1)
	firstTokenStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: firstTokenShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: firstTokenStage, Layout: r.pipelineLayout}}, nil, firstTokenPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines first-token projection: %s", res)
	}
	r.firstTokenPipeline = firstTokenPipelines[0]
	var q8ProjShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(q8ProjSPV) * 4), PCode: q8ProjSPV}, nil, &q8ProjShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q8 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, q8ProjShader, nil)
	q8ProjPipelines := make([]vk.Pipeline, 1)
	q8ProjStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: q8ProjShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: q8ProjStage, Layout: r.pipelineLayout}}, nil, q8ProjPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q8 projection: %s", res)
	}
	r.q8ProjPipeline = q8ProjPipelines[0]
	var q6ProjShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(q6ProjSPV) * 4), PCode: q6ProjSPV}, nil, &q6ProjShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q6 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, q6ProjShader, nil)
	q6ProjPipelines := make([]vk.Pipeline, 1)
	q6ProjStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: q6ProjShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: q6ProjStage, Layout: r.pipelineLayout}}, nil, q6ProjPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q6 projection: %s", res)
	}
	r.q6ProjPipeline = q6ProjPipelines[0]
	var q4ProjShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(q4ProjSPV) * 4), PCode: q4ProjSPV}, nil, &q4ProjShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q4 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, q4ProjShader, nil)
	q4ProjPipelines := make([]vk.Pipeline, 1)
	q4ProjStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: q4ProjShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: q4ProjStage, Layout: r.pipelineLayout}}, nil, q4ProjPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q4 projection: %s", res)
	}
	r.q4ProjPipeline = q4ProjPipelines[0]
	var firstTokenQ8Shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(firstTokenQ8SPV) * 4), PCode: firstTokenQ8SPV}, nil, &firstTokenQ8Shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule first-token q8 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, firstTokenQ8Shader, nil)
	firstTokenQ8Pipelines := make([]vk.Pipeline, 1)
	firstTokenQ8Stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: firstTokenQ8Shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: firstTokenQ8Stage, Layout: r.pipelineLayout}}, nil, firstTokenQ8Pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines first-token q8 projection: %s", res)
	}
	r.firstTokenQ8Pipeline = firstTokenQ8Pipelines[0]
	var firstTokenQ6Shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(firstTokenQ6SPV) * 4), PCode: firstTokenQ6SPV}, nil, &firstTokenQ6Shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule first-token q6 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, firstTokenQ6Shader, nil)
	firstTokenQ6Pipelines := make([]vk.Pipeline, 1)
	firstTokenQ6Stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: firstTokenQ6Shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: firstTokenQ6Stage, Layout: r.pipelineLayout}}, nil, firstTokenQ6Pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines first-token q6 projection: %s", res)
	}
	r.firstTokenQ6Pipeline = firstTokenQ6Pipelines[0]
	var firstTokenQ4Shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(firstTokenQ4SPV) * 4), PCode: firstTokenQ4SPV}, nil, &firstTokenQ4Shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule first-token q4 projection: %s", res)
	}
	defer vk.DestroyShaderModule(device, firstTokenQ4Shader, nil)
	firstTokenQ4Pipelines := make([]vk.Pipeline, 1)
	firstTokenQ4Stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: firstTokenQ4Shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: firstTokenQ4Stage, Layout: r.pipelineLayout}}, nil, firstTokenQ4Pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines first-token q4 projection: %s", res)
	}
	r.firstTokenQ4Pipeline = firstTokenQ4Pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanTextAttentionF32LinuxRunner) uploadTextCacheLocked(kCache, vCache []float32, cacheEpoch uint64, cacheLen, kvDim int) error {
	cacheElems, err := checkedTextAttentionLinuxCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	kBufferBytes, ok := textAttentionCacheBufferBytesLinux(kCache, cacheElems)
	if !ok {
		return fmt.Errorf("text attention key cache byte size overflows: elems=%d len=%d", cacheElems, len(kCache))
	}
	vBufferBytes, ok := textAttentionCacheBufferBytesLinux(vCache, cacheElems)
	if !ok {
		return fmt.Errorf("text attention value cache byte size overflows: elems=%d len=%d", cacheElems, len(vCache))
	}
	kKey := float32SliceKeyLinux(kCache)
	vKey := float32SliceKeyLinux(vCache)
	fullCacheWrite := kKey == 0 || vKey == 0 ||
		kKey != r.kCacheKey || vKey != r.vCacheKey ||
		r.cacheEpoch != cacheEpoch || r.cacheKVDim != kvDim || cacheLen < r.cacheUploaded ||
		r.kBuf.buffer == vk.NullBuffer || r.vBuf.buffer == vk.NullBuffer ||
		r.kBuf.size < kBufferBytes || r.vBuf.size < vBufferBytes
	if err := r.ensureHostBuffer(&r.kBuf, kBufferBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, vBufferBytes); err != nil {
		return err
	}
	if fullCacheWrite {
		if err := r.kBuf.writeFloat32(r.device, kCache[:cacheElems]); err != nil {
			return err
		}
		if err := r.vBuf.writeFloat32(r.device, vCache[:cacheElems]); err != nil {
			return err
		}
	} else if cacheLen > r.cacheUploaded {
		start := r.cacheUploaded * kvDim
		end := cacheElems
		if err := r.kBuf.writeFloat32At(r.device, start, kCache[start:end]); err != nil {
			return err
		}
		if err := r.vBuf.writeFloat32At(r.device, start, vCache[start:end]); err != nil {
			return err
		}
	}
	r.kCacheKey = kKey
	r.vCacheKey = vKey
	r.cacheEpoch = cacheEpoch
	r.cacheUploaded = cacheLen
	r.cacheKVDim = kvDim
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) run(out, q, kCache, vCache []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	bufferInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOnly || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordAttentionCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(r.device, out[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) recordAttentionCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(numHeads), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.rememberAttentionCommand(vulkanTextAttentionLinuxCommandOnly, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, vk.NullPipeline)
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) runOut(out, q, kCache, vCache, w, bias []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	wBytes, ok := checkedFloat32ByteLenLinuxSquare(qRows)
	if !ok {
		return fmt.Errorf("text attention out weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes, ok := checkedFloat32ByteLenLinux(qRows)
	if !ok {
		return fmt.Errorf("text attention out bias byte size overflows: q_rows=%d", qRows)
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("text attention out weight length overflows: q_rows=%d", qRows)
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:weightElems], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:qRows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	bufferInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOut || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != r.projPipeline {
		if err := r.recordAttentionOutCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, r.projPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readFloat32Into(r.device, out[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ8(out, q, kCache, vCache []float32, w *tensor.Q8Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(w.Rows)
	if !ok {
		return fmt.Errorf("text attention q8 scale byte size overflows: rows=%d", w.Rows)
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.q8DataBuffer(w.Data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(w.Scale[:w.Rows], scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	bufferInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOut || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != r.q8ProjPipeline {
		if err := r.recordAttentionOutCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, r.q8ProjPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readFloat32Into(r.device, out[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ6(out, q, kCache, vCache []float32, w *tensor.Q6Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytes(out, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], dataBytes, r.q6ProjPipeline, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ4(out, q, kCache, vCache []float32, w *tensor.Q4Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytes(out, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], dataBytes, r.q4ProjPipeline, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutBytes(out, q, kCache, vCache []float32, data []byte, scale []float32, dataBytes vk.DeviceSize, projPipeline vk.Pipeline, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(len(scale))
	if !ok {
		return fmt.Errorf("text attention quantized scale byte size overflows: len=%d", len(scale))
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.byteDataBuffer(data, dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(scale, scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	bufferInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOut || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != projPipeline {
		if err := r.recordAttentionOutCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.finalBuf.readFloat32Into(r.device, out[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ8AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutQ8BytesAddRMSNorm(normOut, residual, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.q8ProjPipeline, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ6AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutQuantizedAddRMSNorm(normOut, residual, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.q6ProjPipeline, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ4AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutQuantizedAddRMSNorm(normOut, residual, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.q4ProjPipeline, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQ8BytesAddRMSNorm(normOut, residual, q, kCache, vCache []float32, data []int8, scale, normWeight []float32, dataBytes vk.DeviceSize, projPipeline vk.Pipeline, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(len(scale))
	if !ok {
		return fmt.Errorf("text attention q8 norm scale byte size overflows: len=%d", len(scale))
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.q8DataBuffer(data, dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(scale, scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOutNorm || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != projPipeline {
		if err := r.recordAttentionOutAddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutQuantizedAddRMSNorm(normOut, residual, q, kCache, vCache []float32, data []byte, scale, normWeight []float32, dataBytes vk.DeviceSize, projPipeline vk.Pipeline, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(len(scale))
	if !ok {
		return fmt.Errorf("text attention quantized norm scale byte size overflows: len=%d", len(scale))
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.byteDataBuffer(data, dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(scale, scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOutNorm || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != projPipeline {
		if err := r.recordAttentionOutAddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) recordAttentionOutCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int, projPipeline vk.Pipeline) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(numHeads), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(qRows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.rememberAttentionCommand(vulkanTextAttentionLinuxCommandOut, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline)
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) runOutAddRMSNorm(normOut, residual, q, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	wBytes, ok := checkedFloat32ByteLenLinuxSquare(qRows)
	if !ok {
		return fmt.Errorf("text attention norm weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes, ok := checkedFloat32ByteLenLinux(qRows)
	if !ok {
		return fmt.Errorf("text attention norm bias byte size overflows: q_rows=%d", qRows)
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("text attention norm weight length overflows: q_rows=%d", qRows)
	}
	if err := r.ensureHostBuffer(&r.qBuf, qBytes); err != nil {
		return err
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, cacheLen, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:weightElems], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:qRows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.qBuf.writeFloat32(r.device, q[:qRows]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Offset: 0, Range: r.qBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandOutNorm || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != r.projPipeline {
		if err := r.recordAttentionOutAddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, r.projPipeline); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) recordAttentionOutAddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int, projPipeline vk.Pipeline) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(numHeads), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(qRows), 1, 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	binary.LittleEndian.PutUint32(pc[8:12], 0)
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.rememberAttentionCommand(vulkanTextAttentionLinuxCommandOutNorm, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline)
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutAddRMSNorm(normOut, residual, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	wBytes, ok := checkedFloat32ByteLenLinuxSquare(qRows)
	if !ok {
		return fmt.Errorf("first-token text attention weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes, ok := checkedFloat32ByteLenLinux(qRows)
	if !ok {
		return fmt.Errorf("first-token text attention bias byte size overflows: q_rows=%d", qRows)
	}
	weightElems, ok := checkedMulInt(qRows, qRows)
	if !ok {
		return fmt.Errorf("first-token text attention weight length overflows: q_rows=%d", qRows)
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, 1, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:weightElems], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:qRows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionLinuxCommandFirstTokenOutNorm || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordFirstTokenValueOutAddRMSNormCommand(numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutQ8AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutQ8BytesAddRMSNorm(normOut, residual, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.firstTokenQ8Pipeline, vulkanTextAttentionLinuxCommandFirstTokenOutNormQ8, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutQ6AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.firstTokenQ6Pipeline, vulkanTextAttentionLinuxCommandFirstTokenOutNormQ6, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutQ4AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesLinux(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.firstTokenQ4Pipeline, vulkanTextAttentionLinuxCommandFirstTokenOutNormQ4, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutQ8BytesAddRMSNorm(normOut, residual, kCache, vCache []float32, data []int8, scale, normWeight []float32, dataBytes vk.DeviceSize, projPipeline vk.Pipeline, commandKind int, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(len(scale))
	if !ok {
		return fmt.Errorf("first-token q8 text attention scale byte size overflows: len=%d", len(scale))
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, 1, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.q8DataBuffer(data, dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(scale, scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != commandKind || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != projPipeline {
		if err := r.recordFirstTokenValueOutQuantizedAddRMSNormCommand(projPipeline, commandKind, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache []float32, data []byte, scale, normWeight []float32, dataBytes vk.DeviceSize, projPipeline vk.Pipeline, commandKind int, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenLinux(len(scale))
	if !ok {
		return fmt.Errorf("first-token quantized text attention scale byte size overflows: len=%d", len(scale))
	}
	if err := r.uploadTextCacheLocked(kCache, vCache, cacheEpoch, 1, kvDim); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.finalBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, qBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, qBytes); err != nil {
		return err
	}
	dataBuf, err := r.byteDataBuffer(data, dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(scale, scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:qRows]); err != nil {
		return err
	}
	bufferInfos := [10]vk.DescriptorBufferInfo{
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.kBuf.buffer, Offset: 0, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Offset: 0, Range: r.vBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.finalBuf.buffer, Offset: 0, Range: r.finalBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != commandKind || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim || r.commandPipeline != projPipeline {
		if err := r.recordFirstTokenValueOutQuantizedAddRMSNormCommand(projPipeline, commandKind, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(r.device, residual[:qRows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:qRows])
}

func (r *vulkanTextAttentionF32LinuxRunner) recordFirstTokenValueOutAddRMSNormCommand(numHeads, kvHeads, headDim, qRows, kvDim int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [20]byte
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.firstTokenPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(qRows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	binary.LittleEndian.PutUint32(pc[8:12], 0)
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.rememberAttentionCommand(vulkanTextAttentionLinuxCommandFirstTokenOutNorm, 1, numHeads, kvHeads, headDim, qRows, kvDim, r.firstTokenPipeline)
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) recordFirstTokenValueOutQuantizedAddRMSNormCommand(projPipeline vk.Pipeline, commandKind, numHeads, kvHeads, headDim, qRows, kvDim int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [20]byte
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, projPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(qRows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	binary.LittleEndian.PutUint32(pc[8:12], 0)
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.rememberAttentionCommand(commandKind, 1, numHeads, kvHeads, headDim, qRows, kvDim, projPipeline)
	return nil
}

func (r *vulkanTextAttentionF32LinuxRunner) rememberAttentionCommand(kind, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int, projPipeline vk.Pipeline) {
	r.commandKind = kind
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandPipeline = projPipeline
	r.commandRecorded = true
}

func textAttentionCacheBufferBytesLinux(cache []float32, minElems int) (vk.DeviceSize, bool) {
	if len(cache) > minElems {
		return checkedFloat32ByteLenLinux(len(cache))
	}
	return checkedFloat32ByteLenLinux(minElems)
}

func checkedTextAttentionLinuxCacheElems(cacheLen, kvDim int) (int, error) {
	cacheElems, ok := checkedMulInt(cacheLen, kvDim)
	if !ok {
		return 0, fmt.Errorf("text attention cache elements overflow: cacheLen=%d kvDim=%d", cacheLen, kvDim)
	}
	return cacheElems, nil
}

func checkedTextAttentionLinuxDims(numHeads, kvHeads, headDim int) (qRows, kvDim int, qBytes vk.DeviceSize, err error) {
	qRows, ok := checkedMulInt(numHeads, headDim)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention q rows overflow: num_heads=%d head_dim=%d", numHeads, headDim)
	}
	kvDim, ok = checkedMulInt(kvHeads, headDim)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention kv dim overflow: kv_heads=%d head_dim=%d", kvHeads, headDim)
	}
	qBytes, ok = checkedFloat32ByteLenLinux(qRows)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention q byte size overflows: q_rows=%d", qRows)
	}
	return qRows, kvDim, qBytes, nil
}

func checkedFloat32ByteLenLinux(n int) (vk.DeviceSize, bool) {
	if n < 0 || n > maxInt()/4 {
		return 0, false
	}
	return vk.DeviceSize(n * 4), true
}

func checkedFloat32ByteLenLinuxSquare(n int) (vk.DeviceSize, bool) {
	elements, ok := checkedMulInt(n, n)
	if !ok {
		return 0, false
	}
	return checkedFloat32ByteLenLinux(elements)
}

func checkedTextAttentionQ8DataBytesLinux(rows, cols int) (int, vk.DeviceSize, error) {
	dataLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q8 data length overflows: rows=%d cols=%d", rows, cols)
	}
	dataBytes, ok := checkedAlignedByteLenLinux(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q8 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedTextAttentionQ6DataBytesLinux(rows, cols int) (int, vk.DeviceSize, error) {
	packedCols, ok := checkedPackedQ6ColsLinux(cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 packed cols overflow: cols=%d", cols)
	}
	dataLen, ok := checkedMulInt(rows, packedCols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 data length overflows: rows=%d packed_cols=%d", rows, packedCols)
	}
	dataBytes, ok := checkedAlignedByteLenLinux(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedTextAttentionQ4DataBytesLinux(rows, cols int) (int, vk.DeviceSize, error) {
	packedCols, ok := checkedPackedQ4ColsLinux(cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 packed cols overflow: cols=%d", cols)
	}
	dataLen, ok := checkedMulInt(rows, packedCols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 data length overflows: rows=%d packed_cols=%d", rows, packedCols)
	}
	dataBytes, ok := checkedAlignedByteLenLinux(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedPackedQ6ColsLinux(cols int) (int, bool) {
	if cols <= 0 || cols > (maxInt()-7)/6 {
		return 0, false
	}
	return tensor.PackedQ6Cols(cols), true
}

func checkedPackedQ4ColsLinux(cols int) (int, bool) {
	if cols <= 0 || cols == maxInt() {
		return 0, false
	}
	return (cols + 1) / 2, true
}

func checkedPackedQ6DataLenLinux(rows, cols int, label string) (int, error) {
	packedCols, ok := checkedPackedQ6ColsLinux(cols)
	if !ok {
		return 0, fmt.Errorf("%s q6 packed cols overflow: cols=%d", label, cols)
	}
	return checkedPackedRowsLinux(rows, packedCols, label)
}

func checkedPackedQ4DataLenLinux(rows, cols int, label string) (int, error) {
	packedCols, ok := checkedPackedQ4ColsLinux(cols)
	if !ok {
		return 0, fmt.Errorf("%s q4 packed cols overflow: cols=%d", label, cols)
	}
	return checkedPackedRowsLinux(rows, packedCols, label)
}

func checkedPackedRowsLinux(rows, packedCols int, label string) (int, error) {
	dataLen, ok := checkedMulInt(rows, packedCols)
	if !ok {
		return 0, fmt.Errorf("%s data length overflows: rows=%d packed_cols=%d", label, rows, packedCols)
	}
	return dataLen, nil
}

type vulkanMatRowsGELU2DimsLinux struct {
	xLen      int
	hiddenLen int
	outLen    int
	w1Len     int
	w2Len     int
}

func checkedMatRowsGELU2DimsLinux(batches, hiddenRows, cols, outRows int, label string) (vulkanMatRowsGELU2DimsLinux, error) {
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return vulkanMatRowsGELU2DimsLinux{}, fmt.Errorf("%s x length overflows: batches=%d cols=%d", label, batches, cols)
	}
	hiddenLen, ok := checkedMulInt(batches, hiddenRows)
	if !ok {
		return vulkanMatRowsGELU2DimsLinux{}, fmt.Errorf("%s hidden length overflows: batches=%d hiddenRows=%d", label, batches, hiddenRows)
	}
	outLen, ok := checkedMulInt(batches, outRows)
	if !ok {
		return vulkanMatRowsGELU2DimsLinux{}, fmt.Errorf("%s output length overflows: batches=%d outRows=%d", label, batches, outRows)
	}
	w1Len, ok := checkedMulInt(hiddenRows, cols)
	if !ok {
		return vulkanMatRowsGELU2DimsLinux{}, fmt.Errorf("%s w1 length overflows: hiddenRows=%d cols=%d", label, hiddenRows, cols)
	}
	w2Len, ok := checkedMulInt(outRows, hiddenRows)
	if !ok {
		return vulkanMatRowsGELU2DimsLinux{}, fmt.Errorf("%s w2 length overflows: outRows=%d hiddenRows=%d", label, outRows, hiddenRows)
	}
	return vulkanMatRowsGELU2DimsLinux{xLen: xLen, hiddenLen: hiddenLen, outLen: outLen, w1Len: w1Len, w2Len: w2Len}, nil
}

func checkedMatVecF32WeightLenLinux(rows, cols int, label string) (int, error) {
	wLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return 0, fmt.Errorf("%s weight length overflows: rows=%d cols=%d", label, rows, cols)
	}
	return wLen, nil
}

func checkedMatVecTopKCandidateFloatsLinux(rows int, label string) (int, error) {
	if rows <= 0 || rows > maxInt()-255 {
		return 0, fmt.Errorf("%s top-k block count overflows: rows=%d", label, rows)
	}
	blocks := (rows + 255) / 256
	candidates, ok := checkedMulInt(blocks, vulkanMatVecTopKMaxK)
	if !ok {
		return 0, fmt.Errorf("%s top-k candidate count overflows: blocks=%d maxK=%d", label, blocks, vulkanMatVecTopKMaxK)
	}
	candidateFloats, ok := checkedMulInt(candidates, 2)
	if !ok {
		return 0, fmt.Errorf("%s top-k candidate float count overflows: candidates=%d", label, candidates)
	}
	return candidateFloats, nil
}

type vulkanSwiGLUDimsLinux struct {
	gateLen int
	downLen int
}

func checkedSwiGLUDimsLinux(rows, cols, outRows int, label string) (vulkanSwiGLUDimsLinux, error) {
	gateLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return vulkanSwiGLUDimsLinux{}, fmt.Errorf("%s gate/up length overflows: rows=%d cols=%d", label, rows, cols)
	}
	downLen := 0
	if outRows > 0 {
		var ok bool
		downLen, ok = checkedMulInt(outRows, rows)
		if !ok {
			return vulkanSwiGLUDimsLinux{}, fmt.Errorf("%s down length overflows: outRows=%d rows=%d", label, outRows, rows)
		}
	}
	return vulkanSwiGLUDimsLinux{gateLen: gateLen, downLen: downLen}, nil
}

type vulkanProjectImageDimsLinux struct {
	tokens  int
	batches int
	cols    int
	w1Len   int
	w2Len   int
}

func checkedProjectImageDimsLinux(gridT, gridH, gridW, visionDim, hiddenRows, outRows int, label string) (vulkanProjectImageDimsLinux, error) {
	gridHW, ok := checkedMulInt(gridH, gridW)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s token count overflows: gridH=%d gridW=%d", label, gridH, gridW)
	}
	tokens, ok := checkedMulInt(gridT, gridHW)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s token count overflows: gridT=%d gridH=%d gridW=%d", label, gridT, gridH, gridW)
	}
	blocksHW, ok := checkedMulInt(gridH/2, gridW/2)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s batch count overflows: gridH=%d gridW=%d", label, gridH, gridW)
	}
	batches, ok := checkedMulInt(gridT, blocksHW)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s batch count overflows: gridT=%d gridH=%d gridW=%d", label, gridT, gridH, gridW)
	}
	cols, ok := checkedMulInt(visionDim, 4)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s merged cols overflow: visionDim=%d", label, visionDim)
	}
	w1Len, ok := checkedMulInt(hiddenRows, cols)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s w1 length overflows: hiddenRows=%d cols=%d", label, hiddenRows, cols)
	}
	w2Len, ok := checkedMulInt(outRows, hiddenRows)
	if !ok {
		return vulkanProjectImageDimsLinux{}, fmt.Errorf("%s w2 length overflows: outRows=%d hiddenRows=%d", label, outRows, hiddenRows)
	}
	return vulkanProjectImageDimsLinux{tokens: tokens, batches: batches, cols: cols, w1Len: w1Len, w2Len: w2Len}, nil
}

type vulkanVisionAttentionDimsLinux struct {
	hidden    int
	bufLen    int
	wLen      int
	hTableLen int
	wTableLen int
	gridLen   int
}

func checkedVisionAttentionDimsLinux(tokens, heads, headDim, hidden, gridH, gridW int, label string) (vulkanVisionAttentionDimsLinux, error) {
	computedHidden, ok := checkedMulInt(heads, headDim)
	if !ok {
		return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s hidden length overflows: heads=%d headDim=%d", label, heads, headDim)
	}
	if hidden == 0 {
		hidden = computedHidden
	} else if hidden != computedHidden {
		return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s hidden mismatch: hidden=%d heads=%d headDim=%d", label, hidden, heads, headDim)
	}
	bufLen, ok := checkedMulInt(tokens, hidden)
	if !ok {
		return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s token buffer length overflows: tokens=%d hidden=%d", label, tokens, hidden)
	}
	wLen, ok := checkedMulInt(hidden, hidden)
	if !ok {
		return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s projection weight length overflows: hidden=%d", label, hidden)
	}
	dims := vulkanVisionAttentionDimsLinux{hidden: hidden, bufLen: bufLen, wLen: wLen}
	if gridH > 0 || gridW > 0 {
		gridLen, ok := checkedMulInt(gridH, gridW)
		if !ok {
			return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s grid length overflows: gridH=%d gridW=%d", label, gridH, gridW)
		}
		quarter := headDim / 4
		hTableLen, ok := checkedMulInt(gridH, quarter)
		if !ok {
			return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s h rope table length overflows: gridH=%d quarter=%d", label, gridH, quarter)
		}
		wTableLen, ok := checkedMulInt(gridW, quarter)
		if !ok {
			return vulkanVisionAttentionDimsLinux{}, fmt.Errorf("%s w rope table length overflows: gridW=%d quarter=%d", label, gridW, quarter)
		}
		dims.hTableLen = hTableLen
		dims.wTableLen = wTableLen
		dims.gridLen = gridLen
	}
	return dims, nil
}

func checkedAlignedByteLenLinux(n, alignment int) (vk.DeviceSize, bool) {
	if n < 0 || alignment <= 0 || n > maxInt()-(alignment-1) {
		return 0, false
	}
	return vk.DeviceSize(alignUpInt(n, alignment)), true
}

func checkedFloat32ByteLenErrLinux(n int, label string) (vk.DeviceSize, error) {
	bytes, ok := checkedFloat32ByteLenLinux(n)
	if !ok {
		return 0, fmt.Errorf("%s byte size overflows: elements=%d", label, n)
	}
	return bytes, nil
}

func checkedAlignedByteLenErrLinux(n, alignment int, label string) (vk.DeviceSize, error) {
	bytes, ok := checkedAlignedByteLenLinux(n, alignment)
	if !ok {
		return 0, fmt.Errorf("%s aligned byte size overflows: bytes=%d alignment=%d", label, n, alignment)
	}
	return bytes, nil
}

func (r *vulkanTextAttentionF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

type vulkanCachedFloat32BufferLinux struct {
	buffer      vulkanHostBuffer
	length      int
	fingerprint uint64
}

type vulkanCachedInt8BufferLinux struct {
	buffer      vulkanHostBuffer
	length      int
	fingerprint uint64
}

type vulkanCachedByteBufferLinux struct {
	buffer      vulkanHostBuffer
	length      int
	fingerprint uint64
}

func cachedFloat32BufferLinux(device vk.Device, memProps vk.PhysicalDeviceMemoryProperties, data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := cache[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			cache[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(device)
		delete(cache, key)
	}
	buf, err := newVulkanHostBuffer(device, memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(device, data); err != nil {
		buf.destroy(device)
		return vulkanHostBuffer{}, err
	}
	cache[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func cachedInt8BufferLinux(device vk.Device, memProps vk.PhysicalDeviceMemoryProperties, data []int8, size vk.DeviceSize, cache map[uintptr]vulkanCachedInt8BufferLinux) (vulkanHostBuffer, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintInt8ForVulkanCache(data)
	if cached, ok := cache[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeInt8(device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			cache[key] = vulkanCachedInt8BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(device)
		delete(cache, key)
	}
	buf, err := newVulkanHostBuffer(device, memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeInt8(device, data); err != nil {
		buf.destroy(device)
		return vulkanHostBuffer{}, err
	}
	cache[key] = vulkanCachedInt8BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func cachedByteBufferLinux(device vk.Device, memProps vk.PhysicalDeviceMemoryProperties, data []byte, size vk.DeviceSize, cache map[uintptr]vulkanCachedByteBufferLinux) (vulkanHostBuffer, error) {
	key := uintptr(unsafe.Pointer(&data[0]))
	fingerprint := fingerprintBytesForVulkanCache(data)
	if cached, ok := cache[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeBytes(device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			cache[key] = vulkanCachedByteBufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(device)
		delete(cache, key)
	}
	buf, err := newVulkanHostBuffer(device, memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeBytes(device, data); err != nil {
		buf.destroy(device)
		return vulkanHostBuffer{}, err
	}
	cache[key] = vulkanCachedByteBufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanTextAttentionF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanTextAttentionF32LinuxRunner) q8DataBuffer(data []int8, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedInt8BufferLinux(r.device, r.memProps, data, size, r.q8DataBuffers)
}

func (r *vulkanTextAttentionF32LinuxRunner) byteDataBuffer(data []byte, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedByteBufferLinux(r.device, r.memProps, data, size, r.byteDataBuffers)
}

func (r *vulkanTextAttentionF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.projPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.projPipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.firstTokenPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.firstTokenPipeline, nil)
		}
		if r.firstTokenQ8Pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.firstTokenQ8Pipeline, nil)
		}
		if r.firstTokenQ6Pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.firstTokenQ6Pipeline, nil)
		}
		if r.firstTokenQ4Pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.firstTokenQ4Pipeline, nil)
		}
		if r.q8ProjPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.q8ProjPipeline, nil)
		}
		if r.q6ProjPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.q6ProjPipeline, nil)
		}
		if r.q4ProjPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.q4ProjPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.qBuf.destroy(r.device)
		r.kBuf.destroy(r.device)
		r.vBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.finalBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.q8DataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.byteDataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.q8ScaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatVecQ8LinuxRunner() (*vulkanMatVecQ8LinuxRunner, error) {
	vulkanMatVecQ8LinuxRunnerCache.once.Do(func() {
		vulkanMatVecQ8LinuxRunnerCache.runner, vulkanMatVecQ8LinuxRunnerCache.err = newVulkanMatVecQ8LinuxRunner()
	})
	return vulkanMatVecQ8LinuxRunnerCache.runner, vulkanMatVecQ8LinuxRunnerCache.err
}

type vulkanMatVecQ8LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	normPipeline    vk.Pipeline
	argmaxPipeline  vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outBuf          vulkanHostBuffer
	argmaxBuf       vulkanHostBuffer
	residualBuf     vulkanHostBuffer
	normBuf         vulkanHostBuffer
	dataBuffers     map[uintptr]vulkanCachedInt8BufferLinux
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [7]vulkanDescriptorBindingLinux
	commandKind     int
	commandRecorded bool
	commandRows     int
	commandCols     int
	mu              sync.Mutex
}

const (
	vulkanMatVecQ8LinuxCommandDefault = iota + 1
	vulkanMatVecQ8LinuxCommandAddRMSNorm
	vulkanMatVecQ8LinuxCommandArgmax
)

func newVulkanMatVecQ8LinuxRunner() (*vulkanMatVecQ8LinuxRunner, error) {
	spv, err := vulkanMatVecQ8SPV()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanMatVecQ8AddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	argmaxSPV, err := vulkanArgmaxQuantizedF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-q8-matvec\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatVecQ8LinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, 7)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{
			Binding:         uint32(i),
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			DescriptorCount: 1,
			StageFlags:      vk.ShaderStageComputeBit,
		}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(normSPV) * 4),
		PCode:    normSPV,
	}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q8 matvec norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: normShader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  normStage,
		Layout: pipelineLayout,
	}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q8 matvec norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	var argmaxShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(argmaxSPV) * 4),
		PCode:    argmaxSPV,
	}, nil, &argmaxShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q8 matvec argmax: %s", res)
	}
	defer vk.DestroyShaderModule(device, argmaxShader, nil)
	argmaxPipelines := make([]vk.Pipeline, 1)
	argmaxStage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: argmaxShader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  argmaxStage,
		Layout: pipelineLayout,
	}}, nil, argmaxPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q8 matvec argmax: %s", res)
	}
	r.argmaxPipeline = argmaxPipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: 7}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
	}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{setLayout},
	}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        cmdPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanMatVecQ8LinuxRunner) run(out, x []float32, q *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(q.Cols, "Vulkan q8 matvec runner x")
	if err != nil {
		return err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(q.Rows, q.Cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 matvec runner: %w", err)
	}
	scaleBytes, err := checkedFloat32ByteLenErrLinux(q.Rows, "Vulkan q8 matvec runner scale")
	if err != nil {
		return err
	}
	outBytes := scaleBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	dataBuf, err := r.int8WeightBuffer(q.Data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.floatWeightBuffer(q.Scale[:q.Rows], scaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:q.Cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8LinuxCommandDefault || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordCommand(q.Rows, q.Cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(device, out[:q.Rows])
}

func (r *vulkanMatVecQ8LinuxRunner) runArgmax(x []float32, q *tensor.Q8Matrix) (int, float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(q.Cols, "Vulkan q8 matvec argmax runner x")
	if err != nil {
		return 0, 0, err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(q.Rows, q.Cols)
	if err != nil {
		return 0, 0, fmt.Errorf("Vulkan q8 matvec argmax runner: %w", err)
	}
	scaleBytes, err := checkedFloat32ByteLenErrLinux(q.Rows, "Vulkan q8 matvec argmax runner scale")
	if err != nil {
		return 0, 0, err
	}
	outBytes := scaleBytes
	resultBytes, err := checkedFloat32ByteLenErrLinux(2, "Vulkan q8 matvec argmax runner result")
	if err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.argmaxBuf, resultBytes); err != nil {
		return 0, 0, err
	}
	dataBuf, err := r.int8WeightBuffer(q.Data[:dataLen], dataBytes)
	if err != nil {
		return 0, 0, err
	}
	scaleBuf, err := r.floatWeightBuffer(q.Scale[:q.Rows], scaleBytes)
	if err != nil {
		return 0, 0, err
	}
	if err := r.xBuf.writeFloat32(device, x[:q.Cols]); err != nil {
		return 0, 0, err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.argmaxBuf.buffer, Offset: 0, Range: r.argmaxBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8LinuxCommandArgmax || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordArgmaxCommand(q.Rows, q.Cols); err != nil {
			return 0, 0, err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return 0, 0, fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return 0, 0, fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return 0, 0, fmt.Errorf("vkWaitForFences: %s", res)
	}
	var result [2]float32
	if err := r.argmaxBuf.readFloat32Into(device, result[:]); err != nil {
		return 0, 0, err
	}
	return int(result[1]), result[0], nil
}

func (r *vulkanMatVecQ8LinuxRunner) runAddRMSNorm(normOut, residual, x []float32, q *tensor.Q8Matrix, normWeight []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(q.Cols, "Vulkan q8 matvec+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesLinux(q.Rows, q.Cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 matvec+add+rmsnorm runner: %w", err)
	}
	rowsBytes, err := checkedFloat32ByteLenErrLinux(q.Rows, "Vulkan q8 matvec+add+rmsnorm runner rows")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, rowsBytes); err != nil {
		return err
	}
	dataBuf, err := r.int8WeightBuffer(q.Data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.floatWeightBuffer(q.Scale[:q.Rows], rowsBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.floatWeightBuffer(normWeight[:q.Rows], rowsBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:q.Cols]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(device, residual[:q.Rows]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8LinuxCommandAddRMSNorm || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordAddRMSNormCommand(q.Rows, q.Cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(device, residual[:q.Rows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(device, normOut[:q.Rows])
}

func (r *vulkanMatVecQ8LinuxRunner) recordCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8LinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8LinuxRunner) recordArgmaxCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.argmaxPipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8LinuxCommandArgmax
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8LinuxRunner) recordAddRMSNormCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8LinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatVecQ8LinuxRunner) int8WeightBuffer(data []int8, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedInt8BufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanMatVecQ8LinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanMatVecQ8LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.argmaxPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.argmaxPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.argmaxBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanFusedMatVec3Q8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	vulkanFusedMatVec3Q8LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec3Q8LinuxRunnerCache.runner, vulkanFusedMatVec3Q8LinuxRunnerCache.err = newVulkanFusedMatVec3Q8LinuxRunner()
	})
	return vulkanFusedMatVec3Q8LinuxRunnerCache.runner, vulkanFusedMatVec3Q8LinuxRunnerCache.err
}

func getVulkanFusedMatVec2Q8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	vulkanFusedMatVec2Q8LinuxRunnerCache.once.Do(func() {
		vulkanFusedMatVec2Q8LinuxRunnerCache.runner, vulkanFusedMatVec2Q8LinuxRunnerCache.err = newVulkanFusedMatVec2Q8LinuxRunner()
	})
	return vulkanFusedMatVec2Q8LinuxRunnerCache.runner, vulkanFusedMatVec2Q8LinuxRunnerCache.err
}

func getVulkanFusedMatVec3MRoPEQ8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec3MRoPEQ8SPV()
		if err != nil {
			vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.err = newVulkanFusedMatVec3Q8LinuxRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec3-mrope\x00", spv, 12, 20, true)
	})
	return vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ8LinuxRunnerCache.err
}

func getVulkanFusedMatVec2MRoPEQ8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec2MRoPEQ8SPV()
		if err != nil {
			vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.err = newVulkanFusedMatVec3Q8LinuxRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec2-mrope\x00", spv, 9, 16, true)
	})
	return vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ8LinuxRunnerCache.err
}

type vulkanFusedMatVec3Q8LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	cosBuf          vulkanHostBuffer
	sinBuf          vulkanHostBuffer
	outABuf         vulkanHostBuffer
	outBBuf         vulkanHostBuffer
	outCBuf         vulkanHostBuffer
	dataBuffers     map[uintptr]vulkanCachedInt8BufferLinux
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [12]vulkanDescriptorBindingLinux
	descriptorCount int
	mrope           bool
	commandRecorded bool
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	commandPacked   int
	mu              sync.Mutex
}

func newVulkanFusedMatVec3Q8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	spv, err := vulkanFusedMatVec3Q8SPV()
	if err != nil {
		return nil, err
	}
	return newVulkanFusedMatVec3Q8LinuxRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec3\x00", spv, 10, 16, false)
}

func newVulkanFusedMatVec2Q8LinuxRunner() (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	spv, err := vulkanFusedMatVec2Q8SPV()
	if err != nil {
		return nil, err
	}
	return newVulkanFusedMatVec3Q8LinuxRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec2\x00", spv, 7, 12, false)
}

func newVulkanFusedMatVec3Q8LinuxRunnerWithShader(appName string, spv []uint32, descriptorCount, pushConstantBytes int, mrope bool) (*vulkanFusedMatVec3Q8LinuxRunner, error) {
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanFusedMatVec3Q8LinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux), descriptorCount: descriptorCount, mrope: mrope}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: uint32(pushConstantBytes)}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
	}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{setLayout},
	}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: cmdPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) run(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	cols := a.Cols
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan q8 fused matvec3 runner x")
	if err != nil {
		return err
	}
	dataALen, dataABytes, err := checkedTextAttentionQ8DataBytesLinux(a.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3 runner a: %w", err)
	}
	dataBLen, dataBBytes, err := checkedTextAttentionQ8DataBytesLinux(b.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3 runner b: %w", err)
	}
	dataCLen, dataCBytes, err := checkedTextAttentionQ8DataBytesLinux(c.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3 runner c: %w", err)
	}
	scaleABytes, err := checkedFloat32ByteLenErrLinux(a.Rows, "Vulkan q8 fused matvec3 runner a scale")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(b.Rows, "Vulkan q8 fused matvec3 runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(c.Rows, "Vulkan q8 fused matvec3 runner c scale")
	if err != nil {
		return err
	}
	outABytes, outBBytes, outCBytes := scaleABytes, scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataABuf, err := r.int8WeightBuffer(a.Data[:dataALen], dataABytes)
	if err != nil {
		return err
	}
	dataBBuf, err := r.int8WeightBuffer(b.Data[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.int8WeightBuffer(c.Data[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleABuf, err := r.floatWeightBuffer(a.Scale[:a.Rows], scaleABytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataABuf.buffer, Offset: 0, Range: dataABuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleABuf.buffer, Offset: 0, Range: scaleABuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsA != a.Rows || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols {
		if err := r.recordCommand(a.Rows, b.Rows, c.Rows, cols, 0); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:a.Rows]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) run2(outB, outC, x []float32, b, c *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	cols := b.Cols
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan q8 fused matvec2 runner x")
	if err != nil {
		return err
	}
	dataBLen, dataBBytes, err := checkedTextAttentionQ8DataBytesLinux(b.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec2 runner b: %w", err)
	}
	dataCLen, dataCBytes, err := checkedTextAttentionQ8DataBytesLinux(c.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec2 runner c: %w", err)
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(b.Rows, "Vulkan q8 fused matvec2 runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(c.Rows, "Vulkan q8 fused matvec2 runner c scale")
	if err != nil {
		return err
	}
	outBBytes, outCBytes := scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataBBuf, err := r.int8WeightBuffer(b.Data[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.int8WeightBuffer(c.Data[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols {
		if err := r.recordCommand2(b.Rows, c.Rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:b.Rows]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:c.Rows])
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) runMRoPE(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	cols := a.Cols
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan q8 fused matvec3+mrope runner x")
	if err != nil {
		return err
	}
	dataALen, dataABytes, err := checkedTextAttentionQ8DataBytesLinux(a.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3+mrope runner a: %w", err)
	}
	dataBLen, dataBBytes, err := checkedTextAttentionQ8DataBytesLinux(b.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3+mrope runner b: %w", err)
	}
	dataCLen, dataCBytes, err := checkedTextAttentionQ8DataBytesLinux(c.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec3+mrope runner c: %w", err)
	}
	scaleABytes, err := checkedFloat32ByteLenErrLinux(a.Rows, "Vulkan q8 fused matvec3+mrope runner a scale")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(b.Rows, "Vulkan q8 fused matvec3+mrope runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(c.Rows, "Vulkan q8 fused matvec3+mrope runner c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan q8 fused matvec3+mrope runner table")
	if err != nil {
		return err
	}
	outABytes, outBBytes, outCBytes := scaleABytes, scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataABuf, err := r.int8WeightBuffer(a.Data[:dataALen], dataABytes)
	if err != nil {
		return err
	}
	dataBBuf, err := r.int8WeightBuffer(b.Data[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.int8WeightBuffer(c.Data[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleABuf, err := r.floatWeightBuffer(a.Scale[:a.Rows], scaleABytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataABuf.buffer, Offset: 0, Range: dataABuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleABuf.buffer, Offset: 0, Range: scaleABuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != a.Rows || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(a.Rows, b.Rows, c.Rows, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:a.Rows]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) run2MRoPE(outB, outC, x []float32, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	cols := b.Cols
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan q8 fused matvec2+mrope runner x")
	if err != nil {
		return err
	}
	dataBLen, dataBBytes, err := checkedTextAttentionQ8DataBytesLinux(b.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec2+mrope runner b: %w", err)
	}
	dataCLen, dataCBytes, err := checkedTextAttentionQ8DataBytesLinux(c.Rows, cols)
	if err != nil {
		return fmt.Errorf("Vulkan q8 fused matvec2+mrope runner c: %w", err)
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(b.Rows, "Vulkan q8 fused matvec2+mrope runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(c.Rows, "Vulkan q8 fused matvec2+mrope runner c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan q8 fused matvec2+mrope runner table")
	if err != nil {
		return err
	}
	outBBytes, outCBytes := scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataBBuf, err := r.int8WeightBuffer(b.Data[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.int8WeightBuffer(c.Data[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand2MRoPE(b.Rows, c.Rows, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:b.Rows]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:c.Rows])
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) recordCommand2MRoPE(rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(packed))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB/2+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) recordCommand(rowsA, rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsA))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(cols))
	pushBytes := 16
	groups := rowsA + rowsB + rowsC
	if r.mrope {
		binary.LittleEndian.PutUint32(pc[16:20], uint32(packed))
		pushBytes = 20
		groups = rowsA/2 + rowsB/2 + rowsC
	}
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(pushBytes), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(groups), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) recordCommand2(rowsB, rowsC, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [12]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) int8WeightBuffer(data []int8, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedInt8BufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanFusedMatVec3Q8LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.cosBuf.destroy(r.device)
		r.sinBuf.destroy(r.device)
		r.outABuf.destroy(r.device)
		r.outBBuf.destroy(r.device)
		r.outCBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanSwiGLUDownQ8LinuxRunner() (*vulkanSwiGLUDownQ8LinuxRunner, error) {
	vulkanSwiGLUDownQ8LinuxRunnerCache.once.Do(func() {
		vulkanSwiGLUDownQ8LinuxRunnerCache.runner, vulkanSwiGLUDownQ8LinuxRunnerCache.err = newVulkanSwiGLUDownQ8LinuxRunner()
	})
	return vulkanSwiGLUDownQ8LinuxRunnerCache.runner, vulkanSwiGLUDownQ8LinuxRunnerCache.err
}

type vulkanSwiGLUDownQ8LinuxRunner struct {
	instance            vk.Instance
	device              vk.Device
	queue               vk.Queue
	queueFamily         uint32
	memProps            vk.PhysicalDeviceMemoryProperties
	setLayout           vk.DescriptorSetLayout
	downSetLayout       vk.DescriptorSetLayout
	descriptorPool      vk.DescriptorPool
	descriptorSet       vk.DescriptorSet
	downDescriptorSet   vk.DescriptorSet
	pipelineLayout      vk.PipelineLayout
	pipeline            vk.Pipeline
	downPipelineLayout  vk.PipelineLayout
	downPipeline        vk.Pipeline
	normPipeline        vk.Pipeline
	commandPool         vk.CommandPool
	commandBuffer       vk.CommandBuffer
	fence               vk.Fence
	xBuf                vulkanHostBuffer
	interBuf            vulkanHostBuffer
	outBuf              vulkanHostBuffer
	residualBuf         vulkanHostBuffer
	normBuf             vulkanHostBuffer
	dataBuffers         map[uintptr]vulkanCachedInt8BufferLinux
	scaleBuffers        map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache     [6]vulkanDescriptorBindingLinux
	downDescriptorCache [7]vulkanDescriptorBindingLinux
	commandKind         int
	commandRecorded     bool
	commandRows         int
	commandCols         int
	commandOutRows      int
	mu                  sync.Mutex
}

const (
	vulkanSwiGLUDownQ8LinuxCommandDefault = iota + 1
	vulkanSwiGLUDownQ8LinuxCommandAddRMSNorm
	vulkanSwiGLUDownQ8LinuxCommandGateUp
)

func newVulkanSwiGLUDownQ8LinuxRunner() (*vulkanSwiGLUDownQ8LinuxRunner, error) {
	spv, err := vulkanSwiGLUGateUpQ8SPV()
	if err != nil {
		return nil, err
	}
	downSPV, err := vulkanMatVecQ8SPV()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanMatVecQ8AddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-q8-swiglu-down\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanSwiGLUDownQ8LinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 6)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	downBindings := make([]vk.DescriptorSetLayoutBinding, 7)
	for i := range downBindings {
		downBindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(downBindings)), PBindings: downBindings}, nil, &r.downSetLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout down: %s", res)
	}
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings) + len(downBindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 2, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	descSets := make([]vk.DescriptorSet, 2)
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 2, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout, r.downSetLayout}}, descSets); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSets[0]
	r.downDescriptorSet = descSets[1]
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.downSetLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.downPipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout down: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var downShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(downSPV) * 4), PCode: downSPV}, nil, &downShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule down: %s", res)
	}
	defer vk.DestroyShaderModule(device, downShader, nil)
	downPipelines := make([]vk.Pipeline, 1)
	downStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: downShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: downStage, Layout: r.downPipelineLayout}}, nil, downPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines down: %s", res)
	}
	r.downPipeline = downPipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(normSPV) * 4), PCode: normSPV}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule q8 swiglu norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: normShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.downPipelineLayout}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines q8 swiglu norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) runGateUp(out, x []float32, gate, up *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols := gate.Rows, gate.Cols
	xBytes := float32ByteLen(cols)
	interBytes := float32ByteLen(rows)
	gateLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan q8 swiglu gate/up runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedMatVecF32WeightLenLinux(up.Rows, up.Cols, "Vulkan q8 swiglu gate/up runner up")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan q8 swiglu gate/up runner gate")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan q8 swiglu gate/up runner up")
	if err != nil {
		return err
	}
	gateScaleBytes := float32ByteLen(gate.Rows)
	upScaleBytes := float32ByteLen(up.Rows)
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	gateBuf, err := r.int8WeightBuffer(gate.Data[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.int8WeightBuffer(up.Data[:upLen], upBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gate.Scale[:gate.Rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(up.Scale[:up.Rows], upScaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownQ8LinuxCommandGateUp || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordGateUpCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.interBuf.readFloat32Into(r.device, out[:rows])
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) run(out, x []float32, gate, up, down *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols, outRows := gate.Rows, gate.Cols, down.Rows
	xBytes := float32ByteLen(cols)
	interBytes := float32ByteLen(rows)
	outBytes := float32ByteLen(outRows)
	gateLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan q8 swiglu/down runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedMatVecF32WeightLenLinux(up.Rows, up.Cols, "Vulkan q8 swiglu/down runner up")
	if err != nil {
		return err
	}
	downLen, err := checkedMatVecF32WeightLenLinux(down.Rows, down.Cols, "Vulkan q8 swiglu/down runner down")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan q8 swiglu/down runner gate")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan q8 swiglu/down runner up")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrLinux(downLen, 4, "Vulkan q8 swiglu/down runner down")
	if err != nil {
		return err
	}
	gateScaleBytes := float32ByteLen(gate.Rows)
	upScaleBytes := float32ByteLen(up.Rows)
	downScaleBytes := float32ByteLen(down.Rows)
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.int8WeightBuffer(gate.Data[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.int8WeightBuffer(up.Data[:upLen], upBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.int8WeightBuffer(down.Data[:downLen], downBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gate.Scale[:gate.Rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(up.Scale[:up.Rows], upScaleBytes)
	if err != nil {
		return err
	}
	downScaleBuf, err := r.floatWeightBuffer(down.Scale[:down.Rows], downScaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	downInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: downScaleBuf.buffer, Offset: 0, Range: downScaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownQ8LinuxCommandDefault || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(r.device, out[:outRows])
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) runAddRMSNorm(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32, updateResidual bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols, outRows := gate.Rows, gate.Cols, down.Rows
	xBytes := float32ByteLen(cols)
	interBytes := float32ByteLen(rows)
	outBytes := float32ByteLen(outRows)
	gateLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan q8 swiglu/down+add+rmsnorm runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedMatVecF32WeightLenLinux(up.Rows, up.Cols, "Vulkan q8 swiglu/down+add+rmsnorm runner up")
	if err != nil {
		return err
	}
	downLen, err := checkedMatVecF32WeightLenLinux(down.Rows, down.Cols, "Vulkan q8 swiglu/down+add+rmsnorm runner down")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner gate")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner up")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrLinux(downLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner down")
	if err != nil {
		return err
	}
	gateScaleBytes := float32ByteLen(gate.Rows)
	upScaleBytes := float32ByteLen(up.Rows)
	downScaleBytes := float32ByteLen(down.Rows)
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.int8WeightBuffer(gate.Data[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.int8WeightBuffer(up.Data[:upLen], upBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.int8WeightBuffer(down.Data[:downLen], downBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gate.Scale[:gate.Rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(up.Scale[:up.Rows], upScaleBytes)
	if err != nil {
		return err
	}
	downScaleBuf, err := r.floatWeightBuffer(down.Scale[:down.Rows], downScaleBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.floatWeightBuffer(normWeight[:outRows], outBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:outRows]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	downInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: downScaleBuf.buffer, Offset: 0, Range: downScaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownQ8LinuxCommandAddRMSNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordAddRMSNormCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if updateResidual {
		if err := r.residualBuf.readFloat32Into(r.device, residual[:outRows]); err != nil {
			return err
		}
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:outRows])
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) recordCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownQ8LinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) recordGateUpCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	cbi := vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}
	if res := vk.BeginCommandBuffer(cmd, &cbi); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandKind = vulkanSwiGLUDownQ8LinuxCommandGateUp
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) recordAddRMSNormCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownQ8LinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) int8WeightBuffer(data []int8, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedInt8BufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanSwiGLUDownQ8LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.downPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.downPipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.interBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.downPipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.downPipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		if r.downSetLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.downSetLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanSwiGLUDownQ4LinuxRunner() (*vulkanSwiGLUDownPackedBytesLinuxRunner, error) {
	vulkanSwiGLUDownQ4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanSwiGLUGateUpQ4SPV()
		if err != nil {
			vulkanSwiGLUDownQ4LinuxRunnerCache.err = err
			return
		}
		downSPV, err := vulkanMatVecQ4SPV()
		if err != nil {
			vulkanSwiGLUDownQ4LinuxRunnerCache.err = err
			return
		}
		vulkanSwiGLUDownQ4LinuxRunnerCache.runner, vulkanSwiGLUDownQ4LinuxRunnerCache.err = newVulkanSwiGLUDownPackedBytesLinuxRunner("rapidocrvl-vulkan-q4-swiglu-down\x00", spv, downSPV)
	})
	return vulkanSwiGLUDownQ4LinuxRunnerCache.runner, vulkanSwiGLUDownQ4LinuxRunnerCache.err
}

func getVulkanSwiGLUDownQ6LinuxRunner() (*vulkanSwiGLUDownPackedBytesLinuxRunner, error) {
	vulkanSwiGLUDownQ6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanSwiGLUGateUpQ6SPV()
		if err != nil {
			vulkanSwiGLUDownQ6LinuxRunnerCache.err = err
			return
		}
		downSPV, err := vulkanMatVecQ6SPV()
		if err != nil {
			vulkanSwiGLUDownQ6LinuxRunnerCache.err = err
			return
		}
		vulkanSwiGLUDownQ6LinuxRunnerCache.runner, vulkanSwiGLUDownQ6LinuxRunnerCache.err = newVulkanSwiGLUDownPackedBytesLinuxRunner("rapidocrvl-vulkan-q6-swiglu-down\x00", spv, downSPV)
	})
	return vulkanSwiGLUDownQ6LinuxRunnerCache.runner, vulkanSwiGLUDownQ6LinuxRunnerCache.err
}

type vulkanSwiGLUDownPackedBytesLinuxRunner struct {
	instance            vk.Instance
	device              vk.Device
	queue               vk.Queue
	queueFamily         uint32
	memProps            vk.PhysicalDeviceMemoryProperties
	setLayout           vk.DescriptorSetLayout
	downSetLayout       vk.DescriptorSetLayout
	descriptorPool      vk.DescriptorPool
	descriptorSet       vk.DescriptorSet
	downDescriptorSet   vk.DescriptorSet
	pipelineLayout      vk.PipelineLayout
	pipeline            vk.Pipeline
	downPipelineLayout  vk.PipelineLayout
	downPipeline        vk.Pipeline
	normPipeline        vk.Pipeline
	commandPool         vk.CommandPool
	commandBuffer       vk.CommandBuffer
	fence               vk.Fence
	xBuf                vulkanHostBuffer
	interBuf            vulkanHostBuffer
	outBuf              vulkanHostBuffer
	residualBuf         vulkanHostBuffer
	normBuf             vulkanHostBuffer
	dataBuffers         map[uintptr]vulkanCachedByteBufferLinux
	scaleBuffers        map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache     [6]vulkanDescriptorBindingLinux
	downDescriptorCache [7]vulkanDescriptorBindingLinux
	commandKind         int
	commandRecorded     bool
	commandRows         int
	commandCols         int
	commandOutRows      int
	mu                  sync.Mutex
}

const (
	vulkanSwiGLUDownPackedLinuxCommandDefault = iota + 1
	vulkanSwiGLUDownPackedLinuxCommandAddRMSNorm
	vulkanSwiGLUDownPackedLinuxCommandGateUp
)

func newVulkanSwiGLUDownPackedBytesLinuxRunner(appName string, spv, downSPV []uint32) (*vulkanSwiGLUDownPackedBytesLinuxRunner, error) {
	normSPV, err := vulkanMatVecQ8AddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanSwiGLUDownPackedBytesLinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedByteBufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 6)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	downBindings := make([]vk.DescriptorSetLayoutBinding, 7)
	for i := range downBindings {
		downBindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(downBindings)), PBindings: downBindings}, nil, &r.downSetLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout down: %s", res)
	}
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings) + len(downBindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 2, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	descSets := make([]vk.DescriptorSet, 2)
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 2, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout, r.downSetLayout}}, descSets); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSets[0]
	r.downDescriptorSet = descSets[1]
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.setLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: []vk.DescriptorSetLayout{r.downSetLayout}, PushConstantRangeCount: uint32(len(pushRanges)), PPushConstantRanges: pushRanges}, nil, &r.downPipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout down: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var downShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(downSPV) * 4), PCode: downSPV}, nil, &downShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule down: %s", res)
	}
	defer vk.DestroyShaderModule(device, downShader, nil)
	downPipelines := make([]vk.Pipeline, 1)
	downStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: downShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: downStage, Layout: r.downPipelineLayout}}, nil, downPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines down: %s", res)
	}
	r.downPipeline = downPipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(normSPV) * 4), PCode: normSPV}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule packed swiglu norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: normShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.downPipelineLayout}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines packed swiglu norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) runGateUp(out, x []float32, gateData, upData []byte, gateScale, upScale []float32, rows, cols, gateLen, upLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed swiglu gate/up runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu gate/up runner intermediate")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan packed swiglu gate/up runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan packed swiglu gate/up runner up data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu gate/up runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu gate/up runner up scale")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	gateBuf, err := r.byteWeightBuffer(gateData[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.byteWeightBuffer(upData[:upLen], upBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gateScale[:rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(upScale[:rows], upScaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownPackedLinuxCommandGateUp || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordGateUpCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.interBuf.readFloat32Into(r.device, out[:rows])
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) run(out, x []float32, gateData, upData, downData []byte, gateScale, upScale, downScale []float32, rows, cols, outRows, gateLen, upLen, downLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed swiglu/down runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down runner intermediate")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan packed swiglu/down runner output")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan packed swiglu/down runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan packed swiglu/down runner up data")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrLinux(downLen, 4, "Vulkan packed swiglu/down runner down data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down runner up scale")
	if err != nil {
		return err
	}
	downScaleBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan packed swiglu/down runner down scale")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.byteWeightBuffer(gateData[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.byteWeightBuffer(upData[:upLen], upBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.byteWeightBuffer(downData[:downLen], downBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gateScale[:rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(upScale[:rows], upScaleBytes)
	if err != nil {
		return err
	}
	downScaleBuf, err := r.floatWeightBuffer(downScale[:outRows], downScaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	downInfos := [4]vk.DescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: downScaleBuf.buffer, Offset: 0, Range: downScaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownPackedLinuxCommandDefault || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(r.device, out[:outRows])
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) runAddRMSNorm(normOut, residual, x []float32, gateData, upData, downData []byte, gateScale, upScale, downScale, normWeight []float32, rows, cols, outRows, gateLen, upLen, downLen int, updateResidual bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed swiglu/down+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down+add+rmsnorm runner intermediate")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan packed swiglu/down+add+rmsnorm runner output")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrLinux(gateLen, 4, "Vulkan packed swiglu/down+add+rmsnorm runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrLinux(upLen, 4, "Vulkan packed swiglu/down+add+rmsnorm runner up data")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrLinux(downLen, 4, "Vulkan packed swiglu/down+add+rmsnorm runner down data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down+add+rmsnorm runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed swiglu/down+add+rmsnorm runner up scale")
	if err != nil {
		return err
	}
	downScaleBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan packed swiglu/down+add+rmsnorm runner down scale")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.interBuf, interBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, outBytes); err != nil {
		return err
	}
	gateBuf, err := r.byteWeightBuffer(gateData[:gateLen], gateBytes)
	if err != nil {
		return err
	}
	upBuf, err := r.byteWeightBuffer(upData[:upLen], upBytes)
	if err != nil {
		return err
	}
	downBuf, err := r.byteWeightBuffer(downData[:downLen], downBytes)
	if err != nil {
		return err
	}
	gateScaleBuf, err := r.floatWeightBuffer(gateScale[:rows], gateScaleBytes)
	if err != nil {
		return err
	}
	upScaleBuf, err := r.floatWeightBuffer(upScale[:rows], upScaleBytes)
	if err != nil {
		return err
	}
	downScaleBuf, err := r.floatWeightBuffer(downScale[:outRows], downScaleBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.floatWeightBuffer(normWeight[:outRows], outBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(r.device, x[:cols]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(r.device, residual[:outRows]); err != nil {
		return err
	}
	swiInfos := [6]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: gateBuf.buffer, Offset: 0, Range: gateBuf.size},
		{Buffer: upBuf.buffer, Offset: 0, Range: upBuf.size},
		{Buffer: gateScaleBuf.buffer, Offset: 0, Range: gateScaleBuf.size},
		{Buffer: upScaleBuf.buffer, Offset: 0, Range: upScaleBuf.size},
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
	}
	downInfos := [7]vk.DescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Offset: 0, Range: r.interBuf.size},
		{Buffer: downBuf.buffer, Offset: 0, Range: downBuf.size},
		{Buffer: downScaleBuf.buffer, Offset: 0, Range: downScaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersLinux(r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUDownPackedLinuxCommandAddRMSNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordAddRMSNormCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if updateResidual {
		if err := r.residualBuf.readFloat32Into(r.device, residual[:outRows]); err != nil {
			return err
		}
	}
	return r.normBuf.readFloat32Into(r.device, normOut[:outRows])
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) recordCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownPackedLinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) recordGateUpCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	cbi := vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}
	if res := vk.BeginCommandBuffer(cmd, &cbi); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandKind = vulkanSwiGLUDownPackedLinuxCommandGateUp
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) recordAddRMSNormCommand(rows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.downPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.downPipelineLayout, 0, 1, []vk.DescriptorSet{r.downDescriptorSet}, 0, nil)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), 1, 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.downPipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandKind = vulkanSwiGLUDownPackedLinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) byteWeightBuffer(data []byte, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedByteBufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanSwiGLUDownPackedBytesLinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.downPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.downPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.interBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.downPipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.downPipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		if r.downSetLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.downSetLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatVecQ4LinuxRunner() (*vulkanMatVecPackedBytesLinuxRunner, error) {
	vulkanMatVecQ4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanMatVecQ4SPV()
		if err != nil {
			vulkanMatVecQ4LinuxRunnerCache.err = err
			return
		}
		vulkanMatVecQ4LinuxRunnerCache.runner, vulkanMatVecQ4LinuxRunnerCache.err = newVulkanMatVecPackedBytesLinuxRunner("rapidocrvl-vulkan-q4-matvec\x00", spv)
	})
	return vulkanMatVecQ4LinuxRunnerCache.runner, vulkanMatVecQ4LinuxRunnerCache.err
}

func getVulkanMatVecQ6LinuxRunner() (*vulkanMatVecPackedBytesLinuxRunner, error) {
	vulkanMatVecQ6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanMatVecQ6SPV()
		if err != nil {
			vulkanMatVecQ6LinuxRunnerCache.err = err
			return
		}
		vulkanMatVecQ6LinuxRunnerCache.runner, vulkanMatVecQ6LinuxRunnerCache.err = newVulkanMatVecPackedBytesLinuxRunner("rapidocrvl-vulkan-q6-matvec\x00", spv)
	})
	return vulkanMatVecQ6LinuxRunnerCache.runner, vulkanMatVecQ6LinuxRunnerCache.err
}

type vulkanMatVecPackedBytesLinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	normPipeline    vk.Pipeline
	argmaxPipeline  vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outBuf          vulkanHostBuffer
	argmaxBuf       vulkanHostBuffer
	residualBuf     vulkanHostBuffer
	normBuf         vulkanHostBuffer
	dataBuffers     map[uintptr]vulkanCachedByteBufferLinux
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferLinux
	topKReadback    []float32
	topKCandidates  []VulkanTokenScore
	descriptorCache [7]vulkanDescriptorBindingLinux
	commandKind     int
	commandRecorded bool
	commandRows     int
	commandCols     int
	mu              sync.Mutex
}

const (
	vulkanMatVecPackedLinuxCommandDefault = iota + 1
	vulkanMatVecPackedLinuxCommandAddRMSNorm
	vulkanMatVecPackedLinuxCommandArgmax
	vulkanMatVecPackedLinuxCommandTopK
)

func newVulkanMatVecPackedBytesLinuxRunner(appName string, spv []uint32) (*vulkanMatVecPackedBytesLinuxRunner, error) {
	normSPV, err := vulkanMatVecQ8AddRMSNormF32SPV()
	if err != nil {
		return nil, err
	}
	argmaxSPV, err := vulkanArgmaxQuantizedF32SPV()
	if err != nil {
		return nil, err
	}
	topKSPV, err := vulkanBlockTopKQuantizedF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatVecPackedBytesLinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedByteBufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, 7)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 8}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var normShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(normSPV) * 4), PCode: normSPV}, nil, &normShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule packed matvec norm: %s", res)
	}
	defer vk.DestroyShaderModule(device, normShader, nil)
	normPipelines := make([]vk.Pipeline, 1)
	normStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: normShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.pipelineLayout}}, nil, normPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines packed matvec norm: %s", res)
	}
	r.normPipeline = normPipelines[0]
	var argmaxShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(argmaxSPV) * 4), PCode: argmaxSPV}, nil, &argmaxShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule packed matvec argmax: %s", res)
	}
	defer vk.DestroyShaderModule(device, argmaxShader, nil)
	argmaxPipelines := make([]vk.Pipeline, 1)
	argmaxStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: argmaxShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: argmaxStage, Layout: r.pipelineLayout}}, nil, argmaxPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines packed matvec argmax: %s", res)
	}
	r.argmaxPipeline = argmaxPipelines[0]
	var topKShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(topKSPV) * 4), PCode: topKSPV}, nil, &topKShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule packed matvec top-k: %s", res)
	}
	defer vk.DestroyShaderModule(device, topKShader, nil)
	topKPipelines := make([]vk.Pipeline, 1)
	topKStage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: topKShader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: topKStage, Layout: r.pipelineLayout}}, nil, topKPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines packed matvec top-k: %s", res)
	}
	r.topKPipeline = topKPipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: 7}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSizes)),
		PPoolSizes:    poolSizes,
	}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     descPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{setLayout},
	}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: cmdPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) run(out, x []float32, data []byte, scale []float32, rows, cols, dataLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed matvec runner x")
	if err != nil {
		return err
	}
	dataBytes, err := checkedAlignedByteLenErrLinux(dataLen, 4, "Vulkan packed matvec runner data")
	if err != nil {
		return err
	}
	scaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed matvec runner scale")
	if err != nil {
		return err
	}
	outBytes := scaleBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	dataBuf, err := r.byteWeightBuffer(data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.floatWeightBuffer(scale[:rows], scaleBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readFloat32Into(device, out[:rows])
}

func (r *vulkanMatVecPackedBytesLinuxRunner) runArgmax(x []float32, data []byte, scale []float32, rows, cols, dataLen int) (int, float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed matvec argmax runner x")
	if err != nil {
		return 0, 0, err
	}
	dataBytes, err := checkedAlignedByteLenErrLinux(dataLen, 4, "Vulkan packed matvec argmax runner data")
	if err != nil {
		return 0, 0, err
	}
	scaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed matvec argmax runner scale")
	if err != nil {
		return 0, 0, err
	}
	outBytes := scaleBytes
	resultBytes, err := checkedFloat32ByteLenErrLinux(2, "Vulkan packed matvec argmax runner result")
	if err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return 0, 0, err
	}
	if err := r.ensureHostBuffer(&r.argmaxBuf, resultBytes); err != nil {
		return 0, 0, err
	}
	dataBuf, err := r.byteWeightBuffer(data[:dataLen], dataBytes)
	if err != nil {
		return 0, 0, err
	}
	scaleBuf, err := r.floatWeightBuffer(scale[:rows], scaleBytes)
	if err != nil {
		return 0, 0, err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return 0, 0, err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.argmaxBuf.buffer, Offset: 0, Range: r.argmaxBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecPackedLinuxCommandArgmax || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordArgmaxCommand(rows, cols); err != nil {
			return 0, 0, err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return 0, 0, fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return 0, 0, fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return 0, 0, fmt.Errorf("vkWaitForFences: %s", res)
	}
	var result [2]float32
	if err := r.argmaxBuf.readFloat32Into(device, result[:]); err != nil {
		return 0, 0, err
	}
	return int(result[1]), result[0], nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) runTopK(x []float32, data []byte, scale []float32, rows, cols, dataLen, k int) ([]VulkanTokenScore, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	blocks := (rows + 255) / 256
	candidateFloats, ok := checkedMulInt(blocks, vulkanMatVecTopKMaxK)
	if !ok {
		return nil, fmt.Errorf("Vulkan packed matvec top-k runner candidate count overflows: blocks=%d k=%d", blocks, vulkanMatVecTopKMaxK)
	}
	candidateFloats, ok = checkedMulInt(candidateFloats, 2)
	if !ok {
		return nil, fmt.Errorf("Vulkan packed matvec top-k runner candidate count overflows: blocks=%d k=%d", blocks, vulkanMatVecTopKMaxK)
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed matvec top-k runner x")
	if err != nil {
		return nil, err
	}
	dataBytes, err := checkedAlignedByteLenErrLinux(dataLen, 4, "Vulkan packed matvec top-k runner data")
	if err != nil {
		return nil, err
	}
	scaleBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed matvec top-k runner scale")
	if err != nil {
		return nil, err
	}
	outBytes := scaleBytes
	resultBytes, err := checkedFloat32ByteLenErrLinux(candidateFloats, "Vulkan packed matvec top-k runner result")
	if err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return nil, err
	}
	if err := r.ensureHostBuffer(&r.argmaxBuf, resultBytes); err != nil {
		return nil, err
	}
	dataBuf, err := r.byteWeightBuffer(data[:dataLen], dataBytes)
	if err != nil {
		return nil, err
	}
	scaleBuf, err := r.floatWeightBuffer(scale[:rows], scaleBytes)
	if err != nil {
		return nil, err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return nil, err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.argmaxBuf.buffer, Offset: 0, Range: r.argmaxBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecPackedLinuxCommandTopK || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordTopKCommand(rows, cols); err != nil {
			return nil, err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return nil, fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return nil, fmt.Errorf("vkWaitForFences: %s", res)
	}
	r.topKReadback = ensureVulkanFloat32Scratch(r.topKReadback, candidateFloats)
	candidateData := r.topKReadback
	if err := r.argmaxBuf.readFloat32Into(device, candidateData); err != nil {
		return nil, err
	}
	r.topKCandidates = selectVulkanTopKCandidatesInto(r.topKCandidates, candidateData, rows, k)
	return r.topKCandidates, nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) runAddRMSNorm(normOut, residual, x []float32, data []byte, scale, normWeight []float32, rows, cols, dataLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed matvec+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	dataBytes, err := checkedAlignedByteLenErrLinux(dataLen, 4, "Vulkan packed matvec+add+rmsnorm runner data")
	if err != nil {
		return err
	}
	rowsBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan packed matvec+add+rmsnorm runner rows")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, rowsBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.normBuf, rowsBytes); err != nil {
		return err
	}
	dataBuf, err := r.byteWeightBuffer(data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.floatWeightBuffer(scale[:rows], rowsBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.floatWeightBuffer(normWeight[:rows], rowsBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.residualBuf.writeFloat32(device, residual[:rows]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBuf.buffer, Offset: 0, Range: dataBuf.size},
		{Buffer: scaleBuf.buffer, Offset: 0, Range: scaleBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: normWeightBuf.buffer, Offset: 0, Range: normWeightBuf.size},
		{Buffer: r.normBuf.buffer, Offset: 0, Range: r.normBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecPackedLinuxCommandAddRMSNorm || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordAddRMSNormCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{r.commandBuffer}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.residualBuf.readFloat32Into(device, residual[:rows]); err != nil {
		return err
	}
	return r.normBuf.readFloat32Into(device, normOut[:rows])
}

func (r *vulkanMatVecPackedBytesLinuxRunner) recordCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecPackedLinuxCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) recordArgmaxCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.argmaxPipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecPackedLinuxCommandArgmax
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) recordTopKCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.topKPipeline)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	blocks := (rows + 255) / 256
	vk.CmdDispatch(cmd, uint32(blocks), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecPackedLinuxCommandTopK
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) recordAddRMSNormCommand(rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	barrier := []vk.MemoryBarrier{{SType: vk.StructureTypeMemoryBarrier, SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 1, 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecPackedLinuxCommandAddRMSNorm
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) byteWeightBuffer(data []byte, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedByteBufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanMatVecPackedBytesLinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanMatVecPackedBytesLinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.normPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.normPipeline, nil)
		}
		if r.argmaxPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.argmaxPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.argmaxBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.normBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanFusedMatVec3Q4LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec3Q4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec3Q4SPV()
		if err != nil {
			vulkanFusedMatVec3Q4LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec3Q4LinuxRunnerCache.runner, vulkanFusedMatVec3Q4LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q4-fused-matvec3\x00", spv, 10, 16, false)
	})
	return vulkanFusedMatVec3Q4LinuxRunnerCache.runner, vulkanFusedMatVec3Q4LinuxRunnerCache.err
}

func getVulkanFusedMatVec2Q4LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec2Q4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec2Q4SPV()
		if err != nil {
			vulkanFusedMatVec2Q4LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec2Q4LinuxRunnerCache.runner, vulkanFusedMatVec2Q4LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q4-fused-matvec2\x00", spv, 7, 12, false)
	})
	return vulkanFusedMatVec2Q4LinuxRunnerCache.runner, vulkanFusedMatVec2Q4LinuxRunnerCache.err
}

func getVulkanFusedMatVec3MRoPEQ4LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec3MRoPEQ4SPV()
		if err != nil {
			vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q4-fused-matvec3-mrope\x00", spv, 12, 20, true)
	})
	return vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ4LinuxRunnerCache.err
}

func getVulkanFusedMatVec2MRoPEQ4LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec2MRoPEQ4SPV()
		if err != nil {
			vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q4-fused-matvec2-mrope\x00", spv, 9, 16, true)
	})
	return vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ4LinuxRunnerCache.err
}

func getVulkanFusedMatVec3Q6LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec3Q6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec3Q6SPV()
		if err != nil {
			vulkanFusedMatVec3Q6LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec3Q6LinuxRunnerCache.runner, vulkanFusedMatVec3Q6LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q6-fused-matvec3\x00", spv, 10, 16, false)
	})
	return vulkanFusedMatVec3Q6LinuxRunnerCache.runner, vulkanFusedMatVec3Q6LinuxRunnerCache.err
}

func getVulkanFusedMatVec2Q6LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec2Q6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec2Q6SPV()
		if err != nil {
			vulkanFusedMatVec2Q6LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec2Q6LinuxRunnerCache.runner, vulkanFusedMatVec2Q6LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q6-fused-matvec2\x00", spv, 7, 12, false)
	})
	return vulkanFusedMatVec2Q6LinuxRunnerCache.runner, vulkanFusedMatVec2Q6LinuxRunnerCache.err
}

func getVulkanFusedMatVec3MRoPEQ6LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec3MRoPEQ6SPV()
		if err != nil {
			vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q6-fused-matvec3-mrope\x00", spv, 12, 20, true)
	})
	return vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.runner, vulkanFusedMatVec3MRoPEQ6LinuxRunnerCache.err
}

func getVulkanFusedMatVec2MRoPEQ6LinuxRunner() (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.once.Do(func() {
		spv, err := vulkanFusedMatVec2MRoPEQ6SPV()
		if err != nil {
			vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.err = err
			return
		}
		vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.err = newVulkanFusedMatVec3PackedBytesLinuxRunner("rapidocrvl-vulkan-q6-fused-matvec2-mrope\x00", spv, 9, 16, true)
	})
	return vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.runner, vulkanFusedMatVec2MRoPEQ6LinuxRunnerCache.err
}

type vulkanFusedMatVec3PackedBytesLinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	cosBuf          vulkanHostBuffer
	sinBuf          vulkanHostBuffer
	outABuf         vulkanHostBuffer
	outBBuf         vulkanHostBuffer
	outCBuf         vulkanHostBuffer
	dataBuffers     map[uintptr]vulkanCachedByteBufferLinux
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [12]vulkanDescriptorBindingLinux
	descriptorCount int
	mrope           bool
	commandRecorded bool
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	commandPacked   int
	mu              sync.Mutex
}

func newVulkanFusedMatVec3PackedBytesLinuxRunner(appName string, spv []uint32, descriptorCount, pushConstantBytes int, mrope bool) (*vulkanFusedMatVec3PackedBytesLinuxRunner, error) {
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanFusedMatVec3PackedBytesLinuxRunner{instance: instance, dataBuffers: make(map[uintptr]vulkanCachedByteBufferLinux), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux), descriptorCount: descriptorCount, mrope: mrope}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{SType: vk.StructureTypeDeviceQueueCreateInfo, QueueFamilyIndex: queueFamily, QueueCount: 1, PQueuePriorities: priority}
	dci := vk.DeviceCreateInfo{SType: vk.StructureTypeDeviceCreateInfo, QueueCreateInfoCount: 1, PQueueCreateInfos: []vk.DeviceQueueCreateInfo{qci}}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	var queue vk.Queue
	vk.GetDeviceQueue(device, queueFamily, 0, &queue)
	r.queue = queue

	bindings := make([]vk.DescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	var setLayout vk.DescriptorSetLayout
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: bindings}, nil, &setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	r.setLayout = setLayout
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: uint32(pushConstantBytes)}}
	var pipelineLayout vk.PipelineLayout
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	r.pipelineLayout = pipelineLayout
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{SType: vk.StructureTypeShaderModuleCreateInfo, CodeSize: uint(len(spv) * 4), PCode: spv}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageComputeBit, Module: shader, PName: "main\x00"}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{SType: vk.StructureTypeComputePipelineCreateInfo, Stage: stage, Layout: pipelineLayout}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var descPool vk.DescriptorPool
	poolSizes := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: uint32(len(poolSizes)), PPoolSizes: poolSizes}, nil, &descPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	r.descriptorPool = descPool
	var descSet vk.DescriptorSet
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: descPool, DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{setLayout}}, &descSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	r.descriptorSet = descSet
	var cmdPool vk.CommandPool
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}, nil, &cmdPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	r.commandPool = cmdPool
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: cmdPool, Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	var fence vk.Fence
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	r.fence = fence
	success = true
	return r, nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) run(outA, outB, outC, x []float32, dataA, dataB, dataC []byte, scaleA, scaleB, scaleC []float32, rowsA, rowsB, rowsC, cols, dataALen, dataBLen, dataCLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed fused matvec3 runner x")
	if err != nil {
		return err
	}
	dataABytes, err := checkedAlignedByteLenErrLinux(dataALen, 4, "Vulkan packed fused matvec3 runner a data")
	if err != nil {
		return err
	}
	dataBBytes, err := checkedAlignedByteLenErrLinux(dataBLen, 4, "Vulkan packed fused matvec3 runner b data")
	if err != nil {
		return err
	}
	dataCBytes, err := checkedAlignedByteLenErrLinux(dataCLen, 4, "Vulkan packed fused matvec3 runner c data")
	if err != nil {
		return err
	}
	scaleABytes, err := checkedFloat32ByteLenErrLinux(rowsA, "Vulkan packed fused matvec3 runner a scale")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan packed fused matvec3 runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan packed fused matvec3 runner c scale")
	if err != nil {
		return err
	}
	outABytes, outBBytes, outCBytes := scaleABytes, scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataABuf, err := r.byteWeightBuffer(dataA[:dataALen], dataABytes)
	if err != nil {
		return err
	}
	dataBBuf, err := r.byteWeightBuffer(dataB[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.byteWeightBuffer(dataC[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleABuf, err := r.floatWeightBuffer(scaleA[:rowsA], scaleABytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(scaleB[:rowsB], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(scaleC[:rowsC], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataABuf.buffer, Offset: 0, Range: dataABuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleABuf.buffer, Offset: 0, Range: scaleABuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:10], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, 0); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:rowsA]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) run2(outB, outC, x []float32, dataB, dataC []byte, scaleB, scaleC []float32, rowsB, rowsC, cols, dataBLen, dataCLen int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed fused matvec2 runner x")
	if err != nil {
		return err
	}
	dataBBytes, err := checkedAlignedByteLenErrLinux(dataBLen, 4, "Vulkan packed fused matvec2 runner b data")
	if err != nil {
		return err
	}
	dataCBytes, err := checkedAlignedByteLenErrLinux(dataCLen, 4, "Vulkan packed fused matvec2 runner c data")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan packed fused matvec2 runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan packed fused matvec2 runner c scale")
	if err != nil {
		return err
	}
	outBBytes, outCBytes := scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataBBuf, err := r.byteWeightBuffer(dataB[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.byteWeightBuffer(dataC[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(scaleB[:rowsB], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(scaleC[:rowsC], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	if !r.commandRecorded || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand2(rowsB, rowsC, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:rowsC])
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) runMRoPE(outA, outB, outC, x []float32, dataA, dataB, dataC []byte, scaleA, scaleB, scaleC, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, dataALen, dataBLen, dataCLen, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed fused matvec3+mrope runner x")
	if err != nil {
		return err
	}
	dataABytes, err := checkedAlignedByteLenErrLinux(dataALen, 4, "Vulkan packed fused matvec3+mrope runner a data")
	if err != nil {
		return err
	}
	dataBBytes, err := checkedAlignedByteLenErrLinux(dataBLen, 4, "Vulkan packed fused matvec3+mrope runner b data")
	if err != nil {
		return err
	}
	dataCBytes, err := checkedAlignedByteLenErrLinux(dataCLen, 4, "Vulkan packed fused matvec3+mrope runner c data")
	if err != nil {
		return err
	}
	scaleABytes, err := checkedFloat32ByteLenErrLinux(rowsA, "Vulkan packed fused matvec3+mrope runner a scale")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan packed fused matvec3+mrope runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan packed fused matvec3+mrope runner c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan packed fused matvec3+mrope runner table")
	if err != nil {
		return err
	}
	outABytes, outBBytes, outCBytes := scaleABytes, scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataABuf, err := r.byteWeightBuffer(dataA[:dataALen], dataABytes)
	if err != nil {
		return err
	}
	dataBBuf, err := r.byteWeightBuffer(dataB[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.byteWeightBuffer(dataC[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleABuf, err := r.floatWeightBuffer(scaleA[:rowsA], scaleABytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(scaleB[:rowsB], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(scaleC[:rowsC], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataABuf.buffer, Offset: 0, Range: dataABuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleABuf.buffer, Offset: 0, Range: scaleABuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readFloat32Into(device, outA[:rowsA]); err != nil {
		return err
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	if err := r.outCBuf.readFloat32Into(device, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) run2MRoPE(outB, outC, x []float32, dataB, dataC []byte, scaleB, scaleC, cosTable, sinTable []float32, rowsB, rowsC, cols, dataBLen, dataCLen, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	device := r.device
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan packed fused matvec2+mrope runner x")
	if err != nil {
		return err
	}
	dataBBytes, err := checkedAlignedByteLenErrLinux(dataBLen, 4, "Vulkan packed fused matvec2+mrope runner b data")
	if err != nil {
		return err
	}
	dataCBytes, err := checkedAlignedByteLenErrLinux(dataCLen, 4, "Vulkan packed fused matvec2+mrope runner c data")
	if err != nil {
		return err
	}
	scaleBBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan packed fused matvec2+mrope runner b scale")
	if err != nil {
		return err
	}
	scaleCBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan packed fused matvec2+mrope runner c scale")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrLinux(half, "Vulkan packed fused matvec2+mrope runner table")
	if err != nil {
		return err
	}
	outBBytes, outCBytes := scaleBBytes, scaleCBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	dataBBuf, err := r.byteWeightBuffer(dataB[:dataBLen], dataBBytes)
	if err != nil {
		return err
	}
	dataCBuf, err := r.byteWeightBuffer(dataC[:dataCLen], dataCBytes)
	if err != nil {
		return err
	}
	scaleBBuf, err := r.floatWeightBuffer(scaleB[:rowsB], scaleBBytes)
	if err != nil {
		return err
	}
	scaleCBuf, err := r.floatWeightBuffer(scaleC[:rowsC], scaleCBytes)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeFloat32(device, x[:cols]); err != nil {
		return err
	}
	if err := r.cosBuf.writeFloat32(device, cosTable[:half]); err != nil {
		return err
	}
	if err := r.sinBuf.writeFloat32(device, sinTable[:half]); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: dataBBuf.buffer, Offset: 0, Range: dataBBuf.size},
		{Buffer: dataCBuf.buffer, Offset: 0, Range: dataCBuf.size},
		{Buffer: scaleBBuf.buffer, Offset: 0, Range: scaleBBuf.size},
		{Buffer: scaleCBuf.buffer, Offset: 0, Range: scaleCBuf.size},
		{Buffer: r.cosBuf.buffer, Offset: 0, Range: r.cosBuf.size},
		{Buffer: r.sinBuf.buffer, Offset: 0, Range: r.sinBuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufferInfos[:])
	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand2MRoPE(rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outBBuf.readFloat32Into(device, outB[:rowsB]); err != nil {
		return err
	}
	return r.outCBuf.readFloat32Into(device, outC[:rowsC])
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) recordCommand(rowsA, rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsA))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(cols))
	pushBytes := 16
	groups := rowsA + rowsB + rowsC
	if r.mrope {
		binary.LittleEndian.PutUint32(pc[16:20], uint32(packed))
		pushBytes = 20
		groups = rowsA/2 + rowsB/2 + rowsC
	}
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(pushBytes), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(groups), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) recordCommand2(rowsB, rowsC, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [12]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) recordCommand2MRoPE(rowsB, rowsC, cols, packed int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(packed))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsB/2+rowsC), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRowsA = 0
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) byteWeightBuffer(data []byte, size vk.DeviceSize) (vulkanHostBuffer, error) {
	return cachedByteBufferLinux(r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) floatWeightBuffer(data []float32, size vk.DeviceSize) (vulkanHostBuffer, error) {
	key := float32SliceKeyLinux(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := r.scaleBuffers[key]; ok {
		if cached.buffer.size >= size {
			if cached.length == len(data) && cached.fingerprint == fingerprint {
				return cached.buffer, nil
			}
			if err := cached.buffer.writeFloat32(r.device, data); err != nil {
				return vulkanHostBuffer{}, err
			}
			r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: cached.buffer, length: len(data), fingerprint: fingerprint}
			return cached.buffer, nil
		}
		cached.buffer.destroy(r.device)
		delete(r.scaleBuffers, key)
	}
	buf, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return vulkanHostBuffer{}, err
	}
	if err := buf.writeFloat32(r.device, data); err != nil {
		buf.destroy(r.device)
		return vulkanHostBuffer{}, err
	}
	r.scaleBuffers[key] = vulkanCachedFloat32BufferLinux{buffer: buf, length: len(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanFusedMatVec3PackedBytesLinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.cosBuf.destroy(r.device)
		r.sinBuf.destroy(r.device)
		r.outABuf.destroy(r.device)
		r.outBBuf.destroy(r.device)
		r.outCBuf.destroy(r.device)
		for _, cached := range r.dataBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.scaleBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatRowsBiasF32LinuxRunner() (*vulkanMatRowsBiasF32LinuxRunner, error) {
	vulkanMatRowsBiasF32LinuxRunnerCache.once.Do(func() {
		vulkanMatRowsBiasF32LinuxRunnerCache.runner, vulkanMatRowsBiasF32LinuxRunnerCache.err = newVulkanMatRowsBiasF32LinuxRunner()
	})
	return vulkanMatRowsBiasF32LinuxRunnerCache.runner, vulkanMatRowsBiasF32LinuxRunnerCache.err
}

type vulkanMatRowsBiasF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	addPipeline     vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outBuf          vulkanHostBuffer
	addBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [5]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandBatches  int
	commandRows     int
	commandCols     int
	commandAddRows  int
	mu              sync.Mutex
}

func newVulkanMatRowsBiasF32LinuxRunner() (*vulkanMatRowsBiasF32LinuxRunner, error) {
	spv, err := vulkanMatRowsBiasF32SPV()
	if err != nil {
		return nil, err
	}
	addSPV, err := vulkanMatRowsBiasAddRowsF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-matrows-bias-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatRowsBiasF32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 5)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 16}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	var addShader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(addSPV) * 4),
		PCode:    addSPV,
	}, nil, &addShader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule addrows: %s", res)
	}
	defer vk.DestroyShaderModule(device, addShader, nil)
	addPipelines := make([]vk.Pipeline, 1)
	addStage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: addShader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  addStage,
		Layout: r.pipelineLayout,
	}}, nil, addPipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines addrows: %s", res)
	}
	r.addPipeline = addPipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanMatRowsBiasF32LinuxRunner) run(out, xs [][]float32, w, bias []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias runner x length overflows: batches=%d cols=%d", batches, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matrows+bias runner")
	if err != nil {
		return err
	}
	outLen, ok := checkedMulInt(batches, rows)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias runner output length overflows: batches=%d rows=%d", batches, rows)
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(xLen, "Vulkan matrows+bias runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan matrows+bias runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan matrows+bias runner bias")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outLen, "Vulkan matrows+bias runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:rows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, xs, batches, cols); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandBatches != batches || r.commandRows != rows || r.commandCols != cols || r.commandAddRows != 0 {
		if err := r.recordCommand(batches, rows, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, batches, rows)
}

func (r *vulkanMatRowsBiasF32LinuxRunner) runAddRows(out, xs [][]float32, w, bias []float32, add [][]float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	addRows := len(add)
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias+addrows runner x length overflows: batches=%d cols=%d", batches, cols)
	}
	wLen, err := checkedMatVecF32WeightLenLinux(rows, cols, "Vulkan matrows+bias+addrows runner")
	if err != nil {
		return err
	}
	addLen, ok := checkedMulInt(addRows, rows)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias+addrows runner add length overflows: addRows=%d rows=%d", addRows, rows)
	}
	outLen, ok := checkedMulInt(batches, rows)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias+addrows runner output length overflows: batches=%d rows=%d", batches, rows)
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(xLen, "Vulkan matrows+bias+addrows runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrLinux(wLen, "Vulkan matrows+bias+addrows runner weight")
	if err != nil {
		return err
	}
	biasBytes, err := checkedFloat32ByteLenErrLinux(rows, "Vulkan matrows+bias+addrows runner bias")
	if err != nil {
		return err
	}
	addBytes, err := checkedFloat32ByteLenErrLinux(addLen, "Vulkan matrows+bias+addrows runner add")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outLen, "Vulkan matrows+bias+addrows runner output")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, addBytes); err != nil {
		return err
	}
	wBuf, err := r.cachedBuffer(w[:wLen], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:rows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, xs, batches, cols); err != nil {
		return err
	}
	if err := r.addBuf.writeRowsPrefix(r.device, add, addRows, rows); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: wBuf.buffer, Offset: 0, Range: wBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandBatches != batches || r.commandRows != rows || r.commandCols != cols || r.commandAddRows != addRows {
		if err := r.recordAddRowsCommand(batches, rows, cols, addRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, batches, rows)
}

func (r *vulkanMatRowsBiasF32LinuxRunner) recordCommand(batches, rows, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [12]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), uint32(batches), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandRows = rows
	r.commandCols = cols
	r.commandAddRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBiasF32LinuxRunner) recordAddRowsCommand(batches, rows, cols, addRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.addPipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(addRows))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), uint32(batches), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandRows = rows
	r.commandCols = cols
	r.commandAddRows = addRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBiasF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatRowsBiasF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanMatRowsBiasF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.addPipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.addPipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		r.addBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

const (
	vulkanLayerNormRowsLinuxModePlain = iota
	vulkanLayerNormRowsLinuxModeAdd
)

func getVulkanLayerNormRowsF32LinuxRunner() (*vulkanLayerNormRowsF32LinuxRunner, error) {
	vulkanLayerNormRowsF32LinuxRunnerCache.once.Do(func() {
		vulkanLayerNormRowsF32LinuxRunnerCache.runner, vulkanLayerNormRowsF32LinuxRunnerCache.err = newVulkanLayerNormRowsF32LinuxRunner()
	})
	return vulkanLayerNormRowsF32LinuxRunnerCache.runner, vulkanLayerNormRowsF32LinuxRunnerCache.err
}

type vulkanLayerNormRowsF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	addBuf          vulkanHostBuffer
	outBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [5]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandRows     int
	commandCols     int
	commandMode     int
	commandEps      uint32
	mu              sync.Mutex
}

func newVulkanLayerNormRowsF32LinuxRunner() (*vulkanLayerNormRowsF32LinuxRunner, error) {
	spv, err := vulkanLayerNormRowsF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-layernorm-rows-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanLayerNormRowsF32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 5)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 16}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanLayerNormRowsF32LinuxRunner) run(out, x, add [][]float32, weight, bias []float32, rows, cols, mode int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	bufLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return fmt.Errorf("Vulkan layernorm rows runner buffer length overflows: rows=%d cols=%d", rows, cols)
	}
	bufBytes, err := checkedFloat32ByteLenErrLinux(bufLen, "Vulkan layernorm rows runner buffer")
	if err != nil {
		return err
	}
	paramBytes, err := checkedFloat32ByteLenErrLinux(cols, "Vulkan layernorm rows runner params")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.addBuf, bufBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, bufBytes); err != nil {
		return err
	}
	weightBuf, err := r.cachedBuffer(weight[:cols], paramBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:cols], paramBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, x, rows, cols); err != nil {
		return err
	}
	if mode == vulkanLayerNormRowsLinuxModeAdd {
		if err := r.addBuf.writeRowsPrefix(r.device, add, rows, cols); err != nil {
			return err
		}
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: r.addBuf.buffer, Offset: 0, Range: r.addBuf.size},
		{Buffer: weightBuf.buffer, Offset: 0, Range: weightBuf.size},
		{Buffer: biasBuf.buffer, Offset: 0, Range: biasBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandRows != rows || r.commandCols != cols || r.commandMode != mode || r.commandEps != epsBits {
		if err := r.recordCommand(rows, cols, mode, eps); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, rows, cols)
}

func (r *vulkanLayerNormRowsF32LinuxRunner) recordCommand(rows, cols, mode int, eps float32) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [16]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(mode))
	binary.LittleEndian.PutUint32(pc[12:16], math.Float32bits(eps))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rows), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandMode = mode
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanLayerNormRowsF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanLayerNormRowsF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanLayerNormRowsF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.addBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatRowsBias3F32LinuxRunner() (*vulkanMatRowsBias3F32LinuxRunner, error) {
	vulkanMatRowsBias3F32LinuxRunnerCache.once.Do(func() {
		vulkanMatRowsBias3F32LinuxRunnerCache.runner, vulkanMatRowsBias3F32LinuxRunnerCache.err = newVulkanMatRowsBias3F32LinuxRunner()
	})
	return vulkanMatRowsBias3F32LinuxRunnerCache.runner, vulkanMatRowsBias3F32LinuxRunnerCache.err
}

type vulkanMatRowsBias3F32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	outABuf         vulkanHostBuffer
	outBBuf         vulkanHostBuffer
	outCBuf         vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [10]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandBatches  int
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	mu              sync.Mutex
}

func newVulkanMatRowsBias3F32LinuxRunner() (*vulkanMatRowsBias3F32LinuxRunner, error) {
	spv, err := vulkanMatRowsBias3F32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-matrows-bias3-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatRowsBias3F32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 10)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 20}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanMatRowsBias3F32LinuxRunner) run(outA, outB, outC, xs [][]float32, wa, ba, wb, bb, wc, bc []float32, rowsA, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	xLen, ok := checkedMulInt(batches, cols)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias3 runner x length overflows: batches=%d cols=%d", batches, cols)
	}
	waLen, err := checkedMatVecF32WeightLenLinux(rowsA, cols, "Vulkan matrows+bias3 runner wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenLinux(rowsB, cols, "Vulkan matrows+bias3 runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenLinux(rowsC, cols, "Vulkan matrows+bias3 runner wc")
	if err != nil {
		return err
	}
	outALen, ok := checkedMulInt(batches, rowsA)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias3 runner outA length overflows: batches=%d rowsA=%d", batches, rowsA)
	}
	outBLen, ok := checkedMulInt(batches, rowsB)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias3 runner outB length overflows: batches=%d rowsB=%d", batches, rowsB)
	}
	outCLen, ok := checkedMulInt(batches, rowsC)
	if !ok {
		return fmt.Errorf("Vulkan matrows+bias3 runner outC length overflows: batches=%d rowsC=%d", batches, rowsC)
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(xLen, "Vulkan matrows+bias3 runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrLinux(waLen, "Vulkan matrows+bias3 runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrLinux(wbLen, "Vulkan matrows+bias3 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrLinux(wcLen, "Vulkan matrows+bias3 runner wc")
	if err != nil {
		return err
	}
	baBytes, err := checkedFloat32ByteLenErrLinux(rowsA, "Vulkan matrows+bias3 runner ba")
	if err != nil {
		return err
	}
	bbBytes, err := checkedFloat32ByteLenErrLinux(rowsB, "Vulkan matrows+bias3 runner bb")
	if err != nil {
		return err
	}
	bcBytes, err := checkedFloat32ByteLenErrLinux(rowsC, "Vulkan matrows+bias3 runner bc")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrLinux(outALen, "Vulkan matrows+bias3 runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrLinux(outBLen, "Vulkan matrows+bias3 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrLinux(outCLen, "Vulkan matrows+bias3 runner outC")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, outABytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, outBBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, outCBytes); err != nil {
		return err
	}
	waBuf, err := r.cachedBuffer(wa[:waLen], waBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	baBuf, err := r.cachedBuffer(ba[:rowsA], baBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	wbBuf, err := r.cachedBuffer(wb[:wbLen], wbBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	bbBuf, err := r.cachedBuffer(bb[:rowsB], bbBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	wcBuf, err := r.cachedBuffer(wc[:wcLen], wcBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	bcBuf, err := r.cachedBuffer(bc[:rowsC], bcBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, xs, batches, cols); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: waBuf.buffer, Offset: 0, Range: waBuf.size},
		{Buffer: baBuf.buffer, Offset: 0, Range: baBuf.size},
		{Buffer: wbBuf.buffer, Offset: 0, Range: wbBuf.size},
		{Buffer: bbBuf.buffer, Offset: 0, Range: bbBuf.size},
		{Buffer: wcBuf.buffer, Offset: 0, Range: wcBuf.size},
		{Buffer: bcBuf.buffer, Offset: 0, Range: bcBuf.size},
		{Buffer: r.outABuf.buffer, Offset: 0, Range: r.outABuf.size},
		{Buffer: r.outBBuf.buffer, Offset: 0, Range: r.outBBuf.size},
		{Buffer: r.outCBuf.buffer, Offset: 0, Range: r.outCBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandBatches != batches || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(batches, rowsA, rowsB, rowsC, cols); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	if err := r.outABuf.readRowsPrefixInto(r.device, outA, batches, rowsA); err != nil {
		return err
	}
	if err := r.outBBuf.readRowsPrefixInto(r.device, outB, batches, rowsB); err != nil {
		return err
	}
	if err := r.outCBuf.readRowsPrefixInto(r.device, outC, batches, rowsC); err != nil {
		return err
	}
	return nil
}

func (r *vulkanMatRowsBias3F32LinuxRunner) recordCommand(batches, rowsA, rowsB, rowsC, cols int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rowsA))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(rowsB))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(rowsC))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(cols))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(rowsA+rowsB+rowsC), uint32(batches), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsBias3F32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatRowsBias3F32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanMatRowsBias3F32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.outABuf.destroy(r.device)
		r.outBBuf.destroy(r.device)
		r.outCBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanProjectImageF32LinuxRunner() (*vulkanProjectImageF32LinuxRunner, error) {
	vulkanProjectImageF32LinuxRunnerCache.once.Do(func() {
		vulkanProjectImageF32LinuxRunnerCache.runner, vulkanProjectImageF32LinuxRunnerCache.err = newVulkanProjectImageF32LinuxRunner()
	})
	return vulkanProjectImageF32LinuxRunnerCache.runner, vulkanProjectImageF32LinuxRunnerCache.err
}

type vulkanProjectImageF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	mergedBuf       vulkanHostBuffer
	hiddenBuf       vulkanHostBuffer
	outBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [10]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandBatches  int
	commandGridH    int
	commandGridW    int
	commandVision   int
	commandHidden   int
	commandOutRows  int
	commandEps      uint32
	mu              sync.Mutex
}

func newVulkanProjectImageF32LinuxRunner() (*vulkanProjectImageF32LinuxRunner, error) {
	spv, err := vulkanProjectImageF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-project-image-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanProjectImageF32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 10)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 32}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanProjectImageF32LinuxRunner) run(out, x [][]float32, normW, normB, w1, b1, w2, b2 []float32, batches, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	tokens := len(x)
	dims, err := checkedProjectImageDimsLinux(1, gridH, gridW, visionDim, hiddenRows, outRows, "Vulkan project image runner")
	if err != nil {
		return err
	}
	if dims.tokens == 0 || dims.batches == 0 || tokens%dims.tokens != 0 || batches%dims.batches != 0 || tokens/dims.tokens != batches/dims.batches {
		return fmt.Errorf("invalid Vulkan project image runner tokens=%d batches=%d gridH=%d gridW=%d", tokens, batches, gridH, gridW)
	}
	cols := dims.cols
	xElems, ok := checkedMulInt(tokens, visionDim)
	if !ok {
		return fmt.Errorf("Vulkan project image runner x length overflows: tokens=%d visionDim=%d", tokens, visionDim)
	}
	mergedElems, ok := checkedMulInt(batches, cols)
	if !ok {
		return fmt.Errorf("Vulkan project image runner merged length overflows: batches=%d cols=%d", batches, cols)
	}
	hiddenElems, ok := checkedMulInt(batches, hiddenRows)
	if !ok {
		return fmt.Errorf("Vulkan project image runner hidden length overflows: batches=%d hiddenRows=%d", batches, hiddenRows)
	}
	outElems, ok := checkedMulInt(batches, outRows)
	if !ok {
		return fmt.Errorf("Vulkan project image runner output length overflows: batches=%d outRows=%d", batches, outRows)
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(xElems, "Vulkan project image runner x")
	if err != nil {
		return err
	}
	paramBytes, err := checkedFloat32ByteLenErrLinux(visionDim, "Vulkan project image runner norm param")
	if err != nil {
		return err
	}
	mergedBytes, err := checkedFloat32ByteLenErrLinux(mergedElems, "Vulkan project image runner merged")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrLinux(hiddenElems, "Vulkan project image runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(outElems, "Vulkan project image runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrLinux(dims.w1Len, "Vulkan project image runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrLinux(hiddenRows, "Vulkan project image runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrLinux(dims.w2Len, "Vulkan project image runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan project image runner b2")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.mergedBuf, mergedBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	normWBuf, err := r.cachedBuffer(normW, paramBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	normBBuf, err := r.cachedBuffer(normB, paramBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w1Buf, err := r.cachedBuffer(w1, w1Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b1Buf, err := r.cachedBuffer(b1, b1Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w2Buf, err := r.cachedBuffer(w2, w2Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b2Buf, err := r.cachedBuffer(b2, b2Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, x, tokens, visionDim); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: normWBuf.buffer, Offset: 0, Range: normWBuf.size},
		{Buffer: normBBuf.buffer, Offset: 0, Range: normBBuf.size},
		{Buffer: w1Buf.buffer, Offset: 0, Range: w1Buf.size},
		{Buffer: b1Buf.buffer, Offset: 0, Range: b1Buf.size},
		{Buffer: w2Buf.buffer, Offset: 0, Range: w2Buf.size},
		{Buffer: b2Buf.buffer, Offset: 0, Range: b2Buf.size},
		{Buffer: r.mergedBuf.buffer, Offset: 0, Range: r.mergedBuf.size},
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandBatches != batches || r.commandGridH != gridH || r.commandGridW != gridW || r.commandVision != visionDim || r.commandHidden != hiddenRows || r.commandOutRows != outRows || r.commandEps != epsBits {
		if err := r.recordCommand(batches, gridH, gridW, visionDim, hiddenRows, outRows, eps); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, batches, outRows)
}

func (r *vulkanProjectImageF32LinuxRunner) recordCommand(batches, gridH, gridW, visionDim, hiddenRows, outRows int, eps float32) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
	}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [32]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(gridH))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(gridW))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(visionDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[20:24], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[24:28], 0)
	binary.LittleEndian.PutUint32(pc[28:32], math.Float32bits(eps))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, 4, uint32(batches), 1)
	barrier := []vk.MemoryBarrier{{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit),
	}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[24:28], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(hiddenRows), uint32(batches), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[24:28], 2)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), uint32(batches), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandGridH = gridH
	r.commandGridW = gridW
	r.commandVision = visionDim
	r.commandHidden = hiddenRows
	r.commandOutRows = outRows
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanProjectImageF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanProjectImageF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanProjectImageF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.mergedBuf.destroy(r.device)
		r.hiddenBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatRowsGELU2F32LinuxRunner() (*vulkanMatRowsGELU2F32LinuxRunner, error) {
	vulkanMatRowsGELU2F32LinuxRunnerCache.once.Do(func() {
		vulkanMatRowsGELU2F32LinuxRunnerCache.runner, vulkanMatRowsGELU2F32LinuxRunnerCache.err = newVulkanMatRowsGELU2F32LinuxRunner()
	})
	return vulkanMatRowsGELU2F32LinuxRunnerCache.runner, vulkanMatRowsGELU2F32LinuxRunnerCache.err
}

type vulkanMatRowsGELU2F32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	hiddenBuf       vulkanHostBuffer
	outBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [7]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandBatches  int
	commandHidden   int
	commandCols     int
	commandOutRows  int
	mu              sync.Mutex
}

func newVulkanMatRowsGELU2F32LinuxRunner() (*vulkanMatRowsGELU2F32LinuxRunner, error) {
	spv, err := vulkanMatRowsGELU2F32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-matrows-gelu2-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatRowsGELU2F32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 7)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 20}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanMatRowsGELU2F32LinuxRunner) run(out, xs [][]float32, w1, b1, w2, b2 []float32, hiddenRows, cols, outRows int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(xs)
	dims, err := checkedMatRowsGELU2DimsLinux(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2 runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(dims.xLen, "Vulkan matrows gelu2 runner x")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrLinux(dims.hiddenLen, "Vulkan matrows gelu2 runner hidden")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(dims.outLen, "Vulkan matrows gelu2 runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrLinux(dims.w1Len, "Vulkan matrows gelu2 runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrLinux(hiddenRows, "Vulkan matrows gelu2 runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrLinux(dims.w2Len, "Vulkan matrows gelu2 runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan matrows gelu2 runner b2")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	w1Buf, err := r.cachedBuffer(w1[:dims.w1Len], w1Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b1Buf, err := r.cachedBuffer(b1[:hiddenRows], b1Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w2Buf, err := r.cachedBuffer(w2[:dims.w2Len], w2Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b2Buf, err := r.cachedBuffer(b2[:outRows], b2Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, xs, batches, cols); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: w1Buf.buffer, Offset: 0, Range: w1Buf.size},
		{Buffer: b1Buf.buffer, Offset: 0, Range: b1Buf.size},
		{Buffer: w2Buf.buffer, Offset: 0, Range: w2Buf.size},
		{Buffer: b2Buf.buffer, Offset: 0, Range: b2Buf.size},
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	if !r.commandRecorded || r.commandBatches != batches || r.commandHidden != hiddenRows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(batches, hiddenRows, cols, outRows); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, batches, outRows)
}

func (r *vulkanMatRowsGELU2F32LinuxRunner) recordCommand(batches, hiddenRows, cols, outRows int) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [20]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(hiddenRows), uint32(batches), 1)
	barrier := []vk.MemoryBarrier{{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit),
	}}
	vk.CmdPipelineBarrier(
		cmd,
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit),
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit),
		0,
		uint32(len(barrier)),
		barrier,
		0,
		nil,
		0,
		nil,
	)
	binary.LittleEndian.PutUint32(pc[16:20], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), uint32(batches), 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandHidden = hiddenRows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsGELU2F32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatRowsGELU2F32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanMatRowsGELU2F32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.hiddenBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func getVulkanMatRowsGELU2AddLayerNormF32LinuxRunner() (*vulkanMatRowsGELU2AddLayerNormF32LinuxRunner, error) {
	vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache.once.Do(func() {
		vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache.runner, vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache.err = newVulkanMatRowsGELU2AddLayerNormF32LinuxRunner()
	})
	return vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache.runner, vulkanMatRowsGELU2AddLayerNormF32LinuxRunnerCache.err
}

type vulkanMatRowsGELU2AddLayerNormF32LinuxRunner struct {
	instance        vk.Instance
	device          vk.Device
	queue           vk.Queue
	queueFamily     uint32
	memProps        vk.PhysicalDeviceMemoryProperties
	setLayout       vk.DescriptorSetLayout
	descriptorPool  vk.DescriptorPool
	descriptorSet   vk.DescriptorSet
	pipelineLayout  vk.PipelineLayout
	pipeline        vk.Pipeline
	commandPool     vk.CommandPool
	commandBuffer   vk.CommandBuffer
	fence           vk.Fence
	xBuf            vulkanHostBuffer
	residualBuf     vulkanHostBuffer
	hiddenBuf       vulkanHostBuffer
	mlpBuf          vulkanHostBuffer
	outBuf          vulkanHostBuffer
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferLinux
	biasBuffers     map[uintptr]vulkanCachedFloat32BufferLinux
	descriptorCache [11]vulkanDescriptorBindingLinux
	commandRecorded bool
	commandBatches  int
	commandHidden   int
	commandCols     int
	commandOutRows  int
	commandEps      uint32
	mu              sync.Mutex
}

func newVulkanMatRowsGELU2AddLayerNormF32LinuxRunner() (*vulkanMatRowsGELU2AddLayerNormF32LinuxRunner, error) {
	spv, err := vulkanMatRowsGELU2AddLayerNormF32SPV()
	if err != nil {
		return nil, err
	}
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan init: %w", err)
	}
	app := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   "rapidocrvl-vulkan-matrows-gelu2-add-layernorm-f32\x00",
		ApplicationVersion: vk.MakeVersion(0, 1, 0),
		PEngineName:        "rapidocrvl\x00",
		EngineVersion:      vk.MakeVersion(0, 1, 0),
		ApiVersion:         vk.MakeVersion(1, 0, 0),
	}
	ici := vk.InstanceCreateInfo{SType: vk.StructureTypeInstanceCreateInfo, PApplicationInfo: &app}
	var instance vk.Instance
	if res := vk.CreateInstance(&ici, nil, &instance); res != vk.Success {
		return nil, fmt.Errorf("vkCreateInstance: %s", res)
	}
	r := &vulkanMatRowsGELU2AddLayerNormF32LinuxRunner{
		instance:      instance,
		weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferLinux),
		biasBuffers:   make(map[uintptr]vulkanCachedFloat32BufferLinux),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	if err := vk.InitInstance(instance); err != nil {
		return nil, fmt.Errorf("vulkan init instance: %w", err)
	}
	var gpuCount uint32
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, nil); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices count: %s", res)
	}
	if gpuCount == 0 {
		return nil, fmt.Errorf("no Vulkan physical devices")
	}
	gpus := make([]vk.PhysicalDevice, gpuCount)
	if res := vk.EnumeratePhysicalDevices(instance, &gpuCount, gpus); res != vk.Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices: %s", res)
	}
	gpu, queueFamily, memProps, err := selectVulkanComputeDevice(gpus)
	if err != nil {
		return nil, err
	}
	priority := []float32{1}
	qci := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: priority,
	}
	dci := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    []vk.DeviceQueueCreateInfo{qci},
	}
	var device vk.Device
	if res := vk.CreateDevice(gpu, &dci, nil, &device); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDevice: %s", res)
	}
	r.device = device
	r.queueFamily = queueFamily
	r.memProps = memProps
	vk.GetDeviceQueue(device, queueFamily, 0, &r.queue)

	bindings := make([]vk.DescriptorSetLayoutBinding, 11)
	for i := range bindings {
		bindings[i] = vk.DescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageComputeBit}
	}
	if res := vk.CreateDescriptorSetLayout(device, &vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    bindings,
	}, nil, &r.setLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %s", res)
	}
	poolSize := []vk.DescriptorPoolSize{{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}}
	if res := vk.CreateDescriptorPool(device, &vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: uint32(len(poolSize)),
		PPoolSizes:    poolSize,
	}, nil, &r.descriptorPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %s", res)
	}
	if res := vk.AllocateDescriptorSets(device, &vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        []vk.DescriptorSetLayout{r.setLayout},
	}, &r.descriptorSet); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %s", res)
	}
	pushRanges := []vk.PushConstantRange{{StageFlags: vk.ShaderStageComputeBit, Offset: 0, Size: 24}}
	if res := vk.CreatePipelineLayout(device, &vk.PipelineLayoutCreateInfo{
		SType:                  vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            []vk.DescriptorSetLayout{r.setLayout},
		PushConstantRangeCount: uint32(len(pushRanges)),
		PPushConstantRanges:    pushRanges,
	}, nil, &r.pipelineLayout); res != vk.Success {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %s", res)
	}
	var shader vk.ShaderModule
	if res := vk.CreateShaderModule(device, &vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uint(len(spv) * 4),
		PCode:    spv,
	}, nil, &shader); res != vk.Success {
		return nil, fmt.Errorf("vkCreateShaderModule: %s", res)
	}
	defer vk.DestroyShaderModule(device, shader, nil)
	pipelines := make([]vk.Pipeline, 1)
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: shader,
		PName:  "main\x00",
	}
	if res := vk.CreateComputePipelines(device, vk.NullPipelineCache, 1, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: r.pipelineLayout,
	}}, nil, pipelines); res != vk.Success {
		return nil, fmt.Errorf("vkCreateComputePipelines: %s", res)
	}
	r.pipeline = pipelines[0]
	if res := vk.CreateCommandPool(device, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: queueFamily,
	}, nil, &r.commandPool); res != vk.Success {
		return nil, fmt.Errorf("vkCreateCommandPool: %s", res)
	}
	cmds := make([]vk.CommandBuffer, 1)
	if res := vk.AllocateCommandBuffers(device, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds); res != vk.Success {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %s", res)
	}
	r.commandBuffer = cmds[0]
	if res := vk.CreateFence(device, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &r.fence); res != vk.Success {
		return nil, fmt.Errorf("vkCreateFence: %s", res)
	}
	success = true
	return r, nil
}

func (r *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner) run(out, x, residual [][]float32, w1, b1, w2, b2, normW, normB []float32, hiddenRows, cols, outRows int, eps float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	batches := len(x)
	dims, err := checkedMatRowsGELU2DimsLinux(batches, hiddenRows, cols, outRows, "Vulkan matrows gelu2+add layernorm runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrLinux(dims.xLen, "Vulkan matrows gelu2+add layernorm runner x")
	if err != nil {
		return err
	}
	residualBytes, err := checkedFloat32ByteLenErrLinux(dims.outLen, "Vulkan matrows gelu2+add layernorm runner residual")
	if err != nil {
		return err
	}
	hiddenBytes, err := checkedFloat32ByteLenErrLinux(dims.hiddenLen, "Vulkan matrows gelu2+add layernorm runner hidden")
	if err != nil {
		return err
	}
	mlpBytes, err := checkedFloat32ByteLenErrLinux(dims.outLen, "Vulkan matrows gelu2+add layernorm runner mlp")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrLinux(dims.outLen, "Vulkan matrows gelu2+add layernorm runner output")
	if err != nil {
		return err
	}
	w1Bytes, err := checkedFloat32ByteLenErrLinux(dims.w1Len, "Vulkan matrows gelu2+add layernorm runner w1")
	if err != nil {
		return err
	}
	b1Bytes, err := checkedFloat32ByteLenErrLinux(hiddenRows, "Vulkan matrows gelu2+add layernorm runner b1")
	if err != nil {
		return err
	}
	w2Bytes, err := checkedFloat32ByteLenErrLinux(dims.w2Len, "Vulkan matrows gelu2+add layernorm runner w2")
	if err != nil {
		return err
	}
	b2Bytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan matrows gelu2+add layernorm runner b2")
	if err != nil {
		return err
	}
	normBytes, err := checkedFloat32ByteLenErrLinux(outRows, "Vulkan matrows gelu2+add layernorm runner norm")
	if err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.residualBuf, residualBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.hiddenBuf, hiddenBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.mlpBuf, mlpBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBuf, outBytes); err != nil {
		return err
	}
	w1Buf, err := r.cachedBuffer(w1, w1Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b1Buf, err := r.cachedBuffer(b1, b1Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	w2Buf, err := r.cachedBuffer(w2, w2Bytes, r.weightBuffers)
	if err != nil {
		return err
	}
	b2Buf, err := r.cachedBuffer(b2, b2Bytes, r.biasBuffers)
	if err != nil {
		return err
	}
	normWBuf, err := r.cachedBuffer(normW, normBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	normBBuf, err := r.cachedBuffer(normB, normBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.xBuf.writeRowsPrefix(r.device, x, batches, cols); err != nil {
		return err
	}
	if err := r.residualBuf.writeRowsPrefix(r.device, residual, batches, outRows); err != nil {
		return err
	}
	bufferInfos := [...]vk.DescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Offset: 0, Range: r.xBuf.size},
		{Buffer: r.residualBuf.buffer, Offset: 0, Range: r.residualBuf.size},
		{Buffer: w1Buf.buffer, Offset: 0, Range: w1Buf.size},
		{Buffer: b1Buf.buffer, Offset: 0, Range: b1Buf.size},
		{Buffer: w2Buf.buffer, Offset: 0, Range: w2Buf.size},
		{Buffer: b2Buf.buffer, Offset: 0, Range: b2Buf.size},
		{Buffer: normWBuf.buffer, Offset: 0, Range: normWBuf.size},
		{Buffer: normBBuf.buffer, Offset: 0, Range: normBBuf.size},
		{Buffer: r.hiddenBuf.buffer, Offset: 0, Range: r.hiddenBuf.size},
		{Buffer: r.mlpBuf.buffer, Offset: 0, Range: r.mlpBuf.size},
		{Buffer: r.outBuf.buffer, Offset: 0, Range: r.outBuf.size},
	}
	updateVulkanDescriptorBuffersLinux(r.device, r.descriptorSet, r.descriptorCache[:], bufferInfos[:])
	epsBits := math.Float32bits(eps)
	if !r.commandRecorded || r.commandBatches != batches || r.commandHidden != hiddenRows || r.commandCols != cols || r.commandOutRows != outRows || r.commandEps != epsBits {
		if err := r.recordCommand(batches, hiddenRows, cols, outRows, eps); err != nil {
			return err
		}
	}
	if res := vk.ResetFences(r.device, 1, []vk.Fence{r.fence}); res != vk.Success {
		return fmt.Errorf("vkResetFences: %s", res)
	}
	cmd := r.commandBuffer
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: []vk.CommandBuffer{cmd}}
	if res := vk.QueueSubmit(r.queue, 1, []vk.SubmitInfo{submit}, r.fence); res != vk.Success {
		return fmt.Errorf("vkQueueSubmit: %s", res)
	}
	if res := vk.WaitForFences(r.device, 1, []vk.Fence{r.fence}, vk.True, math.MaxUint64); res != vk.Success {
		return fmt.Errorf("vkWaitForFences: %s", res)
	}
	return r.outBuf.readRowsPrefixInto(r.device, out, batches, outRows)
}

func (r *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner) recordCommand(batches, hiddenRows, cols, outRows int, eps float32) error {
	if res := vk.ResetCommandPool(r.device, r.commandPool, 0); res != vk.Success {
		return fmt.Errorf("vkResetCommandPool: %s", res)
	}
	cmd := r.commandBuffer
	if res := vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo}); res != vk.Success {
		return fmt.Errorf("vkBeginCommandBuffer: %s", res)
	}
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointCompute, r.pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointCompute, r.pipelineLayout, 0, 1, []vk.DescriptorSet{r.descriptorSet}, 0, nil)
	var pc [24]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(batches))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(hiddenRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(cols))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	binary.LittleEndian.PutUint32(pc[20:24], math.Float32bits(eps))
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(hiddenRows), uint32(batches), 1)
	barrier := []vk.MemoryBarrier{{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit),
	}}
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[16:20], 1)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(outRows), uint32(batches), 1)
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit), 0, uint32(len(barrier)), barrier, 0, nil, 0, nil)
	binary.LittleEndian.PutUint32(pc[16:20], 2)
	vk.CmdPushConstants(cmd, r.pipelineLayout, vk.ShaderStageComputeBit, 0, uint32(len(pc)), unsafe.Pointer(&pc[0]))
	vk.CmdDispatch(cmd, uint32(batches), 1, 1)
	if res := vk.EndCommandBuffer(cmd); res != vk.Success {
		return fmt.Errorf("vkEndCommandBuffer: %s", res)
	}
	r.commandBatches = batches
	r.commandHidden = hiddenRows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandEps = math.Float32bits(eps)
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner) ensureHostBuffer(buf *vulkanHostBuffer, size vk.DeviceSize) error {
	if buf.buffer != vk.NullBuffer && buf.size >= size {
		return nil
	}
	if buf.buffer != vk.NullBuffer || buf.memory != vk.NullDeviceMemory {
		buf.destroy(r.device)
		*buf = vulkanHostBuffer{}
	}
	next, err := newVulkanHostBuffer(r.device, r.memProps, size, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner) cachedBuffer(data []float32, size vk.DeviceSize, cache map[uintptr]vulkanCachedFloat32BufferLinux) (vulkanHostBuffer, error) {
	return cachedFloat32BufferLinux(r.device, r.memProps, data, size, cache)
}

func (r *vulkanMatRowsGELU2AddLayerNormF32LinuxRunner) destroy() {
	if r == nil {
		return
	}
	if r.device != nil {
		if r.pipeline != vk.NullPipeline {
			vk.DestroyPipeline(r.device, r.pipeline, nil)
		}
		if r.fence != vk.NullFence {
			vk.DestroyFence(r.device, r.fence, nil)
		}
		if r.commandPool != vk.NullCommandPool {
			vk.DestroyCommandPool(r.device, r.commandPool, nil)
		}
		r.xBuf.destroy(r.device)
		r.residualBuf.destroy(r.device)
		r.hiddenBuf.destroy(r.device)
		r.mlpBuf.destroy(r.device)
		r.outBuf.destroy(r.device)
		for _, cached := range r.weightBuffers {
			cached.buffer.destroy(r.device)
		}
		for _, cached := range r.biasBuffers {
			cached.buffer.destroy(r.device)
		}
		if r.descriptorPool != vk.NullDescriptorPool {
			vk.DestroyDescriptorPool(r.device, r.descriptorPool, nil)
		}
		if r.pipelineLayout != vk.NullPipelineLayout {
			vk.DestroyPipelineLayout(r.device, r.pipelineLayout, nil)
		}
		if r.setLayout != vk.NullDescriptorSetLayout {
			vk.DestroyDescriptorSetLayout(r.device, r.setLayout, nil)
		}
		vk.DestroyDevice(r.device, nil)
	}
	if r.instance != nil {
		vk.DestroyInstance(r.instance, nil)
	}
}

func vulkanMatVecF32SPV() ([]uint32, error) {
	vulkanMatVecF32SPVCache.once.Do(func() {
		vulkanMatVecF32SPVCache.spv, vulkanMatVecF32SPVCache.err = compileVulkanGLSL(vulkanMatVecF32GLSL)
	})
	return vulkanMatVecF32SPVCache.spv, vulkanMatVecF32SPVCache.err
}

func vulkanArgmaxF32SPV() ([]uint32, error) {
	vulkanArgmaxF32SPVCache.once.Do(func() {
		vulkanArgmaxF32SPVCache.spv, vulkanArgmaxF32SPVCache.err = compileVulkanGLSL(vulkanArgmaxF32GLSL)
	})
	return vulkanArgmaxF32SPVCache.spv, vulkanArgmaxF32SPVCache.err
}

func vulkanArgmaxQuantizedF32SPV() ([]uint32, error) {
	vulkanArgmaxQuantizedF32SPVCache.once.Do(func() {
		vulkanArgmaxQuantizedF32SPVCache.spv, vulkanArgmaxQuantizedF32SPVCache.err = compileVulkanGLSL(vulkanArgmaxQuantizedF32GLSL)
	})
	return vulkanArgmaxQuantizedF32SPVCache.spv, vulkanArgmaxQuantizedF32SPVCache.err
}

func vulkanBlockTopKF32SPV() ([]uint32, error) {
	vulkanBlockTopKF32SPVCache.once.Do(func() {
		vulkanBlockTopKF32SPVCache.spv, vulkanBlockTopKF32SPVCache.err = compileVulkanGLSL(vulkanBlockTopKF32GLSL)
	})
	return vulkanBlockTopKF32SPVCache.spv, vulkanBlockTopKF32SPVCache.err
}

func vulkanBlockTopKQuantizedF32SPV() ([]uint32, error) {
	vulkanBlockTopKQuantizedF32SPVCache.once.Do(func() {
		vulkanBlockTopKQuantizedF32SPVCache.spv, vulkanBlockTopKQuantizedF32SPVCache.err = compileVulkanGLSL(vulkanBlockTopKQuantizedF32GLSL)
	})
	return vulkanBlockTopKQuantizedF32SPVCache.spv, vulkanBlockTopKQuantizedF32SPVCache.err
}

func vulkanRMSNormF32SPV() ([]uint32, error) {
	vulkanRMSNormF32SPVCache.once.Do(func() {
		vulkanRMSNormF32SPVCache.spv, vulkanRMSNormF32SPVCache.err = compileVulkanGLSL(vulkanRMSNormF32PlanGLSL)
	})
	return vulkanRMSNormF32SPVCache.spv, vulkanRMSNormF32SPVCache.err
}

func vulkanAddRMSNormF32SPV() ([]uint32, error) {
	vulkanAddRMSNormF32SPVCache.once.Do(func() {
		vulkanAddRMSNormF32SPVCache.spv, vulkanAddRMSNormF32SPVCache.err = compileVulkanGLSL(vulkanAddRMSNormF32GLSL)
	})
	return vulkanAddRMSNormF32SPVCache.spv, vulkanAddRMSNormF32SPVCache.err
}

func vulkanMatVecAddRMSNormF32SPV() ([]uint32, error) {
	vulkanMatVecAddRMSNormF32SPVCache.once.Do(func() {
		vulkanMatVecAddRMSNormF32SPVCache.spv, vulkanMatVecAddRMSNormF32SPVCache.err = compileVulkanGLSL(vulkanMatVecAddRMSNormF32GLSL)
	})
	return vulkanMatVecAddRMSNormF32SPVCache.spv, vulkanMatVecAddRMSNormF32SPVCache.err
}

func vulkanMatVecQ8AddRMSNormF32SPV() ([]uint32, error) {
	vulkanMatVecQ8AddRMSNormF32SPVCache.once.Do(func() {
		vulkanMatVecQ8AddRMSNormF32SPVCache.spv, vulkanMatVecQ8AddRMSNormF32SPVCache.err = compileVulkanGLSL(vulkanMatVecQ8AddRMSNormF32GLSL)
	})
	return vulkanMatVecQ8AddRMSNormF32SPVCache.spv, vulkanMatVecQ8AddRMSNormF32SPVCache.err
}

func vulkanMRoPEF32SPV() ([]uint32, error) {
	vulkanMRoPEF32SPVCache.once.Do(func() {
		vulkanMRoPEF32SPVCache.spv, vulkanMRoPEF32SPVCache.err = compileVulkanGLSL(vulkanMRoPEF32GLSL)
	})
	return vulkanMRoPEF32SPVCache.spv, vulkanMRoPEF32SPVCache.err
}

func vulkanMRoPEPairF32SPV() ([]uint32, error) {
	vulkanMRoPEPairF32SPVCache.once.Do(func() {
		vulkanMRoPEPairF32SPVCache.spv, vulkanMRoPEPairF32SPVCache.err = compileVulkanGLSL(vulkanMRoPEPairF32GLSL)
	})
	return vulkanMRoPEPairF32SPVCache.spv, vulkanMRoPEPairF32SPVCache.err
}

func vulkanFusedMatVec3F32SPV() ([]uint32, error) {
	vulkanFusedMatVec3F32SPVCache.once.Do(func() {
		vulkanFusedMatVec3F32SPVCache.spv, vulkanFusedMatVec3F32SPVCache.err = compileVulkanGLSL(vulkanFusedQKVF32GLSL)
	})
	return vulkanFusedMatVec3F32SPVCache.spv, vulkanFusedMatVec3F32SPVCache.err
}

func vulkanFusedMatVec2F32SPV() ([]uint32, error) {
	vulkanFusedMatVec2F32SPVCache.once.Do(func() {
		vulkanFusedMatVec2F32SPVCache.spv, vulkanFusedMatVec2F32SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2F32GLSLLinux)
	})
	return vulkanFusedMatVec2F32SPVCache.spv, vulkanFusedMatVec2F32SPVCache.err
}

func vulkanFusedMatVec3MRoPEF32SPV() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEF32SPVCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEF32SPVCache.spv, vulkanFusedMatVec3MRoPEF32SPVCache.err = compileVulkanGLSL(vulkanFusedQKVMRoPEF32GLSL)
	})
	return vulkanFusedMatVec3MRoPEF32SPVCache.spv, vulkanFusedMatVec3MRoPEF32SPVCache.err
}

func vulkanFusedMatVec2MRoPEF32SPV() ([]uint32, error) {
	vulkanFusedMatVec2MRoPEF32SPVCache.once.Do(func() {
		vulkanFusedMatVec2MRoPEF32SPVCache.spv, vulkanFusedMatVec2MRoPEF32SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2MRoPEF32GLSLLinux)
	})
	return vulkanFusedMatVec2MRoPEF32SPVCache.spv, vulkanFusedMatVec2MRoPEF32SPVCache.err
}

func vulkanSwiGLUGateUpF32SPV() ([]uint32, error) {
	vulkanSwiGLUGateUpF32SPVCache.once.Do(func() {
		vulkanSwiGLUGateUpF32SPVCache.spv, vulkanSwiGLUGateUpF32SPVCache.err = compileVulkanGLSL(vulkanFusedSwiGLUF32GLSL)
	})
	return vulkanSwiGLUGateUpF32SPVCache.spv, vulkanSwiGLUGateUpF32SPVCache.err
}

func vulkanVisionAttentionF32SPV() ([]uint32, error) {
	vulkanVisionAttentionF32SPVCache.once.Do(func() {
		vulkanVisionAttentionF32SPVCache.spv, vulkanVisionAttentionF32SPVCache.err = compileVulkanGLSL(vulkanVisionAttentionF32GLSL)
	})
	return vulkanVisionAttentionF32SPVCache.spv, vulkanVisionAttentionF32SPVCache.err
}

func vulkanVisionAttentionOutF32SPV() ([]uint32, error) {
	vulkanVisionAttentionOutF32SPVCache.once.Do(func() {
		vulkanVisionAttentionOutF32SPVCache.spv, vulkanVisionAttentionOutF32SPVCache.err = compileVulkanGLSL(vulkanVisionAttentionOutF32GLSL)
	})
	return vulkanVisionAttentionOutF32SPVCache.spv, vulkanVisionAttentionOutF32SPVCache.err
}

func vulkanVisionRoPEPairF32SPV() ([]uint32, error) {
	vulkanVisionRoPEPairF32SPVCache.once.Do(func() {
		vulkanVisionRoPEPairF32SPVCache.spv, vulkanVisionRoPEPairF32SPVCache.err = compileVulkanGLSL(vulkanVisionRoPEPairF32GLSL)
	})
	return vulkanVisionRoPEPairF32SPVCache.spv, vulkanVisionRoPEPairF32SPVCache.err
}

func vulkanVisionQKVF32SPV() ([]uint32, error) {
	vulkanVisionQKVF32SPVCache.once.Do(func() {
		vulkanVisionQKVF32SPVCache.spv, vulkanVisionQKVF32SPVCache.err = compileVulkanGLSL(vulkanVisionQKVF32GLSL)
	})
	return vulkanVisionQKVF32SPVCache.spv, vulkanVisionQKVF32SPVCache.err
}

func vulkanTextAttentionF32SPV() ([]uint32, error) {
	vulkanTextAttentionF32SPVCache.once.Do(func() {
		vulkanTextAttentionF32SPVCache.spv, vulkanTextAttentionF32SPVCache.err = compileVulkanGLSL(vulkanTextAttentionF32GLSL)
	})
	return vulkanTextAttentionF32SPVCache.spv, vulkanTextAttentionF32SPVCache.err
}

func vulkanTextAttentionOutF32SPV() ([]uint32, error) {
	vulkanTextAttentionOutF32SPVCache.once.Do(func() {
		vulkanTextAttentionOutF32SPVCache.spv, vulkanTextAttentionOutF32SPVCache.err = compileVulkanGLSL(vulkanTextAttentionOutF32GLSL)
	})
	return vulkanTextAttentionOutF32SPVCache.spv, vulkanTextAttentionOutF32SPVCache.err
}

func vulkanTextAttentionOutAddRMSNormF32SPV() ([]uint32, error) {
	vulkanTextAttentionOutAddRMSNormF32SPVCache.once.Do(func() {
		vulkanTextAttentionOutAddRMSNormF32SPVCache.spv, vulkanTextAttentionOutAddRMSNormF32SPVCache.err = compileVulkanGLSL(vulkanTextAttentionOutAddRMSNormF32GLSL)
	})
	return vulkanTextAttentionOutAddRMSNormF32SPVCache.spv, vulkanTextAttentionOutAddRMSNormF32SPVCache.err
}

func vulkanTextFirstTokenValueOutF32SPV() ([]uint32, error) {
	vulkanTextFirstTokenValueOutF32SPVCache.once.Do(func() {
		vulkanTextFirstTokenValueOutF32SPVCache.spv, vulkanTextFirstTokenValueOutF32SPVCache.err = compileVulkanGLSL(vulkanTextFirstTokenValueOutF32GLSLLinux)
	})
	return vulkanTextFirstTokenValueOutF32SPVCache.spv, vulkanTextFirstTokenValueOutF32SPVCache.err
}

func vulkanTextFirstTokenValueOutQ8SPV() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ8SPVCache.once.Do(func() {
		vulkanTextFirstTokenValueOutQ8SPVCache.spv, vulkanTextFirstTokenValueOutQ8SPVCache.err = compileVulkanGLSL(vulkanTextFirstTokenValueOutQ8GLSLLinux)
	})
	return vulkanTextFirstTokenValueOutQ8SPVCache.spv, vulkanTextFirstTokenValueOutQ8SPVCache.err
}

func vulkanTextFirstTokenValueOutQ6SPV() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ6SPVCache.once.Do(func() {
		vulkanTextFirstTokenValueOutQ6SPVCache.spv, vulkanTextFirstTokenValueOutQ6SPVCache.err = compileVulkanGLSL(vulkanTextFirstTokenValueOutQ6GLSLLinux)
	})
	return vulkanTextFirstTokenValueOutQ6SPVCache.spv, vulkanTextFirstTokenValueOutQ6SPVCache.err
}

func vulkanTextFirstTokenValueOutQ4SPV() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ4SPVCache.once.Do(func() {
		vulkanTextFirstTokenValueOutQ4SPVCache.spv, vulkanTextFirstTokenValueOutQ4SPVCache.err = compileVulkanGLSL(vulkanTextFirstTokenValueOutQ4GLSLLinux)
	})
	return vulkanTextFirstTokenValueOutQ4SPVCache.spv, vulkanTextFirstTokenValueOutQ4SPVCache.err
}

func vulkanTextAttentionOutQ8SPV() ([]uint32, error) {
	vulkanTextAttentionOutQ8SPVCache.once.Do(func() {
		vulkanTextAttentionOutQ8SPVCache.spv, vulkanTextAttentionOutQ8SPVCache.err = compileVulkanGLSL(vulkanTextAttentionOutQ8GLSL)
	})
	return vulkanTextAttentionOutQ8SPVCache.spv, vulkanTextAttentionOutQ8SPVCache.err
}

func vulkanTextAttentionOutQ6SPV() ([]uint32, error) {
	vulkanTextAttentionOutQ6SPVCache.once.Do(func() {
		vulkanTextAttentionOutQ6SPVCache.spv, vulkanTextAttentionOutQ6SPVCache.err = compileVulkanGLSL(vulkanTextAttentionOutQ6GLSL)
	})
	return vulkanTextAttentionOutQ6SPVCache.spv, vulkanTextAttentionOutQ6SPVCache.err
}

func vulkanTextAttentionOutQ4SPV() ([]uint32, error) {
	vulkanTextAttentionOutQ4SPVCache.once.Do(func() {
		vulkanTextAttentionOutQ4SPVCache.spv, vulkanTextAttentionOutQ4SPVCache.err = compileVulkanGLSL(vulkanTextAttentionOutQ4GLSL)
	})
	return vulkanTextAttentionOutQ4SPVCache.spv, vulkanTextAttentionOutQ4SPVCache.err
}

func vulkanMatRowsBiasF32SPV() ([]uint32, error) {
	vulkanMatRowsBiasF32SPVCache.once.Do(func() {
		vulkanMatRowsBiasF32SPVCache.spv, vulkanMatRowsBiasF32SPVCache.err = compileVulkanGLSL(vulkanMatRowsBiasF32GLSL)
	})
	return vulkanMatRowsBiasF32SPVCache.spv, vulkanMatRowsBiasF32SPVCache.err
}

func vulkanMatRowsBiasAddRowsF32SPV() ([]uint32, error) {
	vulkanMatRowsBiasAddRowsF32SPVCache.once.Do(func() {
		vulkanMatRowsBiasAddRowsF32SPVCache.spv, vulkanMatRowsBiasAddRowsF32SPVCache.err = compileVulkanGLSL(vulkanMatRowsBiasAddRowsF32GLSL)
	})
	return vulkanMatRowsBiasAddRowsF32SPVCache.spv, vulkanMatRowsBiasAddRowsF32SPVCache.err
}

func vulkanMatRowsBias3F32SPV() ([]uint32, error) {
	vulkanMatRowsBias3F32SPVCache.once.Do(func() {
		vulkanMatRowsBias3F32SPVCache.spv, vulkanMatRowsBias3F32SPVCache.err = compileVulkanGLSL(vulkanMatRowsBias3F32GLSL)
	})
	return vulkanMatRowsBias3F32SPVCache.spv, vulkanMatRowsBias3F32SPVCache.err
}

func vulkanMatRowsGELU2F32SPV() ([]uint32, error) {
	vulkanMatRowsGELU2F32SPVCache.once.Do(func() {
		vulkanMatRowsGELU2F32SPVCache.spv, vulkanMatRowsGELU2F32SPVCache.err = compileVulkanGLSL(vulkanMatRowsGELU2F32GLSL)
	})
	return vulkanMatRowsGELU2F32SPVCache.spv, vulkanMatRowsGELU2F32SPVCache.err
}

func vulkanMatRowsGELU2AddLayerNormF32SPV() ([]uint32, error) {
	vulkanMatRowsGELU2AddLayerNormF32SPVCache.once.Do(func() {
		vulkanMatRowsGELU2AddLayerNormF32SPVCache.spv, vulkanMatRowsGELU2AddLayerNormF32SPVCache.err = compileVulkanGLSL(vulkanMatRowsGELU2AddLayerNormF32GLSL)
	})
	return vulkanMatRowsGELU2AddLayerNormF32SPVCache.spv, vulkanMatRowsGELU2AddLayerNormF32SPVCache.err
}

func vulkanProjectImageF32SPV() ([]uint32, error) {
	vulkanProjectImageF32SPVCache.once.Do(func() {
		vulkanProjectImageF32SPVCache.spv, vulkanProjectImageF32SPVCache.err = compileVulkanGLSL(vulkanProjectImageF32GLSL)
	})
	return vulkanProjectImageF32SPVCache.spv, vulkanProjectImageF32SPVCache.err
}

func vulkanLayerNormRowsF32SPV() ([]uint32, error) {
	vulkanLayerNormRowsF32SPVCache.once.Do(func() {
		vulkanLayerNormRowsF32SPVCache.spv, vulkanLayerNormRowsF32SPVCache.err = compileVulkanGLSL(vulkanLayerNormRowsF32GLSL)
	})
	return vulkanLayerNormRowsF32SPVCache.spv, vulkanLayerNormRowsF32SPVCache.err
}

func vulkanMatVecQ8SPV() ([]uint32, error) {
	vulkanMatVecQ8SPVCache.once.Do(func() {
		vulkanMatVecQ8SPVCache.spv, vulkanMatVecQ8SPVCache.err = compileVulkanGLSL(vulkanMatVecQ8GLSL)
	})
	return vulkanMatVecQ8SPVCache.spv, vulkanMatVecQ8SPVCache.err
}

func vulkanFusedMatVec3Q8SPV() ([]uint32, error) {
	vulkanFusedMatVec3Q8SPVCache.once.Do(func() {
		vulkanFusedMatVec3Q8SPVCache.spv, vulkanFusedMatVec3Q8SPVCache.err = compileVulkanGLSL(vulkanFusedQKVQ8GLSL)
	})
	return vulkanFusedMatVec3Q8SPVCache.spv, vulkanFusedMatVec3Q8SPVCache.err
}

func vulkanFusedMatVec2Q8SPV() ([]uint32, error) {
	vulkanFusedMatVec2Q8SPVCache.once.Do(func() {
		vulkanFusedMatVec2Q8SPVCache.spv, vulkanFusedMatVec2Q8SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2Q8GLSLLinux)
	})
	return vulkanFusedMatVec2Q8SPVCache.spv, vulkanFusedMatVec2Q8SPVCache.err
}

func vulkanFusedMatVec3MRoPEQ8SPV() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEQ8SPVCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEQ8SPVCache.spv, vulkanFusedMatVec3MRoPEQ8SPVCache.err = compileVulkanGLSL(vulkanFusedQKVMRoPEQ8GLSL)
	})
	return vulkanFusedMatVec3MRoPEQ8SPVCache.spv, vulkanFusedMatVec3MRoPEQ8SPVCache.err
}

func vulkanFusedMatVec2MRoPEQ8SPV() ([]uint32, error) {
	vulkanFusedMatVec2MRoPEQ8SPVCache.once.Do(func() {
		vulkanFusedMatVec2MRoPEQ8SPVCache.spv, vulkanFusedMatVec2MRoPEQ8SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2MRoPEQ8GLSLLinux)
	})
	return vulkanFusedMatVec2MRoPEQ8SPVCache.spv, vulkanFusedMatVec2MRoPEQ8SPVCache.err
}

func vulkanSwiGLUGateUpQ8SPV() ([]uint32, error) {
	vulkanSwiGLUGateUpQ8SPVCache.once.Do(func() {
		vulkanSwiGLUGateUpQ8SPVCache.spv, vulkanSwiGLUGateUpQ8SPVCache.err = compileVulkanGLSL(vulkanFusedSwiGLUQ8GLSL)
	})
	return vulkanSwiGLUGateUpQ8SPVCache.spv, vulkanSwiGLUGateUpQ8SPVCache.err
}

func vulkanSwiGLUGateUpQ4SPV() ([]uint32, error) {
	vulkanSwiGLUGateUpQ4SPVCache.once.Do(func() {
		vulkanSwiGLUGateUpQ4SPVCache.spv, vulkanSwiGLUGateUpQ4SPVCache.err = compileVulkanGLSL(vulkanFusedSwiGLUQ4GLSL)
	})
	return vulkanSwiGLUGateUpQ4SPVCache.spv, vulkanSwiGLUGateUpQ4SPVCache.err
}

func vulkanSwiGLUGateUpQ6SPV() ([]uint32, error) {
	vulkanSwiGLUGateUpQ6SPVCache.once.Do(func() {
		vulkanSwiGLUGateUpQ6SPVCache.spv, vulkanSwiGLUGateUpQ6SPVCache.err = compileVulkanGLSL(vulkanFusedSwiGLUQ6GLSL)
	})
	return vulkanSwiGLUGateUpQ6SPVCache.spv, vulkanSwiGLUGateUpQ6SPVCache.err
}

func vulkanMatVecQ4SPV() ([]uint32, error) {
	vulkanMatVecQ4SPVCache.once.Do(func() {
		vulkanMatVecQ4SPVCache.spv, vulkanMatVecQ4SPVCache.err = compileVulkanGLSL(vulkanMatVecQ4GLSL)
	})
	return vulkanMatVecQ4SPVCache.spv, vulkanMatVecQ4SPVCache.err
}

func vulkanMatVecQ6SPV() ([]uint32, error) {
	vulkanMatVecQ6SPVCache.once.Do(func() {
		vulkanMatVecQ6SPVCache.spv, vulkanMatVecQ6SPVCache.err = compileVulkanGLSL(vulkanMatVecQ6GLSL)
	})
	return vulkanMatVecQ6SPVCache.spv, vulkanMatVecQ6SPVCache.err
}

func vulkanFusedMatVec3Q4SPV() ([]uint32, error) {
	vulkanFusedMatVec3Q4SPVCache.once.Do(func() {
		vulkanFusedMatVec3Q4SPVCache.spv, vulkanFusedMatVec3Q4SPVCache.err = compileVulkanGLSL(vulkanFusedQKVQ4GLSL)
	})
	return vulkanFusedMatVec3Q4SPVCache.spv, vulkanFusedMatVec3Q4SPVCache.err
}

func vulkanFusedMatVec2Q4SPV() ([]uint32, error) {
	vulkanFusedMatVec2Q4SPVCache.once.Do(func() {
		vulkanFusedMatVec2Q4SPVCache.spv, vulkanFusedMatVec2Q4SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2Q4GLSLLinux)
	})
	return vulkanFusedMatVec2Q4SPVCache.spv, vulkanFusedMatVec2Q4SPVCache.err
}

func vulkanFusedMatVec3Q6SPV() ([]uint32, error) {
	vulkanFusedMatVec3Q6SPVCache.once.Do(func() {
		vulkanFusedMatVec3Q6SPVCache.spv, vulkanFusedMatVec3Q6SPVCache.err = compileVulkanGLSL(vulkanFusedQKVQ6GLSL)
	})
	return vulkanFusedMatVec3Q6SPVCache.spv, vulkanFusedMatVec3Q6SPVCache.err
}

func vulkanFusedMatVec2Q6SPV() ([]uint32, error) {
	vulkanFusedMatVec2Q6SPVCache.once.Do(func() {
		vulkanFusedMatVec2Q6SPVCache.spv, vulkanFusedMatVec2Q6SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2Q6GLSLLinux)
	})
	return vulkanFusedMatVec2Q6SPVCache.spv, vulkanFusedMatVec2Q6SPVCache.err
}

func vulkanFusedMatVec3MRoPEQ4SPV() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEQ4SPVCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEQ4SPVCache.spv, vulkanFusedMatVec3MRoPEQ4SPVCache.err = compileVulkanGLSL(vulkanFusedQKVMRoPEQ4GLSL)
	})
	return vulkanFusedMatVec3MRoPEQ4SPVCache.spv, vulkanFusedMatVec3MRoPEQ4SPVCache.err
}

func vulkanFusedMatVec2MRoPEQ4SPV() ([]uint32, error) {
	vulkanFusedMatVec2MRoPEQ4SPVCache.once.Do(func() {
		vulkanFusedMatVec2MRoPEQ4SPVCache.spv, vulkanFusedMatVec2MRoPEQ4SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2MRoPEQ4GLSLLinux)
	})
	return vulkanFusedMatVec2MRoPEQ4SPVCache.spv, vulkanFusedMatVec2MRoPEQ4SPVCache.err
}

func vulkanFusedMatVec3MRoPEQ6SPV() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEQ6SPVCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEQ6SPVCache.spv, vulkanFusedMatVec3MRoPEQ6SPVCache.err = compileVulkanGLSL(vulkanFusedQKVMRoPEQ6GLSL)
	})
	return vulkanFusedMatVec3MRoPEQ6SPVCache.spv, vulkanFusedMatVec3MRoPEQ6SPVCache.err
}

func vulkanFusedMatVec2MRoPEQ6SPV() ([]uint32, error) {
	vulkanFusedMatVec2MRoPEQ6SPVCache.once.Do(func() {
		vulkanFusedMatVec2MRoPEQ6SPVCache.spv, vulkanFusedMatVec2MRoPEQ6SPVCache.err = compileVulkanGLSL(vulkanFusedMatVec2MRoPEQ6GLSLLinux)
	})
	return vulkanFusedMatVec2MRoPEQ6SPVCache.spv, vulkanFusedMatVec2MRoPEQ6SPVCache.err
}

type vulkanHostBuffer struct {
	buffer vk.Buffer
	memory vk.DeviceMemory
	size   vk.DeviceSize
	mapped unsafe.Pointer
}

func (b *vulkanHostBuffer) destroy(device vk.Device) {
	if b.mapped != nil {
		vk.UnmapMemory(device, b.memory)
		b.mapped = nil
	}
	if b.buffer != vk.NullBuffer {
		vk.DestroyBuffer(device, b.buffer, nil)
	}
	if b.memory != vk.NullDeviceMemory {
		vk.FreeMemory(device, b.memory, nil)
	}
}

func newVulkanHostBuffer(device vk.Device, memProps vk.PhysicalDeviceMemoryProperties, size vk.DeviceSize, usage vk.BufferUsageFlagBits) (vulkanHostBuffer, error) {
	var out vulkanHostBuffer
	out.size = size
	if res := vk.CreateBuffer(device, &vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       vk.BufferUsageFlags(usage),
		SharingMode: vk.SharingModeExclusive,
	}, nil, &out.buffer); res != vk.Success {
		return out, fmt.Errorf("vkCreateBuffer: %s", res)
	}
	var req vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(device, out.buffer, &req)
	memType, ok := findVulkanMemoryType(memProps, req.MemoryTypeBits, vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit)
	if !ok {
		out.destroy(device)
		return vulkanHostBuffer{}, fmt.Errorf("no host visible coherent memory type")
	}
	if res := vk.AllocateMemory(device, &vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  req.Size,
		MemoryTypeIndex: memType,
	}, nil, &out.memory); res != vk.Success {
		out.destroy(device)
		return vulkanHostBuffer{}, fmt.Errorf("vkAllocateMemory: %s", res)
	}
	if res := vk.BindBufferMemory(device, out.buffer, out.memory, 0); res != vk.Success {
		out.destroy(device)
		return vulkanHostBuffer{}, fmt.Errorf("vkBindBufferMemory: %s", res)
	}
	if res := vk.MapMemory(device, out.memory, 0, out.size, 0, &out.mapped); res != vk.Success {
		out.destroy(device)
		return vulkanHostBuffer{}, fmt.Errorf("vkMapMemory persistent: %s", res)
	}
	return out, nil
}

func (b vulkanHostBuffer) writeFloat32(device vk.Device, values []float32) error {
	return b.writeFloat32At(device, 0, values)
}

func (b vulkanHostBuffer) writeFloat32At(device vk.Device, offsetFloats int, values []float32) error {
	if offsetFloats < 0 || len(values) > maxInt()-offsetFloats {
		return fmt.Errorf("vulkan float32 write overflow offset=%d values=%d buffer=%d", offsetFloats, len(values), b.size)
	}
	total := offsetFloats + len(values)
	bytes, err := checkedFloat32ByteLenErrLinux(total, "vulkan float32 write")
	if err != nil || bytes > b.size {
		return fmt.Errorf("vulkan float32 write overflow offset=%d values=%d buffer=%d", offsetFloats, len(values), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan float32 write on unmapped buffer")
	}
	dst := unsafe.Slice((*float32)(b.mapped), int(b.size/4))
	copy(dst[offsetFloats:offsetFloats+len(values)], values)
	return nil
}

func (b vulkanHostBuffer) writeRowsPrefix(device vk.Device, rows [][]float32, n, cols int) error {
	total, ok := checkedMulInt(n, cols)
	if n < 0 || cols < 0 || !ok || len(rows) < n {
		return fmt.Errorf("vulkan rows write overflow rows=%d n=%d cols=%d buffer=%d", len(rows), n, cols, b.size)
	}
	bytes, err := checkedFloat32ByteLenErrLinux(total, "vulkan rows write")
	if err != nil || bytes > b.size {
		return fmt.Errorf("vulkan rows write overflow rows=%d n=%d cols=%d buffer=%d", len(rows), n, cols, b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan rows write on unmapped buffer")
	}
	dst := unsafe.Slice((*float32)(b.mapped), int(b.size/4))
	for i := 0; i < n; i++ {
		if len(rows[i]) < cols {
			return fmt.Errorf("vulkan rows write short row=%d len=%d cols=%d", i, len(rows[i]), cols)
		}
		copy(dst[i*cols:(i+1)*cols], rows[i][:cols])
	}
	return nil
}

func (b vulkanHostBuffer) writeInt8(device vk.Device, values []int8) error {
	if vk.DeviceSize(len(values)) > b.size {
		return fmt.Errorf("vulkan int8 write overflow values=%d buffer=%d", len(values), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan int8 write on unmapped buffer")
	}
	dst := unsafe.Slice((*int8)(b.mapped), int(b.size))
	clear(dst)
	copy(dst[:len(values)], values)
	return nil
}

func (b vulkanHostBuffer) writeBytes(device vk.Device, values []byte) error {
	if vk.DeviceSize(len(values)) > b.size {
		return fmt.Errorf("vulkan byte write overflow values=%d buffer=%d", len(values), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan byte write on unmapped buffer")
	}
	dst := unsafe.Slice((*byte)(b.mapped), int(b.size))
	clear(dst)
	copy(dst[:len(values)], values)
	return nil
}

func (b vulkanHostBuffer) readFloat32(device vk.Device, n int) ([]float32, error) {
	bytes, err := checkedFloat32ByteLenErrLinux(n, "vulkan float32 read")
	if err != nil || bytes > b.size {
		return nil, fmt.Errorf("vulkan float32 read overflow n=%d buffer=%d", n, b.size)
	}
	if b.mapped == nil {
		return nil, fmt.Errorf("vulkan float32 read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), n)
	out := make([]float32, n)
	copy(out, src)
	return out, nil
}

func (b vulkanHostBuffer) readFloat32Into(device vk.Device, out []float32) error {
	bytes, err := checkedFloat32ByteLenErrLinux(len(out), "vulkan float32 read")
	if err != nil || bytes > b.size {
		return fmt.Errorf("vulkan float32 read overflow out=%d buffer=%d", len(out), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan float32 read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), len(out))
	copy(out, src)
	return nil
}

func (b vulkanHostBuffer) readRowsPrefixInto(device vk.Device, out [][]float32, n, cols int) error {
	total, ok := checkedMulInt(n, cols)
	if n < 0 || cols < 0 || !ok || len(out) < n {
		return fmt.Errorf("vulkan rows read overflow rows=%d n=%d cols=%d buffer=%d", len(out), n, cols, b.size)
	}
	bytes, err := checkedFloat32ByteLenErrLinux(total, "vulkan rows read")
	if err != nil || bytes > b.size {
		return fmt.Errorf("vulkan rows read overflow rows=%d n=%d cols=%d buffer=%d", len(out), n, cols, b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vulkan rows read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), total)
	for i := 0; i < n; i++ {
		if len(out[i]) < cols {
			return fmt.Errorf("vulkan rows read short row=%d len=%d cols=%d", i, len(out[i]), cols)
		}
		copy(out[i][:cols], src[i*cols:(i+1)*cols])
	}
	return nil
}

func selectVulkanComputeDevice(gpus []vk.PhysicalDevice) (vk.PhysicalDevice, uint32, vk.PhysicalDeviceMemoryProperties, error) {
	for _, gpu := range gpus {
		var count uint32
		vk.GetPhysicalDeviceQueueFamilyProperties(gpu, &count, nil)
		if count == 0 {
			continue
		}
		props := make([]vk.QueueFamilyProperties, count)
		vk.GetPhysicalDeviceQueueFamilyProperties(gpu, &count, props)
		for i, p := range props {
			if p.QueueCount > 0 && p.QueueFlags&vk.QueueFlags(vk.QueueComputeBit) != 0 {
				var mem vk.PhysicalDeviceMemoryProperties
				vk.GetPhysicalDeviceMemoryProperties(gpu, &mem)
				return gpu, uint32(i), mem, nil
			}
		}
	}
	return nil, 0, vk.PhysicalDeviceMemoryProperties{}, fmt.Errorf("no Vulkan compute queue found")
}

func findVulkanMemoryType(mem vk.PhysicalDeviceMemoryProperties, typeBits uint32, flags vk.MemoryPropertyFlagBits) (uint32, bool) {
	want := vk.MemoryPropertyFlags(flags)
	for i := uint32(0); i < mem.MemoryTypeCount; i++ {
		if typeBits&(1<<i) != 0 && mem.MemoryTypes[i].PropertyFlags&want == want {
			return i, true
		}
	}
	return 0, false
}

func compileVulkanGLSL(source string) ([]uint32, error) {
	compiler, err := findVulkanShaderCompiler()
	if err != nil {
		return nil, err
	}
	dir, err := os.MkdirTemp("", "rapidocrvl-vulkan-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "probe.comp")
	dst := filepath.Join(dir, "probe.spv")
	if err := os.WriteFile(src, []byte(source), 0o600); err != nil {
		return nil, err
	}
	cmd := exec.Command(compiler, "-fshader-stage=compute", src, "-o", dst)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("%s failed: %w: %s", filepath.Base(compiler), err, string(out))
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		return nil, err
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("SPIR-V bytecode size %d is not uint32-aligned", len(raw))
	}
	words := make([]uint32, len(raw)/4)
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(raw[i*4 : i*4+4])
	}
	return words, nil
}

func findVulkanShaderCompiler() (string, error) {
	for _, name := range []string{"glslc", "glslangValidator"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("glslc or glslangValidator not found")
}

func float32ByteLen(n int) vk.DeviceSize {
	return vk.DeviceSize(n * 4)
}

func alignUpInt(v, alignment int) int {
	if rem := v % alignment; rem != 0 {
		return v + alignment - rem
	}
	return v
}

func float32SliceKeyLinux(v []float32) uintptr {
	if len(v) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&v[0]))
}

const vulkanMatVecAddRMSNormF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=2) readonly buffer A { float addv[]; };
layout(set=0,binding=3) buffer D { float dst[]; };
layout(set=0,binding=4) readonly buffer W { float w[]; };
layout(set=0,binding=5) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  for (uint i = lid; i < pc.rows; i += 256) {
    float v = dst[i] + addv[i];
    dst[i] = v;
    sum += v * v;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) {
      scratch[lid] += scratch[lid + stride];
    }
    barrier();
  }
  float scale = inversesqrt(scratch[0] / float(pc.rows) + 0.000001);
  for (uint i = lid; i < pc.rows; i += 256) outv[i] = dst[i] * scale * w[i];
}`

const vulkanMatVecQ8AddRMSNormF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=3) readonly buffer A { float addv[]; };
layout(set=0,binding=4) buffer D { float dst[]; };
layout(set=0,binding=5) readonly buffer W { float w[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  for (uint i = lid; i < pc.rows; i += 256) {
    float v = dst[i] + addv[i];
    dst[i] = v;
    sum += v * v;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) {
      scratch[lid] += scratch[lid + stride];
    }
    barrier();
  }
  float scale = inversesqrt(scratch[0] / float(pc.rows) + 0.000001);
  for (uint i = lid; i < pc.rows; i += 256) outv[i] = dst[i] * scale * w[i];
}`

const vulkanTextFirstTokenValueOutF32GLSLLinux = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint cacheLen; uint numHeads; uint kvHeads; uint headDim; uint kvDim; } pc;
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=4) readonly buffer W { float w[]; };
layout(set=0,binding=5) readonly buffer B { float bias[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint group = pc.numHeads / pc.kvHeads;
  uint tokenBase = (pc.cacheLen - 1u) * pc.kvDim;
  float sum = 0.0;
  for (uint c = lid; c < pc.numHeads * pc.headDim; c += 256) {
    uint srcHead = c / pc.headDim;
    uint elem = c - srcHead * pc.headDim;
    uint kvHead = srcHead / group;
    sum += w[row * pc.numHeads * pc.headDim + c] * v[tokenBase + kvHead * pc.headDim + elem];
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[row] = scratch[0] + bias[row];
}`

const vulkanTextFirstTokenValueOutQ8GLSLLinux = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint cacheLen; uint numHeads; uint kvHeads; uint headDim; uint kvDim; } pc;
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q8(uint idx) {
  uint word = packed[idx >> 2];
  uint shift = (idx & 3u) * 8u;
  uint v = (word >> shift) & 255u;
  return float(int(v << 24) >> 24);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint group = pc.numHeads / pc.kvHeads;
  uint cols = pc.numHeads * pc.headDim;
  uint tokenBase = (pc.cacheLen - 1u) * pc.kvDim;
  uint wBase = row * cols;
  float sum = 0.0;
  for (uint c = lid; c < cols; c += 256) {
    uint srcHead = c / pc.headDim;
    uint elem = c - srcHead * pc.headDim;
    uint kvHead = srcHead / group;
    sum += q8(wBase + c) * v[tokenBase + kvHead * pc.headDim + elem];
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanTextFirstTokenValueOutQ6GLSLLinux = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint cacheLen; uint numHeads; uint kvHeads; uint headDim; uint kvDim; } pc;
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
uint byteAt(uint byteIdx) {
  uint word = packed[byteIdx >> 2];
  uint shift = (byteIdx & 3u) * 8u;
  return (word >> shift) & 255u;
}
float q6(uint rowByteBase, uint packedCols, uint col) {
  uint bit = col * 6u;
  uint idx = rowByteBase + (bit >> 3);
  uint shift = bit & 7u;
  uint x = byteAt(idx);
  if (idx + 1u < rowByteBase + packedCols) x |= byteAt(idx + 1u) << 8;
  uint v = (x >> shift) & 63u;
  return float(int(v) - 32);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint group = pc.numHeads / pc.kvHeads;
  uint cols = pc.numHeads * pc.headDim;
  uint packedCols = (cols * 6u + 7u) >> 3;
  uint rowBase = row * packedCols;
  uint tokenBase = (pc.cacheLen - 1u) * pc.kvDim;
  float sum = 0.0;
  for (uint c = lid; c < cols; c += 256) {
    uint srcHead = c / pc.headDim;
    uint elem = c - srcHead * pc.headDim;
    uint kvHead = srcHead / group;
    sum += q6(rowBase, packedCols, c) * v[tokenBase + kvHead * pc.headDim + elem];
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanTextFirstTokenValueOutQ4GLSLLinux = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint cacheLen; uint numHeads; uint kvHeads; uint headDim; uint kvDim; } pc;
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q4(uint idx) {
  uint word = packed[idx >> 3];
  uint shift = (idx & 7u) * 4u;
  uint v = (word >> shift) & 15u;
  return float(int(v) - 8);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint group = pc.numHeads / pc.kvHeads;
  uint cols = pc.numHeads * pc.headDim;
  uint packedStride = (((cols + 1u) >> 1) << 1);
  uint rowBase = row * packedStride;
  uint tokenBase = (pc.cacheLen - 1u) * pc.kvDim;
  float sum = 0.0;
  for (uint c = lid; c < cols; c += 256) {
    uint srcHead = c / pc.headDim;
    uint elem = c - srcHead * pc.headDim;
    uint kvHead = srcHead / group;
    sum += q4(rowBase + c) * v[tokenBase + kvHead * pc.headDim + elem];
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[row] = scratch[0] * scale[row];
}`

const vulkanLayerNormRowsF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; uint mode; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer A { float addv[]; };
layout(set=0,binding=2) readonly buffer W { float weight[]; };
layout(set=0,binding=3) readonly buffer B { float bias[]; };
layout(set=0,binding=4) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
void main() {
  uint row = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint base = row * pc.cols;
  float sum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    sum += v;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0u) mean = scratch[0] / float(pc.cols);
  barrier();
  float varSum = 0.0;
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    float d = v - mean;
    varSum += d * d;
  }
  scratch[lid] = varSum;
  barrier();
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  float scale = inversesqrt(scratch[0] / float(pc.cols) + pc.eps);
  for (uint c = lid; c < pc.cols; c += 256u) {
    float v = x[base + c];
    if (pc.mode == 1u) v += addv[base + c];
    outv[base + c] = (v - mean) * scale * weight[c] + bias[c];
  }
}`

const vulkanProjectImageF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint gridH; uint gridW; uint vd; uint hiddenRows; uint outRows; uint stage; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer NW { float normW[]; };
layout(set=0,binding=2) readonly buffer NB { float normB[]; };
layout(set=0,binding=3) readonly buffer W1 { float w1[]; };
layout(set=0,binding=4) readonly buffer B1 { float b1[]; };
layout(set=0,binding=5) readonly buffer W2 { float w2[]; };
layout(set=0,binding=6) readonly buffer B2 { float b2[]; };
layout(set=0,binding=7) buffer M { float merged[]; };
layout(set=0,binding=8) buffer H { float hidden[]; };
layout(set=0,binding=9) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
uint patch_base(uint batch, uint patchIdx) {
  uint blocksW = pc.gridW >> 1;
  uint blocksPerT = (pc.gridH >> 1) * blocksW;
  uint t = batch / blocksPerT;
  uint local = batch - t * blocksPerT;
  uint by = local / blocksW;
  uint bx = local - by * blocksW;
  uint base = t * pc.gridH * pc.gridW + by * 2u * pc.gridW + bx * 2u;
  if (patchIdx == 1u) return base + 1u;
  if (patchIdx == 2u) return base + pc.gridW;
  if (patchIdx == 3u) return base + pc.gridW + 1u;
  return base;
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  if (pc.stage == 0u) {
    uint patchIdx = row;
    uint srcBase = patch_base(batch, patchIdx) * pc.vd;
    float sum = 0.0;
    for (uint c = lid; c < pc.vd; c += 256u) sum += x[srcBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mean = scratch[0] / float(pc.vd);
    barrier();
    float varSum = 0.0;
    for (uint c = lid; c < pc.vd; c += 256u) {
      float d = x[srcBase + c] - mean;
      varSum += d * d;
    }
    scratch[lid] = varSum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    float scale = inversesqrt(scratch[0] / float(pc.vd) + pc.eps);
    uint dstBase = batch * pc.vd * 4u + patchIdx * pc.vd;
    for (uint c = lid; c < pc.vd; c += 256u) {
      float v = x[srcBase + c];
      merged[dstBase + c] = (v - mean) * scale * normW[c] + normB[c];
    }
    return;
  }
  float sum = 0.0;
  if (pc.stage == 1u) {
    uint cols = pc.vd * 4u;
    uint mBase = batch * cols;
    uint wBase = row * cols;
    for (uint c = lid; c < cols; c += 256u) sum += w1[wBase + c] * merged[mBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256u) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) outv[batch * pc.outRows + row] = scratch[0] + b2[row];
  }
}`

func fingerprintFloat32ForVulkanCache(values []float32) uint64 {
	n := len(values)
	h := uint64(n) * 1099511628211
	if n == 0 {
		return h
	}
	h = vulkanMixFloat32Fingerprint(h, values[0])
	h = vulkanMixFloat32Fingerprint(h, values[n/2])
	h = vulkanMixFloat32Fingerprint(h, values[n-1])
	if n > 3 {
		h = vulkanMixFloat32Fingerprint(h, values[n/3])
		h = vulkanMixFloat32Fingerprint(h, values[(2*n)/3])
	}
	return h
}

func vulkanMixFloat32Fingerprint(h uint64, v float32) uint64 {
	h ^= uint64(math.Float32bits(v))
	h *= 1099511628211
	return h
}

func fingerprintInt8ForVulkanCache(values []int8) uint64 {
	n := len(values)
	h := uint64(n) * 1099511628211
	if n == 0 {
		return h
	}
	h = vulkanMixByteFingerprint(h, byte(values[0]))
	h = vulkanMixByteFingerprint(h, byte(values[n/2]))
	h = vulkanMixByteFingerprint(h, byte(values[n-1]))
	if n > 3 {
		h = vulkanMixByteFingerprint(h, byte(values[n/3]))
		h = vulkanMixByteFingerprint(h, byte(values[(2*n)/3]))
	}
	return h
}

func fingerprintBytesForVulkanCache(values []byte) uint64 {
	n := len(values)
	h := uint64(n) * 1099511628211
	if n == 0 {
		return h
	}
	h = vulkanMixByteFingerprint(h, values[0])
	h = vulkanMixByteFingerprint(h, values[n/2])
	h = vulkanMixByteFingerprint(h, values[n-1])
	if n > 3 {
		h = vulkanMixByteFingerprint(h, values[n/3])
		h = vulkanMixByteFingerprint(h, values[(2*n)/3])
	}
	return h
}

func vulkanMixByteFingerprint(h uint64, v byte) uint64 {
	h ^= uint64(v)
	h *= 1099511628211
	return h
}

func absFloat32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

const vulkanVisionAttentionF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint tokens; uint heads; uint headDim; uint hidden; } pc;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float maxScore;
shared float denom;
shared float weight;
void main() {
  uint token = gl_WorkGroupID.x;
  uint head = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint headBase = head * pc.headDim;
  uint qBase = token * pc.hidden + headBase;
  float scale = inversesqrt(float(pc.headDim));
  if (lid == 0) maxScore = -3.4028234663852886e38;
  barrier();
  for (uint key = 0; key < pc.tokens; key++) {
    uint kBase = key * pc.hidden + headBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) maxScore = max(maxScore, scratch[0] * scale);
    barrier();
  }
  float acc = 0.0;
  if (lid == 0) denom = 0.0;
  barrier();
  for (uint key = 0; key < pc.tokens; key++) {
    uint kBase = key * pc.hidden + headBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) {
      weight = exp(scratch[0] * scale - maxScore);
      denom += weight;
    }
    barrier();
    if (lid < pc.headDim) acc += weight * v[key * pc.hidden + headBase + lid];
    barrier();
  }
  if (lid < pc.headDim) outv[token * pc.hidden + headBase + lid] = acc / denom;
}`

const vulkanVisionAttentionOutF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { float w[]; };
layout(set=0,binding=5) readonly buffer B { float bias[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] + bias[row];
}`

const vulkanVisionRoPEPairF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint gridH; uint gridW; uint heads; uint headDim; } pc;
layout(set=0,binding=0) buffer Q { float q[]; };
layout(set=0,binding=1) buffer K { float k[]; };
layout(set=0,binding=7) readonly buffer CH { float cosH[]; };
layout(set=0,binding=8) readonly buffer SH { float sinH[]; };
layout(set=0,binding=9) readonly buffer CW { float cosW[]; };
layout(set=0,binding=10) readonly buffer SW { float sinW[]; };
void rotatePair(uint base, uint aOff, uint bOff, float cs, float sn) {
  float qa = q[base + aOff];
  float qb = q[base + bOff];
  float ka = k[base + aOff];
  float kb = k[base + bOff];
  q[base + aOff] = qa * cs - qb * sn;
  q[base + bOff] = qb * cs + qa * sn;
  k[base + aOff] = ka * cs - kb * sn;
  k[base + bOff] = kb * cs + ka * sn;
}
void main() {
  uint token = gl_WorkGroupID.x;
  uint head = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint halfDim = pc.headDim >> 1;
  uint quarter = halfDim >> 1;
  if (quarter == 0u || pc.gridH == 0u || pc.gridW == 0u) {
    return;
  }
  uint period = pc.gridH * pc.gridW;
  uint pos = token - (token / period) * period;
  uint hy = pos / pc.gridW;
  uint wx = pos - hy * pc.gridW;
  uint hidden = pc.heads * pc.headDim;
  uint base = token * hidden + head * pc.headDim;
  for (uint i = lid; i < quarter; i += 256u) {
    uint hIdx = hy * quarter + i;
    rotatePair(base, i, quarter + i, cosH[hIdx], sinH[hIdx]);
    uint wIdx = wx * quarter + i;
    rotatePair(base, halfDim + i, halfDim + quarter + i, cosW[wIdx], sinW[wIdx]);
  }
}`

const vulkanVisionQKVF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rowsA; uint rowsB; uint rowsC; uint cols; } pc;
layout(set=0,binding=0) writeonly buffer OA { float oa[]; };
layout(set=0,binding=1) writeonly buffer OB { float ob[]; };
layout(set=0,binding=2) writeonly buffer OC { float oc[]; };
layout(set=0,binding=11) readonly buffer X { float x[]; };
layout(set=0,binding=12) readonly buffer WA { float wa[]; };
layout(set=0,binding=13) readonly buffer BA { float ba[]; };
layout(set=0,binding=14) readonly buffer WB { float wb[]; };
layout(set=0,binding=15) readonly buffer BB { float bb[]; };
layout(set=0,binding=16) readonly buffer WC { float wc[]; };
layout(set=0,binding=17) readonly buffer BC { float bc[]; };
shared float scratch[256];
void main() {
  uint globalRow = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint row = globalRow;
  uint segment = 0;
  if (row >= pc.rowsA) {
    row -= pc.rowsA;
    segment = 1;
    if (row >= pc.rowsB) {
      row -= pc.rowsB;
      segment = 2;
    }
  }
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    float xv = x[xBase + c];
    if (segment == 0) sum += wa[wBase + c] * xv;
    else if (segment == 1) sum += wb[wBase + c] * xv;
    else sum += wc[wBase + c] * xv;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    if (segment == 0) oa[batch * pc.rowsA + row] = scratch[0] + ba[row];
    else if (segment == 1) ob[batch * pc.rowsB + row] = scratch[0] + bb[row];
    else oc[batch * pc.rowsC + row] = scratch[0] + bc[row];
  }
}`

const vulkanTextAttentionF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint cacheLen; uint numHeads; uint kvHeads; uint headDim; uint kvDim; } pc;
layout(set=0,binding=0) readonly buffer Q { float q[]; };
layout(set=0,binding=1) readonly buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer V { float v[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float maxScore;
shared float denom;
shared float weight;
void main() {
  uint head = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint group = pc.numHeads / pc.kvHeads;
  uint kvHead = head / group;
  uint qBase = head * pc.headDim;
  uint kvHeadBase = kvHead * pc.headDim;
  float scale = inversesqrt(float(pc.headDim));
  if (lid == 0) maxScore = -3.4028234663852886e38;
  barrier();
  for (uint token = 0; token < pc.cacheLen; token++) {
    uint kBase = token * pc.kvDim + kvHeadBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) maxScore = max(maxScore, scratch[0] * scale);
    barrier();
  }
  float acc = 0.0;
  if (lid == 0) denom = 0.0;
  barrier();
  for (uint token = 0; token < pc.cacheLen; token++) {
    uint kBase = token * pc.kvDim + kvHeadBase;
    float part = 0.0;
    if (lid < pc.headDim) part = q[qBase + lid] * k[kBase + lid];
    scratch[lid] = part;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) {
      weight = exp(scratch[0] * scale - maxScore);
      denom += weight;
    }
    barrier();
    if (lid < pc.headDim) acc += weight * v[token * pc.kvDim + kvHeadBase + lid];
    barrier();
  }
  if (lid < pc.headDim) outv[qBase + lid] = acc / denom;
}`

const vulkanTextAttentionOutF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad0; uint pad1; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { float w[]; };
layout(set=0,binding=5) readonly buffer B { float bias[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] + bias[row];
}`

const vulkanTextAttentionOutQ8GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad0; uint pad1; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q8(uint idx) {
  uint word = packed[idx >> 2];
  uint shift = (idx & 3u) * 8u;
  uint v = (word >> shift) & 255u;
  return float(int(v << 24) >> 24);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += q8(wBase + c) * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] * scale[row];
}`

const vulkanTextAttentionOutQ6GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad0; uint pad1; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
uint byteAt(uint byteIdx) {
  uint word = packed[byteIdx >> 2];
  uint shift = (byteIdx & 3u) * 8u;
  return (word >> shift) & 255u;
}
float q6(uint rowByteBase, uint packedCols, uint col) {
  uint bit = col * 6u;
  uint idx = rowByteBase + (bit >> 3);
  uint shift = bit & 7u;
  uint x = byteAt(idx);
  if (idx + 1u < rowByteBase + packedCols) x |= byteAt(idx + 1u) << 8;
  uint v = (x >> shift) & 63u;
  return float(int(v) - 32);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint packedCols = (pc.cols * 6u + 7u) >> 3;
  uint rowBase = row * packedCols;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += q6(rowBase, packedCols, c) * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] * scale[row];
}`

const vulkanTextAttentionOutQ4GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint pad0; uint pad1; } pc;
layout(set=0,binding=3) readonly buffer H { float head[]; };
layout(set=0,binding=4) readonly buffer W { uint packed[]; };
layout(set=0,binding=5) readonly buffer S { float scale[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float q4(uint idx) {
  uint word = packed[idx >> 3];
  uint shift = (idx & 7u) * 4u;
  uint v = (word >> shift) & 15u;
  return float(int(v) - 8);
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint packedStride = (((pc.cols + 1u) >> 1) << 1);
  uint rowBase = row * packedStride;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += q4(rowBase + c) * head[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] * scale[row];
}`

const vulkanMatRowsBiasF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * x[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) outv[batch * pc.rows + row] = scratch[0] + bias[row];
}`

const vulkanMatRowsBiasAddRowsF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rows; uint cols; uint addRows; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) readonly buffer B { float bias[]; };
layout(set=0,binding=3) writeonly buffer O { float outv[]; };
layout(set=0,binding=4) readonly buffer A { float add[]; };
shared float scratch[256];
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) sum += w[wBase + c] * x[xBase + c];
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    uint addRow = batch % pc.addRows;
    outv[batch * pc.rows + row] = scratch[0] + bias[row] + add[addRow * pc.rows + row];
  }
}`

const vulkanMatRowsBias3F32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint rowsA; uint rowsB; uint rowsC; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer WA { float wa[]; };
layout(set=0,binding=2) readonly buffer BA { float ba[]; };
layout(set=0,binding=3) readonly buffer WB { float wb[]; };
layout(set=0,binding=4) readonly buffer BB { float bb[]; };
layout(set=0,binding=5) readonly buffer WC { float wc[]; };
layout(set=0,binding=6) readonly buffer BC { float bc[]; };
layout(set=0,binding=7) writeonly buffer OA { float oa[]; };
layout(set=0,binding=8) writeonly buffer OB { float ob[]; };
layout(set=0,binding=9) writeonly buffer OC { float oc[]; };
shared float scratch[256];
void main() {
  uint globalRow = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  uint row = globalRow;
  uint segment = 0;
  if (row >= pc.rowsA) {
    row -= pc.rowsA;
    segment = 1;
    if (row >= pc.rowsB) {
      row -= pc.rowsB;
      segment = 2;
    }
  }
  float sum = 0.0;
  uint xBase = batch * pc.cols;
  uint wBase = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    float xv = x[xBase + c];
    if (segment == 0) sum += wa[wBase + c] * xv;
    else if (segment == 1) sum += wb[wBase + c] * xv;
    else sum += wc[wBase + c] * xv;
  }
  scratch[lid] = sum;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) scratch[lid] += scratch[lid + stride];
    barrier();
  }
  if (lid == 0) {
    if (segment == 0) oa[batch * pc.rowsA + row] = scratch[0] + ba[row];
    else if (segment == 1) ob[batch * pc.rowsB + row] = scratch[0] + bb[row];
    else oc[batch * pc.rowsC + row] = scratch[0] + bc[row];
  }
}`

const vulkanMatRowsGELU2F32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint hiddenRows; uint cols; uint outRows; uint stage; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W1 { float w1[]; };
layout(set=0,binding=2) readonly buffer B1 { float b1[]; };
layout(set=0,binding=3) readonly buffer W2 { float w2[]; };
layout(set=0,binding=4) readonly buffer B2 { float b2[]; };
layout(set=0,binding=5) buffer H { float hidden[]; };
layout(set=0,binding=6) writeonly buffer O { float outv[]; };
shared float scratch[256];
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  if (pc.stage == 0) {
    uint xBase = batch * pc.cols;
    uint wBase = row * pc.cols;
    for (uint c = lid; c < pc.cols; c += 256) sum += w1[wBase + c] * x[xBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0) outv[batch * pc.outRows + row] = scratch[0] + b2[row];
  }
}`

const vulkanMatRowsGELU2AddLayerNormF32GLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint batches; uint hiddenRows; uint cols; uint outRows; uint stage; float eps; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer R { float residual[]; };
layout(set=0,binding=2) readonly buffer W1 { float w1[]; };
layout(set=0,binding=3) readonly buffer B1 { float b1[]; };
layout(set=0,binding=4) readonly buffer W2 { float w2[]; };
layout(set=0,binding=5) readonly buffer B2 { float b2[]; };
layout(set=0,binding=6) readonly buffer NW { float normW[]; };
layout(set=0,binding=7) readonly buffer NB { float normB[]; };
layout(set=0,binding=8) buffer H { float hidden[]; };
layout(set=0,binding=9) buffer M { float mlp[]; };
layout(set=0,binding=10) writeonly buffer O { float outv[]; };
shared float scratch[256];
shared float mean;
float gelu_tanh(float v) {
  return 0.5 * v * (1.0 + tanh(v * (0.7978845608028654 + 0.035677408136300125 * v * v)));
}
void main() {
  uint row = gl_WorkGroupID.x;
  uint batch = gl_WorkGroupID.y;
  uint lid = gl_LocalInvocationID.x;
  float sum = 0.0;
  if (pc.stage == 0u) {
    uint xBase = batch * pc.cols;
    uint wBase = row * pc.cols;
    for (uint c = lid; c < pc.cols; c += 256u) sum += w1[wBase + c] * x[xBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) hidden[batch * pc.hiddenRows + row] = gelu_tanh(scratch[0] + b1[row]);
  } else if (pc.stage == 1u) {
    uint hBase = batch * pc.hiddenRows;
    uint wBase = row * pc.hiddenRows;
    for (uint c = lid; c < pc.hiddenRows; c += 256u) sum += w2[wBase + c] * hidden[hBase + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mlp[batch * pc.outRows + row] = scratch[0] + b2[row];
  } else {
    uint base = row * pc.outRows;
    for (uint c = lid; c < pc.outRows; c += 256u) sum += residual[base + c] + mlp[base + c];
    scratch[lid] = sum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    if (lid == 0u) mean = scratch[0] / float(pc.outRows);
    barrier();
    float varSum = 0.0;
    for (uint c = lid; c < pc.outRows; c += 256u) {
      float v = residual[base + c] + mlp[base + c];
      float d = v - mean;
      varSum += d * d;
    }
    scratch[lid] = varSum;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
      if (lid < stride) scratch[lid] += scratch[lid + stride];
      barrier();
    }
    float scale = inversesqrt(scratch[0] / float(pc.outRows) + pc.eps);
    for (uint c = lid; c < pc.outRows; c += 256u) {
      float v = residual[base + c] + mlp[base + c];
      outv[base + c] = (v - mean) * scale * normW[c] + normB[c];
    }
  }
}`
