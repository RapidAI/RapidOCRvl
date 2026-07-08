//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
)

var vulkanDispatchProbe struct {
	once sync.Once
	err  error
}

var vulkanMatVecF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanArgmaxF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanArgmaxQuantizedF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanBlockTopKF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanBlockTopKQuantizedF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanRMSNormF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanAddRMSNormF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMRoPEF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMRoPEPairF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanFusedMatVec3F32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatVecF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanMatVecTopKF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanRMSNormF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanAddRMSNormF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanMRoPEF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanMRoPEPairF32RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecF32WinRunner
	err    error
}

var vulkanFusedMatVec3F32RunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32WinRunner
	err    error
}

var vulkanFusedMatVec3MRoPEF32RunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3F32WinRunner
	err    error
}

func VulkanDispatchSmokeTest() error {
	vulkanDispatchProbe.once.Do(func() {
		vulkanDispatchProbe.err = runVulkanDispatchSmokeTestWindows()
	})
	return vulkanDispatchProbe.err
}

func runVulkanDispatchSmokeTestWindows() error {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1}
	out := make([]float32, 3)
	if err := VulkanMatVecF32(out, x, w, 3, 4); err != nil {
		return err
	}
	want := []float32{1, 2, 10}
	for i := range want {
		if absFloat32Windows(out[i]-want[i]) > 1e-4 {
			return fmt.Errorf("vulkan matvec mismatch at %d: got %.6f want %.6f", i, out[i], want[i])
		}
	}
	return nil
}

func VulkanMatVecF32(out, x, w []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan matvec shape rows=%d cols=%d", rows, cols)
	}
	if len(out) < rows || len(x) < cols || len(w) < rows*cols {
		return fmt.Errorf("invalid Vulkan matvec buffers out=%d x=%d w=%d rows=%d cols=%d", len(out), len(x), len(w), rows, cols)
	}
	runner, err := getVulkanMatVecF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(out, x, w, rows, cols)
}

func VulkanMatVecArgmaxF32(x, w []float32, rows, cols int) (int, float32, error) {
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("invalid Vulkan matvec argmax shape rows=%d cols=%d", rows, cols)
	}
	if len(x) < cols || len(w) < rows*cols {
		return 0, 0, fmt.Errorf("invalid Vulkan matvec argmax buffers x=%d w=%d rows=%d cols=%d", len(x), len(w), rows, cols)
	}
	runner, err := getVulkanMatVecArgmaxF32RunnerWindows()
	if err != nil {
		return 0, 0, err
	}
	return runner.runMatVecArgmax(x, w, rows, cols)
}

func VulkanMatVecTopKF32(x, w []float32, rows, cols, k int) ([]VulkanTokenScore, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k shape rows=%d cols=%d", rows, cols)
	}
	if k <= 0 || k > vulkanMatVecTopKMaxK {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k k=%d max=%d", k, vulkanMatVecTopKMaxK)
	}
	if len(x) < cols || len(w) < rows*cols {
		return nil, fmt.Errorf("invalid Vulkan matvec top-k buffers x=%d w=%d rows=%d cols=%d", len(x), len(w), rows, cols)
	}
	runner, err := getVulkanMatVecTopKF32RunnerWindows()
	if err != nil {
		return nil, err
	}
	return runner.runMatVecTopK(x, w, rows, cols, k)
}

func VulkanRMSNormF32(out, x, weight []float32) error {
	n := len(x)
	if n <= 0 {
		return fmt.Errorf("invalid Vulkan rmsnorm shape n=%d", n)
	}
	if len(out) < n || len(weight) < n {
		return fmt.Errorf("invalid Vulkan rmsnorm buffers out=%d x=%d weight=%d", len(out), len(x), len(weight))
	}
	runner, err := getVulkanRMSNormF32RunnerWindows()
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
	runner, err := getVulkanAddRMSNormF32RunnerWindows()
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
	runner, err := getVulkanAddRMSNormF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runAddOutOnly(out, dst, add, weight, n)
}

func VulkanMRoPEF32(x, cosTable, sinTable []float32, heads, dim int) error {
	if heads <= 0 || dim <= 0 || dim%2 != 0 {
		return fmt.Errorf("invalid Vulkan mrope shape heads=%d dim=%d", heads, dim)
	}
	n := heads * dim
	half := dim / 2
	if len(x) < n || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan mrope buffers x=%d cos=%d sin=%d heads=%d dim=%d", len(x), len(cosTable), len(sinTable), heads, dim)
	}
	runner, err := getVulkanMRoPEF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runMRoPE(x, cosTable, sinTable, heads, dim)
}

func VulkanMRoPEPairF32(q, k, cosTable, sinTable []float32, qHeads, kvHeads, dim int) error {
	if qHeads <= 0 || kvHeads <= 0 || dim <= 0 || dim%2 != 0 || dim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan mrope pair shape qHeads=%d kvHeads=%d dim=%d", qHeads, kvHeads, dim)
	}
	qRows := qHeads * dim
	kvRows := kvHeads * dim
	half := dim / 2
	if len(q) < qRows || len(k) < kvRows || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan mrope pair buffers q=%d k=%d cos=%d sin=%d qHeads=%d kvHeads=%d dim=%d", len(q), len(k), len(cosTable), len(sinTable), qHeads, kvHeads, dim)
	}
	runner, err := getVulkanMRoPEPairF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runMRoPEPair(q, k, cosTable, sinTable, qHeads, kvHeads, dim)
}

func VulkanFusedMatVec3F32(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) error {
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 {
		return fmt.Errorf("invalid Vulkan fused matvec3 shape rowsA=%d rowsB=%d rowsC=%d cols=%d", rowsA, rowsB, rowsC, cols)
	}
	if len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC || len(x) < cols ||
		len(wa) < rowsA*cols || len(wb) < rowsB*cols || len(wc) < rowsC*cols {
		return fmt.Errorf("invalid Vulkan fused matvec3 buffers outA=%d outB=%d outC=%d x=%d wa=%d wb=%d wc=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			len(outA), len(outB), len(outC), len(x), len(wa), len(wb), len(wc), rowsA, rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec3F32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.run(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, rowsC, cols)
}

func VulkanFusedMatVec3MRoPEF32(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim int) error {
	if rowsA <= 0 || rowsB <= 0 || rowsC <= 0 || cols <= 0 || qHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%2 != 0 || headDim > 65535 || kvHeads > 65535 {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope shape rowsA=%d rowsB=%d rowsC=%d cols=%d qHeads=%d kvHeads=%d headDim=%d", rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim)
	}
	if rowsA != qHeads*headDim || rowsB != kvHeads*headDim {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope rows rowsA=%d rowsB=%d want=%d/%d", rowsA, rowsB, qHeads*headDim, kvHeads*headDim)
	}
	half := headDim / 2
	if len(outA) < rowsA || len(outB) < rowsB || len(outC) < rowsC || len(x) < cols ||
		len(wa) < rowsA*cols || len(wb) < rowsB*cols || len(wc) < rowsC*cols || len(cosTable) < half || len(sinTable) < half {
		return fmt.Errorf("invalid Vulkan fused matvec3+mrope buffers outA=%d outB=%d outC=%d x=%d wa=%d wb=%d wc=%d cos=%d sin=%d rowsA=%d rowsB=%d rowsC=%d cols=%d",
			len(outA), len(outB), len(outC), len(x), len(wa), len(wb), len(wc), len(cosTable), len(sinTable), rowsA, rowsB, rowsC, cols)
	}
	runner, err := getVulkanFusedMatVec3MRoPEF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runMRoPE(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, rowsA, rowsB, rowsC, cols, kvHeads, headDim)
}

func getVulkanMatVecF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanMatVecF32RunnerCache.once.Do(func() {
		vulkanMatVecF32RunnerCache.runner, vulkanMatVecF32RunnerCache.err = newVulkanMatVecF32WinRunner()
	})
	return vulkanMatVecF32RunnerCache.runner, vulkanMatVecF32RunnerCache.err
}

func getVulkanMatVecArgmaxF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	return getVulkanMatVecF32RunnerWindows()
}

func getVulkanMatVecTopKF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanMatVecTopKF32RunnerCache.once.Do(func() {
		vulkanMatVecTopKF32RunnerCache.runner, vulkanMatVecTopKF32RunnerCache.err = newVulkanMatVecTopKF32WinRunner()
	})
	return vulkanMatVecTopKF32RunnerCache.runner, vulkanMatVecTopKF32RunnerCache.err
}

func getVulkanRMSNormF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanRMSNormF32RunnerCache.once.Do(func() {
		vulkanRMSNormF32RunnerCache.runner, vulkanRMSNormF32RunnerCache.err = newVulkanRMSNormF32WinRunner()
	})
	return vulkanRMSNormF32RunnerCache.runner, vulkanRMSNormF32RunnerCache.err
}

func getVulkanAddRMSNormF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanAddRMSNormF32RunnerCache.once.Do(func() {
		vulkanAddRMSNormF32RunnerCache.runner, vulkanAddRMSNormF32RunnerCache.err = newVulkanAddRMSNormF32WinRunner()
	})
	return vulkanAddRMSNormF32RunnerCache.runner, vulkanAddRMSNormF32RunnerCache.err
}

func getVulkanMRoPEF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanMRoPEF32RunnerCache.once.Do(func() {
		vulkanMRoPEF32RunnerCache.runner, vulkanMRoPEF32RunnerCache.err = newVulkanMRoPEF32WinRunner()
	})
	return vulkanMRoPEF32RunnerCache.runner, vulkanMRoPEF32RunnerCache.err
}

func getVulkanMRoPEPairF32RunnerWindows() (*vulkanMatVecF32WinRunner, error) {
	vulkanMRoPEPairF32RunnerCache.once.Do(func() {
		vulkanMRoPEPairF32RunnerCache.runner, vulkanMRoPEPairF32RunnerCache.err = newVulkanMRoPEPairF32WinRunner()
	})
	return vulkanMRoPEPairF32RunnerCache.runner, vulkanMRoPEPairF32RunnerCache.err
}

func getVulkanFusedMatVec3F32RunnerWindows() (*vulkanFusedMatVec3F32WinRunner, error) {
	vulkanFusedMatVec3F32RunnerCache.once.Do(func() {
		vulkanFusedMatVec3F32RunnerCache.runner, vulkanFusedMatVec3F32RunnerCache.err = newVulkanFusedMatVec3F32WinRunner()
	})
	return vulkanFusedMatVec3F32RunnerCache.runner, vulkanFusedMatVec3F32RunnerCache.err
}

func getVulkanFusedMatVec3MRoPEF32RunnerWindows() (*vulkanFusedMatVec3F32WinRunner, error) {
	vulkanFusedMatVec3MRoPEF32RunnerCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEF32RunnerCache.runner, vulkanFusedMatVec3MRoPEF32RunnerCache.err = newVulkanFusedMatVec3MRoPEF32WinRunner()
	})
	return vulkanFusedMatVec3MRoPEF32RunnerCache.runner, vulkanFusedMatVec3MRoPEF32RunnerCache.err
}

type vulkanMatVecF32WinRunner struct {
	vk              *vulkanWin
	instance        uintptr
	device          uintptr
	queue           uintptr
	queueFamily     uint32
	memProps        vkPhysicalDeviceMemoryProperties
	setLayout       uintptr
	descriptorPool  uintptr
	descriptorSet   uintptr
	pipelineLayout  uintptr
	pipeline        uintptr
	secondPipeline  uintptr
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	addBuf          vkHostBufferWin
	outBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	topKReadback    []float32
	topKCandidates  []VulkanTokenScore
	descriptorCache [5]vulkanDescriptorBindingWin
	descriptorCount int
	commandKind     int
	commandRecorded bool
	commandRows     int
	commandCols     int
	dispatchOnce    bool
	sharedDevice    bool
	mu              sync.Mutex
}

const (
	vulkanMatVecF32WinCommandDefault = iota + 1
	vulkanMatVecF32WinCommandArgmax
	vulkanMatVecF32WinCommandTopK
)

func newVulkanMatVecF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunnerWithSecondPipeline("rapidocrvl-vulkan-matvec", vulkanMatVecF32ShaderCodeWindows, vulkanArgmaxF32ShaderCodeWindows, false, 4)
}

func newVulkanMatVecArgmaxF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunnerWithSecondPipeline("rapidocrvl-vulkan-matvec-argmax", vulkanMatVecF32ShaderCodeWindows, vulkanArgmaxF32ShaderCodeWindows, false, 4)
}

func newVulkanMatVecTopKF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunnerWithSecondPipeline("rapidocrvl-vulkan-matvec-topk", vulkanMatVecF32ShaderCodeWindows, vulkanBlockTopKF32ShaderCodeWindows, false, 5)
}

func newVulkanRMSNormF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunner("rapidocrvl-vulkan-rmsnorm", vulkanRMSNormF32ShaderCodeWindows, true, 3)
}

func newVulkanAddRMSNormF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunner("rapidocrvl-vulkan-add-rmsnorm", vulkanAddRMSNormF32ShaderCodeWindows, true, 4)
}

func newVulkanMRoPEF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunner("rapidocrvl-vulkan-mrope", vulkanMRoPEF32ShaderCodeWindows, true, 4)
}

func newVulkanMRoPEPairF32WinRunner() (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunner("rapidocrvl-vulkan-mrope-pair", vulkanMRoPEPairF32ShaderCodeWindows, true, 4)
}

func newVulkanF32TripleBufferWinRunner(appLabel string, shaderCode func() ([]uint32, error), dispatchOnce bool, descriptorCount int) (*vulkanMatVecF32WinRunner, error) {
	return newVulkanF32TripleBufferWinRunnerWithSecondPipeline(appLabel, shaderCode, nil, dispatchOnce, descriptorCount)
}

func newVulkanF32TripleBufferWinRunnerWithSecondPipeline(appLabel string, shaderCode, secondShaderCode func() ([]uint32, error), dispatchOnce bool, descriptorCount int) (*vulkanMatVecF32WinRunner, error) {
	spv, err := shaderCode()
	if err != nil {
		return nil, err
	}
	var secondSPV []uint32
	if secondShaderCode != nil {
		secondSPV, err = secondShaderCode()
		if err != nil {
			return nil, err
		}
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	instance := ctx.instance
	device := ctx.device
	queue := ctx.queue
	queueFamily := ctx.queueFamily
	memProps := ctx.memProps
	entryName := append([]byte("main"), 0)
	r := &vulkanMatVecF32WinRunner{vk: vk, instance: instance, device: device, queue: queue, queueFamily: queueFamily, memProps: memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), dispatchOnce: dispatchOnce, descriptorCount: descriptorCount}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{
		SType:        vkStructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    uintptr(unsafe.Pointer(&bindings[0])),
	}
	var setLayout uintptr
	if res := vk.call(vk.createDescriptorSetLayout, device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	r.setLayout = setLayout

	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}
	dpci := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize)),
	}
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&r.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}

	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 8}
	plci := vkPipelineLayoutCreateInfo{
		SType:                  vkStructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            uintptr(unsafe.Pointer(&setLayout)),
		PushConstantRangeCount: 1,
		PPushConstantRanges:    uintptr(unsafe.Pointer(&pushRange)),
	}
	var pipelineLayout uintptr
	if res := vk.call(vk.createPipelineLayout, device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	r.pipelineLayout = pipelineLayout

	smci := vkShaderModuleCreateInfo{
		SType:    vkStructureTypeShaderModuleCreateInfo,
		CodeSize: uintptr(len(spv) * 4),
		PCode:    uintptr(unsafe.Pointer(&spv[0])),
	}
	var shader uintptr
	if res := vk.call(vk.createShaderModule, device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, device, shader, 0)

	stage := vkPipelineShaderStageCreateInfo{
		SType:  vkStructureTypePipelineShaderStageCreateInfo,
		Stage:  vkShaderStageComputeBit,
		Module: shader,
		PName:  uintptr(unsafe.Pointer(&entryName[0])),
	}
	cpci := vkComputePipelineCreateInfo{
		SType:  vkStructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: pipelineLayout,
	}
	var pipeline uintptr
	if res := vk.call(vk.createComputePipelines, device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&pipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	r.pipeline = pipeline
	if len(secondSPV) > 0 {
		smci2 := vkShaderModuleCreateInfo{
			SType:    vkStructureTypeShaderModuleCreateInfo,
			CodeSize: uintptr(len(secondSPV) * 4),
			PCode:    uintptr(unsafe.Pointer(&secondSPV[0])),
		}
		var shader2 uintptr
		if res := vk.call(vk.createShaderModule, device, uintptr(unsafe.Pointer(&smci2)), 0, uintptr(unsafe.Pointer(&shader2))); res != vkSuccess {
			return nil, fmt.Errorf("vkCreateShaderModule second: %d", int32(res))
		}
		defer vk.callVoid(vk.destroyShaderModule, device, shader2, 0)
		stage2 := vkPipelineShaderStageCreateInfo{
			SType:  vkStructureTypePipelineShaderStageCreateInfo,
			Stage:  vkShaderStageComputeBit,
			Module: shader2,
			PName:  uintptr(unsafe.Pointer(&entryName[0])),
		}
		cpciSecond := vkComputePipelineCreateInfo{
			SType:  vkStructureTypeComputePipelineCreateInfo,
			Stage:  stage2,
			Layout: pipelineLayout,
		}
		if res := vk.call(vk.createComputePipelines, device, 0, 1, uintptr(unsafe.Pointer(&cpciSecond)), 0, uintptr(unsafe.Pointer(&r.secondPipeline))); res != vkSuccess {
			return nil, fmt.Errorf("vkCreateComputePipelines second: %d", int32(res))
		}
	}

	cpci2 := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, device, uintptr(unsafe.Pointer(&cpci2)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{
		SType:              vkStructureTypeCommandBufferAllocateInfo,
		CommandPool:        r.commandPool,
		Level:              vkCommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	if res := vk.call(vk.allocateCommandBuffers, device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanMatVecF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.secondPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.secondPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.addBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	for _, b := range r.weightBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	if r.descriptorPool != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorPool, r.device, r.descriptorPool, 0)
	}
	if r.pipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.pipelineLayout, 0)
	}
	if r.setLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.setLayout, 0)
	}
	if !r.sharedDevice {
		if r.device != 0 {
			r.vk.callVoid(r.vk.destroyDevice, r.device, 0)
		}
		if r.instance != 0 {
			r.vk.callVoid(r.vk.destroyInstance, r.instance, 0)
		}
	}
}

func (r *vulkanMatVecF32WinRunner) run(out, x, w []float32, rows, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	xLen := cols
	if r.dispatchOnce {
		xLen = rows
	}
	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan f32 matvec runner")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(xLen, "Vulkan f32 matvec runner x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan f32 matvec runner weight")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan f32 matvec runner output")
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
	if err := vk.writeFloat32(device, r.xBuf, x[:xLen]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanMatVecF32WinCommandDefault || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return vk.readFloat32Into(device, r.outBuf, out[:rows])
}

func (r *vulkanMatVecF32WinRunner) runMatVecArgmax(x, w []float32, rows, cols int) (int, float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.secondPipeline == 0 {
		return 0, 0, fmt.Errorf("vulkan matvec argmax pipeline is unavailable")
	}
	vk := r.vk
	device := r.device
	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan f32 matvec argmax runner")
	if err != nil {
		return 0, 0, err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 matvec argmax runner x")
	if err != nil {
		return 0, 0, err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan f32 matvec argmax runner weight")
	if err != nil {
		return 0, 0, err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan f32 matvec argmax runner output")
	if err != nil {
		return 0, 0, err
	}
	resultBytes, err := checkedFloat32ByteLenErrWin(2, "Vulkan f32 matvec argmax runner result")
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
	if err := vk.writeFloat32(device, r.xBuf, x[:cols]); err != nil {
		return 0, 0, err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: r.addBuf.buffer, Range: resultBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32WinCommandArgmax || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordMatVecArgmaxCommand(rows, cols); err != nil {
			return 0, 0, err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	var result [2]float32
	if err := vk.readFloat32Into(device, r.addBuf, result[:]); err != nil {
		return 0, 0, err
	}
	return int(result[1]), result[0], nil
}

func (r *vulkanMatVecF32WinRunner) runMatVecTopK(x, w []float32, rows, cols, k int) ([]VulkanTokenScore, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.secondPipeline == 0 {
		return nil, fmt.Errorf("vulkan matvec top-k pipeline is unavailable")
	}
	vk := r.vk
	device := r.device
	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan f32 matvec top-k runner")
	if err != nil {
		return nil, err
	}
	if rows <= 0 || rows > maxInt()-255 {
		return nil, fmt.Errorf("Vulkan f32 matvec top-k runner block count overflows: rows=%d", rows)
	}
	blocks := (rows + 255) / 256
	candidates, ok := checkedMulInt(blocks, vulkanMatVecTopKMaxK)
	if !ok {
		return nil, fmt.Errorf("Vulkan f32 matvec top-k runner candidate count overflows: blocks=%d k=%d", blocks, vulkanMatVecTopKMaxK)
	}
	candidateFloats, ok := checkedMulInt(candidates, 2)
	if !ok {
		return nil, fmt.Errorf("Vulkan f32 matvec top-k runner candidate float count overflows: candidates=%d", candidates)
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 matvec top-k runner x")
	if err != nil {
		return nil, err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan f32 matvec top-k runner weight")
	if err != nil {
		return nil, err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan f32 matvec top-k runner output")
	if err != nil {
		return nil, err
	}
	resultBytes, err := checkedFloat32ByteLenErrWin(candidateFloats, "Vulkan f32 matvec top-k runner result")
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
	if err := vk.writeFloat32(device, r.xBuf, x[:cols]); err != nil {
		return nil, err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: r.addBuf.buffer, Range: resultBytes},
	}
	bindings := [...]uint32{0, 1, 2, 3}
	updateVulkanDescriptorBindingsWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bindings[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecF32WinCommandTopK || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordMatVecTopKCommand(rows, cols); err != nil {
			return nil, err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return nil, fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return nil, fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	r.topKReadback = ensureVulkanFloat32Scratch(r.topKReadback, candidateFloats)
	candidateData := r.topKReadback
	if err := vk.readFloat32Into(device, r.addBuf, candidateData); err != nil {
		return nil, err
	}
	r.topKCandidates = selectVulkanTopKCandidatesInto(r.topKCandidates, candidateData, rows, k)
	return r.topKCandidates, nil
}

func (r *vulkanMatVecF32WinRunner) runAdd(out, dst, add, weight []float32, n int) error {
	return r.runAddMaybeReadDst(out, dst, add, weight, n, true)
}

func (r *vulkanMatVecF32WinRunner) runAddOutOnly(out, dst, add, weight []float32, n int) error {
	return r.runAddMaybeReadDst(out, dst, add, weight, n, false)
}

func (r *vulkanMatVecF32WinRunner) runAddMaybeReadDst(out, dst, add, weight []float32, n int, readDst bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	bytes, err := checkedFloat32ByteLenErrWin(n, "Vulkan add rmsnorm runner")
	if err != nil {
		return err
	}
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
	if err := vk.writeFloat32(device, r.xBuf, dst[:n]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.addBuf, add[:n]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: bytes},
		{Buffer: r.addBuf.buffer, Range: bytes},
		{Buffer: wBuf.buffer, Range: bytes},
		{Buffer: r.outBuf.buffer, Range: bytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])
	if !r.commandRecorded || r.commandRows != n || r.commandCols != 1 {
		if err := r.recordCommand(n, 1); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if readDst {
		if err := vk.readFloat32Into(device, r.xBuf, dst[:n]); err != nil {
			return err
		}
	}
	return vk.readFloat32Into(device, r.outBuf, out[:n])
}

func (r *vulkanMatVecF32WinRunner) runMRoPE(x, cosTable, sinTable []float32, heads, dim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	n, ok := checkedMulInt(heads, dim)
	if !ok {
		return fmt.Errorf("Vulkan mrope runner length overflows: heads=%d dim=%d", heads, dim)
	}
	half := dim / 2
	xBytes, err := checkedFloat32ByteLenErrWin(n, "Vulkan mrope runner x")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan mrope runner table")
	if err != nil {
		return err
	}
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
	if err := vk.writeFloat32(device, r.xBuf, x[:n]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.addBuf, cosTable[:half]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: r.addBuf.buffer, Range: tableBytes},
		{Buffer: sinBuf.buffer, Range: tableBytes},
		{Buffer: r.outBuf.buffer, Range: xBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])
	if !r.commandRecorded || r.commandRows != n || r.commandCols != dim {
		if err := r.recordCommand(n, dim); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return vk.readFloat32Into(device, r.xBuf, x[:n])
}

func (r *vulkanMatVecF32WinRunner) runMRoPEPair(q, k, cosTable, sinTable []float32, qHeads, kvHeads, dim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	qRows := qHeads * dim
	kvRows := kvHeads * dim
	half := dim / 2
	qBytes, err := checkedFloat32ByteLenErrWin(qRows, "Vulkan mrope pair runner q")
	if err != nil {
		return err
	}
	kBytes, err := checkedFloat32ByteLenErrWin(kvRows, "Vulkan mrope pair runner k")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan mrope pair runner table")
	if err != nil {
		return err
	}
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
	if err := vk.writeFloat32(device, r.xBuf, q[:qRows]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.addBuf, k[:kvRows]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.outBuf, sinTable[:half]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: qBytes},
		{Buffer: r.addBuf.buffer, Range: kBytes},
		{Buffer: cosBuf.buffer, Range: tableBytes},
		{Buffer: r.outBuf.buffer, Range: tableBytes},
	}
	encodedCols := (kvHeads << 16) | dim
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])
	if !r.commandRecorded || r.commandRows != qHeads || r.commandCols != encodedCols {
		if err := r.recordCommand(qHeads, encodedCols); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := vk.readFloat32Into(device, r.xBuf, q[:qRows]); err != nil {
		return err
	}
	return vk.readFloat32Into(device, r.addBuf, k[:kvRows])
}

func (r *vulkanMatVecF32WinRunner) recordCommand(rows, cols int) error {
	vk := r.vk
	if res := vk.call(vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	groupsX := rows
	if r.dispatchOnce {
		groupsX = 1
	}
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groupsX), 1, 1)
	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32WinCommandDefault
	r.commandRecorded = true
	return nil
}

// dispatchInto records a matvec F32 dispatch into an external command buffer
// (cmd) without submitting or waiting.  The caller is responsible for
// submitting the command buffer and waiting for the fence.
//
// The input data must already be written to xBuf and the output will be
// written to outBuf by the GPU.  The descriptor set must already be updated
// to point to the correct buffers.
func (r *vulkanMatVecF32WinRunner) dispatchInto(cmd uintptr, rows, cols int) error {
	// Note: do NOT call recordCommand here — it would reset the command pool
	// and destroy the command buffer we're recording into.  The caller must
	// ensure the pipeline and descriptor set are already set up.
	vk := r.vk
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	groupsX := rows
	if r.dispatchOnce {
		groupsX = 1
	}
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groupsX), 1, 1)
	return nil
}

// prepareMatVecF32 sets up the host buffers and descriptor set for a matvec
// dispatch, writing input data to xBuf and binding xBuf/wBuf/outBuf to the
// descriptor set.  After this, call dispatchInto to record the actual dispatch.
func (r *vulkanMatVecF32WinRunner) prepareMatVecF32(x, w []float32, rows, cols int) error {
	xLen := cols
	if r.dispatchOnce {
		xLen = rows
	}
	wLen, err := checkedMatVecF32WeightLenWin(rows, cols, "Vulkan f32 matvec prepare")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(xLen, "Vulkan f32 matvec prepare x")
	if err != nil {
		return err
	}
	wBytes, err := checkedFloat32ByteLenErrWin(wLen, "Vulkan f32 matvec prepare weight")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan f32 matvec prepare output")
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:xLen]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])
	return nil
}

// outBufferHandle returns the Vulkan buffer handle of the output buffer,
// for use by callers that need to chain the output into another runner's
// descriptor set.
func (r *vulkanMatVecF32WinRunner) outBufferHandle() uintptr {
	return r.outBuf.buffer
}

// readOut reads the output buffer into the given slice.
func (r *vulkanMatVecF32WinRunner) readOut(out []float32, rows int) error {
	return r.vk.readFloat32Into(r.device, r.outBuf, out[:rows])
}

func (r *vulkanMatVecF32WinRunner) recordMatVecArgmaxCommand(rows, cols int) error {
	vk := r.vk
	if res := vk.call(vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit}
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.secondPipeline)
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, 1, 1, 1)
	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32WinCommandArgmax
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32WinRunner) recordMatVecTopKCommand(rows, cols int) error {
	vk := r.vk
	if res := vk.call(vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit}
	vk.callVoid(vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.secondPipeline)
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	blocks := (rows + 255) / 256
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(blocks), 1, 1)
	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecF32WinCommandTopK
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
	if buf.buffer != 0 && buf.size >= size {
		return nil
	}
	if buf.buffer != 0 || buf.memory != 0 {
		r.vk.destroyBuffer(r.device, *buf)
		*buf = vkHostBufferWin{}
	}
	next, err := r.vk.newHostBuffer(r.device, r.memProps, size)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanMatVecF32WinRunner) weightBuffer(w []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, w, size, r.weightBuffers)
}

type vulkanFusedMatVec3F32WinRunner struct {
	vk              *vulkanWin
	instance        uintptr
	device          uintptr
	queue           uintptr
	queueFamily     uint32
	memProps        vkPhysicalDeviceMemoryProperties
	setLayout       uintptr
	descriptorPool  uintptr
	descriptorSet   uintptr
	pipelineLayout  uintptr
	pipeline        uintptr
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	outABuf         vkHostBufferWin
	outBBuf         vkHostBufferWin
	outCBuf         vkHostBufferWin
	cosBuf          vkHostBufferWin
	sinBuf          vkHostBufferWin
	weightBuffers   map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [9]vulkanDescriptorBindingWin
	descriptorCount int
	mrope           bool
	sharedDevice    bool
	commandRecorded bool
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	commandPacked   int
	mu              sync.Mutex
}

func newVulkanFusedMatVec3F32WinRunner() (*vulkanFusedMatVec3F32WinRunner, error) {
	return newVulkanFusedMatVec3F32WinRunnerWithShader("rapidocrvl-vulkan-fused-matvec3", vulkanFusedMatVec3F32ShaderCodeWindows, 7, 16, false)
}

func newVulkanFusedMatVec3MRoPEF32WinRunner() (*vulkanFusedMatVec3F32WinRunner, error) {
	return newVulkanFusedMatVec3F32WinRunnerWithShader("rapidocrvl-vulkan-fused-matvec3-mrope", vulkanFusedMatVec3MRoPEF32ShaderCodeWindows, 9, 20, true)
}

func newVulkanFusedMatVec3F32WinRunnerWithShader(appLabel string, shaderCode func() ([]uint32, error), descriptorCount, pushConstantBytes int, mrope bool) (*vulkanFusedMatVec3F32WinRunner, error) {
	spv, err := shaderCode()
	if err != nil {
		return nil, err
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	instance := ctx.instance
	device := ctx.device
	queue := ctx.queue
	queueFamily := ctx.queueFamily
	memProps := ctx.memProps
	entryName := append([]byte("main"), 0)
	r := &vulkanFusedMatVec3F32WinRunner{vk: vk, instance: instance, device: device, queue: queue, queueFamily: queueFamily, memProps: memProps, sharedDevice: true, weightBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), descriptorCount: descriptorCount, mrope: mrope}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, descriptorCount)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{
		SType:        vkStructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
		PBindings:    uintptr(unsafe.Pointer(&bindings[0])),
	}
	if res := vk.call(vk.createDescriptorSetLayout, device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}
	dpci := vkDescriptorPoolCreateInfo{
		SType:         vkStructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    uintptr(unsafe.Pointer(&poolSize)),
	}
	if res := vk.call(vk.createDescriptorPool, device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{
		SType:              vkStructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     r.descriptorPool,
		DescriptorSetCount: 1,
		PSetLayouts:        uintptr(unsafe.Pointer(&r.setLayout)),
	}
	if res := vk.call(vk.allocateDescriptorSets, device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}

	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: uint32(pushConstantBytes)}
	plci := vkPipelineLayoutCreateInfo{
		SType:                  vkStructureTypePipelineLayoutCreateInfo,
		SetLayoutCount:         1,
		PSetLayouts:            uintptr(unsafe.Pointer(&r.setLayout)),
		PushConstantRangeCount: 1,
		PPushConstantRanges:    uintptr(unsafe.Pointer(&pushRange)),
	}
	if res := vk.call(vk.createPipelineLayout, device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := vk.call(vk.createShaderModule, device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.pipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	cpci2 := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, device, uintptr(unsafe.Pointer(&cpci2)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{SType: vkStructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vkCommandBufferLevelPrimary, CommandBufferCount: 1}
	if res := vk.call(vk.allocateCommandBuffers, device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanFusedMatVec3F32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.outABuf)
	r.vk.destroyBuffer(r.device, r.outBBuf)
	r.vk.destroyBuffer(r.device, r.outCBuf)
	r.vk.destroyBuffer(r.device, r.cosBuf)
	r.vk.destroyBuffer(r.device, r.sinBuf)
	for _, b := range r.weightBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	if r.descriptorPool != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorPool, r.device, r.descriptorPool, 0)
	}
	if r.pipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.pipelineLayout, 0)
	}
	if r.setLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.setLayout, 0)
	}
	if !r.sharedDevice {
		if r.device != 0 {
			r.vk.callVoid(r.vk.destroyDevice, r.device, 0)
		}
		if r.instance != 0 {
			r.vk.callVoid(r.vk.destroyInstance, r.instance, 0)
		}
	}
}

func (r *vulkanFusedMatVec3F32WinRunner) run(outA, outB, outC, x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	waLen, err := checkedMatVecF32WeightLenWin(rowsA, cols, "Vulkan f32 fused matvec3 runner wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenWin(rowsB, cols, "Vulkan f32 fused matvec3 runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenWin(rowsC, cols, "Vulkan f32 fused matvec3 runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 fused matvec3 runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrWin(waLen, "Vulkan f32 fused matvec3 runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrWin(wbLen, "Vulkan f32 fused matvec3 runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrWin(wcLen, "Vulkan f32 fused matvec3 runner wc")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrWin(rowsA, "Vulkan f32 fused matvec3 runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan f32 fused matvec3 runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan f32 fused matvec3 runner outC")
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
	waBuf, err := r.weightBuffer(wa[:rowsA*cols], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:rowsB*cols], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:rowsC*cols], wcBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.xBuf, x[:cols]); err != nil {
		return err
	}

	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: waBuf.buffer, Range: waBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: r.outABuf.buffer, Range: outABytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outCBuf.buffer, Range: outCBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, 0); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := vk.readFloat32Into(device, r.outABuf, outA[:rowsA]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, r.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, r.outCBuf, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3F32WinRunner) runMRoPE(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable []float32, rowsA, rowsB, rowsC, cols, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	half := headDim / 2
	waLen, err := checkedMatVecF32WeightLenWin(rowsA, cols, "Vulkan f32 fused matvec3 mrope runner wa")
	if err != nil {
		return err
	}
	wbLen, err := checkedMatVecF32WeightLenWin(rowsB, cols, "Vulkan f32 fused matvec3 mrope runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenWin(rowsC, cols, "Vulkan f32 fused matvec3 mrope runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 fused matvec3 mrope runner x")
	if err != nil {
		return err
	}
	waBytes, err := checkedFloat32ByteLenErrWin(waLen, "Vulkan f32 fused matvec3 mrope runner wa")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrWin(wbLen, "Vulkan f32 fused matvec3 mrope runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrWin(wcLen, "Vulkan f32 fused matvec3 mrope runner wc")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan f32 fused matvec3 mrope runner table")
	if err != nil {
		return err
	}
	outABytes, err := checkedFloat32ByteLenErrWin(rowsA, "Vulkan f32 fused matvec3 mrope runner outA")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan f32 fused matvec3 mrope runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan f32 fused matvec3 mrope runner outC")
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
	waBuf, err := r.weightBuffer(wa[:rowsA*cols], waBytes)
	if err != nil {
		return err
	}
	wbBuf, err := r.weightBuffer(wb[:rowsB*cols], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:rowsC*cols], wcBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.sinBuf, sinTable[:half]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: waBuf.buffer, Range: waBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: r.cosBuf.buffer, Range: tableBytes},
		{Buffer: r.sinBuf.buffer, Range: tableBytes},
		{Buffer: r.outABuf.buffer, Range: outABytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outCBuf.buffer, Range: outCBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != rowsA || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(rowsA, rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := vk.readFloat32Into(device, r.outABuf, outA[:rowsA]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, r.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, r.outCBuf, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3F32WinRunner) run2MRoPE(outB, outC, x, wb, wc, cosTable, sinTable []float32, rowsB, rowsC, cols, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	vk := r.vk
	device := r.device
	half := headDim / 2
	wbLen, err := checkedMatVecF32WeightLenWin(rowsB, cols, "Vulkan f32 fused matvec2 mrope runner wb")
	if err != nil {
		return err
	}
	wcLen, err := checkedMatVecF32WeightLenWin(rowsC, cols, "Vulkan f32 fused matvec2 mrope runner wc")
	if err != nil {
		return err
	}
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan f32 fused matvec2 mrope runner x")
	if err != nil {
		return err
	}
	wbBytes, err := checkedFloat32ByteLenErrWin(wbLen, "Vulkan f32 fused matvec2 mrope runner wb")
	if err != nil {
		return err
	}
	wcBytes, err := checkedFloat32ByteLenErrWin(wcLen, "Vulkan f32 fused matvec2 mrope runner wc")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan f32 fused matvec2 mrope runner table")
	if err != nil {
		return err
	}
	outBBytes, err := checkedFloat32ByteLenErrWin(rowsB, "Vulkan f32 fused matvec2 mrope runner outB")
	if err != nil {
		return err
	}
	outCBytes, err := checkedFloat32ByteLenErrWin(rowsC, "Vulkan f32 fused matvec2 mrope runner outC")
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
	wbBuf, err := r.weightBuffer(wb[:rowsB*cols], wbBytes)
	if err != nil {
		return err
	}
	wcBuf, err := r.weightBuffer(wc[:rowsC*cols], wcBytes)
	if err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := vk.writeFloat32(device, r.sinBuf, sinTable[:half]); err != nil {
		return err
	}
	bufInfos := [...]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wbBuf.buffer, Range: wbBytes},
		{Buffer: wcBuf.buffer, Range: wcBytes},
		{Buffer: r.cosBuf.buffer, Range: tableBytes},
		{Buffer: r.sinBuf.buffer, Range: tableBytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outBBuf.buffer, Range: outBBytes},
		{Buffer: r.outCBuf.buffer, Range: outCBytes},
	}
	updateVulkanDescriptorBuffersWin(vk, device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != 0 || r.commandRowsB != rowsB || r.commandRowsC != rowsC || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(0, rowsB, rowsC, cols, packed); err != nil {
			return err
		}
	}
	if res := vk.call(vk.resetFences, device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := vk.call(vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := vk.call(vk.waitForFences, device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := vk.readFloat32Into(device, r.outBBuf, outB[:rowsB]); err != nil {
		return err
	}
	if err := vk.readFloat32Into(device, r.outCBuf, outC[:rowsC]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3F32WinRunner) recordCommand(rowsA, rowsB, rowsC, cols, packed int) error {
	vk := r.vk
	if res := vk.call(vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := vk.call(vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
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
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(pushBytes), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groups), 1, 1)
	if res := vk.call(vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRowsA = rowsA
	r.commandRowsB = rowsB
	r.commandRowsC = rowsC
	r.commandCols = cols
	r.commandPacked = packed
	r.commandRecorded = true
	return nil
}

// dispatchIntoFused3 records a fused matvec3 dispatch into an external command
// buffer (for chaining).  Does NOT call recordCommand or resetCommandPool.
// The caller must have already set up the descriptor set and input buffers.
func (r *vulkanFusedMatVec3F32WinRunner) dispatchIntoFused3(cmd uintptr, rowsA, rowsB, rowsC, cols, packed int) {
	vk := r.vk
	vk.callVoid(vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	vk.callVoid(vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
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
	vk.callVoid(vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(pushBytes), uintptr(unsafe.Pointer(&pc[0])))
	vk.callVoid(vk.cmdDispatch, cmd, uintptr(groups), 1, 1)
}

func (r *vulkanFusedMatVec3F32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
	if buf.buffer != 0 && buf.size >= size {
		return nil
	}
	if buf.buffer != 0 || buf.memory != 0 {
		r.vk.destroyBuffer(r.device, *buf)
		*buf = vkHostBufferWin{}
	}
	next, err := r.vk.newHostBuffer(r.device, r.memProps, size)
	if err != nil {
		return err
	}
	*buf = next
	return nil
}

func (r *vulkanFusedMatVec3F32WinRunner) weightBuffer(w []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, w, size, r.weightBuffers)
}

type vulkanWin struct {
	dll                        *syscall.LazyDLL
	createInstance             *syscall.LazyProc
	destroyInstance            *syscall.LazyProc
	enumeratePhysicalDevices   *syscall.LazyProc
	getPhysicalDeviceQueues    *syscall.LazyProc
	getPhysicalDeviceMemory    *syscall.LazyProc
	createDevice               *syscall.LazyProc
	destroyDevice              *syscall.LazyProc
	getDeviceQueue             *syscall.LazyProc
	createBuffer               *syscall.LazyProc
	destroyBufferProc          *syscall.LazyProc
	getBufferMemoryReqs        *syscall.LazyProc
	allocateMemory             *syscall.LazyProc
	freeMemory                 *syscall.LazyProc
	bindBufferMemory           *syscall.LazyProc
	mapMemory                  *syscall.LazyProc
	unmapMemory                *syscall.LazyProc
	createDescriptorSetLayout  *syscall.LazyProc
	destroyDescriptorSetLayout *syscall.LazyProc
	createPipelineLayout       *syscall.LazyProc
	destroyPipelineLayout      *syscall.LazyProc
	createShaderModule         *syscall.LazyProc
	destroyShaderModule        *syscall.LazyProc
	createComputePipelines     *syscall.LazyProc
	destroyPipeline            *syscall.LazyProc
	createDescriptorPool       *syscall.LazyProc
	destroyDescriptorPool      *syscall.LazyProc
	allocateDescriptorSets     *syscall.LazyProc
	updateDescriptorSets       *syscall.LazyProc
	createCommandPool          *syscall.LazyProc
	destroyCommandPool         *syscall.LazyProc
	resetCommandPool           *syscall.LazyProc
	allocateCommandBuffers     *syscall.LazyProc
	beginCommandBuffer         *syscall.LazyProc
	endCommandBuffer           *syscall.LazyProc
	cmdBindPipeline            *syscall.LazyProc
	cmdBindDescriptorSets      *syscall.LazyProc
	cmdPushConstants           *syscall.LazyProc
	cmdDispatch                *syscall.LazyProc
	cmdPipelineBarrier         *syscall.LazyProc
	cmdCopyBuffer              *syscall.LazyProc
	createFence                *syscall.LazyProc
	destroyFence               *syscall.LazyProc
	resetFences                *syscall.LazyProc
	queueSubmit                *syscall.LazyProc
	waitForFences              *syscall.LazyProc
}

func newVulkanWin(name string) *vulkanWin {
	return &vulkanWin{dll: syscall.NewLazyDLL(name)}
}

func (v *vulkanWin) load() error {
	v.createInstance = v.dll.NewProc("vkCreateInstance")
	v.destroyInstance = v.dll.NewProc("vkDestroyInstance")
	v.enumeratePhysicalDevices = v.dll.NewProc("vkEnumeratePhysicalDevices")
	v.getPhysicalDeviceQueues = v.dll.NewProc("vkGetPhysicalDeviceQueueFamilyProperties")
	v.getPhysicalDeviceMemory = v.dll.NewProc("vkGetPhysicalDeviceMemoryProperties")
	v.createDevice = v.dll.NewProc("vkCreateDevice")
	v.destroyDevice = v.dll.NewProc("vkDestroyDevice")
	v.getDeviceQueue = v.dll.NewProc("vkGetDeviceQueue")
	v.createBuffer = v.dll.NewProc("vkCreateBuffer")
	v.destroyBufferProc = v.dll.NewProc("vkDestroyBuffer")
	v.getBufferMemoryReqs = v.dll.NewProc("vkGetBufferMemoryRequirements")
	v.allocateMemory = v.dll.NewProc("vkAllocateMemory")
	v.freeMemory = v.dll.NewProc("vkFreeMemory")
	v.bindBufferMemory = v.dll.NewProc("vkBindBufferMemory")
	v.mapMemory = v.dll.NewProc("vkMapMemory")
	v.unmapMemory = v.dll.NewProc("vkUnmapMemory")
	v.createDescriptorSetLayout = v.dll.NewProc("vkCreateDescriptorSetLayout")
	v.destroyDescriptorSetLayout = v.dll.NewProc("vkDestroyDescriptorSetLayout")
	v.createPipelineLayout = v.dll.NewProc("vkCreatePipelineLayout")
	v.destroyPipelineLayout = v.dll.NewProc("vkDestroyPipelineLayout")
	v.createShaderModule = v.dll.NewProc("vkCreateShaderModule")
	v.destroyShaderModule = v.dll.NewProc("vkDestroyShaderModule")
	v.createComputePipelines = v.dll.NewProc("vkCreateComputePipelines")
	v.destroyPipeline = v.dll.NewProc("vkDestroyPipeline")
	v.createDescriptorPool = v.dll.NewProc("vkCreateDescriptorPool")
	v.destroyDescriptorPool = v.dll.NewProc("vkDestroyDescriptorPool")
	v.allocateDescriptorSets = v.dll.NewProc("vkAllocateDescriptorSets")
	v.updateDescriptorSets = v.dll.NewProc("vkUpdateDescriptorSets")
	v.createCommandPool = v.dll.NewProc("vkCreateCommandPool")
	v.destroyCommandPool = v.dll.NewProc("vkDestroyCommandPool")
	v.resetCommandPool = v.dll.NewProc("vkResetCommandPool")
	v.allocateCommandBuffers = v.dll.NewProc("vkAllocateCommandBuffers")
	v.beginCommandBuffer = v.dll.NewProc("vkBeginCommandBuffer")
	v.endCommandBuffer = v.dll.NewProc("vkEndCommandBuffer")
	v.cmdBindPipeline = v.dll.NewProc("vkCmdBindPipeline")
	v.cmdBindDescriptorSets = v.dll.NewProc("vkCmdBindDescriptorSets")
	v.cmdPushConstants = v.dll.NewProc("vkCmdPushConstants")
	v.cmdDispatch = v.dll.NewProc("vkCmdDispatch")
	v.cmdPipelineBarrier = v.dll.NewProc("vkCmdPipelineBarrier")
	v.cmdCopyBuffer = v.dll.NewProc("vkCmdCopyBuffer")
	v.createFence = v.dll.NewProc("vkCreateFence")
	v.destroyFence = v.dll.NewProc("vkDestroyFence")
	v.resetFences = v.dll.NewProc("vkResetFences")
	v.queueSubmit = v.dll.NewProc("vkQueueSubmit")
	v.waitForFences = v.dll.NewProc("vkWaitForFences")
	for _, p := range []*syscall.LazyProc{v.createInstance, v.destroyInstance, v.enumeratePhysicalDevices, v.getPhysicalDeviceQueues, v.getPhysicalDeviceMemory, v.createDevice, v.destroyDevice, v.getDeviceQueue, v.createBuffer, v.destroyBufferProc, v.getBufferMemoryReqs, v.allocateMemory, v.freeMemory, v.bindBufferMemory, v.mapMemory, v.unmapMemory, v.createDescriptorSetLayout, v.destroyDescriptorSetLayout, v.createPipelineLayout, v.destroyPipelineLayout, v.createShaderModule, v.destroyShaderModule, v.createComputePipelines, v.destroyPipeline, v.createDescriptorPool, v.destroyDescriptorPool, v.allocateDescriptorSets, v.updateDescriptorSets, v.createCommandPool, v.destroyCommandPool, v.resetCommandPool, v.allocateCommandBuffers, v.beginCommandBuffer, v.endCommandBuffer, v.cmdBindPipeline, v.cmdBindDescriptorSets, v.cmdPushConstants, v.cmdDispatch, v.cmdPipelineBarrier, v.cmdCopyBuffer, v.createFence, v.destroyFence, v.resetFences, v.queueSubmit, v.waitForFences} {
		if err := p.Find(); err != nil {
			return err
		}
	}
	return nil
}

func (v *vulkanWin) call(p *syscall.LazyProc, args ...uintptr) uintptr {
	r1, _, _ := p.Call(args...)
	return r1
}

func (v *vulkanWin) callVoid(p *syscall.LazyProc, args ...uintptr) {
	p.Call(args...)
}

func (v *vulkanWin) selectComputeDevice(gpus []uintptr) (uintptr, uint32, vkPhysicalDeviceMemoryProperties, error) {
	for _, gpu := range gpus {
		var count uint32
		v.callVoid(v.getPhysicalDeviceQueues, gpu, uintptr(unsafe.Pointer(&count)), 0)
		if count == 0 {
			continue
		}
		props := make([]vkQueueFamilyProperties, count)
		v.callVoid(v.getPhysicalDeviceQueues, gpu, uintptr(unsafe.Pointer(&count)), uintptr(unsafe.Pointer(&props[0])))
		for i, p := range props {
			if p.QueueCount > 0 && p.QueueFlags&vkQueueComputeBit != 0 {
				var mem vkPhysicalDeviceMemoryProperties
				v.callVoid(v.getPhysicalDeviceMemory, gpu, uintptr(unsafe.Pointer(&mem)))
				return gpu, uint32(i), mem, nil
			}
		}
	}
	return 0, 0, vkPhysicalDeviceMemoryProperties{}, fmt.Errorf("no Vulkan compute queue found")
}

type vkHostBufferWin struct {
	buffer uintptr
	memory uintptr
	size   uint64
	mapped unsafe.Pointer
}

func (v *vulkanWin) newHostBuffer(device uintptr, memProps vkPhysicalDeviceMemoryProperties, size uint64) (vkHostBufferWin, error) {
	var out vkHostBufferWin
	out.size = size
	bci := vkBufferCreateInfo{SType: vkStructureTypeBufferCreateInfo, Size: size, Usage: vkBufferUsageStorageBufferBit, SharingMode: vkSharingModeExclusive}
	if res := v.call(v.createBuffer, device, uintptr(unsafe.Pointer(&bci)), 0, uintptr(unsafe.Pointer(&out.buffer))); res != vkSuccess {
		return out, fmt.Errorf("vkCreateBuffer: %d", int32(res))
	}
	var req vkMemoryRequirements
	v.callVoid(v.getBufferMemoryReqs, device, out.buffer, uintptr(unsafe.Pointer(&req)))
	memType, ok := findVulkanMemoryTypeWindows(memProps, req.MemoryTypeBits, vkMemoryPropertyHostVisibleBit|vkMemoryPropertyHostCoherentBit)
	if !ok {
		v.destroyBuffer(device, out)
		return vkHostBufferWin{}, fmt.Errorf("no host visible coherent memory type")
	}
	mai := vkMemoryAllocateInfo{SType: vkStructureTypeMemoryAllocateInfo, AllocationSize: req.Size, MemoryTypeIndex: memType}
	if res := v.call(v.allocateMemory, device, uintptr(unsafe.Pointer(&mai)), 0, uintptr(unsafe.Pointer(&out.memory))); res != vkSuccess {
		v.destroyBuffer(device, out)
		return vkHostBufferWin{}, fmt.Errorf("vkAllocateMemory: %d", int32(res))
	}
	if res := v.call(v.bindBufferMemory, device, out.buffer, out.memory, 0); res != vkSuccess {
		v.destroyBuffer(device, out)
		return vkHostBufferWin{}, fmt.Errorf("vkBindBufferMemory: %d", int32(res))
	}
	if res := v.call(v.mapMemory, device, out.memory, 0, uintptr(out.size), 0, uintptr(unsafe.Pointer(&out.mapped))); res != vkSuccess {
		v.destroyBuffer(device, out)
		return vkHostBufferWin{}, fmt.Errorf("vkMapMemory persistent: %d", int32(res))
	}
	return out, nil
}

func (v *vulkanWin) destroyBuffer(device uintptr, b vkHostBufferWin) {
	if b.mapped != nil {
		v.callVoid(v.unmapMemory, device, b.memory)
	}
	if b.buffer != 0 {
		v.callVoid(v.destroyBufferProc, device, b.buffer, 0)
	}
	if b.memory != 0 {
		v.callVoid(v.freeMemory, device, b.memory, 0)
	}
}

func (v *vulkanWin) writeFloat32(device uintptr, b vkHostBufferWin, values []float32) error {
	return v.writeFloat32At(device, b, 0, values)
}

func (v *vulkanWin) writeFloat32At(device uintptr, b vkHostBufferWin, offsetFloats int, values []float32) error {
	if offsetFloats < 0 || len(values) > maxInt()-offsetFloats {
		return fmt.Errorf("vkMapMemory write range offset=%d len=%d bufferBytes=%d", offsetFloats, len(values), b.size)
	}
	totalBytes, err := checkedFloat32ByteLenErrWin(offsetFloats+len(values), "vkMapMemory write")
	if err != nil || totalBytes > b.size {
		return fmt.Errorf("vkMapMemory write range offset=%d len=%d bufferBytes=%d", offsetFloats, len(values), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vkMapMemory write on unmapped buffer")
	}
	base := unsafe.Slice((*float32)(b.mapped), int(b.size/4))
	dst := base[offsetFloats : offsetFloats+len(values)]
	copy(dst, values)
	return nil
}

func (v *vulkanWin) writeRowsPrefix(device uintptr, b vkHostBufferWin, rows [][]float32, n, cols int) error {
	total, ok := checkedMulInt(n, cols)
	if n < 0 || cols < 0 || !ok || len(rows) < n {
		return fmt.Errorf("vkMapMemory rows write range rows=%d n=%d cols=%d bufferBytes=%d", len(rows), n, cols, b.size)
	}
	totalBytes, err := checkedFloat32ByteLenErrWin(total, "vkMapMemory rows write")
	if err != nil || totalBytes > b.size {
		return fmt.Errorf("vkMapMemory rows write range rows=%d n=%d cols=%d bufferBytes=%d", len(rows), n, cols, b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vkMapMemory rows write on unmapped buffer")
	}
	dst := unsafe.Slice((*float32)(b.mapped), int(b.size/4))
	for i := 0; i < n; i++ {
		if len(rows[i]) < cols {
			return fmt.Errorf("vkMapMemory rows write short row=%d len=%d cols=%d", i, len(rows[i]), cols)
		}
		copy(dst[i*cols:(i+1)*cols], rows[i][:cols])
	}
	return nil
}

func (v *vulkanWin) readFloat32(device uintptr, b vkHostBufferWin, n int) ([]float32, error) {
	if b.mapped == nil {
		return nil, fmt.Errorf("vkMapMemory read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), n)
	out := make([]float32, n)
	copy(out, src)
	return out, nil
}

func (v *vulkanWin) readFloat32Into(device uintptr, b vkHostBufferWin, out []float32) error {
	readBytes, err := checkedFloat32ByteLenErrWin(len(out), "vkMapMemory read")
	if err != nil || readBytes > b.size {
		return fmt.Errorf("vkMapMemory read range len=%d bufferBytes=%d", len(out), b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vkMapMemory read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), len(out))
	copy(out, src)
	return nil
}

func (v *vulkanWin) readRowsPrefixInto(device uintptr, b vkHostBufferWin, out [][]float32, n, cols int) error {
	readTotal, ok := checkedMulInt(n, cols)
	if n < 0 || cols < 0 || !ok || len(out) < n {
		return fmt.Errorf("vkMapMemory rows read range rows=%d n=%d cols=%d bufferBytes=%d", len(out), n, cols, b.size)
	}
	readBytes, err := checkedFloat32ByteLenErrWin(readTotal, "vkMapMemory rows read")
	if err != nil || readBytes > b.size {
		return fmt.Errorf("vkMapMemory rows read range rows=%d n=%d cols=%d bufferBytes=%d", len(out), n, cols, b.size)
	}
	if b.mapped == nil {
		return fmt.Errorf("vkMapMemory rows read on unmapped buffer")
	}
	src := unsafe.Slice((*float32)(b.mapped), n*cols)
	for i := 0; i < n; i++ {
		if len(out[i]) < cols {
			return fmt.Errorf("vkMapMemory rows read short row=%d len=%d cols=%d", i, len(out[i]), cols)
		}
		copy(out[i][:cols], src[i*cols:(i+1)*cols])
	}
	return nil
}

func findVulkanMemoryTypeWindows(mem vkPhysicalDeviceMemoryProperties, typeBits, flags uint32) (uint32, bool) {
	for i := uint32(0); i < mem.MemoryTypeCount; i++ {
		if typeBits&(1<<i) != 0 && mem.MemoryTypes[i].PropertyFlags&flags == flags {
			return i, true
		}
	}
	return 0, false
}

func compileVulkanGLSLWindows(source string) ([]uint32, error) {
	compiler, err := findVulkanShaderCompilerWindows()
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
	cmd := exec.Command(compiler, "-fshader-stage=compute", "-std=450core", src, "-o", dst)
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

func vulkanMatVecF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMatVecF32SPV.once.Do(func() {
		vulkanMatVecF32SPV.code, vulkanMatVecF32SPV.err = compileVulkanGLSLWindows(vulkanMatVecF32GLSL)
	})
	return vulkanMatVecF32SPV.code, vulkanMatVecF32SPV.err
}

func vulkanArgmaxF32ShaderCodeWindows() ([]uint32, error) {
	vulkanArgmaxF32SPV.once.Do(func() {
		vulkanArgmaxF32SPV.code, vulkanArgmaxF32SPV.err = compileVulkanGLSLWindows(vulkanArgmaxF32GLSL)
	})
	return vulkanArgmaxF32SPV.code, vulkanArgmaxF32SPV.err
}

func vulkanArgmaxQuantizedF32ShaderCodeWindows() ([]uint32, error) {
	vulkanArgmaxQuantizedF32SPV.once.Do(func() {
		vulkanArgmaxQuantizedF32SPV.code, vulkanArgmaxQuantizedF32SPV.err = compileVulkanGLSLWindows(vulkanArgmaxQuantizedF32GLSL)
	})
	return vulkanArgmaxQuantizedF32SPV.code, vulkanArgmaxQuantizedF32SPV.err
}

func vulkanBlockTopKF32ShaderCodeWindows() ([]uint32, error) {
	vulkanBlockTopKF32SPV.once.Do(func() {
		vulkanBlockTopKF32SPV.code, vulkanBlockTopKF32SPV.err = compileVulkanGLSLWindows(vulkanBlockTopKF32GLSL)
	})
	return vulkanBlockTopKF32SPV.code, vulkanBlockTopKF32SPV.err
}

func vulkanBlockTopKQuantizedF32ShaderCodeWindows() ([]uint32, error) {
	vulkanBlockTopKQuantizedF32SPV.once.Do(func() {
		vulkanBlockTopKQuantizedF32SPV.code, vulkanBlockTopKQuantizedF32SPV.err = compileVulkanGLSLWindows(vulkanBlockTopKQuantizedF32GLSL)
	})
	return vulkanBlockTopKQuantizedF32SPV.code, vulkanBlockTopKQuantizedF32SPV.err
}

func vulkanRMSNormF32ShaderCodeWindows() ([]uint32, error) {
	vulkanRMSNormF32SPV.once.Do(func() {
		vulkanRMSNormF32SPV.code, vulkanRMSNormF32SPV.err = compileVulkanGLSLWindows(vulkanRMSNormF32PlanGLSL)
	})
	return vulkanRMSNormF32SPV.code, vulkanRMSNormF32SPV.err
}

func vulkanAddRMSNormF32ShaderCodeWindows() ([]uint32, error) {
	vulkanAddRMSNormF32SPV.once.Do(func() {
		vulkanAddRMSNormF32SPV.code, vulkanAddRMSNormF32SPV.err = compileVulkanGLSLWindows(vulkanAddRMSNormF32GLSL)
	})
	return vulkanAddRMSNormF32SPV.code, vulkanAddRMSNormF32SPV.err
}

func vulkanMRoPEF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMRoPEF32SPV.once.Do(func() {
		vulkanMRoPEF32SPV.code, vulkanMRoPEF32SPV.err = compileVulkanGLSLWindows(vulkanMRoPEF32GLSL)
	})
	return vulkanMRoPEF32SPV.code, vulkanMRoPEF32SPV.err
}

func vulkanMRoPEPairF32ShaderCodeWindows() ([]uint32, error) {
	vulkanMRoPEPairF32SPV.once.Do(func() {
		vulkanMRoPEPairF32SPV.code, vulkanMRoPEPairF32SPV.err = compileVulkanGLSLWindows(vulkanMRoPEPairF32GLSL)
	})
	return vulkanMRoPEPairF32SPV.code, vulkanMRoPEPairF32SPV.err
}

func vulkanFusedMatVec3F32ShaderCodeWindows() ([]uint32, error) {
	vulkanFusedMatVec3F32SPV.once.Do(func() {
		vulkanFusedMatVec3F32SPV.code, vulkanFusedMatVec3F32SPV.err = compileVulkanGLSLWindows(vulkanFusedQKVF32GLSL)
	})
	return vulkanFusedMatVec3F32SPV.code, vulkanFusedMatVec3F32SPV.err
}

func vulkanFusedMatVec3MRoPEF32ShaderCodeWindows() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEF32SPV.once.Do(func() {
		vulkanFusedMatVec3MRoPEF32SPV.code, vulkanFusedMatVec3MRoPEF32SPV.err = compileVulkanGLSLWindows(vulkanFusedQKVMRoPEF32GLSL)
	})
	return vulkanFusedMatVec3MRoPEF32SPV.code, vulkanFusedMatVec3MRoPEF32SPV.err
}

func findVulkanShaderCompilerWindows() (string, error) {
	for _, name := range []string{"glslc", "glslangValidator"} {
		if path, err := exec.LookPath(name); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("glslc or glslangValidator not found")
}

func vkMakeVersion(major, minor, patch uint32) uint32 {
	return major<<22 | minor<<12 | patch
}

func absFloat32Windows(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

const (
	vkSuccess                                    = 0
	vkStructureTypeApplicationInfo               = 0
	vkStructureTypeInstanceCreateInfo            = 1
	vkStructureTypeDeviceQueueCreateInfo         = 2
	vkStructureTypeDeviceCreateInfo              = 3
	vkStructureTypeMemoryAllocateInfo            = 5
	vkStructureTypeFenceCreateInfo               = 8
	vkStructureTypeSubmitInfo                    = 4
	vkStructureTypeBufferCreateInfo              = 12
	vkStructureTypeShaderModuleCreateInfo        = 16
	vkStructureTypePipelineShaderStageCreateInfo = 18
	vkStructureTypeComputePipelineCreateInfo     = 29
	vkStructureTypePipelineLayoutCreateInfo      = 30
	vkStructureTypeDescriptorSetLayoutCreateInfo = 32
	vkStructureTypeDescriptorPoolCreateInfo      = 33
	vkStructureTypeDescriptorSetAllocateInfo     = 34
	vkStructureTypeWriteDescriptorSet            = 35
	vkStructureTypeCommandPoolCreateInfo         = 39
	vkStructureTypeCommandBufferAllocateInfo     = 40
	vkStructureTypeCommandBufferBeginInfo        = 42
	vkStructureTypeMemoryBarrier                 = 46
	vkQueueComputeBit                            = 2
	vkMemoryPropertyHostVisibleBit               = 2
	vkMemoryPropertyHostCoherentBit              = 4
	vkAccessShaderReadBit                        = 0x00000020
	vkAccessShaderWriteBit                       = 0x00000040
	vkAccessTransferReadBit                      = 0x00000800
	vkAccessTransferWriteBit                     = 0x00001000
	vkPipelineStageComputeShaderBit              = 0x00000800
	vkPipelineStageTransferBit                   = 0x00001000
	vkPipelineStageHostBit                       = 0x00000000
	vkSharingModeExclusive                       = 0
	vkDescriptorTypeStorageBuffer                = 7
	vkPipelineBindPointCompute                   = 1
	vkCommandBufferLevelPrimary                  = 0
	vkCommandBufferUsageOneTimeSubmitBit         = 1
	vkBufferUsageStorageBufferBit                = 32
	vkShaderStageComputeBit                      = 32
)

type vkApplicationInfo struct {
	SType              uint32
	PNext              uintptr
	PApplicationName   uintptr
	ApplicationVersion uint32
	PEngineName        uintptr
	EngineVersion      uint32
	APIVersion         uint32
}

type vkInstanceCreateInfo struct {
	SType                   uint32
	PNext                   uintptr
	Flags                   uint32
	PApplicationInfo        uintptr
	EnabledLayerCount       uint32
	PpEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames uintptr
}

type vkExtent3D struct{ Width, Height, Depth uint32 }

type vkQueueFamilyProperties struct {
	QueueFlags                  uint32
	QueueCount                  uint32
	TimestampValidBits          uint32
	MinImageTransferGranularity vkExtent3D
}

type vkMemoryType struct{ PropertyFlags, HeapIndex uint32 }
type vkMemoryHeap struct {
	Size  uint64
	Flags uint32
	_     uint32
}
type vkPhysicalDeviceMemoryProperties struct {
	MemoryTypeCount uint32
	MemoryTypes     [32]vkMemoryType
	MemoryHeapCount uint32
	_               uint32
	MemoryHeaps     [16]vkMemoryHeap
}

type vkDeviceQueueCreateInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
	QueueCount       uint32
	_                uint32
	PQueuePriorities uintptr
}

type vkDeviceCreateInfo struct {
	SType                   uint32
	PNext                   uintptr
	Flags                   uint32
	QueueCreateInfoCount    uint32
	PQueueCreateInfos       uintptr
	EnabledLayerCount       uint32
	PpEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames uintptr
	PEnabledFeatures        uintptr
}

type vkBufferCreateInfo struct {
	SType                 uint32
	PNext                 uintptr
	Flags                 uint32
	_                     uint32
	Size                  uint64
	Usage                 uint32
	SharingMode           uint32
	QueueFamilyIndexCount uint32
	_                     uint32
	PQueueFamilyIndices   uintptr
}

type vkMemoryRequirements struct {
	Size           uint64
	Alignment      uint64
	MemoryTypeBits uint32
	_              uint32
}

type vkMemoryAllocateInfo struct {
	SType           uint32
	PNext           uintptr
	AllocationSize  uint64
	MemoryTypeIndex uint32
	_               uint32
}

type vkMemoryBarrier struct {
	SType         uint32
	PNext         uintptr
	SrcAccessMask uint32
	DstAccessMask uint32
}

type vkDescriptorSetLayoutBinding struct {
	Binding            uint32
	DescriptorType     uint32
	DescriptorCount    uint32
	StageFlags         uint32
	PImmutableSamplers uintptr
}

type vkDescriptorSetLayoutCreateInfo struct {
	SType        uint32
	PNext        uintptr
	Flags        uint32
	BindingCount uint32
	PBindings    uintptr
}

type vkPushConstantRange struct{ StageFlags, Offset, Size uint32 }

type vkPipelineLayoutCreateInfo struct {
	SType                  uint32
	PNext                  uintptr
	Flags                  uint32
	SetLayoutCount         uint32
	PSetLayouts            uintptr
	PushConstantRangeCount uint32
	PPushConstantRanges    uintptr
}

type vkShaderModuleCreateInfo struct {
	SType    uint32
	PNext    uintptr
	Flags    uint32
	_        uint32
	CodeSize uintptr
	PCode    uintptr
}

type vkPipelineShaderStageCreateInfo struct {
	SType               uint32
	PNext               uintptr
	Flags               uint32
	Stage               uint32
	Module              uintptr
	PName               uintptr
	PSpecializationInfo uintptr
}

type vkComputePipelineCreateInfo struct {
	SType              uint32
	PNext              uintptr
	Flags              uint32
	_                  uint32
	Stage              vkPipelineShaderStageCreateInfo
	Layout             uintptr
	BasePipelineHandle uintptr
	BasePipelineIndex  int32
	_                  uint32
}

type vkDescriptorPoolSize struct{ Type, DescriptorCount uint32 }
type vkDescriptorPoolCreateInfo struct {
	SType         uint32
	PNext         uintptr
	Flags         uint32
	MaxSets       uint32
	PoolSizeCount uint32
	_             uint32
	PPoolSizes    uintptr
}

type vkDescriptorSetAllocateInfo struct {
	SType              uint32
	PNext              uintptr
	DescriptorPool     uintptr
	DescriptorSetCount uint32
	_                  uint32
	PSetLayouts        uintptr
}

type vkDescriptorBufferInfo struct {
	Buffer uintptr
	Offset uint64
	Range  uint64
}

type vkWriteDescriptorSet struct {
	SType            uint32
	PNext            uintptr
	DstSet           uintptr
	DstBinding       uint32
	DstArrayElement  uint32
	DescriptorCount  uint32
	DescriptorType   uint32
	PImageInfo       uintptr
	PBufferInfo      uintptr
	PTexelBufferView uintptr
}

type vkCommandPoolCreateInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
}

type vkCommandBufferAllocateInfo struct {
	SType              uint32
	PNext              uintptr
	CommandPool        uintptr
	Level              uint32
	CommandBufferCount uint32
}

type vkCommandBufferBeginInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	_                uint32
	PInheritanceInfo uintptr
}

type vkFenceCreateInfo struct {
	SType uint32
	PNext uintptr
	Flags uint32
	_     uint32
}

type vkSubmitInfo struct {
	SType                uint32
	PNext                uintptr
	WaitSemaphoreCount   uint32
	_                    uint32
	PWaitSemaphores      uintptr
	PWaitDstStageMask    uintptr
	CommandBufferCount   uint32
	_                    uint32
	PCommandBuffers      uintptr
	SignalSemaphoreCount uint32
	_                    uint32
	PSignalSemaphores    uintptr
}
