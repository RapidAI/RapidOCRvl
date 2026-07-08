//go:build windows

package backend

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"paddleocrvl-go/internal/tensor"
)

var vulkanMatVecQ8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanFusedMatVec3Q8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanFusedMatVec3MRoPEQ8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanSwiGLUGateUpQ8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanMatVecQ8RunnerCache struct {
	once   sync.Once
	runner *vulkanMatVecQ8WinRunner
	err    error
}

var vulkanFusedMatVec3Q8RunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8WinRunner
	err    error
}

var vulkanFusedMatVec3MRoPEQ8RunnerCache struct {
	once   sync.Once
	runner *vulkanFusedMatVec3Q8WinRunner
	err    error
}

var vulkanSwiGLUDownQ8RunnerCache struct {
	once   sync.Once
	runner *vulkanSwiGLUDownQ8WinRunner
	err    error
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
	runner, err := getVulkanMatVecQ8RunnerWindows()
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
	runner, err := getVulkanMatVecArgmaxQ8RunnerWindows()
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
	runner, err := getVulkanMatVecQ8RunnerWindows()
	if err != nil {
		return nil, err
	}
	return runner.runTopK(x, q, k)
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
	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runMatVecAddRMSNorm(normOut, residual, x, q, normWeight)
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
	runner, err := getVulkanFusedMatVec3Q8RunnerWindows()
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
	runner, err := getVulkanFusedMatVec3Q8RunnerWindows()
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
	runner, err := getVulkanFusedMatVec3MRoPEQ8RunnerWindows()
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
	runner, err := getVulkanFusedMatVec3MRoPEQ8RunnerWindows()
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
	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
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
	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runGateUp(out, x, gate, up)
}

func VulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
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
	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, gate, up, down, normWeight, true)
}

func VulkanSwiGLUDownAddRMSNormQ8OutOnly(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32) error {
	if gate == nil || up == nil || down == nil {
		return fmt.Errorf("nil Vulkan q8 swiglu/down+add+rmsnorm matrix")
	}
	if gate.Rows <= 0 || gate.Cols <= 0 || up.Rows != gate.Rows || up.Cols != gate.Cols || down.Rows <= 0 || down.Cols != gate.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down+add+rmsnorm shape gate=%dx%d up=%dx%d down=%dx%d", gate.Rows, gate.Cols, up.Rows, up.Cols, down.Rows, down.Cols)
	}
	gateLen, upLen, downLen := gate.Rows*gate.Cols, up.Rows*up.Cols, down.Rows*down.Cols
	if len(normOut) < down.Rows || len(residual) < down.Rows || len(x) < gate.Cols || len(normWeight) < down.Rows ||
		len(gate.Data) < gateLen || len(up.Data) < upLen || len(down.Data) < downLen ||
		len(gate.Scale) < gate.Rows || len(up.Scale) < up.Rows || len(down.Scale) < down.Rows {
		return fmt.Errorf("invalid Vulkan q8 swiglu/down+add+rmsnorm buffers")
	}
	runner, err := getVulkanSwiGLUDownQ8RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runAddRMSNorm(normOut, residual, x, gate, up, down, normWeight, false)
}

func getVulkanMatVecQ8RunnerWindows() (*vulkanMatVecQ8WinRunner, error) {
	vulkanMatVecQ8RunnerCache.once.Do(func() {
		vulkanMatVecQ8RunnerCache.runner, vulkanMatVecQ8RunnerCache.err = newVulkanMatVecQ8WinRunner()
	})
	return vulkanMatVecQ8RunnerCache.runner, vulkanMatVecQ8RunnerCache.err
}

func getVulkanMatVecArgmaxQ8RunnerWindows() (*vulkanMatVecQ8WinRunner, error) {
	return getVulkanMatVecQ8RunnerWindows()
}

func getVulkanFusedMatVec3Q8RunnerWindows() (*vulkanFusedMatVec3Q8WinRunner, error) {
	vulkanFusedMatVec3Q8RunnerCache.once.Do(func() {
		vulkanFusedMatVec3Q8RunnerCache.runner, vulkanFusedMatVec3Q8RunnerCache.err = newVulkanFusedMatVec3Q8WinRunner()
	})
	return vulkanFusedMatVec3Q8RunnerCache.runner, vulkanFusedMatVec3Q8RunnerCache.err
}

func getVulkanFusedMatVec3MRoPEQ8RunnerWindows() (*vulkanFusedMatVec3Q8WinRunner, error) {
	vulkanFusedMatVec3MRoPEQ8RunnerCache.once.Do(func() {
		vulkanFusedMatVec3MRoPEQ8RunnerCache.runner, vulkanFusedMatVec3MRoPEQ8RunnerCache.err = newVulkanFusedMatVec3MRoPEQ8WinRunner()
	})
	return vulkanFusedMatVec3MRoPEQ8RunnerCache.runner, vulkanFusedMatVec3MRoPEQ8RunnerCache.err
}

func getVulkanSwiGLUDownQ8RunnerWindows() (*vulkanSwiGLUDownQ8WinRunner, error) {
	vulkanSwiGLUDownQ8RunnerCache.once.Do(func() {
		vulkanSwiGLUDownQ8RunnerCache.runner, vulkanSwiGLUDownQ8RunnerCache.err = newVulkanSwiGLUDownQ8WinRunner()
	})
	return vulkanSwiGLUDownQ8RunnerCache.runner, vulkanSwiGLUDownQ8RunnerCache.err
}

type vulkanMatVecQ8WinRunner struct {
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
	argmaxPipeline  uintptr
	topKPipeline    uintptr
	commandPool     uintptr
	commandBuffer   uintptr
	fence           uintptr
	xBuf            vkHostBufferWin
	outBuf          vkHostBufferWin
	argmaxBuf       vkHostBufferWin
	dataBuffers     map[uintptr]vulkanCachedInt8BufferWin
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferWin
	topKReadback    []float32
	topKCandidates  []VulkanTokenScore
	descriptorCache [5]vulkanDescriptorBindingWin
	commandKind     int
	commandRecorded bool
	commandRows     int
	commandCols     int
	sharedDevice bool
	mu          sync.Mutex
}

const (
	vulkanMatVecQ8WinCommandDefault = iota + 1
	vulkanMatVecQ8WinCommandArgmax
	vulkanMatVecQ8WinCommandTopK
)

func newVulkanMatVecQ8WinRunner() (*vulkanMatVecQ8WinRunner, error) {
	spv, err := vulkanMatVecQ8ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	argmaxSPV, err := vulkanArgmaxQuantizedF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	topKSPV, err := vulkanBlockTopKQuantizedF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	instance := ctx.instance
	queueFamily := ctx.queueFamily
	entryName := append([]byte("main"), 0)
	r := &vulkanMatVecQ8WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferWin), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 5)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 5}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 8}
	plci := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.pipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	smciArgmax := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(argmaxSPV) * 4), PCode: uintptr(unsafe.Pointer(&argmaxSPV[0]))}
	var argmaxShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smciArgmax)), 0, uintptr(unsafe.Pointer(&argmaxShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule argmax: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, argmaxShader, 0)
	argmaxStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: argmaxShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	argmaxCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: argmaxStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&argmaxCPCI)), 0, uintptr(unsafe.Pointer(&r.argmaxPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines argmax: %d", int32(res))
	}
	smciTopK := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(topKSPV) * 4), PCode: uintptr(unsafe.Pointer(&topKSPV[0]))}
	var topKShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smciTopK)), 0, uintptr(unsafe.Pointer(&topKShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule top-k: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, topKShader, 0)
	topKStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: topKShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	topKCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: topKStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&topKCPCI)), 0, uintptr(unsafe.Pointer(&r.topKPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines top-k: %d", int32(res))
	}
	cpci2 := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, r.device, uintptr(unsafe.Pointer(&cpci2)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{SType: vkStructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vkCommandBufferLevelPrimary, CommandBufferCount: 1}
	if res := vk.call(vk.allocateCommandBuffers, r.device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, r.device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanMatVecQ8WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.argmaxPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.argmaxPipeline, 0)
	}
	if r.topKPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.topKPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.argmaxBuf)
	for _, b := range r.dataBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.scaleBuffers {
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

func (r *vulkanMatVecQ8WinRunner) run(out, x []float32, q *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrWin(q.Cols, "Vulkan q8 matvec runner x")
	if err != nil {
		return err
	}
	dataLen, err := checkedQ8DataLenWin(q.Rows, q.Cols, "Vulkan q8 matvec runner")
	if err != nil {
		return err
	}
	dataBytes, err := checkedAlignedByteLenErrWin(dataLen, 4, "Vulkan q8 matvec runner data")
	if err != nil {
		return err
	}
	scaleBytes, err := checkedFloat32ByteLenErrWin(q.Rows, "Vulkan q8 matvec runner scale")
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:q.Cols]); err != nil {
		return err
	}
	bufInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8WinCommandDefault || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordCommand(q.Rows, q.Cols); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readFloat32Into(r.device, r.outBuf, out[:q.Rows])
}

func (r *vulkanMatVecQ8WinRunner) runArgmax(x []float32, q *tensor.Q8Matrix) (int, float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrWin(q.Cols, "Vulkan q8 matvec argmax runner x")
	if err != nil {
		return 0, 0, err
	}
	dataLen, err := checkedQ8DataLenWin(q.Rows, q.Cols, "Vulkan q8 matvec argmax runner")
	if err != nil {
		return 0, 0, err
	}
	dataBytes, err := checkedAlignedByteLenErrWin(dataLen, 4, "Vulkan q8 matvec argmax runner data")
	if err != nil {
		return 0, 0, err
	}
	scaleBytes, err := checkedFloat32ByteLenErrWin(q.Rows, "Vulkan q8 matvec argmax runner scale")
	if err != nil {
		return 0, 0, err
	}
	outBytes := scaleBytes
	resultBytes, err := checkedFloat32ByteLenErrWin(2, "Vulkan q8 matvec argmax runner result")
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:q.Cols]); err != nil {
		return 0, 0, err
	}
	bufInfos := [5]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: r.argmaxBuf.buffer, Range: resultBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8WinCommandArgmax || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordArgmaxCommand(q.Rows, q.Cols); err != nil {
			return 0, 0, err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return 0, 0, fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	var result [2]float32
	if err := r.vk.readFloat32Into(r.device, r.argmaxBuf, result[:]); err != nil {
		return 0, 0, err
	}
	return int(result[1]), result[0], nil
}

func (r *vulkanMatVecQ8WinRunner) runTopK(x []float32, q *tensor.Q8Matrix, k int) ([]VulkanTokenScore, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	xBytes, err := checkedFloat32ByteLenErrWin(q.Cols, "Vulkan q8 matvec top-k runner x")
	if err != nil {
		return nil, err
	}
	dataLen, err := checkedQ8DataLenWin(q.Rows, q.Cols, "Vulkan q8 matvec top-k runner")
	if err != nil {
		return nil, err
	}
	dataBytes, err := checkedAlignedByteLenErrWin(dataLen, 4, "Vulkan q8 matvec top-k runner data")
	if err != nil {
		return nil, err
	}
	scaleBytes, err := checkedFloat32ByteLenErrWin(q.Rows, "Vulkan q8 matvec top-k runner scale")
	if err != nil {
		return nil, err
	}
	outBytes := scaleBytes
	blocks := (q.Rows + 255) / 256
	candidateFloats, ok := checkedMulInt(blocks, vulkanMatVecTopKMaxK)
	if !ok {
		return nil, fmt.Errorf("Vulkan q8 matvec top-k runner candidate count overflows: blocks=%d k=%d", blocks, vulkanMatVecTopKMaxK)
	}
	candidateFloats, ok = checkedMulInt(candidateFloats, 2)
	if !ok {
		return nil, fmt.Errorf("Vulkan q8 matvec top-k runner candidate count overflows: blocks=%d k=%d", blocks, vulkanMatVecTopKMaxK)
	}
	resultBytes, err := checkedFloat32ByteLenErrWin(candidateFloats, "Vulkan q8 matvec top-k runner result")
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
	dataBuf, err := r.int8WeightBuffer(q.Data[:dataLen], dataBytes)
	if err != nil {
		return nil, err
	}
	scaleBuf, err := r.floatWeightBuffer(q.Scale[:q.Rows], scaleBytes)
	if err != nil {
		return nil, err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:q.Cols]); err != nil {
		return nil, err
	}
	bufInfos := [5]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: r.argmaxBuf.buffer, Range: resultBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanMatVecQ8WinCommandTopK || r.commandRows != q.Rows || r.commandCols != q.Cols {
		if err := r.recordTopKCommand(q.Rows, q.Cols); err != nil {
			return nil, err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return nil, fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return nil, fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	r.topKReadback = ensureVulkanFloat32Scratch(r.topKReadback, candidateFloats)
	candidateData := r.topKReadback
	if err := r.vk.readFloat32Into(r.device, r.argmaxBuf, candidateData); err != nil {
		return nil, err
	}
	r.topKCandidates = selectVulkanTopKCandidatesInto(r.topKCandidates, candidateData, q.Rows, k)
	return r.topKCandidates, nil
}

func (r *vulkanMatVecQ8WinRunner) recordCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8WinCommandDefault
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8WinRunner) recordArgmaxCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.argmaxPipeline)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8WinCommandArgmax
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8WinRunner) recordTopKCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit | vkAccessShaderWriteBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.topKPipeline)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	blocks := (rows + 255) / 256
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(blocks), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandRows = rows
	r.commandCols = cols
	r.commandKind = vulkanMatVecQ8WinCommandTopK
	r.commandRecorded = true
	return nil
}

func (r *vulkanMatVecQ8WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanMatVecQ8WinRunner) int8WeightBuffer(data []int8, size uint64) (vkHostBufferWin, error) {
	return cachedInt8BufferWin(r.vk, r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanMatVecQ8WinRunner) floatWeightBuffer(data []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, r.scaleBuffers)
}

func (r *vulkanMatVecQ8WinRunner) writeInt8(buf vkHostBufferWin, values []int8) error {
	if buf.mapped == nil {
		return fmt.Errorf("vkMapMemory int8 write on unmapped buffer")
	}
	dst := unsafe.Slice((*int8)(buf.mapped), int(buf.size))
	clear(dst)
	dst = dst[:len(values)]
	copy(dst, values)
	return nil
}

type vulkanFusedMatVec3Q8WinRunner struct {
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
	dataBuffers     map[uintptr]vulkanCachedInt8BufferWin
	scaleBuffers    map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache [12]vulkanDescriptorBindingWin
	descriptorCount int
	mrope           bool
	commandRecorded bool
	commandRowsA    int
	commandRowsB    int
	commandRowsC    int
	commandCols     int
	commandPacked   int
	sharedDevice bool
	mu          sync.Mutex
}

func newVulkanFusedMatVec3Q8WinRunner() (*vulkanFusedMatVec3Q8WinRunner, error) {
	return newVulkanFusedMatVec3Q8WinRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec3", vulkanFusedMatVec3Q8ShaderCodeWindows, 10, 16, false)
}

func newVulkanFusedMatVec3MRoPEQ8WinRunner() (*vulkanFusedMatVec3Q8WinRunner, error) {
	return newVulkanFusedMatVec3Q8WinRunnerWithShader("rapidocrvl-vulkan-q8-fused-matvec3-mrope", vulkanFusedMatVec3MRoPEQ8ShaderCodeWindows, 12, 20, true)
}

func newVulkanFusedMatVec3Q8WinRunnerWithShader(appLabel string, shaderCode func() ([]uint32, error), descriptorCount, pushConstantBytes int, mrope bool) (*vulkanFusedMatVec3Q8WinRunner, error) {
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
	queueFamily := ctx.queueFamily
	entryName := append([]byte("main"), 0)
	r := &vulkanFusedMatVec3Q8WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferWin), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin), descriptorCount: descriptorCount, mrope: mrope}
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
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(descriptorCount)}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: uint32(pushConstantBytes)}
	plci := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.pipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	cpci2 := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, r.device, uintptr(unsafe.Pointer(&cpci2)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{SType: vkStructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vkCommandBufferLevelPrimary, CommandBufferCount: 1}
	if res := vk.call(vk.allocateCommandBuffers, r.device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, r.device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanFusedMatVec3Q8WinRunner) destroy() {
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
	for _, b := range r.dataBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.scaleBuffers {
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

func (r *vulkanFusedMatVec3Q8WinRunner) run(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	cols := a.Cols
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 fused matvec3 runner x")
	if err != nil {
		return err
	}
	aLen, err := checkedQ8DataLenWin(a.Rows, cols, "Vulkan q8 fused matvec3 runner a")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(b.Rows, cols, "Vulkan q8 fused matvec3 runner b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(c.Rows, cols, "Vulkan q8 fused matvec3 runner c")
	if err != nil {
		return err
	}
	aBytes, err := checkedAlignedByteLenErrWin(aLen, 4, "Vulkan q8 fused matvec3 runner a data")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 fused matvec3 runner b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 fused matvec3 runner c data")
	if err != nil {
		return err
	}
	saBytes, err := checkedFloat32ByteLenErrWin(a.Rows, "Vulkan q8 fused matvec3 runner a scale")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(b.Rows, "Vulkan q8 fused matvec3 runner b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(c.Rows, "Vulkan q8 fused matvec3 runner c scale")
	if err != nil {
		return err
	}
	oaBytes, obBytes, ocBytes := saBytes, sbBytes, scBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, oaBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, obBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, ocBytes); err != nil {
		return err
	}
	abuf, err := r.int8WeightBuffer(a.Data[:aLen], aBytes)
	if err != nil {
		return err
	}
	bbuf, err := r.int8WeightBuffer(b.Data[:bLen], bBytes)
	if err != nil {
		return err
	}
	cbuf, err := r.int8WeightBuffer(c.Data[:cLen], cBytes)
	if err != nil {
		return err
	}
	asbuf, err := r.floatWeightBuffer(a.Scale[:a.Rows], saBytes)
	if err != nil {
		return err
	}
	bsbuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: abuf.buffer, Range: aBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: cbuf.buffer, Range: cBytes},
		{Buffer: asbuf.buffer, Range: saBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: csbuf.buffer, Range: scBytes},
		{Buffer: r.outABuf.buffer, Range: oaBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outCBuf.buffer, Range: ocBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	if !r.commandRecorded || r.commandRowsA != a.Rows || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols {
		if err := r.recordCommand(a.Rows, b.Rows, c.Rows, cols, 0); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readFloat32Into(r.device, r.outABuf, outA[:a.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outBBuf, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outCBuf, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8WinRunner) run2(outB, outC, x []float32, b, c *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	cols := b.Cols
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 fused matvec2 runner x")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(b.Rows, cols, "Vulkan q8 fused matvec2 runner b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(c.Rows, cols, "Vulkan q8 fused matvec2 runner c")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 fused matvec2 runner b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 fused matvec2 runner c data")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(b.Rows, "Vulkan q8 fused matvec2 runner b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(c.Rows, "Vulkan q8 fused matvec2 runner c scale")
	if err != nil {
		return err
	}
	obBytes, ocBytes := sbBytes, scBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, obBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, ocBytes); err != nil {
		return err
	}
	bbuf, err := r.int8WeightBuffer(b.Data[:bLen], bBytes)
	if err != nil {
		return err
	}
	cbuf, err := r.int8WeightBuffer(c.Data[:cLen], cBytes)
	if err != nil {
		return err
	}
	bsbuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: cbuf.buffer, Range: cBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: csbuf.buffer, Range: scBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outCBuf.buffer, Range: ocBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	if !r.commandRecorded || r.commandRowsA != 0 || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols {
		if err := r.recordCommand(0, b.Rows, c.Rows, cols, 0); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readFloat32Into(r.device, r.outBBuf, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outCBuf, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8WinRunner) runMRoPE(outA, outB, outC, x []float32, a, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	cols := a.Cols
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 fused matvec3+mrope runner x")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan q8 fused matvec3+mrope runner table")
	if err != nil {
		return err
	}
	aLen, err := checkedQ8DataLenWin(a.Rows, cols, "Vulkan q8 fused matvec3+mrope runner a")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(b.Rows, cols, "Vulkan q8 fused matvec3+mrope runner b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(c.Rows, cols, "Vulkan q8 fused matvec3+mrope runner c")
	if err != nil {
		return err
	}
	aBytes, err := checkedAlignedByteLenErrWin(aLen, 4, "Vulkan q8 fused matvec3+mrope runner a data")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 fused matvec3+mrope runner b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 fused matvec3+mrope runner c data")
	if err != nil {
		return err
	}
	saBytes, err := checkedFloat32ByteLenErrWin(a.Rows, "Vulkan q8 fused matvec3+mrope runner a scale")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(b.Rows, "Vulkan q8 fused matvec3+mrope runner b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(c.Rows, "Vulkan q8 fused matvec3+mrope runner c scale")
	if err != nil {
		return err
	}
	oaBytes, obBytes, ocBytes := saBytes, sbBytes, scBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outABuf, oaBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, obBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, ocBytes); err != nil {
		return err
	}
	abuf, err := r.int8WeightBuffer(a.Data[:aLen], aBytes)
	if err != nil {
		return err
	}
	bbuf, err := r.int8WeightBuffer(b.Data[:bLen], bBytes)
	if err != nil {
		return err
	}
	cbuf, err := r.int8WeightBuffer(c.Data[:cLen], cBytes)
	if err != nil {
		return err
	}
	asbuf, err := r.floatWeightBuffer(a.Scale[:a.Rows], saBytes)
	if err != nil {
		return err
	}
	bsbuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.sinBuf, sinTable[:half]); err != nil {
		return err
	}
	bufInfos := [12]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: abuf.buffer, Range: aBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: cbuf.buffer, Range: cBytes},
		{Buffer: asbuf.buffer, Range: saBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: csbuf.buffer, Range: scBytes},
		{Buffer: r.cosBuf.buffer, Range: tableBytes},
		{Buffer: r.sinBuf.buffer, Range: tableBytes},
		{Buffer: r.outABuf.buffer, Range: oaBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outCBuf.buffer, Range: ocBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != a.Rows || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(a.Rows, b.Rows, c.Rows, cols, packed); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readFloat32Into(r.device, r.outABuf, outA[:a.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outBBuf, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outCBuf, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8WinRunner) run2MRoPE(outB, outC, x []float32, b, c *tensor.Q8Matrix, cosTable, sinTable []float32, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	cols := b.Cols
	half := headDim / 2
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 fused matvec2+mrope runner x")
	if err != nil {
		return err
	}
	tableBytes, err := checkedFloat32ByteLenErrWin(half, "Vulkan q8 fused matvec2+mrope runner table")
	if err != nil {
		return err
	}
	bLen, err := checkedQ8DataLenWin(b.Rows, cols, "Vulkan q8 fused matvec2+mrope runner b")
	if err != nil {
		return err
	}
	cLen, err := checkedQ8DataLenWin(c.Rows, cols, "Vulkan q8 fused matvec2+mrope runner c")
	if err != nil {
		return err
	}
	bBytes, err := checkedAlignedByteLenErrWin(bLen, 4, "Vulkan q8 fused matvec2+mrope runner b data")
	if err != nil {
		return err
	}
	cBytes, err := checkedAlignedByteLenErrWin(cLen, 4, "Vulkan q8 fused matvec2+mrope runner c data")
	if err != nil {
		return err
	}
	sbBytes, err := checkedFloat32ByteLenErrWin(b.Rows, "Vulkan q8 fused matvec2+mrope runner b scale")
	if err != nil {
		return err
	}
	scBytes, err := checkedFloat32ByteLenErrWin(c.Rows, "Vulkan q8 fused matvec2+mrope runner c scale")
	if err != nil {
		return err
	}
	obBytes, ocBytes := sbBytes, scBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.cosBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.sinBuf, tableBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outBBuf, obBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.outCBuf, ocBytes); err != nil {
		return err
	}
	bbuf, err := r.int8WeightBuffer(b.Data[:bLen], bBytes)
	if err != nil {
		return err
	}
	cbuf, err := r.int8WeightBuffer(c.Data[:cLen], cBytes)
	if err != nil {
		return err
	}
	bsbuf, err := r.floatWeightBuffer(b.Scale[:b.Rows], sbBytes)
	if err != nil {
		return err
	}
	csbuf, err := r.floatWeightBuffer(c.Scale[:c.Rows], scBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.cosBuf, cosTable[:half]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.sinBuf, sinTable[:half]); err != nil {
		return err
	}
	bufInfos := [12]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: bbuf.buffer, Range: bBytes},
		{Buffer: cbuf.buffer, Range: cBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: bsbuf.buffer, Range: sbBytes},
		{Buffer: csbuf.buffer, Range: scBytes},
		{Buffer: r.cosBuf.buffer, Range: tableBytes},
		{Buffer: r.sinBuf.buffer, Range: tableBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outBBuf.buffer, Range: obBytes},
		{Buffer: r.outCBuf.buffer, Range: ocBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:r.descriptorCount], bufInfos[:])

	packed := headDim | (kvHeads << 16)
	if !r.commandRecorded || r.commandRowsA != 0 || r.commandRowsB != b.Rows || r.commandRowsC != c.Rows || r.commandCols != cols || r.commandPacked != packed {
		if err := r.recordCommand(0, b.Rows, c.Rows, cols, packed); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readFloat32Into(r.device, r.outBBuf, outB[:b.Rows]); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.outCBuf, outC[:c.Rows]); err != nil {
		return err
	}
	return nil
}

func (r *vulkanFusedMatVec3Q8WinRunner) recordCommand(rowsA, rowsB, rowsC, cols, packed int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
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
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(pushBytes), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(groups), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
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

func (r *vulkanFusedMatVec3Q8WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanFusedMatVec3Q8WinRunner) int8WeightBuffer(data []int8, size uint64) (vkHostBufferWin, error) {
	return cachedInt8BufferWin(r.vk, r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanFusedMatVec3Q8WinRunner) floatWeightBuffer(data []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, r.scaleBuffers)
}

func writeInt8Windows(vk *vulkanWin, device uintptr, buf vkHostBufferWin, values []int8) error {
	if buf.mapped == nil {
		return fmt.Errorf("vkMapMemory int8 write on unmapped buffer")
	}
	dst := unsafe.Slice((*int8)(buf.mapped), int(buf.size))
	clear(dst)
	copy(dst[:len(values)], values)
	return nil
}

type vulkanSwiGLUDownQ8WinRunner struct {
	vk                  *vulkanWin
	instance            uintptr
	device              uintptr
	queue               uintptr
	queueFamily         uint32
	memProps            vkPhysicalDeviceMemoryProperties
	setLayout           uintptr
	downSetLayout       uintptr
	normSetLayout       uintptr
	descriptorPool      uintptr
	descriptorSet       uintptr
	downDescriptorSet   uintptr
	normDescriptorSet   uintptr
	pipelineLayout      uintptr
	pipeline            uintptr
	downPipelineLayout  uintptr
	downPipeline        uintptr
	normPipelineLayout  uintptr
	normPipeline        uintptr
	commandPool         uintptr
	commandBuffer       uintptr
	fence               uintptr
	xBuf                vkHostBufferWin
	interBuf            vkHostBufferWin
	outBuf              vkHostBufferWin
	residualBuf         vkHostBufferWin
	normBuf             vkHostBufferWin
	dataBuffers         map[uintptr]vulkanCachedInt8BufferWin
	scaleBuffers        map[uintptr]vulkanCachedFloat32BufferWin
	descriptorCache     [6]vulkanDescriptorBindingWin
	downDescriptorCache [4]vulkanDescriptorBindingWin
	normDescriptorCache [4]vulkanDescriptorBindingWin
	commandRecorded     bool
	commandKind         int
	commandRows         int
	commandCols         int
	commandOutRows      int
	sharedDevice bool
	mu          sync.Mutex
}

const (
	vulkanSwiGLUQ8CommandDown     = 1
	vulkanSwiGLUQ8CommandDownNorm = 2
	vulkanSwiGLUQ8CommandMatNorm  = 3
	vulkanSwiGLUQ8CommandGateUp   = 4
)

func newVulkanSwiGLUDownQ8WinRunner() (*vulkanSwiGLUDownQ8WinRunner, error) {
	spv, err := vulkanSwiGLUGateUpQ8ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanAddRMSNormF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	downSPV, err := vulkanMatVecQ8ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	instance := ctx.instance
	queueFamily := ctx.queueFamily
	entryName := append([]byte("main"), 0)
	r := &vulkanSwiGLUDownQ8WinRunner{vk: vk, instance: instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true, dataBuffers: make(map[uintptr]vulkanCachedInt8BufferWin), scaleBuffers: make(map[uintptr]vulkanCachedFloat32BufferWin)}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()

	bindings := make([]vkDescriptorSetLayoutBinding, 6)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	downBindings := make([]vkDescriptorSetLayoutBinding, 4)
	for i := range downBindings {
		downBindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	downDSL := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(downBindings)), PBindings: uintptr(unsafe.Pointer(&downBindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&downDSL)), 0, uintptr(unsafe.Pointer(&r.downSetLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout down: %d", int32(res))
	}
	normBindings := make([]vkDescriptorSetLayoutBinding, 4)
	for i := range normBindings {
		normBindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	normDSL := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(normBindings)), PBindings: uintptr(unsafe.Pointer(&normBindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&normDSL)), 0, uintptr(unsafe.Pointer(&r.normSetLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout norm: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: 14}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 3, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	setLayouts := []uintptr{r.setLayout, r.downSetLayout, r.normSetLayout}
	descSets := make([]uintptr, 3)
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 3, PSetLayouts: uintptr(unsafe.Pointer(&setLayouts[0]))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&descSets[0]))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	r.descriptorSet = descSets[0]
	r.downDescriptorSet = descSets[1]
	r.normDescriptorSet = descSets[2]
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 8}
	plci := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&plci)), 0, uintptr(unsafe.Pointer(&r.pipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout: %d", int32(res))
	}
	downPLCI := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.downSetLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&downPLCI)), 0, uintptr(unsafe.Pointer(&r.downPipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout down: %d", int32(res))
	}
	normPLCI := vkPipelineLayoutCreateInfo{SType: vkStructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.normSetLayout)), PushConstantRangeCount: 1, PPushConstantRanges: uintptr(unsafe.Pointer(&pushRange))}
	if res := vk.call(vk.createPipelineLayout, r.device, uintptr(unsafe.Pointer(&normPLCI)), 0, uintptr(unsafe.Pointer(&r.normPipelineLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreatePipelineLayout norm: %d", int32(res))
	}
	if err := r.createPipelines(spv, downSPV, normSPV, entryName); err != nil {
		return nil, err
	}
	cpci := vkCommandPoolCreateInfo{SType: vkStructureTypeCommandPoolCreateInfo, QueueFamilyIndex: queueFamily}
	if res := vk.call(vk.createCommandPool, r.device, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.commandPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateCommandPool: %d", int32(res))
	}
	cbai := vkCommandBufferAllocateInfo{SType: vkStructureTypeCommandBufferAllocateInfo, CommandPool: r.commandPool, Level: vkCommandBufferLevelPrimary, CommandBufferCount: 1}
	if res := vk.call(vk.allocateCommandBuffers, r.device, uintptr(unsafe.Pointer(&cbai)), uintptr(unsafe.Pointer(&r.commandBuffer))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateCommandBuffers: %d", int32(res))
	}
	fci := vkFenceCreateInfo{SType: vkStructureTypeFenceCreateInfo}
	if res := vk.call(vk.createFence, r.device, uintptr(unsafe.Pointer(&fci)), 0, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateFence: %d", int32(res))
	}
	success = true
	return r, nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) createPipelines(spv, downSPV, normSPV []uint32, entryName []byte) error {
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := r.vk.call(r.vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return fmt.Errorf("vkCreateShaderModule: %d", int32(res))
	}
	defer r.vk.callVoid(r.vk.destroyShaderModule, r.device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	if res := r.vk.call(r.vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&r.pipeline))); res != vkSuccess {
		return fmt.Errorf("vkCreateComputePipelines: %d", int32(res))
	}
	downSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(downSPV) * 4), PCode: uintptr(unsafe.Pointer(&downSPV[0]))}
	var downShader uintptr
	if res := r.vk.call(r.vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&downSMCI)), 0, uintptr(unsafe.Pointer(&downShader))); res != vkSuccess {
		return fmt.Errorf("vkCreateShaderModule down: %d", int32(res))
	}
	defer r.vk.callVoid(r.vk.destroyShaderModule, r.device, downShader, 0)
	downStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: downShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	downCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: downStage, Layout: r.downPipelineLayout}
	if res := r.vk.call(r.vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&downCPCI)), 0, uintptr(unsafe.Pointer(&r.downPipeline))); res != vkSuccess {
		return fmt.Errorf("vkCreateComputePipelines down: %d", int32(res))
	}
	normSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(normSPV) * 4), PCode: uintptr(unsafe.Pointer(&normSPV[0]))}
	var normShader uintptr
	if res := r.vk.call(r.vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&normSMCI)), 0, uintptr(unsafe.Pointer(&normShader))); res != vkSuccess {
		return fmt.Errorf("vkCreateShaderModule norm: %d", int32(res))
	}
	defer r.vk.callVoid(r.vk.destroyShaderModule, r.device, normShader, 0)
	normStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: normShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	normCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.normPipelineLayout}
	if res := r.vk.call(r.vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&normCPCI)), 0, uintptr(unsafe.Pointer(&r.normPipeline))); res != vkSuccess {
		return fmt.Errorf("vkCreateComputePipelines norm: %d", int32(res))
	}
	return nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.downPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.downPipeline, 0)
	}
	if r.normPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.normPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.xBuf)
	r.vk.destroyBuffer(r.device, r.interBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.residualBuf)
	r.vk.destroyBuffer(r.device, r.normBuf)
	for _, b := range r.dataBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.scaleBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	if r.descriptorPool != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorPool, r.device, r.descriptorPool, 0)
	}
	if r.pipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.pipelineLayout, 0)
	}
	if r.downPipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.downPipelineLayout, 0)
	}
	if r.normPipelineLayout != 0 {
		r.vk.callVoid(r.vk.destroyPipelineLayout, r.device, r.normPipelineLayout, 0)
	}
	if r.setLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.setLayout, 0)
	}
	if r.downSetLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.downSetLayout, 0)
	}
	if r.normSetLayout != 0 {
		r.vk.callVoid(r.vk.destroyDescriptorSetLayout, r.device, r.normSetLayout, 0)
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

func (r *vulkanSwiGLUDownQ8WinRunner) runGateUp(out, x []float32, gate, up *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols := gate.Rows, gate.Cols
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 swiglu gate/up runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan q8 swiglu gate/up runner intermediate")
	if err != nil {
		return err
	}
	gateLen, err := checkedQ8DataLenWin(gate.Rows, gate.Cols, "Vulkan q8 swiglu gate/up runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedQ8DataLenWin(up.Rows, up.Cols, "Vulkan q8 swiglu gate/up runner up")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrWin(gateLen, 4, "Vulkan q8 swiglu gate/up runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrWin(upLen, 4, "Vulkan q8 swiglu gate/up runner up data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrWin(gate.Rows, "Vulkan q8 swiglu gate/up runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrWin(up.Rows, "Vulkan q8 swiglu gate/up runner up scale")
	if err != nil {
		return err
	}
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: gateBytes},
		{Buffer: upBuf.buffer, Range: upBytes},
		{Buffer: gateScaleBuf.buffer, Range: gateScaleBytes},
		{Buffer: upScaleBuf.buffer, Range: upScaleBytes},
		{Buffer: r.interBuf.buffer, Range: interBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanSwiGLUQ8CommandGateUp || r.commandRows != rows || r.commandCols != cols {
		if err := r.recordGateUpCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readFloat32Into(r.device, r.interBuf, out[:rows])
}

func (r *vulkanSwiGLUDownQ8WinRunner) run(out, x []float32, gate, up, down *tensor.Q8Matrix) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols, outRows := gate.Rows, gate.Cols, down.Rows
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 swiglu/down runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan q8 swiglu/down runner intermediate")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan q8 swiglu/down runner output")
	if err != nil {
		return err
	}
	gateLen, err := checkedQ8DataLenWin(gate.Rows, gate.Cols, "Vulkan q8 swiglu/down runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedQ8DataLenWin(up.Rows, up.Cols, "Vulkan q8 swiglu/down runner up")
	if err != nil {
		return err
	}
	downLen, err := checkedQ8DataLenWin(down.Rows, down.Cols, "Vulkan q8 swiglu/down runner down")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrWin(gateLen, 4, "Vulkan q8 swiglu/down runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrWin(upLen, 4, "Vulkan q8 swiglu/down runner up data")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrWin(downLen, 4, "Vulkan q8 swiglu/down runner down data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrWin(gate.Rows, "Vulkan q8 swiglu/down runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrWin(up.Rows, "Vulkan q8 swiglu/down runner up scale")
	if err != nil {
		return err
	}
	downScaleBytes, err := checkedFloat32ByteLenErrWin(down.Rows, "Vulkan q8 swiglu/down runner down scale")
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	swiInfos := [6]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: gateBytes},
		{Buffer: upBuf.buffer, Range: upBytes},
		{Buffer: gateScaleBuf.buffer, Range: gateScaleBytes},
		{Buffer: upScaleBuf.buffer, Range: upScaleBytes},
		{Buffer: r.interBuf.buffer, Range: interBytes},
	}
	downInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Range: interBytes},
		{Buffer: downBuf.buffer, Range: downBytes},
		{Buffer: downScaleBuf.buffer, Range: downScaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUQ8CommandDown || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	return r.vk.readFloat32Into(r.device, r.outBuf, out[:outRows])
}

func (r *vulkanSwiGLUDownQ8WinRunner) runAddRMSNorm(normOut, residual, x []float32, gate, up, down *tensor.Q8Matrix, normWeight []float32, readResidual bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols, outRows := gate.Rows, gate.Cols, down.Rows
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 swiglu/down+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	interBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan q8 swiglu/down+add+rmsnorm runner intermediate")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(outRows, "Vulkan q8 swiglu/down+add+rmsnorm runner output")
	if err != nil {
		return err
	}
	gateLen, err := checkedQ8DataLenWin(gate.Rows, gate.Cols, "Vulkan q8 swiglu/down+add+rmsnorm runner gate")
	if err != nil {
		return err
	}
	upLen, err := checkedQ8DataLenWin(up.Rows, up.Cols, "Vulkan q8 swiglu/down+add+rmsnorm runner up")
	if err != nil {
		return err
	}
	downLen, err := checkedQ8DataLenWin(down.Rows, down.Cols, "Vulkan q8 swiglu/down+add+rmsnorm runner down")
	if err != nil {
		return err
	}
	gateBytes, err := checkedAlignedByteLenErrWin(gateLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner gate data")
	if err != nil {
		return err
	}
	upBytes, err := checkedAlignedByteLenErrWin(upLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner up data")
	if err != nil {
		return err
	}
	downBytes, err := checkedAlignedByteLenErrWin(downLen, 4, "Vulkan q8 swiglu/down+add+rmsnorm runner down data")
	if err != nil {
		return err
	}
	gateScaleBytes, err := checkedFloat32ByteLenErrWin(gate.Rows, "Vulkan q8 swiglu/down+add+rmsnorm runner gate scale")
	if err != nil {
		return err
	}
	upScaleBytes, err := checkedFloat32ByteLenErrWin(up.Rows, "Vulkan q8 swiglu/down+add+rmsnorm runner up scale")
	if err != nil {
		return err
	}
	downScaleBytes, err := checkedFloat32ByteLenErrWin(down.Rows, "Vulkan q8 swiglu/down+add+rmsnorm runner down scale")
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
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:outRows]); err != nil {
		return err
	}
	swiInfos := [6]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: gateBuf.buffer, Range: gateBytes},
		{Buffer: upBuf.buffer, Range: upBytes},
		{Buffer: gateScaleBuf.buffer, Range: gateScaleBytes},
		{Buffer: upScaleBuf.buffer, Range: upScaleBytes},
		{Buffer: r.interBuf.buffer, Range: interBytes},
	}
	downInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.interBuf.buffer, Range: interBytes},
		{Buffer: downBuf.buffer, Range: downBytes},
		{Buffer: downScaleBuf.buffer, Range: downScaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.residualBuf.buffer, Range: outBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: normWeightBuf.buffer, Range: outBytes},
		{Buffer: r.normBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], swiInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.normDescriptorSet, r.normDescriptorCache[:], normInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUQ8CommandDownNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != outRows {
		if err := r.recordAddRMSNormCommand(rows, cols, outRows); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if readResidual {
		if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:outRows]); err != nil {
			return err
		}
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:outRows])
}

func (r *vulkanSwiGLUDownQ8WinRunner) runMatVecAddRMSNorm(normOut, residual, x []float32, q *tensor.Q8Matrix, normWeight []float32) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rows, cols := q.Rows, q.Cols
	xBytes, err := checkedFloat32ByteLenErrWin(cols, "Vulkan q8 matvec+add+rmsnorm runner x")
	if err != nil {
		return err
	}
	outBytes, err := checkedFloat32ByteLenErrWin(rows, "Vulkan q8 matvec+add+rmsnorm runner output")
	if err != nil {
		return err
	}
	dataLen, err := checkedQ8DataLenWin(rows, cols, "Vulkan q8 matvec+add+rmsnorm runner")
	if err != nil {
		return err
	}
	dataBytes, err := checkedAlignedByteLenErrWin(dataLen, 4, "Vulkan q8 matvec+add+rmsnorm runner data")
	if err != nil {
		return err
	}
	scaleBytes := outBytes
	if err := r.ensureHostBuffer(&r.xBuf, xBytes); err != nil {
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
	dataBuf, err := r.int8WeightBuffer(q.Data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.floatWeightBuffer(q.Scale[:rows], scaleBytes)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.floatWeightBuffer(normWeight[:rows], outBytes)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.xBuf, x[:cols]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:rows]); err != nil {
		return err
	}
	downInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.xBuf.buffer, Range: xBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
	}
	normInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.residualBuf.buffer, Range: outBytes},
		{Buffer: r.outBuf.buffer, Range: outBytes},
		{Buffer: normWeightBuf.buffer, Range: outBytes},
		{Buffer: r.normBuf.buffer, Range: outBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.downDescriptorSet, r.downDescriptorCache[:], downInfos[:])
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.normDescriptorSet, r.normDescriptorCache[:], normInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanSwiGLUQ8CommandMatNorm || r.commandRows != rows || r.commandCols != cols || r.commandOutRows != rows {
		if err := r.recordMatVecAddRMSNormCommand(rows, cols); err != nil {
			return err
		}
	}
	if res := r.vk.call(r.vk.resetFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence))); res != vkSuccess {
		return fmt.Errorf("vkResetFences: %d", int32(res))
	}
	cmd := r.commandBuffer
	submit := vkSubmitInfo{SType: vkStructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: uintptr(unsafe.Pointer(&cmd))}
	if res := r.vk.call(r.vk.queueSubmit, r.queue, 1, uintptr(unsafe.Pointer(&submit)), r.fence); res != vkSuccess {
		return fmt.Errorf("vkQueueSubmit: %d", int32(res))
	}
	if res := r.vk.call(r.vk.waitForFences, r.device, 1, uintptr(unsafe.Pointer(&r.fence)), 1, uintptr(math.MaxUint64)); res != vkSuccess {
		return fmt.Errorf("vkWaitForFences: %d", int32(res))
	}
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:rows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:rows])
}

func (r *vulkanSwiGLUDownQ8WinRunner) recordCommand(rows, cols, outRows int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUQ8CommandDown
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) recordGateUpCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUQ8CommandGateUp
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = 0
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) recordMatVecAddRMSNormCommand(rows, cols int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.normDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUQ8CommandMatNorm
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = rows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) recordAddRMSNormCommand(rows, cols, outRows int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [8]byte
	binary.LittleEndian.PutUint32(pc[0:4], uint32(rows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(cols))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.pipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(rows), 1, 1)

	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(rows))
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.downPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.downPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.downDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.downPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(outRows), 1, 1)

	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)

	binary.LittleEndian.PutUint32(pc[0:4], uint32(outRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.normPipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.normDescriptorSet)), 0, 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.normPipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanSwiGLUQ8CommandDownNorm
	r.commandRows = rows
	r.commandCols = cols
	r.commandOutRows = outRows
	r.commandRecorded = true
	return nil
}

func (r *vulkanSwiGLUDownQ8WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func (r *vulkanSwiGLUDownQ8WinRunner) int8WeightBuffer(data []int8, size uint64) (vkHostBufferWin, error) {
	return cachedInt8BufferWin(r.vk, r.device, r.memProps, data, size, r.dataBuffers)
}

func (r *vulkanSwiGLUDownQ8WinRunner) floatWeightBuffer(data []float32, size uint64) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, r.scaleBuffers)
}

func alignUpInt(v, alignment int) int {
	if rem := v % alignment; rem != 0 {
		return v + alignment - rem
	}
	return v
}

func vulkanMatVecQ8ShaderCodeWindows() ([]uint32, error) {
	vulkanMatVecQ8SPV.once.Do(func() {
		vulkanMatVecQ8SPV.code, vulkanMatVecQ8SPV.err = compileVulkanGLSLWindows(vulkanMatVecQ8GLSL)
	})
	return vulkanMatVecQ8SPV.code, vulkanMatVecQ8SPV.err
}

func vulkanFusedMatVec3Q8ShaderCodeWindows() ([]uint32, error) {
	vulkanFusedMatVec3Q8SPV.once.Do(func() {
		vulkanFusedMatVec3Q8SPV.code, vulkanFusedMatVec3Q8SPV.err = compileVulkanGLSLWindows(vulkanFusedQKVQ8GLSL)
	})
	return vulkanFusedMatVec3Q8SPV.code, vulkanFusedMatVec3Q8SPV.err
}

func vulkanFusedMatVec3MRoPEQ8ShaderCodeWindows() ([]uint32, error) {
	vulkanFusedMatVec3MRoPEQ8SPV.once.Do(func() {
		vulkanFusedMatVec3MRoPEQ8SPV.code, vulkanFusedMatVec3MRoPEQ8SPV.err = compileVulkanGLSLWindows(vulkanFusedQKVMRoPEQ8GLSL)
	})
	return vulkanFusedMatVec3MRoPEQ8SPV.code, vulkanFusedMatVec3MRoPEQ8SPV.err
}

func vulkanSwiGLUGateUpQ8ShaderCodeWindows() ([]uint32, error) {
	vulkanSwiGLUGateUpQ8SPV.once.Do(func() {
		vulkanSwiGLUGateUpQ8SPV.code, vulkanSwiGLUGateUpQ8SPV.err = compileVulkanGLSLWindows(vulkanFusedSwiGLUQ8GLSL)
	})
	return vulkanSwiGLUGateUpQ8SPV.code, vulkanSwiGLUGateUpQ8SPV.err
}
