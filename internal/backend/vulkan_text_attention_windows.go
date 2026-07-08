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

var vulkanTextAttentionF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionOutF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionOutAddRMSNormF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextFirstTokenValueOutAddRMSNormF32SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionOutQ8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionOutQ6SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionOutQ4SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ8SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ6SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextFirstTokenValueOutQ4SPV struct {
	once sync.Once
	code []uint32
	err  error
}

var vulkanTextAttentionF32RunnerCache struct {
	once   sync.Once
	runner *vulkanTextAttentionF32WinRunner
	err    error
}

func VulkanTextAttentionF32(out, q, kCache, vCache []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	if cacheLen <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 {
		return fmt.Errorf("invalid Vulkan text attention shape cacheLen=%d numHeads=%d kvHeads=%d headDim=%d", cacheLen, numHeads, kvHeads, headDim)
	}
	if numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan text attention head grouping numHeads=%d kvHeads=%d", numHeads, kvHeads)
	}
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems {
		return fmt.Errorf("invalid Vulkan text attention buffers out=%d q=%d k=%d v=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
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
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
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
	runner, err := getVulkanTextAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runOutAddRMSNorm(normOut, residual, q, kCache, vCache, w, bias, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	if numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || numHeads%kvHeads != 0 {
		return fmt.Errorf("invalid Vulkan first-token value+out+add+rmsnorm shape numHeads=%d kvHeads=%d headDim=%d", numHeads, kvHeads, headDim)
	}
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
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
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q8 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q6 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 first-token value+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(kCache) < kvDim || len(vCache) < kvDim || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q4 first-token value+out+add+rmsnorm buffers normOut=%d residual=%d k=%d v=%d data=%d scale=%d normWeight=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q8 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q6 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(out) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out buffers out=%d q=%d k=%d v=%d data=%d scale=%d cacheLen=%d qRows=%d kvDim=%d",
			len(out), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
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
	qRows, kvDim, _, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	if w.Rows != qRows || w.Cols != qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm matrix shape w=%dx%d want=%dx%d", w.Rows, w.Cols, qRows, qRows)
	}
	cacheElems, err := checkedTextAttentionWinCacheElems(cacheLen, kvDim)
	if err != nil {
		return err
	}
	dataLen, _, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	if len(normOut) < qRows || len(residual) < qRows || len(q) < qRows || len(kCache) < cacheElems || len(vCache) < cacheElems || len(w.Data) < dataLen || len(w.Scale) < qRows || len(normWeight) < qRows {
		return fmt.Errorf("invalid Vulkan q4 text attention+out+add+rmsnorm buffers normOut=%d residual=%d q=%d k=%d v=%d data=%d scale=%d normWeight=%d cacheLen=%d qRows=%d kvDim=%d",
			len(normOut), len(residual), len(q), len(kCache), len(vCache), len(w.Data), len(w.Scale), len(normWeight), cacheLen, qRows, kvDim)
	}
	runner, err := getVulkanTextAttentionF32RunnerWindows()
	if err != nil {
		return err
	}
	return runner.runOutQ4AddRMSNorm(normOut, residual, q, kCache, vCache, w, normWeight, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func getVulkanTextAttentionF32RunnerWindows() (*vulkanTextAttentionF32WinRunner, error) {
	vulkanTextAttentionF32RunnerCache.once.Do(func() {
		vulkanTextAttentionF32RunnerCache.runner, vulkanTextAttentionF32RunnerCache.err = newVulkanTextAttentionF32WinRunner()
	})
	return vulkanTextAttentionF32RunnerCache.runner, vulkanTextAttentionF32RunnerCache.err
}

type vulkanTextAttentionF32WinRunner struct {
	vk                     *vulkanWin
	instance               uintptr
	device                 uintptr
	queue                  uintptr
	queueFamily            uint32
	memProps               vkPhysicalDeviceMemoryProperties
	setLayout              uintptr
	descriptorPool         uintptr
	descriptorSet          uintptr
	pipelineLayout         uintptr
	pipeline               uintptr
	projPipeline           uintptr
	normPipeline           uintptr
	firstTokenProjPipeline uintptr
	firstTokenQ8Pipeline   uintptr
	firstTokenQ6Pipeline   uintptr
	firstTokenQ4Pipeline   uintptr
	q8ProjPipeline         uintptr
	q6ProjPipeline         uintptr
	q4ProjPipeline         uintptr
	commandPool            uintptr
	commandBuffer          uintptr
	fence                  uintptr
	qBuf                   vkHostBufferWin
	kBuf                   vkHostBufferWin
	vBuf                   vkHostBufferWin
	outBuf                 vkHostBufferWin
	finalBuf               vkHostBufferWin
	residualBuf            vkHostBufferWin
	normBuf                vkHostBufferWin
	weightBuffers          map[uintptr]vulkanCachedFloat32BufferWin
	biasBuffers            map[uintptr]vulkanCachedFloat32BufferWin
	q8DataBuffers          map[uintptr]vulkanCachedInt8BufferWin
	byteDataBuffers        map[uintptr]vulkanCachedByteBufferWin
	q8ScaleBuffers         map[uintptr]vulkanCachedFloat32BufferWin
	kCacheKey              uintptr
	vCacheKey              uintptr
	cacheEpoch             uint64
	cacheUploaded          int
	cacheKVDim             int
	valueCacheKey          uintptr
	valueCacheEpoch        uint64
	valueCacheUploaded     int
	valueCacheKVDim        int
	descriptorCache        [10]vulkanDescriptorBindingWin
	commandRecorded        bool
	commandKind            int
	commandCacheLen        int
	commandNumHeads        int
	commandKVHeads         int
	commandHeadDim         int
	commandQRows           int
	commandKVDim           int
	sharedDevice bool
	mu          sync.Mutex
}

const (
	vulkanTextAttentionCommandOnly                     = 1
	vulkanTextAttentionCommandOutF32                   = 2
	vulkanTextAttentionCommandOutQ8                    = 3
	vulkanTextAttentionCommandOutQ6                    = 4
	vulkanTextAttentionCommandOutQ4                    = 5
	vulkanTextAttentionCommandOutNorm                  = 6
	vulkanTextAttentionCommandOutNormQ8                = 7
	vulkanTextAttentionCommandOutNormQ6                = 8
	vulkanTextAttentionCommandOutNormQ4                = 9
	vulkanTextAttentionCommandFirstTokenValueOutNorm   = 10
	vulkanTextAttentionCommandFirstTokenValueOutNormQ8 = 11
	vulkanTextAttentionCommandFirstTokenValueOutNormQ6 = 12
	vulkanTextAttentionCommandFirstTokenValueOutNormQ4 = 13
)

func newVulkanTextAttentionF32WinRunner() (*vulkanTextAttentionF32WinRunner, error) {
	spv, err := vulkanTextAttentionF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	projSPV, err := vulkanTextAttentionOutF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	normSPV, err := vulkanTextAttentionOutAddRMSNormF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	firstTokenProjSPV, err := vulkanTextFirstTokenValueOutAddRMSNormF32ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	q8ProjSPV, err := vulkanTextAttentionOutQ8ShaderCodeWindows()
	if err != nil {
		return nil, err
	}
	ctx, err := getVulkanSharedContextWindows()
	if err != nil {
		return nil, err
	}
	vk := ctx.vk
	queueFamily := ctx.queueFamily
	entryName := append([]byte("main"), 0)
	r := &vulkanTextAttentionF32WinRunner{
		vk: ctx.vk, instance: ctx.instance, device: ctx.device, queue: ctx.queue, queueFamily: ctx.queueFamily, memProps: ctx.memProps, sharedDevice: true,
		weightBuffers:   make(map[uintptr]vulkanCachedFloat32BufferWin),
		biasBuffers:     make(map[uintptr]vulkanCachedFloat32BufferWin),
		q8DataBuffers:   make(map[uintptr]vulkanCachedInt8BufferWin),
		byteDataBuffers: make(map[uintptr]vulkanCachedByteBufferWin),
		q8ScaleBuffers:  make(map[uintptr]vulkanCachedFloat32BufferWin),
	}
	success := false
	defer func() {
		if !success {
			r.destroy()
		}
	}()
	bindings := make([]vkDescriptorSetLayoutBinding, 10)
	for i := range bindings {
		bindings[i] = vkDescriptorSetLayoutBinding{Binding: uint32(i), DescriptorType: vkDescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vkShaderStageComputeBit}
	}
	dslci := vkDescriptorSetLayoutCreateInfo{SType: vkStructureTypeDescriptorSetLayoutCreateInfo, BindingCount: uint32(len(bindings)), PBindings: uintptr(unsafe.Pointer(&bindings[0]))}
	if res := vk.call(vk.createDescriptorSetLayout, r.device, uintptr(unsafe.Pointer(&dslci)), 0, uintptr(unsafe.Pointer(&r.setLayout))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorSetLayout: %d", int32(res))
	}
	poolSize := vkDescriptorPoolSize{Type: vkDescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}
	dpci := vkDescriptorPoolCreateInfo{SType: vkStructureTypeDescriptorPoolCreateInfo, MaxSets: 1, PoolSizeCount: 1, PPoolSizes: uintptr(unsafe.Pointer(&poolSize))}
	if res := vk.call(vk.createDescriptorPool, r.device, uintptr(unsafe.Pointer(&dpci)), 0, uintptr(unsafe.Pointer(&r.descriptorPool))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateDescriptorPool: %d", int32(res))
	}
	dsai := vkDescriptorSetAllocateInfo{SType: vkStructureTypeDescriptorSetAllocateInfo, DescriptorPool: r.descriptorPool, DescriptorSetCount: 1, PSetLayouts: uintptr(unsafe.Pointer(&r.setLayout))}
	if res := vk.call(vk.allocateDescriptorSets, r.device, uintptr(unsafe.Pointer(&dsai)), uintptr(unsafe.Pointer(&r.descriptorSet))); res != vkSuccess {
		return nil, fmt.Errorf("vkAllocateDescriptorSets: %d", int32(res))
	}
	pushRange := vkPushConstantRange{StageFlags: vkShaderStageComputeBit, Size: 20}
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
	projSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(projSPV) * 4), PCode: uintptr(unsafe.Pointer(&projSPV[0]))}
	var projShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&projSMCI)), 0, uintptr(unsafe.Pointer(&projShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule projection: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, projShader, 0)
	projStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: projShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	projCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: projStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&projCPCI)), 0, uintptr(unsafe.Pointer(&r.projPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines projection: %d", int32(res))
	}
	normSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(normSPV) * 4), PCode: uintptr(unsafe.Pointer(&normSPV[0]))}
	var normShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&normSMCI)), 0, uintptr(unsafe.Pointer(&normShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule attention norm: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, normShader, 0)
	normStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: normShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	normCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: normStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&normCPCI)), 0, uintptr(unsafe.Pointer(&r.normPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines attention norm: %d", int32(res))
	}
	firstTokenProjSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(firstTokenProjSPV) * 4), PCode: uintptr(unsafe.Pointer(&firstTokenProjSPV[0]))}
	var firstTokenProjShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&firstTokenProjSMCI)), 0, uintptr(unsafe.Pointer(&firstTokenProjShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule first-token projection: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, firstTokenProjShader, 0)
	firstTokenProjStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: firstTokenProjShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	firstTokenProjCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: firstTokenProjStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&firstTokenProjCPCI)), 0, uintptr(unsafe.Pointer(&r.firstTokenProjPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines first-token projection: %d", int32(res))
	}
	q8ProjSMCI := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(q8ProjSPV) * 4), PCode: uintptr(unsafe.Pointer(&q8ProjSPV[0]))}
	var q8ProjShader uintptr
	if res := vk.call(vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&q8ProjSMCI)), 0, uintptr(unsafe.Pointer(&q8ProjShader))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateShaderModule q8 projection: %d", int32(res))
	}
	defer vk.callVoid(vk.destroyShaderModule, r.device, q8ProjShader, 0)
	q8ProjStage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: q8ProjShader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	q8ProjCPCI := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: q8ProjStage, Layout: r.pipelineLayout}
	if res := vk.call(vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&q8ProjCPCI)), 0, uintptr(unsafe.Pointer(&r.q8ProjPipeline))); res != vkSuccess {
		return nil, fmt.Errorf("vkCreateComputePipelines q8 projection: %d", int32(res))
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

func (r *vulkanTextAttentionF32WinRunner) destroy() {
	if r == nil || r.vk == nil {
		return
	}
	if r.pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.pipeline, 0)
	}
	if r.projPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.projPipeline, 0)
	}
	if r.normPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.normPipeline, 0)
	}
	if r.firstTokenProjPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.firstTokenProjPipeline, 0)
	}
	if r.firstTokenQ8Pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.firstTokenQ8Pipeline, 0)
	}
	if r.firstTokenQ6Pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.firstTokenQ6Pipeline, 0)
	}
	if r.firstTokenQ4Pipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.firstTokenQ4Pipeline, 0)
	}
	if r.q8ProjPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.q8ProjPipeline, 0)
	}
	if r.q6ProjPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.q6ProjPipeline, 0)
	}
	if r.q4ProjPipeline != 0 {
		r.vk.callVoid(r.vk.destroyPipeline, r.device, r.q4ProjPipeline, 0)
	}
	if r.fence != 0 {
		r.vk.callVoid(r.vk.destroyFence, r.device, r.fence, 0)
	}
	if r.commandPool != 0 {
		r.vk.callVoid(r.vk.destroyCommandPool, r.device, r.commandPool, 0)
	}
	r.vk.destroyBuffer(r.device, r.qBuf)
	r.vk.destroyBuffer(r.device, r.kBuf)
	r.vk.destroyBuffer(r.device, r.vBuf)
	r.vk.destroyBuffer(r.device, r.outBuf)
	r.vk.destroyBuffer(r.device, r.finalBuf)
	r.vk.destroyBuffer(r.device, r.residualBuf)
	r.vk.destroyBuffer(r.device, r.normBuf)
	for _, b := range r.weightBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.biasBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.q8DataBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.byteDataBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	for _, b := range r.q8ScaleBuffers {
		r.vk.destroyBuffer(r.device, b.buffer)
	}
	r.kCacheKey = 0
	r.vCacheKey = 0
	r.cacheUploaded = 0
	r.cacheKVDim = 0
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

func (r *vulkanTextAttentionF32WinRunner) uploadTextCacheLocked(kCache, vCache []float32, cacheEpoch uint64, cacheLen, kvDim int) error {
	cacheElems, ok := checkedMulInt(cacheLen, kvDim)
	if !ok {
		return fmt.Errorf("text attention cache size overflows: cache_len=%d kv_dim=%d", cacheLen, kvDim)
	}
	kBufferBytes, ok := textAttentionCacheBufferBytesWin(kCache, cacheElems)
	if !ok {
		return fmt.Errorf("text attention k cache byte size overflows: elems=%d len=%d", cacheElems, len(kCache))
	}
	vBufferBytes, ok := textAttentionCacheBufferBytesWin(vCache, cacheElems)
	if !ok {
		return fmt.Errorf("text attention v cache byte size overflows: elems=%d len=%d", cacheElems, len(vCache))
	}
	kKey := float32SliceKey(kCache)
	vKey := float32SliceKey(vCache)
	fullCacheWrite := kKey == 0 || vKey == 0 ||
		kKey != r.kCacheKey || vKey != r.vCacheKey ||
		r.cacheEpoch != cacheEpoch || r.cacheKVDim != kvDim || cacheLen < r.cacheUploaded ||
		r.kBuf.buffer == 0 || r.vBuf.buffer == 0 ||
		r.kBuf.size < kBufferBytes || r.vBuf.size < vBufferBytes
	if err := r.ensureHostBuffer(&r.kBuf, kBufferBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, vBufferBytes); err != nil {
		return err
	}
	if fullCacheWrite {
		if err := r.vk.writeFloat32(r.device, r.kBuf, kCache[:cacheElems]); err != nil {
			return err
		}
		if err := r.vk.writeFloat32(r.device, r.vBuf, vCache[:cacheElems]); err != nil {
			return err
		}
	} else if cacheLen > r.cacheUploaded {
		start, ok := checkedMulInt(r.cacheUploaded, kvDim)
		if !ok {
			return fmt.Errorf("text attention uploaded cache offset overflows: uploaded=%d kv_dim=%d", r.cacheUploaded, kvDim)
		}
		end := cacheElems
		if err := r.vk.writeFloat32At(r.device, r.kBuf, start, kCache[start:end]); err != nil {
			return err
		}
		if err := r.vk.writeFloat32At(r.device, r.vBuf, start, vCache[start:end]); err != nil {
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

func (r *vulkanTextAttentionF32WinRunner) run(out, q, kCache, vCache []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
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
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	bufInfos := [4]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandOnly || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandKVDim != kvDim {
		if err := r.recordAttentionCommand(cacheLen, numHeads, kvHeads, headDim, kvDim); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.outBuf, out[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) recordAttentionCommand(cacheLen, numHeads, kvHeads, headDim, kvDim int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanTextAttentionCommandOnly
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = numHeads * headDim
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) runOut(out, q, kCache, vCache, w, bias []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	wBytes, ok := checkedFloat32ByteLenWinSquare(qRows)
	if !ok {
		return fmt.Errorf("text attention out weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes := qBytes
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
	wBuf, err := r.cachedBuffer(w[:qRows*qRows], wBytes, r.weightBuffers)
	if err != nil {
		return err
	}
	biasBuf, err := r.cachedBuffer(bias[:qRows], biasBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	bufInfos := [7]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandOutF32 || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutF32Command(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.finalBuf, out[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) recordOutF32Command(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanTextAttentionCommandOutF32
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) runOutAddRMSNorm(normOut, residual, q, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	wBytes, ok := checkedFloat32ByteLenWinSquare(qRows)
	if !ok {
		return fmt.Errorf("text attention out+norm weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes := qBytes
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
	wBuf, err := r.cachedBuffer(w[:qRows*qRows], wBytes, r.weightBuffers)
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
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: r.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: r.normBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandOutNorm || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutF32AddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
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
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) runFirstTokenValueOutAddRMSNorm(normOut, residual, kCache, vCache, w, bias, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	kBufferBytes, ok := textAttentionCacheBufferBytesWin(kCache, kvDim)
	if !ok {
		return fmt.Errorf("first-token k cache byte size overflows: elems=%d len=%d", kvDim, len(kCache))
	}
	vBufferBytes, ok := textAttentionCacheBufferBytesWin(vCache, kvDim)
	if !ok {
		return fmt.Errorf("first-token v cache byte size overflows: elems=%d len=%d", kvDim, len(vCache))
	}
	wBytes, ok := checkedFloat32ByteLenWinSquare(qRows)
	if !ok {
		return fmt.Errorf("first-token out weight byte size overflows: q_rows=%d", qRows)
	}
	biasBytes := qBytes
	if err := r.ensureHostBuffer(&r.kBuf, kBufferBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, vBufferBytes); err != nil {
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
	wBuf, err := r.cachedBuffer(w[:qRows*qRows], wBytes, r.weightBuffers)
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
	if err := r.uploadFirstTokenCacheLocked(kCache, vCache, cacheEpoch, kvDim); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: wBuf.buffer, Range: wBytes},
		{Buffer: biasBuf.buffer, Range: biasBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: r.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: r.normBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandFirstTokenValueOutNorm || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordFirstTokenValueOutAddRMSNormCommand(numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
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
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) runFirstTokenValueOutQ8AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache, int8AsBytesWin(w.Data[:dataLen]), w.Scale[:w.Rows], normWeight, dataBytes, r.ensureFirstTokenQ8Pipeline, vulkanTextAttentionCommandFirstTokenValueOutNormQ8, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runFirstTokenValueOutQ6AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.ensureFirstTokenQ6Pipeline, vulkanTextAttentionCommandFirstTokenValueOutNormQ6, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runFirstTokenValueOutQ4AddRMSNorm(normOut, residual, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.ensureFirstTokenQ4Pipeline, vulkanTextAttentionCommandFirstTokenValueOutNormQ4, cacheEpoch, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runFirstTokenValueOutBytesAddRMSNorm(normOut, residual, kCache, vCache []float32, data []byte, scale, normWeight []float32, dataBytes uint64, ensureProjPipeline func() (uintptr, error), commandKind int, cacheEpoch uint64, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	projPipeline, err := ensureProjPipeline()
	if err != nil {
		return err
	}
	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	kBufferBytes, ok := textAttentionCacheBufferBytesWin(kCache, kvDim)
	if !ok {
		return fmt.Errorf("first-token quantized k cache byte size overflows: elems=%d len=%d", kvDim, len(kCache))
	}
	vBufferBytes, ok := textAttentionCacheBufferBytesWin(vCache, kvDim)
	if !ok {
		return fmt.Errorf("first-token quantized v cache byte size overflows: elems=%d len=%d", kvDim, len(vCache))
	}
	scaleBytes, ok := checkedFloat32ByteLenWin(len(scale))
	if !ok {
		return fmt.Errorf("first-token quantized scale byte size overflows: len=%d", len(scale))
	}
	if err := r.ensureHostBuffer(&r.kBuf, kBufferBytes); err != nil {
		return err
	}
	if err := r.ensureHostBuffer(&r.vBuf, vBufferBytes); err != nil {
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
	if err := r.uploadFirstTokenCacheLocked(kCache, vCache, cacheEpoch, kvDim); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: r.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: r.normBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != commandKind || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordFirstTokenValueOutQuantizedAddRMSNormCommand(projPipeline, commandKind, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if err := r.submitTextAttentionCommand(); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) uploadFirstTokenCacheLocked(kCache, vCache []float32, cacheEpoch uint64, kvDim int) error {
	if err := r.vk.writeFloat32(r.device, r.kBuf, kCache[:kvDim]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.vBuf, vCache[:kvDim]); err != nil {
		return err
	}
	r.kCacheKey = float32SliceKey(kCache)
	r.vCacheKey = float32SliceKey(vCache)
	r.cacheEpoch = cacheEpoch
	r.cacheUploaded = 1
	r.cacheKVDim = kvDim
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) recordOutF32AddRMSNormCommand(cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanTextAttentionCommandOutNorm
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) recordFirstTokenValueOutAddRMSNormCommand(numHeads, kvHeads, headDim, qRows, kvDim int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [20]byte
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.firstTokenProjPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = vulkanTextAttentionCommandFirstTokenValueOutNorm
	r.commandCacheLen = 1
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) recordFirstTokenValueOutQuantizedAddRMSNormCommand(projPipeline uintptr, commandKind, numHeads, kvHeads, headDim, qRows, kvDim int) error {
	if res := r.vk.call(r.vk.resetCommandPool, r.device, r.commandPool, 0); res != vkSuccess {
		return fmt.Errorf("vkResetCommandPool: %d", int32(res))
	}
	cmd := r.commandBuffer
	cbi := vkCommandBufferBeginInfo{SType: vkStructureTypeCommandBufferBeginInfo}
	if res := r.vk.call(r.vk.beginCommandBuffer, cmd, uintptr(unsafe.Pointer(&cbi))); res != vkSuccess {
		return fmt.Errorf("vkBeginCommandBuffer: %d", int32(res))
	}
	var pc [20]byte
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, projPipeline)
	r.vk.callVoid(r.vk.cmdBindDescriptorSets, cmd, vkPipelineBindPointCompute, r.pipelineLayout, 0, 1, uintptr(unsafe.Pointer(&r.descriptorSet)), 0, 0)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = commandKind
	r.commandCacheLen = 1
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ8(out, q, kCache, vCache []float32, w *tensor.Q8Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenWin(w.Rows)
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
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	bufInfos := [7]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandOutQ8 || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutQuantizedCommand(r.q8ProjPipeline, vulkanTextAttentionCommandOutQ8, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.finalBuf, out[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ6(out, q, kCache, vCache []float32, w *tensor.Q6Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytes(out, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], dataBytes, r.ensureQ6ProjPipeline, vulkanTextAttentionCommandOutQ6, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ4(out, q, kCache, vCache []float32, w *tensor.Q4Matrix, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytes(out, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], dataBytes, r.ensureQ4ProjPipeline, vulkanTextAttentionCommandOutQ4, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ8AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q8Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	dataLen, dataBytes, err := checkedTextAttentionQ8DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenWin(w.Rows)
	if !ok {
		return fmt.Errorf("text attention q8 out+norm scale byte size overflows: rows=%d", w.Rows)
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
	dataBuf, err := r.q8DataBuffer(w.Data[:dataLen], dataBytes)
	if err != nil {
		return err
	}
	scaleBuf, err := r.cachedBuffer(w.Scale[:w.Rows], scaleBytes, r.q8ScaleBuffers)
	if err != nil {
		return err
	}
	normWeightBuf, err := r.cachedBuffer(normWeight[:qRows], qBytes, r.biasBuffers)
	if err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: r.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: r.normBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != vulkanTextAttentionCommandOutNormQ8 || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutQuantizedAddRMSNormCommand(r.q8ProjPipeline, vulkanTextAttentionCommandOutNormQ8, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if err := r.submitTextAttentionCommand(); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ6AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q6Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ6DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytesAddRMSNorm(normOut, residual, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.ensureQ6ProjPipeline, vulkanTextAttentionCommandOutNormQ6, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runOutQ4AddRMSNorm(normOut, residual, q, kCache, vCache []float32, w *tensor.Q4Matrix, normWeight []float32, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	dataLen, dataBytes, err := checkedTextAttentionQ4DataBytesWin(w.Rows, w.Cols)
	if err != nil {
		return err
	}
	return r.runOutBytesAddRMSNorm(normOut, residual, q, kCache, vCache, w.Data[:dataLen], w.Scale[:w.Rows], normWeight, dataBytes, r.ensureQ4ProjPipeline, vulkanTextAttentionCommandOutNormQ4, cacheEpoch, cacheLen, numHeads, kvHeads, headDim)
}

func (r *vulkanTextAttentionF32WinRunner) runOutBytes(out, q, kCache, vCache []float32, data []byte, scale []float32, dataBytes uint64, ensureProjPipeline func() (uintptr, error), commandKind int, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	projPipeline, err := ensureProjPipeline()
	if err != nil {
		return err
	}
	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenWin(len(scale))
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
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	bufInfos := [7]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])

	if !r.commandRecorded || r.commandKind != commandKind || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutQuantizedCommand(projPipeline, commandKind, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
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
	return r.vk.readFloat32Into(r.device, r.finalBuf, out[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) runOutBytesAddRMSNorm(normOut, residual, q, kCache, vCache []float32, data []byte, scale, normWeight []float32, dataBytes uint64, ensureProjPipeline func() (uintptr, error), commandKind int, cacheEpoch uint64, cacheLen, numHeads, kvHeads, headDim int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	projPipeline, err := ensureProjPipeline()
	if err != nil {
		return err
	}
	qRows, kvDim, qBytes, err := checkedTextAttentionWinDims(numHeads, kvHeads, headDim)
	if err != nil {
		return err
	}
	scaleBytes, ok := checkedFloat32ByteLenWin(len(scale))
	if !ok {
		return fmt.Errorf("text attention quantized out+norm scale byte size overflows: len=%d", len(scale))
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
	if err := r.vk.writeFloat32(r.device, r.qBuf, q[:qRows]); err != nil {
		return err
	}
	if err := r.vk.writeFloat32(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	bufInfos := [10]vkDescriptorBufferInfo{
		{Buffer: r.qBuf.buffer, Range: qBytes},
		{Buffer: r.kBuf.buffer, Range: r.kBuf.size},
		{Buffer: r.vBuf.buffer, Range: r.vBuf.size},
		{Buffer: r.outBuf.buffer, Range: qBytes},
		{Buffer: dataBuf.buffer, Range: dataBytes},
		{Buffer: scaleBuf.buffer, Range: scaleBytes},
		{Buffer: r.finalBuf.buffer, Range: qBytes},
		{Buffer: r.residualBuf.buffer, Range: qBytes},
		{Buffer: normWeightBuf.buffer, Range: qBytes},
		{Buffer: r.normBuf.buffer, Range: qBytes},
	}
	updateVulkanDescriptorBuffersWin(r.vk, r.device, r.descriptorSet, r.descriptorCache[:], bufInfos[:])
	if !r.commandRecorded || r.commandKind != commandKind || r.commandCacheLen != cacheLen || r.commandNumHeads != numHeads || r.commandKVHeads != kvHeads || r.commandHeadDim != headDim || r.commandQRows != qRows || r.commandKVDim != kvDim {
		if err := r.recordOutQuantizedAddRMSNormCommand(projPipeline, commandKind, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim); err != nil {
			return err
		}
	}
	if err := r.submitTextAttentionCommand(); err != nil {
		return err
	}
	if err := r.vk.readFloat32Into(r.device, r.residualBuf, residual[:qRows]); err != nil {
		return err
	}
	return r.vk.readFloat32Into(r.device, r.normBuf, normOut[:qRows])
}

func (r *vulkanTextAttentionF32WinRunner) submitTextAttentionCommand() error {
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
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) recordOutQuantizedCommand(projPipeline uintptr, commandKind, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = commandKind
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) recordOutQuantizedAddRMSNormCommand(projPipeline uintptr, commandKind, cacheLen, numHeads, kvHeads, headDim, qRows, kvDim int) error {
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
	binary.LittleEndian.PutUint32(pc[0:4], uint32(cacheLen))
	binary.LittleEndian.PutUint32(pc[4:8], uint32(numHeads))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(kvHeads))
	binary.LittleEndian.PutUint32(pc[12:16], uint32(headDim))
	binary.LittleEndian.PutUint32(pc[16:20], uint32(kvDim))
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(numHeads), 1, 1)
	barrier := vkMemoryBarrier{SType: vkStructureTypeMemoryBarrier, SrcAccessMask: vkAccessShaderWriteBit, DstAccessMask: vkAccessShaderReadBit}
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, projPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], 1)
	binary.LittleEndian.PutUint32(pc[4:8], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[8:12], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[12:16], 0)
	binary.LittleEndian.PutUint32(pc[16:20], 0)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, uintptr(qRows), 1, 1)
	r.vk.callVoid(r.vk.cmdPipelineBarrier, cmd, vkPipelineStageComputeShaderBit, vkPipelineStageComputeShaderBit, 0, 1, uintptr(unsafe.Pointer(&barrier)), 0, 0, 0, 0)
	r.vk.callVoid(r.vk.cmdBindPipeline, cmd, vkPipelineBindPointCompute, r.normPipeline)
	binary.LittleEndian.PutUint32(pc[0:4], uint32(qRows))
	binary.LittleEndian.PutUint32(pc[4:8], 1)
	r.vk.callVoid(r.vk.cmdPushConstants, cmd, r.pipelineLayout, vkShaderStageComputeBit, 0, uintptr(len(pc)), uintptr(unsafe.Pointer(&pc[0])))
	r.vk.callVoid(r.vk.cmdDispatch, cmd, 1, 1, 1)
	if res := r.vk.call(r.vk.endCommandBuffer, cmd); res != vkSuccess {
		return fmt.Errorf("vkEndCommandBuffer: %d", int32(res))
	}
	r.commandKind = commandKind
	r.commandCacheLen = cacheLen
	r.commandNumHeads = numHeads
	r.commandKVHeads = kvHeads
	r.commandHeadDim = headDim
	r.commandQRows = qRows
	r.commandKVDim = kvDim
	r.commandRecorded = true
	return nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureQ6ProjPipeline() (uintptr, error) {
	if r.q6ProjPipeline != 0 {
		return r.q6ProjPipeline, nil
	}
	spv, err := vulkanTextAttentionOutQ6ShaderCodeWindows()
	if err != nil {
		return 0, err
	}
	pipeline, err := r.createProjectionPipeline(spv, "q6 projection")
	if err != nil {
		return 0, err
	}
	r.q6ProjPipeline = pipeline
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureQ4ProjPipeline() (uintptr, error) {
	if r.q4ProjPipeline != 0 {
		return r.q4ProjPipeline, nil
	}
	spv, err := vulkanTextAttentionOutQ4ShaderCodeWindows()
	if err != nil {
		return 0, err
	}
	pipeline, err := r.createProjectionPipeline(spv, "q4 projection")
	if err != nil {
		return 0, err
	}
	r.q4ProjPipeline = pipeline
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureFirstTokenQ8Pipeline() (uintptr, error) {
	if r.firstTokenQ8Pipeline != 0 {
		return r.firstTokenQ8Pipeline, nil
	}
	spv, err := vulkanTextFirstTokenValueOutQ8ShaderCodeWindows()
	if err != nil {
		return 0, err
	}
	pipeline, err := r.createProjectionPipeline(spv, "first-token q8 projection")
	if err != nil {
		return 0, err
	}
	r.firstTokenQ8Pipeline = pipeline
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureFirstTokenQ6Pipeline() (uintptr, error) {
	if r.firstTokenQ6Pipeline != 0 {
		return r.firstTokenQ6Pipeline, nil
	}
	spv, err := vulkanTextFirstTokenValueOutQ6ShaderCodeWindows()
	if err != nil {
		return 0, err
	}
	pipeline, err := r.createProjectionPipeline(spv, "first-token q6 projection")
	if err != nil {
		return 0, err
	}
	r.firstTokenQ6Pipeline = pipeline
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureFirstTokenQ4Pipeline() (uintptr, error) {
	if r.firstTokenQ4Pipeline != 0 {
		return r.firstTokenQ4Pipeline, nil
	}
	spv, err := vulkanTextFirstTokenValueOutQ4ShaderCodeWindows()
	if err != nil {
		return 0, err
	}
	pipeline, err := r.createProjectionPipeline(spv, "first-token q4 projection")
	if err != nil {
		return 0, err
	}
	r.firstTokenQ4Pipeline = pipeline
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) createProjectionPipeline(spv []uint32, label string) (uintptr, error) {
	entryName := append([]byte("main"), 0)
	smci := vkShaderModuleCreateInfo{SType: vkStructureTypeShaderModuleCreateInfo, CodeSize: uintptr(len(spv) * 4), PCode: uintptr(unsafe.Pointer(&spv[0]))}
	var shader uintptr
	if res := r.vk.call(r.vk.createShaderModule, r.device, uintptr(unsafe.Pointer(&smci)), 0, uintptr(unsafe.Pointer(&shader))); res != vkSuccess {
		return 0, fmt.Errorf("vkCreateShaderModule %s: %d", label, int32(res))
	}
	defer r.vk.callVoid(r.vk.destroyShaderModule, r.device, shader, 0)
	stage := vkPipelineShaderStageCreateInfo{SType: vkStructureTypePipelineShaderStageCreateInfo, Stage: vkShaderStageComputeBit, Module: shader, PName: uintptr(unsafe.Pointer(&entryName[0]))}
	cpci := vkComputePipelineCreateInfo{SType: vkStructureTypeComputePipelineCreateInfo, Stage: stage, Layout: r.pipelineLayout}
	var pipeline uintptr
	if res := r.vk.call(r.vk.createComputePipelines, r.device, 0, 1, uintptr(unsafe.Pointer(&cpci)), 0, uintptr(unsafe.Pointer(&pipeline))); res != vkSuccess {
		return 0, fmt.Errorf("vkCreateComputePipelines %s: %d", label, int32(res))
	}
	return pipeline, nil
}

func (r *vulkanTextAttentionF32WinRunner) ensureHostBuffer(buf *vkHostBufferWin, size uint64) error {
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

func textAttentionCacheBufferBytesWin(cache []float32, minElems int) (uint64, bool) {
	if len(cache) > minElems {
		return checkedFloat32ByteLenWin(len(cache))
	}
	return checkedFloat32ByteLenWin(minElems)
}

func checkedTextAttentionWinCacheElems(cacheLen, kvDim int) (int, error) {
	cacheElems, ok := checkedMulInt(cacheLen, kvDim)
	if !ok {
		return 0, fmt.Errorf("text attention cache elements overflow: cacheLen=%d kvDim=%d", cacheLen, kvDim)
	}
	return cacheElems, nil
}

func checkedTextAttentionWinDims(numHeads, kvHeads, headDim int) (qRows, kvDim int, qBytes uint64, err error) {
	qRows, ok := checkedMulInt(numHeads, headDim)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention q rows overflow: num_heads=%d head_dim=%d", numHeads, headDim)
	}
	kvDim, ok = checkedMulInt(kvHeads, headDim)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention kv dim overflow: kv_heads=%d head_dim=%d", kvHeads, headDim)
	}
	qBytes, ok = checkedFloat32ByteLenWin(qRows)
	if !ok {
		return 0, 0, 0, fmt.Errorf("text attention q byte size overflows: q_rows=%d", qRows)
	}
	return qRows, kvDim, qBytes, nil
}

func checkedFloat32ByteLenWin(n int) (uint64, bool) {
	if n < 0 || n > maxInt()/4 {
		return 0, false
	}
	return uint64(n) * 4, true
}

func checkedFloat32ByteLenWinSquare(n int) (uint64, bool) {
	elements, ok := checkedMulInt(n, n)
	if !ok {
		return 0, false
	}
	return checkedFloat32ByteLenWin(elements)
}

func checkedTextAttentionQ8DataBytesWin(rows, cols int) (int, uint64, error) {
	dataLen, ok := checkedMulInt(rows, cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q8 data length overflows: rows=%d cols=%d", rows, cols)
	}
	dataBytes, ok := checkedAlignedByteLenWin(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q8 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedTextAttentionQ6DataBytesWin(rows, cols int) (int, uint64, error) {
	packedCols, ok := checkedPackedQ6ColsWin(cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 packed cols overflow: cols=%d", cols)
	}
	dataLen, ok := checkedMulInt(rows, packedCols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 data length overflows: rows=%d packed_cols=%d", rows, packedCols)
	}
	dataBytes, ok := checkedAlignedByteLenWin(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q6 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedTextAttentionQ4DataBytesWin(rows, cols int) (int, uint64, error) {
	packedCols, ok := checkedPackedQ4ColsWin(cols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 packed cols overflow: cols=%d", cols)
	}
	dataLen, ok := checkedMulInt(rows, packedCols)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 data length overflows: rows=%d packed_cols=%d", rows, packedCols)
	}
	dataBytes, ok := checkedAlignedByteLenWin(dataLen, 4)
	if !ok {
		return 0, 0, fmt.Errorf("text attention q4 data byte size overflows: elems=%d", dataLen)
	}
	return dataLen, dataBytes, nil
}

func checkedPackedQ6ColsWin(cols int) (int, bool) {
	if cols <= 0 || cols > (maxInt()-7)/6 {
		return 0, false
	}
	return tensor.PackedQ6Cols(cols), true
}

func checkedPackedQ4ColsWin(cols int) (int, bool) {
	if cols <= 0 || cols == maxInt() {
		return 0, false
	}
	return (cols + 1) / 2, true
}

func checkedAlignedByteLenWin(n, alignment int) (uint64, bool) {
	if n < 0 || alignment <= 0 || n > maxInt()-(alignment-1) {
		return 0, false
	}
	return uint64(alignUpInt(n, alignment)), true
}

func (r *vulkanTextAttentionF32WinRunner) cachedBuffer(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	return cachedFloat32BufferWin(r.vk, r.device, r.memProps, data, size, cache)
}

func (r *vulkanTextAttentionF32WinRunner) cachedBufferRefresh(data []float32, size uint64, cache map[uintptr]vulkanCachedFloat32BufferWin) (vkHostBufferWin, error) {
	key := float32SliceKey(data)
	fingerprint := fingerprintFloat32ForVulkanCache(data)
	if cached, ok := cache[key]; ok && cached.buffer.size >= size && cached.length == len(data) {
		if cached.fingerprint == fingerprint {
			return cached.buffer, nil
		}
		hash := hashFloat32ForVulkanCache(data)
		if err := r.vk.writeFloat32(r.device, cached.buffer, data); err != nil {
			return vkHostBufferWin{}, err
		}
		cache[key] = vulkanCachedFloat32BufferWin{buffer: cached.buffer, length: len(data), hash: hash, fingerprint: fingerprint}
		return cached.buffer, nil
	}
	if old, ok := cache[key]; ok {
		r.vk.destroyBuffer(r.device, old.buffer)
		delete(cache, key)
	}
	buf, err := r.vk.newHostBuffer(r.device, r.memProps, size)
	if err != nil {
		return vkHostBufferWin{}, err
	}
	if err := r.vk.writeFloat32(r.device, buf, data); err != nil {
		r.vk.destroyBuffer(r.device, buf)
		return vkHostBufferWin{}, err
	}
	cache[key] = vulkanCachedFloat32BufferWin{buffer: buf, length: len(data), hash: hashFloat32ForVulkanCache(data), fingerprint: fingerprint}
	return buf, nil
}

func (r *vulkanTextAttentionF32WinRunner) q8DataBuffer(data []int8, size uint64) (vkHostBufferWin, error) {
	return cachedInt8BufferWin(r.vk, r.device, r.memProps, data, size, r.q8DataBuffers)
}

func (r *vulkanTextAttentionF32WinRunner) byteDataBuffer(data []byte, size uint64) (vkHostBufferWin, error) {
	return cachedByteBufferWin(r.vk, r.device, r.memProps, data, size, r.byteDataBuffers)
}

func int8AsBytesWin(data []int8) []byte {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data))
}

type vulkanCachedInt8BufferWin struct {
	buffer      vkHostBufferWin
	length      int
	hash        uint64
	fingerprint uint64
}

type vulkanCachedByteBufferWin struct {
	buffer      vkHostBufferWin
	length      int
	hash        uint64
	fingerprint uint64
}

func hashInt8ForVulkanCache(values []int8) uint64 {
	h := uint64(1469598103934665603)
	for _, v := range values {
		h ^= uint64(byte(v))
		h *= 1099511628211
	}
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

func hashBytesForVulkanCache(values []byte) uint64 {
	h := uint64(1469598103934665603)
	for _, v := range values {
		h ^= uint64(v)
		h *= 1099511628211
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

func float32SliceKey(v []float32) uintptr {
	if len(v) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&v[0]))
}

func vulkanTextAttentionF32ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionF32SPV.once.Do(func() {
		vulkanTextAttentionF32SPV.code, vulkanTextAttentionF32SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionF32GLSL)
	})
	return vulkanTextAttentionF32SPV.code, vulkanTextAttentionF32SPV.err
}

func vulkanTextAttentionOutF32ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionOutF32SPV.once.Do(func() {
		vulkanTextAttentionOutF32SPV.code, vulkanTextAttentionOutF32SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionOutF32GLSL)
	})
	return vulkanTextAttentionOutF32SPV.code, vulkanTextAttentionOutF32SPV.err
}

func vulkanTextAttentionOutAddRMSNormF32ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionOutAddRMSNormF32SPV.once.Do(func() {
		vulkanTextAttentionOutAddRMSNormF32SPV.code, vulkanTextAttentionOutAddRMSNormF32SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionOutAddRMSNormF32GLSL)
	})
	return vulkanTextAttentionOutAddRMSNormF32SPV.code, vulkanTextAttentionOutAddRMSNormF32SPV.err
}

func vulkanTextFirstTokenValueOutAddRMSNormF32ShaderCodeWindows() ([]uint32, error) {
	vulkanTextFirstTokenValueOutAddRMSNormF32SPV.once.Do(func() {
		vulkanTextFirstTokenValueOutAddRMSNormF32SPV.code, vulkanTextFirstTokenValueOutAddRMSNormF32SPV.err = compileVulkanGLSLWindows(vulkanTextFirstTokenValueOutF32GLSL)
	})
	return vulkanTextFirstTokenValueOutAddRMSNormF32SPV.code, vulkanTextFirstTokenValueOutAddRMSNormF32SPV.err
}

func vulkanTextAttentionOutQ8ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionOutQ8SPV.once.Do(func() {
		vulkanTextAttentionOutQ8SPV.code, vulkanTextAttentionOutQ8SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionOutQ8GLSL)
	})
	return vulkanTextAttentionOutQ8SPV.code, vulkanTextAttentionOutQ8SPV.err
}

func vulkanTextAttentionOutQ6ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionOutQ6SPV.once.Do(func() {
		vulkanTextAttentionOutQ6SPV.code, vulkanTextAttentionOutQ6SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionOutQ6GLSL)
	})
	return vulkanTextAttentionOutQ6SPV.code, vulkanTextAttentionOutQ6SPV.err
}

func vulkanTextAttentionOutQ4ShaderCodeWindows() ([]uint32, error) {
	vulkanTextAttentionOutQ4SPV.once.Do(func() {
		vulkanTextAttentionOutQ4SPV.code, vulkanTextAttentionOutQ4SPV.err = compileVulkanGLSLWindows(vulkanTextAttentionOutQ4GLSL)
	})
	return vulkanTextAttentionOutQ4SPV.code, vulkanTextAttentionOutQ4SPV.err
}

func vulkanTextFirstTokenValueOutQ8ShaderCodeWindows() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ8SPV.once.Do(func() {
		vulkanTextFirstTokenValueOutQ8SPV.code, vulkanTextFirstTokenValueOutQ8SPV.err = compileVulkanGLSLWindows(vulkanTextFirstTokenValueOutQ8GLSL)
	})
	return vulkanTextFirstTokenValueOutQ8SPV.code, vulkanTextFirstTokenValueOutQ8SPV.err
}

func vulkanTextFirstTokenValueOutQ6ShaderCodeWindows() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ6SPV.once.Do(func() {
		vulkanTextFirstTokenValueOutQ6SPV.code, vulkanTextFirstTokenValueOutQ6SPV.err = compileVulkanGLSLWindows(vulkanTextFirstTokenValueOutQ6GLSL)
	})
	return vulkanTextFirstTokenValueOutQ6SPV.code, vulkanTextFirstTokenValueOutQ6SPV.err
}

func vulkanTextFirstTokenValueOutQ4ShaderCodeWindows() ([]uint32, error) {
	vulkanTextFirstTokenValueOutQ4SPV.once.Do(func() {
		vulkanTextFirstTokenValueOutQ4SPV.code, vulkanTextFirstTokenValueOutQ4SPV.err = compileVulkanGLSLWindows(vulkanTextFirstTokenValueOutQ4GLSL)
	})
	return vulkanTextFirstTokenValueOutQ4SPV.code, vulkanTextFirstTokenValueOutQ4SPV.err
}

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

const vulkanTextFirstTokenValueOutF32GLSL = `#version 450
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

const vulkanTextFirstTokenValueOutQ8GLSL = `#version 450
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

const vulkanTextFirstTokenValueOutQ6GLSL = `#version 450
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

const vulkanTextFirstTokenValueOutQ4GLSL = `#version 450
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
