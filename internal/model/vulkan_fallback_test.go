package model

import (
	"math"
	"os"
	"testing"

	"paddleocrvl-go/internal/config"
	"paddleocrvl-go/internal/tensor"
)

func TestRuntimeVulkanOpDisableMask(t *testing.T) {
	rt := &Runtime{}
	if rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("vulkan op should be disabled when backend is not vulkan")
	}
	rt.backend = "vulkan"
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("vulkan op should start enabled for vulkan backend")
	}
	rt.disableVulkanOp(vulkanOpMatVecQ8)
	if rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("disabled vulkan op should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ8) || !rt.vulkanOpEnabled(vulkanOpMatVecTopKQ8) {
		t.Fatal("disabling q8 matvec should not disable q8 argmax/top-k")
	}
	rt.disableVulkanOp(vulkanOpMatVecArgmaxQ8)
	if rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ8) {
		t.Fatal("disabled q8 argmax should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecTopKQ8) {
		t.Fatal("disabling q8 argmax should not disable q8 top-k")
	}
	rt.disableVulkanOp(vulkanOpMatVecTopKQ8)
	if rt.vulkanOpEnabled(vulkanOpMatVecTopKQ8) {
		t.Fatal("disabled q8 top-k should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ6) {
		t.Fatal("disabling q8 should not disable q6")
	}
	rt.disableVulkanOp(vulkanOpTextAttentionOutQ8)
	if rt.vulkanOpEnabled(vulkanOpTextAttentionOutQ8) {
		t.Fatal("disabled text attention out q8 should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpTextAttentionOutQ6) {
		t.Fatal("disabling text attention out q8 should not disable q6")
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownQ8)
	if rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ8) {
		t.Fatal("disabled swiglu q8 should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ6) {
		t.Fatal("disabling swiglu q8 should not disable q6")
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownF32)
	if rt.vulkanOpEnabled(vulkanOpSwiGLUDownF32) {
		t.Fatal("disabled swiglu down f32 should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpF32) {
		t.Fatal("disabling swiglu down f32 should not disable swiglu gate/up f32")
	}
	rt.disableVulkanOp(vulkanOpSwiGLUGateUpF32)
	if rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpF32) {
		t.Fatal("disabled swiglu gate/up f32 should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ8) {
		t.Fatal("disabling swiglu gate/up f32 should not disable swiglu gate/up q8")
	}
	rt.disableVulkanOp(vulkanOpSwiGLUGateUpQ8)
	if rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ8) {
		t.Fatal("disabled swiglu gate/up q8 should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ6) || !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ4) {
		t.Fatal("disabling swiglu gate/up q8 should not disable q6/q4 gate/up")
	}
	fresh := &Runtime{backend: "vulkan"}
	fresh.disableVulkanOp(vulkanOpSwiGLUGateUpQ8)
	if !fresh.vulkanOpEnabled(vulkanOpSwiGLUDownQ8) {
		t.Fatal("disabling swiglu gate/up q8 should not disable swiglu down q8")
	}
	swigluNormOps := []struct {
		name    string
		full    vulkanOp
		outOnly vulkanOp
	}{
		{name: "f32", full: vulkanOpSwiGLUDownAddRMSNormF32, outOnly: vulkanOpSwiGLUDownAddRMSNormOutOnlyF32},
		{name: "q8", full: vulkanOpSwiGLUDownAddRMSNormQ8, outOnly: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ8},
		{name: "q6", full: vulkanOpSwiGLUDownAddRMSNormQ6, outOnly: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ6},
		{name: "q4", full: vulkanOpSwiGLUDownAddRMSNormQ4, outOnly: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ4},
	}
	for _, ops := range swigluNormOps {
		rt.disableVulkanOp(ops.full)
		if rt.vulkanOpEnabled(ops.full) {
			t.Fatalf("disabled swiglu add rmsnorm %s should not be enabled", ops.name)
		}
		if !rt.vulkanOpEnabled(ops.outOnly) {
			t.Fatalf("disabling swiglu add rmsnorm %s should not disable out-only", ops.name)
		}
		rt.disableVulkanOp(ops.outOnly)
		if rt.vulkanOpEnabled(ops.outOnly) {
			t.Fatalf("disabled swiglu add rmsnorm %s out-only should not be enabled", ops.name)
		}
	}
	rt.disableVulkanOp(vulkanOpVisionMatRowsBiasF32)
	if rt.vulkanOpEnabled(vulkanOpVisionMatRowsBiasF32) {
		t.Fatal("disabled vision matrows bias should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpVisionAttentionF32) {
		t.Fatal("disabling vision matrows bias should not disable vision attention")
	}
	rt.disableVulkanOp(vulkanOpVisionProjectImageF32)
	if rt.vulkanOpEnabled(vulkanOpVisionProjectImageF32) {
		t.Fatal("disabled vision projection should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpVisionMatRowsGELU2F32) {
		t.Fatal("disabling vision projection should not disable vision gelu2")
	}
	rt.disableVulkanOp(vulkanOpAddRMSNormF32)
	if rt.vulkanOpEnabled(vulkanOpAddRMSNormF32) {
		t.Fatal("disabled add rmsnorm should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpAddRMSNormOutOnlyF32) {
		t.Fatal("disabling add rmsnorm should not disable add rmsnorm out-only")
	}
	rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32)
	if rt.vulkanOpEnabled(vulkanOpAddRMSNormOutOnlyF32) {
		t.Fatal("disabled add rmsnorm out-only should not be enabled")
	}
	rt.disableVulkanOp(vulkanOpMRoPEPairF32)
	if rt.vulkanOpEnabled(vulkanOpMRoPEPairF32) {
		t.Fatal("disabled mrope pair should not be enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpMRoPEF32) {
		t.Fatal("disabling mrope pair should not disable single mrope")
	}
	rt.disableVulkanOp(vulkanOpMRoPEF32)
	if rt.vulkanOpEnabled(vulkanOpMRoPEF32) {
		t.Fatal("disabled single mrope should not be enabled")
	}
}

func TestMatVecTopKQuantizedDisableDoesNotDisableMatVec(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpMatVecTopKQ8)
	q := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	candidates, ok := rt.matVecTopKMaybeVulkan([]float32{1, 2}, nil, q, nil, nil, 3, 2, 2, &generationScratch{})
	if ok || candidates != nil {
		t.Fatalf("disabled q8 top-k returned candidates=%v ok=%v", candidates, ok)
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("disabling q8 top-k should leave q8 matvec enabled")
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecArgmaxQ8) {
		t.Fatal("disabling q8 top-k should leave q8 argmax enabled")
	}
}

func TestQuantizedMatVecVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	out := make([]float32, 3)
	expectPanic := func(t *testing.T, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatal("expected CPU fallback to reject invalid quantized matrix")
			}
		}()
		fn()
	}
	badQ8 := &tensor.Q8Matrix{Rows: 3, Cols: 2, Data: make([]int8, 5), Scale: make([]float32, 3)}
	badQ6 := &tensor.Q6Matrix{Rows: 3, Cols: 2, Data: make([]byte, 3*tensor.PackedQ6Cols(2)-1), Scale: make([]float32, 3)}
	badQ4 := &tensor.Q4Matrix{Rows: 3, Cols: 2, Data: make([]byte, 3*((2+1)/2)-1), Scale: make([]float32, 3)}
	cases := []struct {
		name string
		ops  []vulkanOp
		call func(*Runtime) bool
	}{
		{name: "q8", ops: []vulkanOp{vulkanOpMatVecQ8, vulkanOpMatVecArgmaxQ8, vulkanOpMatVecTopKQ8}, call: func(rt *Runtime) bool {
			expectPanic(t, func() { rt.matVecMaybeQuant(out, x, nil, badQ8, nil, nil, 3, 2) })
			if rt.matVecOnlyMaybeVulkan(out, x, nil, badQ8, nil, nil, 3, 2) {
				return true
			}
			_, _, ok := rt.matVecArgmaxMaybeVulkan(x, nil, badQ8, nil, nil, 3, 2)
			if ok {
				return true
			}
			_, ok = rt.matVecTopKMaybeVulkan(x, nil, badQ8, nil, nil, 3, 2, 2, &generationScratch{})
			return ok
		}},
		{name: "q6", ops: []vulkanOp{vulkanOpMatVecQ6, vulkanOpMatVecArgmaxQ6, vulkanOpMatVecTopKQ6}, call: func(rt *Runtime) bool {
			expectPanic(t, func() { rt.matVecMaybeQuant(out, x, nil, nil, badQ6, nil, 3, 2) })
			if rt.matVecOnlyMaybeVulkan(out, x, nil, nil, badQ6, nil, 3, 2) {
				return true
			}
			_, _, ok := rt.matVecArgmaxMaybeVulkan(x, nil, nil, badQ6, nil, 3, 2)
			if ok {
				return true
			}
			_, ok = rt.matVecTopKMaybeVulkan(x, nil, nil, badQ6, nil, 3, 2, 2, &generationScratch{})
			return ok
		}},
		{name: "q4", ops: []vulkanOp{vulkanOpMatVecQ4, vulkanOpMatVecArgmaxQ4, vulkanOpMatVecTopKQ4}, call: func(rt *Runtime) bool {
			expectPanic(t, func() { rt.matVecMaybeQuant(out, x, nil, nil, nil, badQ4, 3, 2) })
			if rt.matVecOnlyMaybeVulkan(out, x, nil, nil, nil, badQ4, 3, 2) {
				return true
			}
			_, _, ok := rt.matVecArgmaxMaybeVulkan(x, nil, nil, nil, badQ4, 3, 2)
			if ok {
				return true
			}
			_, ok = rt.matVecTopKMaybeVulkan(x, nil, nil, nil, badQ4, 3, 2, 2, &generationScratch{})
			return ok
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			if tc.call(rt) {
				t.Fatal("bad quantized matrix unexpectedly reached a Vulkan matvec path")
			}
			for _, op := range tc.ops {
				if !rt.vulkanOpEnabled(op) {
					t.Fatalf("%s op %d was disabled by invalid shape", tc.name, op)
				}
			}
		})
	}
}

func TestQuantizedMatVecVulkanRejectsShortOutputWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	out := make([]float32, 2)
	q8 := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	q6 := tensor.QuantizeQ6Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	q4 := tensor.QuantizeQ4Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	cases := []struct {
		name string
		op   vulkanOp
		call func(*Runtime)
	}{
		{name: "q8", op: vulkanOpMatVecQ8, call: func(rt *Runtime) { rt.matVecMaybeQuant(out, x, nil, q8, nil, nil, 3, 2) }},
		{name: "q6", op: vulkanOpMatVecQ6, call: func(rt *Runtime) { rt.matVecMaybeQuant(out, x, nil, nil, q6, nil, 3, 2) }},
		{name: "q4", op: vulkanOpMatVecQ4, call: func(rt *Runtime) { rt.matVecMaybeQuant(out, x, nil, nil, nil, q4, 3, 2) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			defer func() {
				if recover() == nil {
					t.Fatal("expected CPU fallback to reject short output")
				}
				if !rt.vulkanOpEnabled(tc.op) {
					t.Fatalf("%s matvec op was disabled by short output", tc.name)
				}
			}()
			tc.call(rt)
		})
	}
}

func TestGroupedMatVecVulkanPrechecksAllSegmentsBeforeDispatch(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	x := []float32{1, 2}
	good := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	bad := &tensor.Q8Matrix{Rows: 3, Cols: 2, Data: make([]int8, 5), Scale: make([]float32, 3)}
	outA := []float32{99, 99, 99}
	outB := []float32{88, 88, 88}
	outC := []float32{77, 77, 77}
	if rt.matVec2MaybeVulkan(outA, outB, x, nil, nil, good, bad, nil, nil, nil, nil, 3, 3, 2) {
		t.Fatal("bad second segment unexpectedly reached grouped Vulkan matvec2 path")
	}
	if outA[0] != 99 || outA[1] != 99 || outA[2] != 99 {
		t.Fatalf("first segment was dispatched before full precheck: outA=%v", outA)
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("q8 matvec op was disabled by grouped precheck failure")
	}
	if rt.matVec3MaybeVulkan(outA, outB, outC, x, nil, nil, nil, good, good, bad, nil, nil, nil, nil, nil, nil, 3, 3, 3, 2) {
		t.Fatal("bad third segment unexpectedly reached grouped Vulkan matvec3 path")
	}
	if outA[0] != 99 || outA[1] != 99 || outA[2] != 99 || outB[0] != 88 || outB[1] != 88 || outB[2] != 88 {
		t.Fatalf("earlier segments were dispatched before full precheck: outA=%v outB=%v", outA, outB)
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("q8 matvec op was disabled by grouped precheck failure")
	}
}

func TestGroupedMatVecVulkanUsesCombinedWorkThreshold(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "100")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	x64 := make([]float32, 64)
	out1 := make([]float32, 1)
	w64 := make([]float32, 64)
	if rt.matVecVulkanReady(out1, x64, w64, nil, nil, nil, 1, 64) {
		t.Fatal("single matvec below threshold should not be ready")
	}
	if !rt.matVecVulkanShapeReady(out1, x64, w64, nil, nil, nil, 1, 64) {
		t.Fatal("shape-only readiness should accept a valid segment below the work threshold")
	}
	if !rt.matVec2VulkanReady(out1, out1, x64, w64, w64, nil, nil, nil, nil, nil, nil, 1, 1, 64) {
		t.Fatal("matvec2 should be ready when combined work reaches the threshold")
	}
	if rt.matVec2VulkanReady(out1, out1, x64[:49], w64[:49], w64[:49], nil, nil, nil, nil, nil, nil, 1, 1, 49) {
		t.Fatal("matvec2 below combined work threshold should not be ready")
	}

	x40 := make([]float32, 40)
	w40 := make([]float32, 40)
	if rt.matVecVulkanReady(out1, x40, w40, nil, nil, nil, 1, 40) {
		t.Fatal("single 1x40 matvec below threshold should not be ready")
	}
	if !rt.matVec3VulkanReady(out1, out1, out1, x40, w40, w40, w40, nil, nil, nil, nil, nil, nil, nil, nil, nil, 1, 1, 1, 40) {
		t.Fatal("matvec3 should be ready when combined work reaches the threshold")
	}
}

func TestVulkanMatVecReadinessRejectsOverflowWork(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rows := maxModelInt()/2 + 1
	cols := 3
	if vulkanMatVecWorkReady(rows, cols) {
		t.Fatal("overflowing matvec work should not be Vulkan-ready")
	}
	if groupedVulkanMatVecWorkReady(cols, rows, rows) {
		t.Fatal("overflowing grouped matvec work should not be Vulkan-ready")
	}
	if vulkanMatVecAddNormWorkReady(rows, cols) {
		t.Fatal("overflowing matvec+add+rmsnorm work should not be Vulkan-ready")
	}
	if q8MatVecShapeOK(&tensor.Q8Matrix{Rows: rows, Cols: cols}, rows, cols) {
		t.Fatal("overflowing q8 shape should not be accepted")
	}
	if q6MatVecShapeOK(&tensor.Q6Matrix{Rows: rows, Cols: maxModelInt()}, rows, maxModelInt()) {
		t.Fatal("overflowing q6 packed shape should not be accepted")
	}
	if q4MatVecShapeOK(&tensor.Q4Matrix{Rows: rows, Cols: maxModelInt()}, rows, maxModelInt()) {
		t.Fatal("overflowing q4 packed shape should not be accepted")
	}
	if f32MatVecWeightsReady(nil, rows, cols) {
		t.Fatal("overflowing f32 weights should not be accepted")
	}
	if q8MatVecMinRowsShapeOK(&tensor.Q8Matrix{Rows: rows, Cols: cols}, rows, cols) {
		t.Fatal("overflowing min-rows q8 shape should not be accepted")
	}
	if q6MatVecMinRowsShapeOK(&tensor.Q6Matrix{Rows: rows, Cols: maxModelInt()}, rows, maxModelInt()) {
		t.Fatal("overflowing min-rows q6 shape should not be accepted")
	}
	if q4MatVecMinRowsShapeOK(&tensor.Q4Matrix{Rows: rows, Cols: maxModelInt()}, rows, maxModelInt()) {
		t.Fatal("overflowing min-rows q4 shape should not be accepted")
	}
}

func TestVulkanTextAttentionReadinessRejectsOverflowWork(t *testing.T) {
	t.Setenv(vulkanTextAttentionMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	huge := maxModelInt()/2 + 1
	cache := &kvCache{len: huge, kvDim: 2}
	if _, _, _, ok := checkedTextAttentionDims(cache, huge, 1, 2); ok {
		t.Fatal("overflowing text attention dimensions should not be accepted")
	}
	if textAttentionOnlyWorkReady(huge, 2, 2) {
		t.Fatal("overflowing attention-only work should not be ready")
	}
	if textAttentionOutWorkReady(1, huge, 2, huge, false) {
		t.Fatal("overflowing attention+out work should not be ready")
	}
	if textAttentionOutWorkReady(1, huge, 2, huge, true) {
		t.Fatal("overflowing attention+out+norm work should not be ready")
	}
	if textFirstTokenValueOutNormWorkReady(huge, huge) {
		t.Fatal("overflowing first-token value+out+norm work should not be ready")
	}
}

func TestVulkanVectorReadinessRejectsOverflowWork(t *testing.T) {
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rows := maxModelInt()/2 + 1
	cols := 3
	if vulkanVectorWorkReady(rows, cols) {
		t.Fatal("overflowing vector work should not be ready")
	}
	if groupedVulkanVectorWorkReady(cols, rows, rows) {
		t.Fatal("overflowing grouped vector work should not be ready")
	}
	if mropeShapeOK(nil, rows, cols, []float32{1}, []float32{0}) {
		t.Fatal("overflowing mrope shape should not be accepted")
	}
}

func TestF32SwiGLUShapeReadinessUsesCheckedWeights(t *testing.T) {
	if !f32SwiGLUDownShapeOK(make([]float32, 2), make([]float32, 2), make([]float32, 8), make([]float32, 8), make([]float32, 8), 4, 2, 2) {
		t.Fatal("fused swiglu down should accept output sized by outRows, not intermediate rows")
	}
	rows := maxModelInt()/2 + 1
	cols := 3
	if f32SwiGLUGateUpShapeOK(nil, make([]float32, cols), nil, nil, rows, cols) {
		t.Fatal("overflowing swiglu gate/up weights should not be accepted")
	}
	if f32SwiGLUDownShapeOK(make([]float32, 1), make([]float32, cols), nil, nil, nil, rows, cols, 1) {
		t.Fatal("overflowing swiglu down weights should not be accepted")
	}
}

func TestSwiGLUSplitFallbackUsesCombinedWorkThreshold(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "100")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rows, cols, outRows := 1, 40, 20
	if rows*cols*2 >= vulkanMatVecMinWork() {
		t.Fatal("test shape should keep gate+up work below the threshold")
	}
	if !swiGLUSplitVulkanWorkReady(rows, cols, outRows) {
		t.Fatal("swiglu split fallback should be ready when full gate+up+down work reaches the threshold")
	}
	if swiGLUSplitVulkanWorkReady(rows, cols, 0) {
		t.Fatal("swiglu split fallback with invalid output rows should not be ready")
	}
}

func TestExpandValueHeads(t *testing.T) {
	t.Run("same_head_count", func(t *testing.T) {
		v := []float32{1, 2, 3, 4, 5, 6}
		dst := make([]float32, len(v))
		expandValueHeads(dst, v, 3, 3, 2)
		for i, want := range v {
			if dst[i] != want {
				t.Fatalf("dst[%d]=%g want %g dst=%v", i, dst[i], want, dst)
			}
		}
	})
	t.Run("grouped_value_heads", func(t *testing.T) {
		v := []float32{1, 2, 3, 4}
		dst := make([]float32, 8)
		expandValueHeads(dst, v, 4, 2, 2)
		want := []float32{1, 2, 1, 2, 3, 4, 3, 4}
		for i := range want {
			if dst[i] != want[i] {
				t.Fatalf("dst[%d]=%g want %g dst=%v", i, dst[i], want[i], dst)
			}
		}
	})
}

func TestMatVecAddRMSNormOutOnlySkipsMatVecWhenOutOnlyOpDisabled(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32)
	q8 := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	tmp := []float32{99, 88}
	if rt.matVecAddRMSNormOutOnlyMaybeVulkan(make([]float32, 2), []float32{1, 2}, []float32{3, 4}, nil, q8, nil, nil, []float32{1, 1}, 2, 2, 1e-6, tmp) {
		t.Fatal("disabled out-only add+rmsnorm unexpectedly used Vulkan")
	}
	if tmp[0] != 99 || tmp[1] != 88 {
		t.Fatalf("matvec ran before out-only readiness check: tmp=%v", tmp)
	}
	if !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("disabled out-only op should not trigger or disable q8 matvec")
	}
}

func TestMatVecAddRMSNormReadinessTracksDownstreamOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	q8 := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
	}, 2, 2)
	rt := &Runtime{backend: "vulkan"}
	if !rt.matVecAddRMSNormVulkanReady(make([]float32, 2), []float32{1, 2}, []float32{3, 4}, nil, q8, nil, nil, []float32{1, 1}, 2, 2, 1e-6) {
		t.Fatal("q8 matvec+add+rmsnorm should be ready with valid shapes and enabled op")
	}
	rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ8)
	if rt.matVecAddRMSNormVulkanReady(make([]float32, 2), []float32{1, 2}, []float32{3, 4}, nil, q8, nil, nil, []float32{1, 1}, 2, 2, 1e-6) {
		t.Fatal("q8 matvec+add+rmsnorm should not be ready when downstream op is disabled")
	}

	rt = &Runtime{backend: "vulkan"}
	if !rt.matVecAddRMSNormOutOnlyVulkanReady(make([]float32, 2), []float32{1, 2}, []float32{3, 4}, nil, q8, nil, nil, []float32{1, 1}, 2, 2, 1e-6, make([]float32, 2)) {
		t.Fatal("q8 out-only split should be ready with valid shapes and enabled ops")
	}
	rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32)
	if rt.matVecAddRMSNormOutOnlyVulkanReady(make([]float32, 2), []float32{1, 2}, []float32{3, 4}, nil, q8, nil, nil, []float32{1, 1}, 2, 2, 1e-6, make([]float32, 2)) {
		t.Fatal("out-only split should not be ready when add+rmsnorm out-only op is disabled")
	}
}

func TestMLPSplitFallbackSkipsGateUpWhenDownstreamNormUnavailable(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       2,
			IntermediateSize: 2,
			RMSNormEps:       1e-6,
		},
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownAddRMSNormQ8)
	rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ8)
	weights := []float32{
		1, 0,
		0, 1,
	}
	tl := &textLayer{q8: qLayerWeights{
		gate: tensor.QuantizeQ8Row(weights, 2, 2),
		up:   tensor.QuantizeQ8Row(weights, 2, 2),
		down: tensor.QuantizeQ8Row(weights, 2, 2),
	}}
	sc := &layerScratch{
		gate: []float32{99, 88},
		mlp:  make([]float32, 2),
	}
	if rt.mlpAddRMSNormMaybeVulkan([]float32{1, 2}, tl, sc, []float32{3, 4}, []float32{1, 1}, make([]float32, 2), 1e-6, true) {
		t.Fatal("split MLP+norm unexpectedly used Vulkan with downstream norm op disabled")
	}
	if sc.gate[0] != 99 || sc.gate[1] != 88 {
		t.Fatalf("gate-up ran before downstream readiness check: gate=%v", sc.gate)
	}
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ8) {
		t.Fatal("downstream norm unavailability should not trigger or disable gate-up")
	}
}

func TestF32MatVecVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	x := []float32{1, 2}
	w := []float32{
		1, 0,
		0, 1,
		1, 2,
	}
	out := make([]float32, 2)
	if rt.matVecOnlyMaybeVulkan(out, x, w, nil, nil, nil, 3, 2) {
		t.Fatal("short output unexpectedly reached Vulkan f32 matvec path")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("expected CPU fallback to reject short output")
		}
		if !rt.vulkanOpEnabled(vulkanOpMatVecF32) {
			t.Fatal("f32 matvec op was disabled by invalid shape")
		}
	}()
	rt.matVecMaybeQuant(out, x, w, nil, nil, nil, 3, 2)
}

func TestVectorVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	expectPanic := func(t *testing.T, fn func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatal("expected CPU fallback to reject invalid shape")
			}
		}()
		fn()
	}
	cases := []struct {
		name string
		op   vulkanOp
		call func(*Runtime)
	}{
		{name: "rmsnorm", op: vulkanOpRMSNormF32, call: func(rt *Runtime) {
			rt.rmsNormMaybeVulkan(make([]float32, 2), []float32{1, 2}, []float32{1}, 1e-6)
		}},
		{name: "add_rmsnorm", op: vulkanOpAddRMSNormF32, call: func(rt *Runtime) {
			rt.addRMSNormMaybeVulkan(make([]float32, 2), []float32{1, 2}, []float32{3}, []float32{1, 1}, 1e-6)
		}},
		{name: "add_rmsnorm_out_only", op: vulkanOpAddRMSNormOutOnlyF32, call: func(rt *Runtime) {
			rt.addRMSNormOutOnlyMaybeVulkan(make([]float32, 2), []float32{1, 2}, []float32{3}, []float32{1, 1}, 1e-6)
		}},
		{name: "mrope", op: vulkanOpMRoPEF32, call: func(rt *Runtime) {
			rt.mropeMaybeVulkan([]float32{1, 2, 3, 4}, 1, 4, []float32{1}, []float32{0})
		}},
		{name: "mrope_pair", op: vulkanOpMRoPEPairF32, call: func(rt *Runtime) {
			rt.mropePairMaybeVulkan([]float32{1, 2, 3, 4}, []float32{1, 2}, 1, 1, 4, []float32{1, 1}, []float32{0, 0})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			expectPanic(t, func() { tc.call(rt) })
			if !rt.vulkanOpEnabled(tc.op) {
				t.Fatalf("%s op was disabled by invalid shape", tc.name)
			}
		})
	}
}

func TestVulkanMatVecArgmaxMaybeVulkan(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan matvec argmax smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	w := []float32{
		1, 0,
		0, 1,
		1, 2,
	}
	wantScores := make([]float32, 3)
	tensor.MatVec(wantScores, x, w, 3, 2)
	wantToken := tensor.Argmax(wantScores)
	wantScore := wantScores[wantToken]
	cases := []struct {
		name string
		op   vulkanOp
		w    []float32
		q8   *tensor.Q8Matrix
		q6   *tensor.Q6Matrix
		q4   *tensor.Q4Matrix
		tol  float32
	}{
		{name: "f32", op: vulkanOpMatVecArgmaxF32, w: w, tol: 1e-5},
		{name: "q8", op: vulkanOpMatVecArgmaxQ8, q8: tensor.QuantizeQ8Row(w, 3, 2), tol: 0.05},
		{name: "q6", op: vulkanOpMatVecArgmaxQ6, q6: tensor.QuantizeQ6Row(w, 3, 2), tol: 0.12},
		{name: "q4", op: vulkanOpMatVecArgmaxQ4, q4: tensor.QuantizeQ4Row(w, 3, 2), tol: 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			token, score, ok := rt.matVecArgmaxMaybeVulkan(x, tc.w, tc.q8, tc.q6, tc.q4, 3, 2)
			if !ok {
				t.Fatal("expected Vulkan matvec argmax path")
			}
			if !rt.vulkanOpEnabled(tc.op) {
				t.Fatalf("Vulkan %s argmax op was disabled", tc.name)
			}
			if token != wantToken {
				t.Fatalf("token=%d want %d", token, wantToken)
			}
			if diff := float32(math.Abs(float64(score - wantScore))); diff > tc.tol {
				t.Fatalf("score=%g want %g diff %g > %g", score, wantScore, diff, tc.tol)
			}
		})
	}
}

func TestMatVecMaybeQuantUsesCPUFallbackWhenVulkanOpDisabled(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpMatVecQ8)
	w := []float32{
		1, -2, 0.5,
		0.25, 3, -1,
	}
	q := tensor.QuantizeQ8Row(w, 2, 3)
	x := []float32{2, -1, 4}
	got := make([]float32, 2)
	want := make([]float32, 2)
	rt.matVecMaybeQuant(got, x, nil, q, nil, nil, 2, 3)
	tensor.MatVecQ8(want, x, q)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestFusedQKVUsesCPUFallbackWhenVulkanOpDisabled(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpFusedQKVQ8)
	x := []float32{2, -1, 4}
	qw := tensor.QuantizeQ8Row([]float32{
		1, -2, 0.5,
		0.25, 3, -1,
	}, 2, 3)
	kw := tensor.QuantizeQ8Row([]float32{
		-1, 0.5, 2,
	}, 1, 3)
	vw := tensor.QuantizeQ8Row([]float32{
		0.75, -1.5, 1,
	}, 1, 3)
	tl := &textLayer{q8: qLayerWeights{q: qw, k: kw, v: vw}}
	q := make([]float32, 2)
	k := make([]float32, 1)
	v := make([]float32, 1)
	if !rt.fusedQKV(q, k, v, x, tl, 2, 1, 3) {
		t.Fatal("expected fused qkv CPU fallback")
	}
	wantQ := make([]float32, 2)
	wantK := make([]float32, 1)
	wantV := make([]float32, 1)
	tensor.FusedMatVec3Q8(wantQ, wantK, wantV, x, qw, kw, vw)
	if q[0] != wantQ[0] || q[1] != wantQ[1] || k[0] != wantK[0] || v[0] != wantV[0] {
		t.Fatalf("qkv=(%v,%v,%v) want (%v,%v,%v)", q, k, v, wantQ, wantK, wantV)
	}
}

func TestVulkanFusedQKVFallsBackThroughMatVecTriple(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan Q/K/V matvec triple fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{2, -1, 4}
	qw := []float32{
		1, -2, 0.5,
		0.25, 3, -1,
	}
	kw := []float32{
		-1, 0.5, 2,
		0.25, 1.5, -0.75,
	}
	vw := []float32{
		0.75, -1.5, 1,
		-2, 0.25, 0.5,
	}
	wantQ := make([]float32, 2)
	wantK := make([]float32, 2)
	wantV := make([]float32, 2)
	tensor.FusedMatVec3(wantQ, wantK, wantV, x, qw, kw, vw, 2, 2, 2, 3)

	cases := []struct {
		name  string
		op    vulkanOp
		matOp vulkanOp
		tl    textLayer
		want  func(q, k, v []float32)
		tol   float32
	}{
		{name: "f32", op: vulkanOpFusedQKVF32, matOp: vulkanOpMatVecF32, tl: textLayer{w: layerWeights{q: qw, k: kw, v: vw}}, want: func(q, k, v []float32) {
			tensor.FusedMatVec3(q, k, v, x, qw, kw, vw, 2, 2, 2, 3)
		}, tol: 1e-5},
		{name: "q8", op: vulkanOpFusedQKVQ8, matOp: vulkanOpMatVecQ8, tl: textLayer{q8: qLayerWeights{
			q: tensor.QuantizeQ8Row(qw, 2, 3),
			k: tensor.QuantizeQ8Row(kw, 2, 3),
			v: tensor.QuantizeQ8Row(vw, 2, 3),
		}}, want: func(q, k, v []float32) {
			tensor.FusedMatVec3Q8(q, k, v, x, tensor.QuantizeQ8Row(qw, 2, 3), tensor.QuantizeQ8Row(kw, 2, 3), tensor.QuantizeQ8Row(vw, 2, 3))
		}, tol: 0.05},
		{name: "q6", op: vulkanOpFusedQKVQ6, matOp: vulkanOpMatVecQ6, tl: textLayer{q6: q6LayerWeights{
			q: tensor.QuantizeQ6Row(qw, 2, 3),
			k: tensor.QuantizeQ6Row(kw, 2, 3),
			v: tensor.QuantizeQ6Row(vw, 2, 3),
		}}, want: func(q, k, v []float32) {
			tensor.FusedMatVec3Q6(q, k, v, x, tensor.QuantizeQ6Row(qw, 2, 3), tensor.QuantizeQ6Row(kw, 2, 3), tensor.QuantizeQ6Row(vw, 2, 3))
		}, tol: 0.12},
		{name: "q4", op: vulkanOpFusedQKVQ4, matOp: vulkanOpMatVecQ4, tl: textLayer{q4: q4LayerWeights{
			q: tensor.QuantizeQ4Row(qw, 2, 3),
			k: tensor.QuantizeQ4Row(kw, 2, 3),
			v: tensor.QuantizeQ4Row(vw, 2, 3),
		}}, want: func(q, k, v []float32) {
			tensor.FusedMatVec3Q4(q, k, v, x, tensor.QuantizeQ4Row(qw, 2, 3), tensor.QuantizeQ4Row(kw, 2, 3), tensor.QuantizeQ4Row(vw, 2, 3))
		}, tol: 0.5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			rt.disableVulkanOp(tc.op)
			q := make([]float32, 2)
			k := make([]float32, 2)
			v := make([]float32, 2)
			if !rt.fusedQKV(q, k, v, x, &tc.tl, 2, 2, 3) {
				t.Fatal("expected fused qkv to fall back through Vulkan matvec triple")
			}
			if !rt.vulkanOpEnabled(tc.matOp) {
				t.Fatalf("Vulkan %s matvec fallback op was disabled", tc.name)
			}
			wantQ := make([]float32, 2)
			wantK := make([]float32, 2)
			wantV := make([]float32, 2)
			tc.want(wantQ, wantK, wantV)
			assertCloseSliceTol(t, "q", q, wantQ, tc.tol)
			assertCloseSliceTol(t, "k", k, wantK, tc.tol)
			assertCloseSliceTol(t, "v", v, wantV, tc.tol)
		})
	}
}

func TestFusedKVUsesCPUFallbackWhenVulkanOpDisabled(t *testing.T) {
	x := []float32{2, -1, 4}
	kw := []float32{
		-1, 0.5, 2,
		0.25, 1.5, -0.75,
	}
	vw := []float32{
		0.75, -1.5, 1,
		-2, 0.25, 0.5,
	}
	wantK := make([]float32, 2)
	wantV := make([]float32, 2)
	tensor.FusedMatVec2(wantK, wantV, x, kw, vw, 2, 2, 3)

	cases := []struct {
		name string
		op   vulkanOp
		tl   textLayer
		tol  float32
	}{
		{name: "f32", op: vulkanOpFusedKVF32, tl: textLayer{w: layerWeights{k: kw, v: vw}}, tol: 1e-6},
		{name: "q8", op: vulkanOpFusedKVQ8, tl: textLayer{q8: qLayerWeights{
			q: tensor.QuantizeQ8Row(make([]float32, 6), 2, 3),
			k: tensor.QuantizeQ8Row(kw, 2, 3),
			v: tensor.QuantizeQ8Row(vw, 2, 3),
		}}, tol: 0.05},
		{name: "q6", op: vulkanOpFusedKVQ6, tl: textLayer{q6: q6LayerWeights{
			q: tensor.QuantizeQ6Row(make([]float32, 6), 2, 3),
			k: tensor.QuantizeQ6Row(kw, 2, 3),
			v: tensor.QuantizeQ6Row(vw, 2, 3),
		}}, tol: 0.12},
		{name: "q4", op: vulkanOpFusedKVQ4, tl: textLayer{q4: q4LayerWeights{
			q: tensor.QuantizeQ4Row(make([]float32, 6), 2, 3),
			k: tensor.QuantizeQ4Row(kw, 2, 3),
			v: tensor.QuantizeQ4Row(vw, 2, 3),
		}}, tol: 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			rt.disableVulkanOp(tc.op)
			k := make([]float32, 2)
			v := make([]float32, 2)
			ready, hasRoPE := rt.fusedKV(k, v, x, &tc.tl, 2, 3, 1, 2, false, nil, nil)
			if !ready || hasRoPE {
				t.Fatalf("ready=%v hasRoPE=%v want ready without rope", ready, hasRoPE)
			}
			assertCloseSliceTol(t, tc.name+"K", k, wantK, tc.tol)
			assertCloseSliceTol(t, tc.name+"V", v, wantV, tc.tol)
		})
	}
}

func TestVulkanFusedKVFallsBackThroughMatVecPair(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan K/V matvec pair fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{2, -1, 4}
	kw := []float32{
		-1, 0.5, 2,
		0.25, 1.5, -0.75,
	}
	vw := []float32{
		0.75, -1.5, 1,
		-2, 0.25, 0.5,
	}
	wantK := make([]float32, 2)
	wantV := make([]float32, 2)
	tensor.FusedMatVec2(wantK, wantV, x, kw, vw, 2, 2, 3)

	cases := []struct {
		name  string
		op    vulkanOp
		matOp vulkanOp
		tl    textLayer
		tol   float32
	}{
		{name: "f32", op: vulkanOpFusedKVF32, matOp: vulkanOpMatVecF32, tl: textLayer{w: layerWeights{k: kw, v: vw}}, tol: 1e-5},
		{name: "q8", op: vulkanOpFusedKVQ8, matOp: vulkanOpMatVecQ8, tl: textLayer{q8: qLayerWeights{
			k: tensor.QuantizeQ8Row(kw, 2, 3),
			v: tensor.QuantizeQ8Row(vw, 2, 3),
		}}, tol: 0.05},
		{name: "q6", op: vulkanOpFusedKVQ6, matOp: vulkanOpMatVecQ6, tl: textLayer{q6: q6LayerWeights{
			k: tensor.QuantizeQ6Row(kw, 2, 3),
			v: tensor.QuantizeQ6Row(vw, 2, 3),
		}}, tol: 0.12},
		{name: "q4", op: vulkanOpFusedKVQ4, matOp: vulkanOpMatVecQ4, tl: textLayer{q4: q4LayerWeights{
			k: tensor.QuantizeQ4Row(kw, 2, 3),
			v: tensor.QuantizeQ4Row(vw, 2, 3),
		}}, tol: 0.5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			rt.disableVulkanOp(tc.op)
			k := make([]float32, 2)
			v := make([]float32, 2)
			ready, hasRoPE := rt.fusedKV(k, v, x, &tc.tl, 2, 3, 1, 2, false, nil, nil)
			if !ready || hasRoPE {
				t.Fatalf("ready=%v hasRoPE=%v want ready without rope", ready, hasRoPE)
			}
			if !rt.vulkanOpEnabled(tc.matOp) {
				t.Fatalf("Vulkan %s matvec fallback op was disabled", tc.name)
			}
			assertCloseSliceTol(t, "k", k, wantK, tc.tol)
			assertCloseSliceTol(t, "v", v, wantV, tc.tol)
		})
	}
}

func TestFusedQKVMRoPEReturnsFalseWhenMergedVulkanOpDisabled(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpFusedQKVMRoPEF32)
	x := []float32{2, -1, 4}
	tl := &textLayer{w: layerWeights{
		q: []float32{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
			1, 1, 1,
		},
		k: []float32{
			1, -1, 0,
			0, 1, -1,
		},
		v: []float32{
			0.5, 0.25, -1,
			-1, 0.5, 0.75,
		},
	}}
	q := []float32{9, 9, 9, 9}
	k := []float32{8, 8}
	v := []float32{7, 7}
	if rt.fusedQKVMRoPE(q, k, v, x, tl, 4, 2, 3, 1, 1, 4, []float32{1, 0.5}, []float32{0, 0.5}) {
		t.Fatal("disabled merged qkv+mrope should let fusedQKV and mropePair fallback paths run")
	}
	if q[0] != 9 || k[0] != 8 || v[0] != 7 {
		t.Fatalf("disabled merged qkv+mrope mutated outputs q=%v k=%v v=%v", q, k, v)
	}
}

func TestFusedQKVMRoPESkipsQuantizedQKV(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	x := []float32{2, -1, 4}
	qw := tensor.QuantizeQ8Row([]float32{
		1, -2, 0.5,
		0.25, 3, -1,
	}, 2, 3)
	tl := &textLayer{
		w: layerWeights{
			q: make([]float32, 6),
			k: make([]float32, 3),
			v: make([]float32, 3),
		},
		q8: qLayerWeights{q: qw},
	}
	q := []float32{9, 9}
	k := []float32{8}
	v := []float32{7}
	if rt.fusedQKVMRoPE(q, k, v, x, tl, 2, 1, 3, 1, 1, 2, []float32{1}, []float32{0}) {
		t.Fatal("merged f32 qkv+mrope should not preempt quantized qkv paths")
	}
	if q[0] != 9 || k[0] != 8 || v[0] != 7 {
		t.Fatalf("quantized qkv skip mutated outputs q=%v k=%v v=%v", q, k, v)
	}
}

func TestAttentionCacheLenOneUsesValueProjection(t *testing.T) {
	rt := &Runtime{
		cfg: &config.Config{
			HiddenSize:        4,
			NumAttentionHeads: 2,
			NumKeyValueHeads:  1,
			HeadDim:           2,
			IntermediateSize:  4,
			NumHiddenLayers:   1,
			VocabSize:         8,
			MaxPositionEmb:    8,
			RopeTheta:         10000,
		},
	}
	x := []float32{2, -1, 0.5, 3}
	tl := &textLayer{w: layerWeights{
		k: make([]float32, 8),
		v: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}}
	sc := &layerScratch{
		q:       make([]float32, 4),
		k:       make([]float32, 2),
		v:       make([]float32, 2),
		headOut: make([]float32, 4),
		att:     make([]float32, 4),
	}
	cache := &kvCache{}
	got := rt.attention(x, cache, tl, sc, false, nil, nil)
	want := []float32{2, -1, 2, -1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if cache.len != 1 {
		t.Fatalf("cache len=%d want 1", cache.len)
	}
}

func TestAttentionFirstTokenSkipsQProjectionAndRotatesKCache(t *testing.T) {
	rt := &Runtime{
		cfg: &config.Config{
			HiddenSize:        4,
			NumAttentionHeads: 1,
			NumKeyValueHeads:  1,
			HeadDim:           4,
		},
	}
	x := []float32{1, 2, 3, 4}
	tl := &textLayer{w: layerWeights{
		k: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		v: []float32{
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			1, 0, 0, 0,
		},
		o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}}
	sc := &layerScratch{
		q:       make([]float32, 4),
		k:       make([]float32, 4),
		v:       make([]float32, 4),
		headOut: make([]float32, 4),
		att:     make([]float32, 4),
	}
	cache := &kvCache{}
	cosTable := []float32{0, 0.5}
	sinTable := []float32{1, 0.5}
	got := rt.attention(x, cache, tl, sc, true, cosTable, sinTable)
	wantOut := []float32{2, 3, 4, 1}
	for i := range wantOut {
		if got[i] != wantOut[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], wantOut[i])
		}
	}
	wantK := []float32{-3, -1, 1, 3}
	for i := range wantK {
		if diff := cache.k[i] - wantK[i]; diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("cache.k[%d]=%g want %g", i, cache.k[i], wantK[i])
		}
	}
}

func TestAttentionFirstTokenVulkanFallbackSkipsDummyQProjection(t *testing.T) {
	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:        4,
			NumAttentionHeads: 2,
			NumKeyValueHeads:  1,
			HeadDim:           2,
		},
	}
	rt.disableVulkanOp(vulkanOpFusedQKVF32)
	rt.disableVulkanOp(vulkanOpMatVecF32)
	x := []float32{1, 2, 3, 4}
	tl := &textLayer{w: layerWeights{
		q: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			9, 0, 0, 0,
			0, 9, 0, 0,
		},
		k: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		v: []float32{
			0, 1, 0, 0,
			0, 0, 1, 0,
		},
		o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}}
	sc := &layerScratch{
		q:       make([]float32, 4),
		k:       make([]float32, 4),
		v:       make([]float32, 4),
		headOut: make([]float32, 4),
		att:     make([]float32, 4),
	}
	cache := &kvCache{}
	got := rt.attention(x, cache, tl, sc, false, nil, nil)
	want := []float32{2, 3, 2, 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if sc.q[0] != 0 || sc.q[1] != 0 {
		t.Fatalf("first q head=%v want fallback to leave dummy q zero", sc.q[:2])
	}
	if sc.q[2] != 0 || sc.q[3] != 0 {
		t.Fatalf("second q head=%v want fallback to leave it zero", sc.q[2:4])
	}
}

func TestAttentionFirstTokenVulkanFallbackSkipsQuantizedDummyQProjection(t *testing.T) {
	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:        4,
			NumAttentionHeads: 2,
			NumKeyValueHeads:  1,
			HeadDim:           2,
		},
	}
	rt.disableVulkanOp(vulkanOpFusedQKVQ8)
	rt.disableVulkanOp(vulkanOpMatVecQ8)
	x := []float32{2, -1, 3, 0.5}
	qw := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		9, 0, 0, 0,
		0, 9, 0, 0,
	}, 4, 4)
	kw := tensor.QuantizeQ8Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, 2, 4)
	vw := tensor.QuantizeQ8Row([]float32{
		0, 1, 0, 0,
		1, 0, 0, 0,
	}, 2, 4)
	tl := &textLayer{
		q8: qLayerWeights{q: qw, k: kw, v: vw},
		w: layerWeights{o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		}},
	}
	sc := &layerScratch{
		q:       make([]float32, 4),
		k:       make([]float32, 2),
		v:       make([]float32, 2),
		headOut: make([]float32, 4),
		att:     make([]float32, 4),
	}
	cache := &kvCache{}
	got := rt.attention(x, cache, tl, sc, false, nil, nil)
	want := []float32{-1, 2, -1, 2}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if sc.q[0] != 0 || sc.q[1] != 0 {
		t.Fatalf("first q8 q head=%v want fallback to leave dummy q zero", sc.q[:2])
	}
	if sc.q[2] != 0 || sc.q[3] != 0 {
		t.Fatalf("second q8 q head=%v want fallback to leave it zero", sc.q[2:4])
	}
}

func TestVulkanTextFirstTokenAttentionOutUsesCacheValue(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan first-token attention out smoke test")
	}
	t.Setenv(vulkanTextAttentionMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	const (
		numHeads = 2
		kvHeads  = 1
		headDim  = 2
		qRows    = numHeads * headDim
	)
	cache := &kvCache{kvDim: kvHeads * headDim, len: 1, epoch: 1}
	cache.k = []float32{0.25, -0.5}
	cache.v = []float32{2, -1}
	w := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0, -2,
	}
	want := []float32{2, -1, 1, 2}
	cases := []struct {
		name string
		tl   textLayer
	}{
		{name: "f32", tl: textLayer{w: layerWeights{o: w}}},
		{name: "q8", tl: textLayer{q8: qLayerWeights{o: tensor.QuantizeQ8Row(w, qRows, qRows)}}},
		{name: "q6", tl: textLayer{q6: q6LayerWeights{o: tensor.QuantizeQ6Row(w, qRows, qRows)}}},
		{name: "q4", tl: textLayer{q4: q4LayerWeights{o: tensor.QuantizeQ4Row(w, qRows, qRows)}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			out := make([]float32, qRows)
			q := []float32{
				float32(math.NaN()), float32(math.NaN()),
				float32(math.NaN()), float32(math.NaN()),
			}
			if !rt.vulkanTextFirstTokenAttentionOut(out, cache, &tc.tl, numHeads, kvHeads, headDim) {
				t.Fatal("first-token attention out did not use Vulkan")
			}
			assertCloseSliceTol(t, "out", out, want, 1e-4)
			for i, v := range q {
				if !math.IsNaN(float64(v)) {
					t.Fatalf("dummyQ[%d]=%g want untouched NaN", i, v)
				}
			}
		})
	}
}

func TestVulkanFirstTokenAttentionWithNormFallsBackThroughAttentionOut(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan first-token attention norm fallback smoke test")
	}
	t.Setenv(vulkanTextAttentionMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	k := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}
	v := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}
	o := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 0.5, 0,
		0, 0, 0, -2,
	}
	cases := []struct {
		name     string
		fusedOp  vulkanOp
		matOp    vulkanOp
		tl       textLayer
		normTol  float32
		residTol float32
	}{
		{name: "f32", fusedOp: vulkanOpTextFirstTokenValueOutAddRMSNormF32, matOp: vulkanOpMatVecAddRMSNormF32, tl: textLayer{w: layerWeights{k: k, v: v, o: o}}, normTol: 1e-4, residTol: 1e-4},
		{name: "q8", fusedOp: vulkanOpTextFirstTokenValueOutAddRMSNormQ8, matOp: vulkanOpMatVecAddRMSNormQ8, tl: textLayer{
			w:  layerWeights{k: k, v: v},
			q8: qLayerWeights{o: tensor.QuantizeQ8Row(o, 4, 4)},
		}, normTol: 1e-4, residTol: 1e-4},
		{name: "q6", fusedOp: vulkanOpTextFirstTokenValueOutAddRMSNormQ6, matOp: vulkanOpMatVecAddRMSNormQ6, tl: textLayer{
			w:  layerWeights{k: k, v: v},
			q6: q6LayerWeights{o: tensor.QuantizeQ6Row(o, 4, 4)},
		}, normTol: 0.02, residTol: 0.02},
		{name: "q4", fusedOp: vulkanOpTextFirstTokenValueOutAddRMSNormQ4, matOp: vulkanOpMatVecAddRMSNormQ4, tl: textLayer{
			w:  layerWeights{k: k, v: v},
			q4: q4LayerWeights{o: tensor.QuantizeQ4Row(o, 4, 4)},
		}, normTol: 0.08, residTol: 0.08},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{
				backend: "vulkan",
				cfg: &config.Config{
					HiddenSize:        4,
					NumAttentionHeads: 2,
					NumKeyValueHeads:  1,
					HeadDim:           2,
					RMSNormEps:        1e-6,
				},
			}
			rt.disableVulkanOp(tc.fusedOp)
			rt.disableVulkanOp(tc.matOp)

			sc := &layerScratch{
				q:       make([]float32, 4),
				k:       make([]float32, 2),
				v:       make([]float32, 2),
				headOut: make([]float32, 4),
				att:     make([]float32, 4),
			}
			x := []float32{2, -1, 0.5, 3}
			residual := []float32{1, 2, 3, 4}
			normWeight := []float32{1, 1.1, 0.9, 1.2}
			normOut := make([]float32, 4)
			wantResidual := append([]float32(nil), residual...)
			wantAtt := []float32{2, -1, 1, 2}
			wantNorm := make([]float32, 4)
			tensor.AddRMSNorm(wantNorm, wantResidual, wantAtt, normWeight, 1e-6)

			out, normDone := rt.attentionWithNorm(x, &kvCache{}, &tc.tl, sc, false, nil, nil, residual, normWeight, normOut)
			if !normDone {
				t.Fatalf("normDone=false out=%v, want attention-out fallback to finish add+rmsnorm", out)
			}
			if out != nil {
				t.Fatalf("out=%v want nil when norm is completed", out)
			}
			assertCloseSliceTol(t, "normOut", normOut, wantNorm, tc.normTol)
			assertCloseSliceTol(t, "residual", residual, wantResidual, tc.residTol)
		})
	}
}

func TestTextFirstTokenValueOutAddRMSNormFallbackWhenDisabled(t *testing.T) {
	w := make([]float32, 16)
	cases := []struct {
		name string
		op   vulkanOp
		tl   textLayer
	}{
		{name: "f32", op: vulkanOpTextFirstTokenValueOutAddRMSNormF32, tl: textLayer{w: layerWeights{o: w}}},
		{name: "q8", op: vulkanOpTextFirstTokenValueOutAddRMSNormQ8, tl: textLayer{q8: qLayerWeights{o: tensor.QuantizeQ8Row(w, 4, 4)}}},
		{name: "q6", op: vulkanOpTextFirstTokenValueOutAddRMSNormQ6, tl: textLayer{q6: q6LayerWeights{o: tensor.QuantizeQ6Row(w, 4, 4)}}},
		{name: "q4", op: vulkanOpTextFirstTokenValueOutAddRMSNormQ4, tl: textLayer{q4: q4LayerWeights{o: tensor.QuantizeQ4Row(w, 4, 4)}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{backend: "vulkan"}
			rt.disableVulkanOp(tc.op)
			normOut := make([]float32, 4)
			residual := []float32{1, 2, 3, 4}
			cache := &kvCache{
				k:     []float32{0.75, 0.125},
				v:     []float32{0.25, -0.5},
				len:   1,
				kvDim: 2,
				epoch: 3,
			}
			normWeight := []float32{1, 1, 1, 1}
			if rt.vulkanTextFirstTokenValueOutAddRMSNorm(normOut, residual, cache, &tc.tl, normWeight, 2, 1, 2) {
				t.Fatal("disabled first-token value+out+add+rmsnorm op should fall back")
			}
			assertCloseSliceTol(t, "firstTokenFallbackNormOut", normOut, []float32{0, 0, 0, 0}, 0)
			assertCloseSliceTol(t, "firstTokenFallbackResidual", residual, []float32{1, 2, 3, 4}, 0)
		})
	}
}

func TestFusedFirstTokenKVQ6FallbackSkipsDummyQProjection(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpFusedQKVQ6)
	x := []float32{2, -1, 3, 0.5}
	qw := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		9, 0, 0, 0,
		0, 9, 0, 0,
	}, 4, 4)
	kw := tensor.QuantizeQ6Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, 2, 4)
	vw := tensor.QuantizeQ6Row([]float32{
		0, 1, 0, 0,
		1, 0, 0, 0,
	}, 2, 4)
	tl := &textLayer{q6: q6LayerWeights{q: qw, k: kw, v: vw}}
	q := make([]float32, 4)
	k := make([]float32, 2)
	v := make([]float32, 2)
	ready, hasRoPE := rt.fusedFirstTokenKV(q, k, v, x, tl, 4, 2, 4, 1, 2, false, nil, nil)
	if !ready || hasRoPE {
		t.Fatalf("ready=%v hasRoPE=%v want ready without rope", ready, hasRoPE)
	}
	assertCloseSliceTol(t, "q6FirstQHeadZero", q[:2], []float32{0, 0}, 0)
	assertCloseSliceTol(t, "q6SecondQHeadZero", q[2:], []float32{0, 0}, 0)
	assertCloseSliceTol(t, "q6K", k, []float32{2, -1}, 0.08)
	assertCloseSliceTol(t, "q6V", v, []float32{-1, 2}, 0.08)
}

func TestFusedFirstTokenKVQ4FallbackSkipsDummyQProjection(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpFusedQKVQ4)
	x := []float32{2, -1, 3, 0.5}
	qw := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		9, 0, 0, 0,
		0, 9, 0, 0,
	}, 4, 4)
	kw := tensor.QuantizeQ4Row([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
	}, 2, 4)
	vw := tensor.QuantizeQ4Row([]float32{
		0, 1, 0, 0,
		1, 0, 0, 0,
	}, 2, 4)
	tl := &textLayer{q4: q4LayerWeights{q: qw, k: kw, v: vw}}
	q := make([]float32, 4)
	k := make([]float32, 2)
	v := make([]float32, 2)
	ready, hasRoPE := rt.fusedFirstTokenKV(q, k, v, x, tl, 4, 2, 4, 1, 2, false, nil, nil)
	if !ready || hasRoPE {
		t.Fatalf("ready=%v hasRoPE=%v want ready without rope", ready, hasRoPE)
	}
	assertCloseSliceTol(t, "q4FirstQHeadZero", q[:2], []float32{0, 0}, 0)
	assertCloseSliceTol(t, "q4SecondQHeadZero", q[2:], []float32{0, 0}, 0)
	assertCloseSliceTol(t, "q4K", k, []float32{2, -1}, 0.2)
	assertCloseSliceTol(t, "q4V", v, []float32{-1, 2}, 0.2)
}

func TestFusedFirstTokenKVDoesNotRequireDummyQBuffer(t *testing.T) {
	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpFusedKVF32)
	x := []float32{2, -1, 3, 0.5}
	tl := &textLayer{w: layerWeights{
		q: []float32{
			9, 0, 0, 0,
			0, 9, 0, 0,
		},
		k: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		v: []float32{
			0, 1, 0, 0,
			1, 0, 0, 0,
		},
	}}
	k := make([]float32, 2)
	v := make([]float32, 2)
	ready, hasRoPE := rt.fusedFirstTokenKV(nil, k, v, x, tl, 0, 2, 4, 1, 2, false, nil, nil)
	if !ready || hasRoPE {
		t.Fatalf("ready=%v hasRoPE=%v want ready without rope", ready, hasRoPE)
	}
	assertCloseSliceTol(t, "k", k, []float32{2, -1}, 1e-6)
	assertCloseSliceTol(t, "v", v, []float32{-1, 2}, 1e-6)
}

func TestAttentionSmallCacheUsesCPUAttentionBeforeProjection(t *testing.T) {
	rt := &Runtime{
		cfg: &config.Config{
			HiddenSize:        4,
			NumAttentionHeads: 2,
			NumKeyValueHeads:  1,
			HeadDim:           2,
		},
	}
	x := []float32{2, -1, 0.5, 3}
	tl := &textLayer{w: layerWeights{
		q: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		k: []float32{
			0, 1, 0, 0,
			0, 0, 1, 0,
		},
		v: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}}
	sc := &layerScratch{
		q:       make([]float32, 4),
		k:       make([]float32, 2),
		v:       make([]float32, 2),
		headOut: make([]float32, 4),
		att:     make([]float32, 4),
	}
	cache := &kvCache{}
	cache.append([]float32{0.25, -0.5}, []float32{1, 2})
	got := rt.attention(x, cache, tl, sc, false, nil, nil)

	q := []float32{2, -1, 0.5, 3}
	k0 := []float32{0.25, -0.5}
	k1 := []float32{-1, 0.5}
	v0 := []float32{1, 2}
	v1 := []float32{2, -1}
	want := make([]float32, 4)
	scale := invSqrt(2)
	for h := 0; h < 2; h++ {
		qh := q[h*2 : h*2+2]
		scores := []float32{
			(qh[0]*k0[0] + qh[1]*k0[1]) * scale,
			(qh[0]*k1[0] + qh[1]*k1[1]) * scale,
		}
		tensor.SoftmaxInPlace(scores)
		want[h*2] = scores[0]*v0[0] + scores[1]*v1[0]
		want[h*2+1] = scores[0]*v0[1] + scores[1]*v1[1]
	}
	for i := range want {
		if diff := got[i] - want[i]; diff < -1e-5 || diff > 1e-5 {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if cache.len != 2 {
		t.Fatalf("cache len=%d want 2", cache.len)
	}
}

func TestAttentionWithNormCacheFallbackMatchesCPU(t *testing.T) {
	cfg := &config.Config{
		HiddenSize:        4,
		NumAttentionHeads: 2,
		NumKeyValueHeads:  1,
		HeadDim:           2,
		RMSNormEps:        1e-6,
	}
	tl := &textLayer{w: layerWeights{
		q: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		k: []float32{
			0, 1, 0, 0,
			0, 0, 1, 0,
		},
		v: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		o: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}}
	x := []float32{2, -1, 0.5, 3}
	residual := []float32{0.25, -0.5, 1.25, 2}
	normWeight := []float32{1, 0.75, 1.5, 0.5}
	newScratch := func() *layerScratch {
		return &layerScratch{
			q:       make([]float32, 4),
			k:       make([]float32, 2),
			v:       make([]float32, 2),
			headOut: make([]float32, 4),
			att:     make([]float32, 4),
			norm:    make([]float32, 4),
		}
	}
	newCache := func() *kvCache {
		cache := &kvCache{}
		cache.append([]float32{0.25, -0.5}, []float32{1, 2})
		return cache
	}

	cpu := &Runtime{cfg: cfg}
	cpuScratch := newScratch()
	cpuCache := newCache()
	cpuResidual := make([]float32, len(residual))
	copy(cpuResidual, residual)
	wantAtt, wantNormDone := cpu.attentionWithNorm(x, cpuCache, tl, cpuScratch, false, nil, nil, cpuResidual, normWeight, cpuScratch.norm)

	rt := &Runtime{backend: "vulkan", cfg: cfg}
	rt.disableVulkanOp(vulkanOpTextAttentionOutAddRMSNormF32)
	rt.disableVulkanOp(vulkanOpTextAttentionOutF32)
	rt.disableVulkanOp(vulkanOpTextAttentionF32)
	gotScratch := newScratch()
	gotCache := newCache()
	gotResidual := make([]float32, len(residual))
	copy(gotResidual, residual)
	gotAtt, gotNormDone := rt.attentionWithNorm(x, gotCache, tl, gotScratch, false, nil, nil, gotResidual, normWeight, gotScratch.norm)

	if gotNormDone != wantNormDone {
		t.Fatalf("normDone got=%v want=%v", gotNormDone, wantNormDone)
	}
	if !gotNormDone {
		assertCloseSliceTol(t, "att", gotAtt, wantAtt, 1e-5)
		assertCloseSliceTol(t, "normOutUnchanged", gotScratch.norm, []float32{0, 0, 0, 0}, 0)
	} else {
		assertCloseSliceTol(t, "normOut", gotScratch.norm, cpuScratch.norm, 1e-5)
	}
	assertCloseSliceTol(t, "cacheK", gotCache.k, cpuCache.k, 1e-5)
	assertCloseSliceTol(t, "cacheV", gotCache.v, cpuCache.v, 1e-5)
	if gotCache.len != 2 {
		t.Fatalf("cache len=%d want 2", gotCache.len)
	}
}

func TestMLPUsesCPUFallbackWhenSwiGLUDownDisabled(t *testing.T) {
	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       3,
			IntermediateSize: 4,
		},
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownQ8)
	gate := tensor.QuantizeQ8Row([]float32{
		1, -2, 0.5,
		0.25, 3, -1,
		-0.5, 0.75, 2,
		1.5, -0.25, 0.5,
	}, 4, 3)
	up := tensor.QuantizeQ8Row([]float32{
		0.5, 1, -1,
		-0.75, 0.25, 1.25,
		2, -1, 0.5,
		-0.5, 1.5, -1.5,
	}, 4, 3)
	down := tensor.QuantizeQ8Row([]float32{
		1, -0.5, 0.25, 0.75,
		-1, 0.5, 1.5, -0.25,
		0.25, 1, -0.75, 0.5,
	}, 3, 4)
	tl := &textLayer{q8: qLayerWeights{gate: gate, up: up, down: down}}
	sc := &layerScratch{
		mlp:  make([]float32, 3),
		gate: make([]float32, 4),
	}
	x := []float32{2, -1, 4}
	got := rt.mlp(x, tl, sc)
	want := make([]float32, 3)
	tmp := make([]float32, 4)
	tensor.FusedSwiGLUQ8Scratch(want, x, gate, up, down, tmp)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestMLPAddRMSNormFallbackMatchesCPU(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       3,
			IntermediateSize: 4,
			RMSNormEps:       1e-6,
		},
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownAddRMSNormQ8)
	rt.disableVulkanOp(vulkanOpSwiGLUDownQ8)
	rt.disableVulkanOp(vulkanOpSwiGLUGateUpQ8)
	rt.disableVulkanOp(vulkanOpMatVecAddRMSNormQ8)
	rt.disableVulkanOp(vulkanOpAddRMSNormF32)
	gate := tensor.QuantizeQ8Row([]float32{
		1, -2, 0.5,
		0.25, 3, -1,
		-0.5, 0.75, 2,
		1.5, -0.25, 0.5,
	}, 4, 3)
	up := tensor.QuantizeQ8Row([]float32{
		0.5, 1, -1,
		-0.75, 0.25, 1.25,
		2, -1, 0.5,
		-0.5, 1.5, -1.5,
	}, 4, 3)
	down := tensor.QuantizeQ8Row([]float32{
		1, -0.5, 0.25, 0.75,
		-1, 0.5, 1.5, -0.25,
		0.25, 1, -0.75, 0.5,
	}, 3, 4)
	tl := &textLayer{q8: qLayerWeights{gate: gate, up: up, down: down}}
	sc := &layerScratch{
		mlp:  make([]float32, 3),
		gate: make([]float32, 4),
	}
	x := []float32{2, -1, 4}
	residual := []float32{0.25, -0.5, 1.25}
	normWeight := []float32{1, 0.75, 1.5}
	gotNorm := make([]float32, 3)
	gotResidual := append([]float32(nil), residual...)

	if rt.mlpAddRMSNormMaybeVulkan(x, tl, sc, gotResidual, normWeight, gotNorm, 1e-6, true) {
		t.Fatal("disabled fused mlp+add+rmsnorm should fall back")
	}
	mlp := rt.mlp(x, tl, sc)
	rt.addRMSNormMaybeVulkan(gotNorm, gotResidual, mlp, normWeight, 1e-6)

	wantNorm := make([]float32, 3)
	wantResidual := append([]float32(nil), residual...)
	wantMLP := make([]float32, 3)
	tmp := make([]float32, 4)
	tensor.FusedSwiGLUQ8Scratch(wantMLP, x, gate, up, down, tmp)
	tensor.AddRMSNorm(wantNorm, wantResidual, wantMLP, normWeight, 1e-6)
	assertCloseSliceTol(t, "norm", gotNorm, wantNorm, 1e-5)
	assertCloseSliceTol(t, "residual", gotResidual, wantResidual, 1e-5)
}

func TestVulkanMLPF32FallsBackThroughGateUp(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan MLP gate/up fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       3,
			IntermediateSize: 4,
		},
	}
	rt.disableVulkanOp(vulkanOpSwiGLUDownF32)
	tl := &textLayer{w: layerWeights{
		gate: []float32{
			1, -2, 0.5,
			0.25, 3, -1,
			-0.5, 0.75, 2,
			1.5, -0.25, 0.5,
		},
		up: []float32{
			0.5, 1, -1,
			-0.75, 0.25, 1.25,
			2, -1, 0.5,
			-0.5, 1.5, -1.5,
		},
		down: []float32{
			1, -0.5, 0.25, 0.75,
			-1, 0.5, 1.5, -0.25,
			0.25, 1, -0.75, 0.5,
		},
	}}
	sc := &layerScratch{
		mlp:  make([]float32, 3),
		gate: make([]float32, 4),
	}
	x := []float32{2, -1, 4}
	got := rt.mlp(x, tl, sc)
	want := make([]float32, 3)
	wantGate := make([]float32, 4)
	tensor.FusedSwiGLUF32Scratch(want, x, tl.w.gate, tl.w.up, tl.w.down, 4, 3, 3, wantGate)
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpF32) {
		t.Fatal("Vulkan gate/up fallback op was disabled")
	}
	assertCloseSliceTol(t, "mlp", got, want, 1e-5)
	assertCloseSliceTol(t, "gate", sc.gate, wantGate, 1e-5)
}

func TestVulkanMLPAddRMSNormQuantizedFallsBackThroughGateUpMatVecNorm(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan quantized MLP+norm gate/up fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	gateWeights := []float32{
		1, -2, 0.5,
		0.25, 3, -1,
		-0.5, 0.75, 2,
		1.5, -0.25, 0.5,
	}
	upWeights := []float32{
		0.5, 1, -1,
		-0.75, 0.25, 1.25,
		2, -1, 0.5,
		-0.5, 1.5, -1.5,
	}
	downWeights := []float32{
		1, -0.5, 0.25, 0.75,
		-1, 0.5, 1.5, -0.25,
		0.25, 1, -0.75, 0.5,
	}
	x := []float32{2, -1, 4}
	residual := []float32{0.25, -0.5, 1.25}
	normWeight := []float32{1, 0.75, 1.5}

	cases := []struct {
		name     string
		fullOp   vulkanOp
		gateOp   vulkanOp
		matOp    vulkanOp
		tl       textLayer
		want     func(out, tmp []float32)
		normTol  float32
		gateTol  float32
		residTol float32
	}{
		{
			name:   "q8",
			fullOp: vulkanOpSwiGLUDownAddRMSNormQ8,
			gateOp: vulkanOpSwiGLUGateUpQ8,
			matOp:  vulkanOpMatVecAddRMSNormQ8,
			tl: textLayer{q8: qLayerWeights{
				gate: tensor.QuantizeQ8Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ8Row(upWeights, 4, 3),
				down: tensor.QuantizeQ8Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ8Scratch(out, x, tensor.QuantizeQ8Row(gateWeights, 4, 3), tensor.QuantizeQ8Row(upWeights, 4, 3), tensor.QuantizeQ8Row(downWeights, 3, 4), tmp)
			},
			normTol:  0.05,
			gateTol:  0.05,
			residTol: 0.05,
		},
		{
			name:   "q6",
			fullOp: vulkanOpSwiGLUDownAddRMSNormQ6,
			gateOp: vulkanOpSwiGLUGateUpQ6,
			matOp:  vulkanOpMatVecAddRMSNormQ6,
			tl: textLayer{q6: q6LayerWeights{
				gate: tensor.QuantizeQ6Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ6Row(upWeights, 4, 3),
				down: tensor.QuantizeQ6Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ6Scratch(out, x, tensor.QuantizeQ6Row(gateWeights, 4, 3), tensor.QuantizeQ6Row(upWeights, 4, 3), tensor.QuantizeQ6Row(downWeights, 3, 4), tmp)
			},
			normTol:  0.12,
			gateTol:  0.12,
			residTol: 0.12,
		},
		{
			name:   "q4",
			fullOp: vulkanOpSwiGLUDownAddRMSNormQ4,
			gateOp: vulkanOpSwiGLUGateUpQ4,
			matOp:  vulkanOpMatVecAddRMSNormQ4,
			tl: textLayer{q4: q4LayerWeights{
				gate: tensor.QuantizeQ4Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ4Row(upWeights, 4, 3),
				down: tensor.QuantizeQ4Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ4Scratch(out, x, tensor.QuantizeQ4Row(gateWeights, 4, 3), tensor.QuantizeQ4Row(upWeights, 4, 3), tensor.QuantizeQ4Row(downWeights, 3, 4), tmp)
			},
			normTol:  0.6,
			gateTol:  0.6,
			residTol: 0.6,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{
				backend: "vulkan",
				cfg: &config.Config{
					HiddenSize:       3,
					IntermediateSize: 4,
					RMSNormEps:       1e-6,
				},
			}
			rt.disableVulkanOp(tc.fullOp)
			sc := &layerScratch{
				mlp:  make([]float32, 3),
				gate: make([]float32, 4),
			}
			gotNorm := make([]float32, 3)
			gotResidual := append([]float32(nil), residual...)
			if !rt.mlpAddRMSNormMaybeVulkan(x, &tc.tl, sc, gotResidual, normWeight, gotNorm, 1e-6, true) {
				t.Fatal("expected quantized MLP+norm to fall back through Vulkan gate/up + matvec norm")
			}
			if !rt.vulkanOpEnabled(tc.gateOp) || !rt.vulkanOpEnabled(tc.matOp) {
				t.Fatalf("Vulkan %s gate/up or matvec+norm fallback op was disabled", tc.name)
			}

			wantMLP := make([]float32, 3)
			wantGate := make([]float32, 4)
			tc.want(wantMLP, wantGate)
			wantNorm := make([]float32, 3)
			wantResidual := append([]float32(nil), residual...)
			tensor.AddRMSNorm(wantNorm, wantResidual, wantMLP, normWeight, 1e-6)
			assertCloseSliceTol(t, "norm", gotNorm, wantNorm, tc.normTol)
			assertCloseSliceTol(t, "residual", gotResidual, wantResidual, tc.residTol)
			assertCloseSliceTol(t, "gate", sc.gate, wantGate, tc.gateTol)
		})
	}
}

func TestVulkanMLPAddRMSNormOutOnlyQuantizedFallsBackThroughGateUpMatVecNorm(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan quantized MLP+out-only norm gate/up fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	gateWeights := []float32{
		1, -2, 0.5,
		0.25, 3, -1,
		-0.5, 0.75, 2,
		1.5, -0.25, 0.5,
	}
	upWeights := []float32{
		0.5, 1, -1,
		-0.75, 0.25, 1.25,
		2, -1, 0.5,
		-0.5, 1.5, -1.5,
	}
	downWeights := []float32{
		1, -0.5, 0.25, 0.75,
		-1, 0.5, 1.5, -0.25,
		0.25, 1, -0.75, 0.5,
	}
	x := []float32{2, -1, 4}
	residual := []float32{0.25, -0.5, 1.25}
	normWeight := []float32{1, 0.75, 1.5}

	cases := []struct {
		name    string
		fullOp  vulkanOp
		gateOp  vulkanOp
		matOp   vulkanOp
		tl      textLayer
		want    func(out, tmp []float32)
		normTol float32
		gateTol float32
		mlpTol  float32
	}{
		{
			name:   "q8",
			fullOp: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ8,
			gateOp: vulkanOpSwiGLUGateUpQ8,
			matOp:  vulkanOpMatVecQ8,
			tl: textLayer{q8: qLayerWeights{
				gate: tensor.QuantizeQ8Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ8Row(upWeights, 4, 3),
				down: tensor.QuantizeQ8Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ8Scratch(out, x, tensor.QuantizeQ8Row(gateWeights, 4, 3), tensor.QuantizeQ8Row(upWeights, 4, 3), tensor.QuantizeQ8Row(downWeights, 3, 4), tmp)
			},
			normTol: 0.05,
			gateTol: 0.05,
			mlpTol:  0.05,
		},
		{
			name:   "q6",
			fullOp: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ6,
			gateOp: vulkanOpSwiGLUGateUpQ6,
			matOp:  vulkanOpMatVecQ6,
			tl: textLayer{q6: q6LayerWeights{
				gate: tensor.QuantizeQ6Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ6Row(upWeights, 4, 3),
				down: tensor.QuantizeQ6Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ6Scratch(out, x, tensor.QuantizeQ6Row(gateWeights, 4, 3), tensor.QuantizeQ6Row(upWeights, 4, 3), tensor.QuantizeQ6Row(downWeights, 3, 4), tmp)
			},
			normTol: 0.12,
			gateTol: 0.12,
			mlpTol:  0.12,
		},
		{
			name:   "q4",
			fullOp: vulkanOpSwiGLUDownAddRMSNormOutOnlyQ4,
			gateOp: vulkanOpSwiGLUGateUpQ4,
			matOp:  vulkanOpMatVecQ4,
			tl: textLayer{q4: q4LayerWeights{
				gate: tensor.QuantizeQ4Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ4Row(upWeights, 4, 3),
				down: tensor.QuantizeQ4Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ4Scratch(out, x, tensor.QuantizeQ4Row(gateWeights, 4, 3), tensor.QuantizeQ4Row(upWeights, 4, 3), tensor.QuantizeQ4Row(downWeights, 3, 4), tmp)
			},
			normTol: 0.6,
			gateTol: 0.6,
			mlpTol:  0.6,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{
				backend: "vulkan",
				cfg: &config.Config{
					HiddenSize:       3,
					IntermediateSize: 4,
					RMSNormEps:       1e-6,
				},
			}
			rt.disableVulkanOp(tc.fullOp)
			sc := &layerScratch{
				mlp:  make([]float32, 3),
				gate: make([]float32, 4),
			}
			gotNorm := make([]float32, 3)
			gotResidual := append([]float32(nil), residual...)
			if !rt.mlpAddRMSNormMaybeVulkan(x, &tc.tl, sc, gotResidual, normWeight, gotNorm, 1e-6, false) {
				t.Fatal("expected quantized MLP+out-only norm to fall back through Vulkan gate/up + matvec + out-only norm")
			}
			if !rt.vulkanOpEnabled(tc.gateOp) || !rt.vulkanOpEnabled(tc.matOp) || !rt.vulkanOpEnabled(vulkanOpAddRMSNormOutOnlyF32) {
				t.Fatalf("Vulkan %s gate/up, matvec, or out-only norm fallback op was disabled", tc.name)
			}

			wantMLP := make([]float32, 3)
			wantGate := make([]float32, 4)
			tc.want(wantMLP, wantGate)
			wantNorm := make([]float32, 3)
			tensor.AddRMSNormOutOnly(wantNorm, residual, wantMLP, normWeight, 1e-6)
			assertCloseSliceTol(t, "norm", gotNorm, wantNorm, tc.normTol)
			assertCloseSliceTol(t, "residual", gotResidual, residual, 0)
			assertCloseSliceTol(t, "mlp", sc.mlp[:3], wantMLP, tc.mlpTol)
			assertCloseSliceTol(t, "gate", sc.gate, wantGate, tc.gateTol)
		})
	}
}

func TestVulkanMLPQuantizedFallsBackThroughGateUp(t *testing.T) {
	if os.Getenv("RAPIDOCRVL_VULKAN_SMOKE") != "1" {
		t.Skip("set RAPIDOCRVL_VULKAN_SMOKE=1 to run Vulkan quantized MLP gate/up fallback smoke test")
	}
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	gateWeights := []float32{
		1, -2, 0.5,
		0.25, 3, -1,
		-0.5, 0.75, 2,
		1.5, -0.25, 0.5,
	}
	upWeights := []float32{
		0.5, 1, -1,
		-0.75, 0.25, 1.25,
		2, -1, 0.5,
		-0.5, 1.5, -1.5,
	}
	downWeights := []float32{
		1, -0.5, 0.25, 0.75,
		-1, 0.5, 1.5, -0.25,
		0.25, 1, -0.75, 0.5,
	}
	x := []float32{2, -1, 4}

	cases := []struct {
		name       string
		downOp     vulkanOp
		gateUpOp   vulkanOp
		tl         textLayer
		want       func(out, tmp []float32)
		outputTol  float32
		scratchTol float32
	}{
		{
			name:     "q8",
			downOp:   vulkanOpSwiGLUDownQ8,
			gateUpOp: vulkanOpSwiGLUGateUpQ8,
			tl: textLayer{q8: qLayerWeights{
				gate: tensor.QuantizeQ8Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ8Row(upWeights, 4, 3),
				down: tensor.QuantizeQ8Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ8Scratch(out, x, tensor.QuantizeQ8Row(gateWeights, 4, 3), tensor.QuantizeQ8Row(upWeights, 4, 3), tensor.QuantizeQ8Row(downWeights, 3, 4), tmp)
			},
			outputTol:  0.05,
			scratchTol: 0.05,
		},
		{
			name:     "q6",
			downOp:   vulkanOpSwiGLUDownQ6,
			gateUpOp: vulkanOpSwiGLUGateUpQ6,
			tl: textLayer{q6: q6LayerWeights{
				gate: tensor.QuantizeQ6Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ6Row(upWeights, 4, 3),
				down: tensor.QuantizeQ6Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ6Scratch(out, x, tensor.QuantizeQ6Row(gateWeights, 4, 3), tensor.QuantizeQ6Row(upWeights, 4, 3), tensor.QuantizeQ6Row(downWeights, 3, 4), tmp)
			},
			outputTol:  0.12,
			scratchTol: 0.12,
		},
		{
			name:     "q4",
			downOp:   vulkanOpSwiGLUDownQ4,
			gateUpOp: vulkanOpSwiGLUGateUpQ4,
			tl: textLayer{q4: q4LayerWeights{
				gate: tensor.QuantizeQ4Row(gateWeights, 4, 3),
				up:   tensor.QuantizeQ4Row(upWeights, 4, 3),
				down: tensor.QuantizeQ4Row(downWeights, 3, 4),
			}},
			want: func(out, tmp []float32) {
				tensor.FusedSwiGLUQ4Scratch(out, x, tensor.QuantizeQ4Row(gateWeights, 4, 3), tensor.QuantizeQ4Row(upWeights, 4, 3), tensor.QuantizeQ4Row(downWeights, 3, 4), tmp)
			},
			outputTol:  0.6,
			scratchTol: 0.6,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rt := &Runtime{
				backend: "vulkan",
				cfg: &config.Config{
					HiddenSize:       3,
					IntermediateSize: 4,
				},
			}
			rt.disableVulkanOp(tc.downOp)
			sc := &layerScratch{
				mlp:  make([]float32, 3),
				gate: make([]float32, 4),
			}
			got := rt.mlp(x, &tc.tl, sc)
			want := make([]float32, 3)
			wantGate := make([]float32, 4)
			tc.want(want, wantGate)
			if !rt.vulkanOpEnabled(tc.gateUpOp) {
				t.Fatalf("Vulkan %s gate/up fallback op was disabled", tc.name)
			}
			assertCloseSliceTol(t, "mlp", got, want, tc.outputTol)
			assertCloseSliceTol(t, "gate", sc.gate, wantGate, tc.scratchTol)
		})
	}
}

func TestAddRMSNormOutOnlyFallbackDoesNotMutateDst(t *testing.T) {
	t.Setenv(vulkanVectorMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{backend: "vulkan"}
	rt.disableVulkanOp(vulkanOpAddRMSNormOutOnlyF32)
	dst := []float32{1, -2, 3, -4, 5, -6, 7, -8, 9}
	add := []float32{0.5, 1, -1.5, 2, -2.5, 3, -3.5, 4, -4.5}
	weight := []float32{1, 1.1, 0.9, 1.2, 0.8, 1.3, 0.7, 1.4, 0.6}
	wantDst := append([]float32(nil), dst...)
	wantAdded := append([]float32(nil), dst...)
	tensor.AddInPlace(wantAdded, add)
	wantOut := make([]float32, len(dst))
	tensor.RMSNorm(wantOut, wantAdded, weight, 1e-6)
	gotOut := make([]float32, len(dst))
	rt.addRMSNormOutOnlyMaybeVulkan(gotOut, dst, add, weight, 1e-6)
	assertCloseSliceTol(t, "dst", dst, wantDst, 0)
	assertCloseSliceTol(t, "out", gotOut, wantOut, 1e-6)
}

func TestFusedQKVVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	qOut := make([]float32, 3)
	kOut := make([]float32, 3)
	vOut := make([]float32, 3)
	cosTable := []float32{1}
	sinTable := []float32{0}
	badQ := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, 1,
	}, 3, 2)
	badQ.Data = badQ.Data[:len(badQ.Data)-1]
	goodK := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, -1,
	}, 3, 2)
	goodV := tensor.QuantizeQ8Row([]float32{
		0, 1,
		1, 0,
		-1, 1,
	}, 3, 2)
	rt := &Runtime{backend: "vulkan"}
	tl := textLayer{q8: qLayerWeights{q: badQ, k: goodK, v: goodV}}

	_ = rt.fusedQKV(qOut, kOut, vOut, x, &tl, 3, 3, 2)
	if !rt.vulkanOpEnabled(vulkanOpFusedQKVQ8) || !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("invalid fused qkv shape disabled a q8 Vulkan op")
	}
	if rt.fusedQKVMRoPE(qOut, kOut, vOut, x, &tl, 3, 3, 2, 1, 1, 2, cosTable, sinTable) {
		t.Fatal("invalid fused qkv+mrope shape unexpectedly reached Vulkan")
	}
	if !rt.vulkanOpEnabled(vulkanOpFusedQKVMRoPEQ8) {
		t.Fatal("invalid fused qkv+mrope shape disabled q8 Vulkan op")
	}
}

func TestFusedKVVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	kOut := make([]float32, 3)
	vOut := make([]float32, 3)
	cosTable := []float32{1}
	sinTable := []float32{0}
	badK := tensor.QuantizeQ8Row([]float32{
		1, 0,
		0, 1,
		1, -1,
	}, 3, 2)
	badK.Scale = badK.Scale[:len(badK.Scale)-1]
	goodV := tensor.QuantizeQ8Row([]float32{
		0, 1,
		1, 0,
		-1, 1,
	}, 3, 2)
	rt := &Runtime{backend: "vulkan"}
	tl := textLayer{q8: qLayerWeights{k: badK, v: goodV}}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected CPU fallback to reject invalid q8 fused kv matrix")
			}
		}()
		_, _ = rt.fusedKV(kOut, vOut, x, &tl, 3, 2, 1, 2, true, cosTable, sinTable)
	}()
	if !rt.vulkanOpEnabled(vulkanOpFusedKVQ8) || !rt.vulkanOpEnabled(vulkanOpFusedKVMRoPEQ8) || !rt.vulkanOpEnabled(vulkanOpMatVecQ8) {
		t.Fatal("invalid fused kv shape disabled a q8 Vulkan op")
	}
}

func TestSwiGLUVulkanRejectsBadShapesWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	x := []float32{1, 2}
	gateWeights := []float32{
		1, 0,
		0, 1,
		1, -1,
	}
	upWeights := []float32{
		0, 1,
		1, 0,
		-1, 1,
	}
	downWeights := []float32{
		1, 0, 1,
		0, 1, -1,
	}
	badGate := tensor.QuantizeQ8Row(gateWeights, 3, 2)
	badGate.Data = badGate.Data[:len(badGate.Data)-1]
	goodUp := tensor.QuantizeQ8Row(upWeights, 3, 2)
	goodDown := tensor.QuantizeQ8Row(downWeights, 2, 3)
	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       2,
			IntermediateSize: 3,
		},
	}
	tl := textLayer{q8: qLayerWeights{gate: badGate, up: goodUp, down: goodDown}}

	if rt.swiGLUGateUpQ8MaybeVulkan(make([]float32, 3), x, badGate, goodUp, goodDown) {
		t.Fatal("bad q8 swiglu gate/up unexpectedly reached Vulkan")
	}
	sc := &layerScratch{mlp: make([]float32, 2), gate: make([]float32, 3)}
	_ = rt.mlpAddRMSNormMaybeVulkan(x, &tl, sc, make([]float32, 2), []float32{1, 1}, make([]float32, 2), 1e-6, true)
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUGateUpQ8) ||
		!rt.vulkanOpEnabled(vulkanOpSwiGLUDownQ8) ||
		!rt.vulkanOpEnabled(vulkanOpSwiGLUDownAddRMSNormQ8) {
		t.Fatal("invalid q8 swiglu shape disabled a Vulkan op")
	}
}

func TestF32SwiGLUVulkanRejectsBadInputWithoutDisablingOps(t *testing.T) {
	t.Setenv(vulkanMatVecMinWorkEnv, "1")
	resetVulkanMinWorkCacheForTest()
	t.Cleanup(resetVulkanMinWorkCacheForTest)

	rt := &Runtime{
		backend: "vulkan",
		cfg: &config.Config{
			HiddenSize:       2,
			IntermediateSize: 3,
		},
	}
	tl := textLayer{w: layerWeights{
		gate: []float32{
			1, 0,
			0, 1,
			1, -1,
		},
		up: []float32{
			0, 1,
			1, 0,
			-1, 1,
		},
		down: []float32{
			1, 0, 1,
			0, 1, -1,
		},
	}}
	sc := &layerScratch{mlp: make([]float32, 2), gate: make([]float32, 3)}
	_ = rt.mlp([]float32{1}, &tl, sc)
	if !rt.vulkanOpEnabled(vulkanOpSwiGLUDownF32) {
		t.Fatal("invalid f32 swiglu input disabled Vulkan swiglu/down op")
	}
}

func assertCloseSliceTol(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		diff := got[i] - want[i]
		if diff < -tol || diff > tol {
			t.Fatalf("%s[%d]=%g want %g tol %g", name, i, got[i], want[i], tol)
		}
	}
}
