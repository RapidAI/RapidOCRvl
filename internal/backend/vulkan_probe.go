package backend

import (
	"fmt"
	"math"
	"sort"

	"paddleocrvl-go/internal/tensor"
)

type VulkanDispatchProbeResult struct {
	Name  string `json:"name"`
	OK    bool   `json:"ok"`
	Error string `json:"error,omitempty"`
}

type vulkanDispatchProbeCase struct {
	name string
	run  func() error
}

func VulkanDispatchProbeNames() []string {
	probes := vulkanDispatchProbes()
	names := make([]string, len(probes))
	for i, probe := range probes {
		names[i] = probe.name
	}
	return names
}

func VulkanDispatchSmokeSuite() []VulkanDispatchProbeResult {
	probes := vulkanDispatchProbes()
	out := make([]VulkanDispatchProbeResult, 0, len(probes))
	for _, probe := range probes {
		res := VulkanDispatchProbeResult{Name: probe.name, OK: true}
		if err := probe.run(); err != nil {
			res.OK = false
			res.Error = err.Error()
		}
		out = append(out, res)
	}
	return out
}

func vulkanDispatchProbes() []vulkanDispatchProbeCase {
	return []vulkanDispatchProbeCase{
		{name: "matvec_f32", run: probeVulkanMatVecF32},
		{name: "matvec_argmax_f32", run: probeVulkanMatVecArgmaxF32},
		{name: "matvec_topk_f32", run: probeVulkanMatVecTopKF32},
		{name: "matvec_q8", run: probeVulkanMatVecQ8},
		{name: "matvec_argmax_q8", run: probeVulkanMatVecArgmaxQ8},
		{name: "matvec_topk_q8", run: probeVulkanMatVecTopKQ8},
		{name: "matvec_q6", run: probeVulkanMatVecQ6},
		{name: "matvec_argmax_q6", run: probeVulkanMatVecArgmaxQ6},
		{name: "matvec_topk_q6", run: probeVulkanMatVecTopKQ6},
		{name: "matvec_q4", run: probeVulkanMatVecQ4},
		{name: "matvec_argmax_q4", run: probeVulkanMatVecArgmaxQ4},
		{name: "matvec_topk_q4", run: probeVulkanMatVecTopKQ4},
		{name: "rmsnorm_f32", run: probeVulkanRMSNormF32},
		{name: "add_rmsnorm_f32", run: probeVulkanAddRMSNormF32},
		{name: "add_rmsnorm_out_only_f32", run: probeVulkanAddRMSNormF32OutOnly},
		{name: "mrope_f32", run: probeVulkanMRoPEF32},
		{name: "mrope_pair_f32", run: probeVulkanMRoPEPairF32},
		{name: "text_attention_f32", run: probeVulkanTextAttentionF32},
		{name: "text_attention_out_f32", run: probeVulkanTextAttentionOutF32},
		{name: "text_attention_out_q8", run: probeVulkanTextAttentionOutQ8},
		{name: "text_attention_out_q6", run: probeVulkanTextAttentionOutQ6},
		{name: "text_attention_out_q4", run: probeVulkanTextAttentionOutQ4},
		{name: "text_attention_out_add_rmsnorm_f32", run: probeVulkanTextAttentionOutAddRMSNormF32},
		{name: "text_attention_out_add_rmsnorm_q8", run: probeVulkanTextAttentionOutAddRMSNormQ8},
		{name: "text_attention_out_add_rmsnorm_q6", run: probeVulkanTextAttentionOutAddRMSNormQ6},
		{name: "text_attention_out_add_rmsnorm_q4", run: probeVulkanTextAttentionOutAddRMSNormQ4},
		{name: "text_first_token_value_out_add_rmsnorm_f32", run: probeVulkanTextFirstTokenValueOutAddRMSNormF32},
		{name: "text_first_token_value_out_add_rmsnorm_q8", run: probeVulkanTextFirstTokenValueOutAddRMSNormQ8},
		{name: "text_first_token_value_out_add_rmsnorm_q6", run: probeVulkanTextFirstTokenValueOutAddRMSNormQ6},
		{name: "text_first_token_value_out_add_rmsnorm_q4", run: probeVulkanTextFirstTokenValueOutAddRMSNormQ4},
		{name: "fused_qkv_f32", run: probeVulkanFusedQKVF32},
		{name: "fused_qkv_q8", run: probeVulkanFusedQKVQ8},
		{name: "fused_qkv_q6", run: probeVulkanFusedQKVQ6},
		{name: "fused_qkv_q4", run: probeVulkanFusedQKVQ4},
		{name: "fused_qkv_mrope_f32", run: probeVulkanFusedQKVMRoPEF32},
		{name: "fused_qkv_mrope_q8", run: probeVulkanFusedQKVMRoPEQ8},
		{name: "fused_qkv_mrope_q6", run: probeVulkanFusedQKVMRoPEQ6},
		{name: "fused_qkv_mrope_q4", run: probeVulkanFusedQKVMRoPEQ4},
		{name: "fused_kv_f32", run: probeVulkanFusedKVF32},
		{name: "fused_kv_q8", run: probeVulkanFusedKVQ8},
		{name: "fused_kv_q6", run: probeVulkanFusedKVQ6},
		{name: "fused_kv_q4", run: probeVulkanFusedKVQ4},
		{name: "fused_kv_mrope_f32", run: probeVulkanFusedKVMRoPEF32},
		{name: "fused_kv_mrope_q8", run: probeVulkanFusedKVMRoPEQ8},
		{name: "fused_kv_mrope_q6", run: probeVulkanFusedKVMRoPEQ6},
		{name: "fused_kv_mrope_q4", run: probeVulkanFusedKVMRoPEQ4},
		{name: "matvec_add_rmsnorm_f32", run: probeVulkanMatVecAddRMSNormF32},
		{name: "matvec_add_rmsnorm_q8", run: probeVulkanMatVecAddRMSNormQ8},
		{name: "matvec_add_rmsnorm_q6", run: probeVulkanMatVecAddRMSNormQ6},
		{name: "matvec_add_rmsnorm_q4", run: probeVulkanMatVecAddRMSNormQ4},
		{name: "swiglu_gate_up_f32", run: probeVulkanSwiGLUGateUpF32},
		{name: "swiglu_gate_up_q8", run: probeVulkanSwiGLUGateUpQ8},
		{name: "swiglu_gate_up_q6", run: probeVulkanSwiGLUGateUpQ6},
		{name: "swiglu_gate_up_q4", run: probeVulkanSwiGLUGateUpQ4},
		{name: "swiglu_down_f32", run: probeVulkanSwiGLUDownF32},
		{name: "swiglu_down_q8", run: probeVulkanSwiGLUDownQ8},
		{name: "swiglu_down_q6", run: probeVulkanSwiGLUDownQ6},
		{name: "swiglu_down_q4", run: probeVulkanSwiGLUDownQ4},
		{name: "swiglu_down_add_rmsnorm_f32", run: probeVulkanSwiGLUDownAddRMSNormF32},
		{name: "swiglu_down_add_rmsnorm_q8", run: probeVulkanSwiGLUDownAddRMSNormQ8},
		{name: "swiglu_down_add_rmsnorm_q6", run: probeVulkanSwiGLUDownAddRMSNormQ6},
		{name: "swiglu_down_add_rmsnorm_q4", run: probeVulkanSwiGLUDownAddRMSNormQ4},
		{name: "swiglu_down_add_rmsnorm_out_only_f32", run: probeVulkanSwiGLUDownAddRMSNormF32OutOnly},
		{name: "swiglu_down_add_rmsnorm_out_only_q8", run: probeVulkanSwiGLUDownAddRMSNormQ8OutOnly},
		{name: "swiglu_down_add_rmsnorm_out_only_q6", run: probeVulkanSwiGLUDownAddRMSNormQ6OutOnly},
		{name: "swiglu_down_add_rmsnorm_out_only_q4", run: probeVulkanSwiGLUDownAddRMSNormQ4OutOnly},
		{name: "vision_matrows_bias_f32", run: probeVulkanVisionMatRowsBiasF32},
		{name: "vision_matrows_bias_add_rows_f32", run: probeVulkanVisionMatRowsBiasAddRowsF32},
		{name: "vision_matrows_bias3_f32", run: probeVulkanVisionMatRowsBias3F32},
		{name: "vision_attention_f32", run: probeVulkanVisionAttentionF32},
		{name: "vision_attention_out_f32", run: probeVulkanVisionAttentionOutF32},
		{name: "vision_matrows_gelu2_f32", run: probeVulkanVisionMatRowsGELU2F32},
		{name: "vision_project_image_f32", run: probeVulkanVisionProjectImageF32},
		{name: "vision_project_image_fused_f32", run: probeVulkanVisionProjectImageF32},
		{name: "vision_layernorm_rows_f32", run: probeVulkanVisionLayerNormRowsF32},
		{name: "vision_add_layernorm_rows_f32", run: probeVulkanVisionAddLayerNormRowsF32},
		{name: "vision_rope_pair_f32", run: probeVulkanVisionRoPEPairF32},
		{name: "vision_matrows_gelu2_add_layernorm_f32", run: probeVulkanVisionMatRowsGELU2AddLayerNormF32},
		{name: "vision_rope_attention_out_f32", run: probeVulkanVisionRoPEAttentionOutF32},
		{name: "vision_qkv_rope_attention_out_f32", run: probeVulkanVisionQKVRoPEAttentionOutF32},
		{name: "chained_rmsnorm_matvec_f32", run: probeVulkanChainedRMSNormMatVecF32},
		{name: "chained_matvec_add_rmsnorm_matvec_f32", run: probeVulkanChainedMatVecAddRMSNormMatVecF32},
		{name: "chained_rmsnorm_qkv_mrope_f32", run: probeVulkanChainedRMSNormQKVMRoPEF32},
		{name: "chained_rmsnorm_qkv_mrope_q8", run: probeVulkanChainedRMSNormQKVMRoPEQ8},
		{name: "chained_qkv_attention_out_norm_f32", run: probeVulkanChainedQKVAttentionOutAddRMSNormF32},
	}
}

func probeVulkanMatVecF32() error {
	x := probeVulkanX()
	w := probeVulkanW()
	out := make([]float32, 4)
	if err := VulkanMatVecF32(out, x, w, 4, 5); err != nil {
		return err
	}
	return compareFloat32("matvec_f32", out, []float32{2, 0, -4, 1.25}, 1e-4)
}

func probeVulkanMatVecArgmaxF32() error {
	token, score, err := VulkanMatVecArgmaxF32(probeVulkanX(), probeVulkanW(), 4, 5)
	if err != nil {
		return err
	}
	if token != 0 || math.Abs(float64(score-2)) > 1e-4 {
		return fmt.Errorf("argmax f32 token=%d score=%.6f want token=0 score=2", token, score)
	}
	return nil
}

func probeVulkanMatVecTopKF32() error {
	x, w, rows, cols, k := probeVulkanTopKInput()
	want := probeVulkanTopKFromLogits(matVecLogits(x, w, rows, cols), k)
	got, err := VulkanMatVecTopKF32(x, w, rows, cols, k)
	if err != nil {
		return err
	}
	return compareVulkanTopK("topk_f32", got, want, 1e-4)
}

func probeVulkanMatVecQ8() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ8Row(probeVulkanW(), 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ8(out, x, q); err != nil {
		return err
	}
	want := make([]float32, 4)
	tensor.MatVecQ8(want, x, q)
	return compareFloat32("matvec_q8", out, want, 1e-4)
}

func probeVulkanMatVecArgmaxQ8() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ8Row(probeVulkanW(), 4, 5)
	wantToken, wantScore := tensor.MatVecArgmaxQ8(x, q)
	token, score, err := VulkanMatVecArgmaxQ8(x, q)
	if err != nil {
		return err
	}
	return compareTokenScore("argmax_q8", token, score, wantToken, wantScore, 1e-4)
}

func probeVulkanMatVecTopKQ8() error {
	x, w, rows, cols, k := probeVulkanTopKInput()
	q := tensor.QuantizeQ8Row(w, rows, cols)
	logits := make([]float32, rows)
	tensor.MatVecQ8(logits, x, q)
	want := probeVulkanTopKFromLogits(logits, k)
	got, err := VulkanMatVecTopKQ8(x, q, k)
	if err != nil {
		return err
	}
	return compareVulkanTopK("topk_q8", got, want, 1e-4)
}

func probeVulkanMatVecQ6() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ6Row(probeVulkanW(), 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ6(out, x, q); err != nil {
		return err
	}
	want := make([]float32, 4)
	tensor.MatVecQ6(want, x, q)
	return compareFloat32("matvec_q6", out, want, 1e-4)
}

func probeVulkanMatVecArgmaxQ6() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ6Row(probeVulkanW(), 4, 5)
	wantToken, wantScore := tensor.MatVecArgmaxQ6(x, q)
	token, score, err := VulkanMatVecArgmaxQ6(x, q)
	if err != nil {
		return err
	}
	return compareTokenScore("argmax_q6", token, score, wantToken, wantScore, 1e-4)
}

func probeVulkanMatVecTopKQ6() error {
	x, w, rows, cols, k := probeVulkanTopKInput()
	q := tensor.QuantizeQ6Row(w, rows, cols)
	logits := make([]float32, rows)
	tensor.MatVecQ6(logits, x, q)
	want := probeVulkanTopKFromLogits(logits, k)
	got, err := VulkanMatVecTopKQ6(x, q, k)
	if err != nil {
		return err
	}
	return compareVulkanTopK("topk_q6", got, want, 1e-4)
}

func probeVulkanMatVecQ4() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ4Row(probeVulkanW(), 4, 5)
	out := make([]float32, 4)
	if err := VulkanMatVecQ4(out, x, q); err != nil {
		return err
	}
	want := make([]float32, 4)
	tensor.MatVecQ4(want, x, q)
	return compareFloat32("matvec_q4", out, want, 1e-4)
}

func probeVulkanMatVecArgmaxQ4() error {
	x := probeVulkanX()
	q := tensor.QuantizeQ4Row(probeVulkanW(), 4, 5)
	wantToken, wantScore := tensor.MatVecArgmaxQ4(x, q)
	token, score, err := VulkanMatVecArgmaxQ4(x, q)
	if err != nil {
		return err
	}
	return compareTokenScore("argmax_q4", token, score, wantToken, wantScore, 1e-4)
}

func probeVulkanMatVecTopKQ4() error {
	x, w, rows, cols, k := probeVulkanTopKInput()
	q := tensor.QuantizeQ4Row(w, rows, cols)
	logits := make([]float32, rows)
	tensor.MatVecQ4(logits, x, q)
	want := probeVulkanTopKFromLogits(logits, k)
	got, err := VulkanMatVecTopKQ4(x, q, k)
	if err != nil {
		return err
	}
	return compareVulkanTopK("topk_q4", got, want, 1e-4)
}

func probeVulkanRMSNormF32() error {
	x := probeVulkanVector()
	weight := probeVulkanNormWeight(len(x))
	out := make([]float32, len(x))
	if err := VulkanRMSNormF32(out, x, weight); err != nil {
		return err
	}
	want := make([]float32, len(x))
	tensor.RMSNorm(want, x, weight, 1e-6)
	return compareFloat32("rmsnorm_f32", out, want, 1e-4)
}

func probeVulkanAddRMSNormF32() error {
	dst := probeVulkanVector()
	add := probeVulkanAddVector(len(dst))
	weight := probeVulkanNormWeight(len(dst))
	out := make([]float32, len(dst))
	wantDst := append([]float32(nil), dst...)
	want := make([]float32, len(dst))
	tensor.AddRMSNorm(want, wantDst, add, weight, 1e-6)
	if err := VulkanAddRMSNormF32(out, dst, add, weight); err != nil {
		return err
	}
	if err := compareFloat32("add_rmsnorm_f32/out", out, want, 1e-4); err != nil {
		return err
	}
	return compareFloat32("add_rmsnorm_f32/dst", dst, wantDst, 1e-4)
}

func probeVulkanAddRMSNormF32OutOnly() error {
	dst := probeVulkanVector()
	add := probeVulkanAddVector(len(dst))
	weight := probeVulkanNormWeight(len(dst))
	out := make([]float32, len(dst))
	wantDst := append([]float32(nil), dst...)
	want := make([]float32, len(dst))
	tensor.AddRMSNormOutOnly(want, wantDst, add, weight, 1e-6)
	if err := VulkanAddRMSNormF32OutOnly(out, dst, add, weight); err != nil {
		return err
	}
	if err := compareFloat32("add_rmsnorm_out_only_f32/out", out, want, 1e-4); err != nil {
		return err
	}
	return compareFloat32("add_rmsnorm_out_only_f32/dst", dst, wantDst, 1e-4)
}

func probeVulkanMRoPEF32() error {
	heads, dim := 4, 32
	cosTable, sinTable := probeVulkanRoPETables(dim)
	x := probeVulkanHeadVector(heads, dim)
	want := append([]float32(nil), x...)
	applyProbeMRoPE(want, heads, dim, cosTable, sinTable)
	if err := VulkanMRoPEF32(x, cosTable, sinTable, heads, dim); err != nil {
		return err
	}
	return compareFloat32("mrope_f32", x, want, 1e-4)
}

func probeVulkanMRoPEPairF32() error {
	qHeads, kvHeads, dim := 4, 2, 32
	cosTable, sinTable := probeVulkanRoPETables(dim)
	q := probeVulkanHeadVector(qHeads, dim)
	k := probeVulkanHeadVector(kvHeads, dim)
	wantQ := append([]float32(nil), q...)
	wantK := append([]float32(nil), k...)
	applyProbeMRoPE(wantQ, qHeads, dim, cosTable, sinTable)
	applyProbeMRoPE(wantK, kvHeads, dim, cosTable, sinTable)
	if err := VulkanMRoPEPairF32(q, k, cosTable, sinTable, qHeads, kvHeads, dim); err != nil {
		return err
	}
	if err := compareFloat32("mrope_pair_f32/q", q, wantQ, 1e-4); err != nil {
		return err
	}
	return compareFloat32("mrope_pair_f32/k", k, wantK, 1e-4)
}

func probeVulkanTextAttentionF32() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	out := make([]float32, numHeads*headDim)
	if err := VulkanTextAttentionF32(out, q, k, v, 1, cacheLen, numHeads, kvHeads, headDim); err != nil {
		return err
	}
	want := make([]float32, numHeads*headDim)
	textAttentionProbe(want, q, k, v, cacheLen, numHeads, kvHeads, headDim)
	return compareFloat32("text_attention_f32", out, want, 1e-4)
}

func probeVulkanTextAttentionOutF32() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, bias := probeVulkanTextOutWeights(qRows)
	return probeVulkanTextAttentionOut("text_attention_out_f32", func(out []float32) error {
		return VulkanTextAttentionOutF32(out, q, k, v, w, bias, 2, cacheLen, numHeads, kvHeads, headDim)
	}, func(want []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecBias(want, head, w, bias, qRows, qRows)
	}, qRows)
}

func probeVulkanTextAttentionOutQ8() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ8Row(w, qRows, qRows)
	return probeVulkanTextAttentionOut("text_attention_out_q8", func(out []float32) error {
		return VulkanTextAttentionOutQ8(out, q, k, v, qw, 3, cacheLen, numHeads, kvHeads, headDim)
	}, func(want []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ8(want, head, qw)
	}, qRows)
}

func probeVulkanTextAttentionOutQ6() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ6Row(w, qRows, qRows)
	return probeVulkanTextAttentionOut("text_attention_out_q6", func(out []float32) error {
		return VulkanTextAttentionOutQ6(out, q, k, v, qw, 4, cacheLen, numHeads, kvHeads, headDim)
	}, func(want []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ6(want, head, qw)
	}, qRows)
}

func probeVulkanTextAttentionOutQ4() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ4Row(w, qRows, qRows)
	return probeVulkanTextAttentionOut("text_attention_out_q4", func(out []float32) error {
		return VulkanTextAttentionOutQ4(out, q, k, v, qw, 5, cacheLen, numHeads, kvHeads, headDim)
	}, func(want []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ4(want, head, qw)
	}, qRows)
}

func probeVulkanTextAttentionOutAddRMSNormF32() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, bias := probeVulkanTextOutWeights(qRows)
	return probeVulkanTextAttentionOutAddRMSNorm("text_attention_out_add_rmsnorm_f32", qRows, func(attOut []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecBias(attOut, head, w, bias, qRows, qRows)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextAttentionOutAddRMSNormF32(normOut, residual, q, k, v, w, bias, normWeight, 6, cacheLen, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextAttentionOutAddRMSNormQ8() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ8Row(w, qRows, qRows)
	return probeVulkanTextAttentionOutAddRMSNorm("text_attention_out_add_rmsnorm_q8", qRows, func(attOut []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ8(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ8(normOut, residual, q, k, v, qw, normWeight, 7, cacheLen, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextAttentionOutAddRMSNormQ6() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ6Row(w, qRows, qRows)
	return probeVulkanTextAttentionOutAddRMSNorm("text_attention_out_add_rmsnorm_q6", qRows, func(attOut []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ6(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ6(normOut, residual, q, k, v, qw, normWeight, 8, cacheLen, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextAttentionOutAddRMSNormQ4() error {
	q, k, v, cacheLen, numHeads, kvHeads, headDim := probeVulkanTextAttentionInput()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ4Row(w, qRows, qRows)
	return probeVulkanTextAttentionOutAddRMSNorm("text_attention_out_add_rmsnorm_q4", qRows, func(attOut []float32) {
		head := make([]float32, qRows)
		textAttentionProbe(head, q, k, v, cacheLen, numHeads, kvHeads, headDim)
		tensor.MatVecQ4(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextAttentionOutAddRMSNormQ4(normOut, residual, q, k, v, qw, normWeight, 9, cacheLen, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextFirstTokenValueOutAddRMSNormF32() error {
	k, v, numHeads, kvHeads, headDim := probeVulkanTextFirstTokenKV()
	qRows := numHeads * headDim
	w, bias := probeVulkanTextOutWeights(qRows)
	return probeVulkanTextFirstTokenValueOutAddRMSNorm("text_first_token_value_out_add_rmsnorm_f32", qRows, numHeads, kvHeads, headDim, v, func(attOut []float32, head []float32) {
		tensor.MatVecBias(attOut, head, w, bias, qRows, qRows)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormF32(normOut, residual, k, v, w, bias, normWeight, 10, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextFirstTokenValueOutAddRMSNormQ8() error {
	k, v, numHeads, kvHeads, headDim := probeVulkanTextFirstTokenKV()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ8Row(w, qRows, qRows)
	return probeVulkanTextFirstTokenValueOutAddRMSNorm("text_first_token_value_out_add_rmsnorm_q8", qRows, numHeads, kvHeads, headDim, v, func(attOut []float32, head []float32) {
		tensor.MatVecQ8(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ8(normOut, residual, k, v, qw, normWeight, 11, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextFirstTokenValueOutAddRMSNormQ6() error {
	k, v, numHeads, kvHeads, headDim := probeVulkanTextFirstTokenKV()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ6Row(w, qRows, qRows)
	return probeVulkanTextFirstTokenValueOutAddRMSNorm("text_first_token_value_out_add_rmsnorm_q6", qRows, numHeads, kvHeads, headDim, v, func(attOut []float32, head []float32) {
		tensor.MatVecQ6(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ6(normOut, residual, k, v, qw, normWeight, 12, numHeads, kvHeads, headDim)
	})
}

func probeVulkanTextFirstTokenValueOutAddRMSNormQ4() error {
	k, v, numHeads, kvHeads, headDim := probeVulkanTextFirstTokenKV()
	qRows := numHeads * headDim
	w, _ := probeVulkanTextOutWeights(qRows)
	qw := tensor.QuantizeQ4Row(w, qRows, qRows)
	return probeVulkanTextFirstTokenValueOutAddRMSNorm("text_first_token_value_out_add_rmsnorm_q4", qRows, numHeads, kvHeads, headDim, v, func(attOut []float32, head []float32) {
		tensor.MatVecQ4(attOut, head, qw)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanTextFirstTokenValueOutAddRMSNormQ4(normOut, residual, k, v, qw, normWeight, 13, numHeads, kvHeads, headDim)
	})
}

func probeVulkanFusedQKVF32() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	wantA, wantB, wantC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec3(wantA, wantB, wantC, x, wa, wb, wc, rowsA, rowsB, rowsC, cols)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3F32(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, rowsC, cols); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_f32", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVQ8() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)
	wantA, wantB, wantC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec3Q8(wantA, wantB, wantC, x, qa, qb, qc)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3Q8(outA, outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_q8", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVQ6() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ6Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ6Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ6Row(wc, rowsC, cols)
	wantA, wantB, wantC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec3Q6(wantA, wantB, wantC, x, qa, qb, qc)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3Q6(outA, outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_q6", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVQ4() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ4Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ4Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ4Row(wc, rowsC, cols)
	wantA, wantB, wantC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec3Q4(wantA, wantB, wantC, x, qa, qb, qc)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3Q4(outA, outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_q4", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVMRoPEF32() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	wantA, wantB, wantC := probeFusedQKVMRoPEWantF32(x, wa, wb, wc, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3MRoPEF32(outA, outB, outC, x, wa, wb, wc, cosTable, sinTable, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_mrope_f32", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVMRoPEQ8() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)
	wantA, wantB, wantC := probeFusedQKVMRoPEWantQ8(x, qa, qb, qc, qHeads, kvHeads, headDim, cosTable, sinTable)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3MRoPEQ8(outA, outB, outC, x, qa, qb, qc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_mrope_q8", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVMRoPEQ6() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ6Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ6Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ6Row(wc, rowsC, cols)
	wantA, wantB, wantC := probeFusedQKVMRoPEWantQ6(x, qa, qb, qc, qHeads, kvHeads, headDim, cosTable, sinTable)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3MRoPEQ6(outA, outB, outC, x, qa, qb, qc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_mrope_q6", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedQKVMRoPEQ4() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ4Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ4Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ4Row(wc, rowsC, cols)
	wantA, wantB, wantC := probeFusedQKVMRoPEWantQ4(x, qa, qb, qc, qHeads, kvHeads, headDim, cosTable, sinTable)
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec3MRoPEQ4(outA, outB, outC, x, qa, qb, qc, cosTable, sinTable, qHeads, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec3("fused_qkv_mrope_q4", outA, outB, outC, wantA, wantB, wantC)
}

func probeVulkanFusedKVF32() error {
	x, _, wb, wc, _, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	wantB, wantC := make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec2(wantB, wantC, x, wb, wc, rowsB, rowsC, cols)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2F32(outB, outC, x, wb, wc, rowsB, rowsC, cols); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_f32", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVQ8() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)
	wantB, wantC := make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec2Q8(wantB, wantC, x, qb, qc)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2Q8(outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_q8", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVQ6() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ6Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ6Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ6Row(wc, rowsC, cols)
	wantB, wantC := make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec2Q6(wantB, wantC, x, qb, qc)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2Q6(outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_q6", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVQ4() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols := probeVulkanFusedMatVecInput()
	qa := tensor.QuantizeQ4Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ4Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ4Row(wc, rowsC, cols)
	wantB, wantC := make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec2Q4(wantB, wantC, x, qb, qc)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2Q4(outB, outC, x, qa, qb, qc); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_q4", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVMRoPEF32() error {
	x, wa, wb, wc, _, rowsB, rowsC, cols, _, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	wantB, wantC := probeFusedKVMRoPEWantF32(x, wb, wc, rowsB, rowsC, cols, kvHeads, headDim, cosTable, sinTable)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2MRoPEF32(outB, outC, x, wa, wb, wc, cosTable, sinTable, rowsB, rowsC, cols, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_mrope_f32", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVMRoPEQ8() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, _, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)
	wantB, wantC := probeFusedKVMRoPEWantQ8(x, qb, qc, kvHeads, headDim, cosTable, sinTable)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2MRoPEQ8(outB, outC, x, qa, qb, qc, cosTable, sinTable, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_mrope_q8", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVMRoPEQ6() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, _, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ6Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ6Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ6Row(wc, rowsC, cols)
	wantB, wantC := probeFusedKVMRoPEWantQ6(x, qb, qc, kvHeads, headDim, cosTable, sinTable)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2MRoPEQ6(outB, outC, x, qa, qb, qc, cosTable, sinTable, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_mrope_q6", outB, outC, wantB, wantC)
}

func probeVulkanFusedKVMRoPEQ4() error {
	x, wa, wb, wc, rowsA, rowsB, rowsC, cols, _, kvHeads, headDim, cosTable, sinTable := probeVulkanFusedMRoPEInput()
	qa := tensor.QuantizeQ4Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ4Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ4Row(wc, rowsC, cols)
	wantB, wantC := probeFusedKVMRoPEWantQ4(x, qb, qc, kvHeads, headDim, cosTable, sinTable)
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	if err := VulkanFusedMatVec2MRoPEQ4(outB, outC, x, qa, qb, qc, cosTable, sinTable, kvHeads, headDim); err != nil {
		return err
	}
	return compareFusedMatVec2("fused_kv_mrope_q4", outB, outC, wantB, wantC)
}

func probeVulkanMatVecAddRMSNormF32() error {
	x := probeVulkanX()
	w := probeVulkanW()
	rows, cols := 4, 5
	residual := probeVulkanResidual(rows)
	normWeight := probeVulkanNormWeight(rows)
	wantOut, wantResidual := probeMatVecAddRMSNormWant(x, w, nil, nil, nil, rows, cols, residual, normWeight)
	out := make([]float32, rows)
	if err := VulkanMatVecAddRMSNormF32(out, residual, x, w, normWeight, rows, cols); err != nil {
		return err
	}
	if err := compareFloat32("matvec_add_rmsnorm_f32/out", out, wantOut, 1e-4); err != nil {
		return err
	}
	return compareFloat32("matvec_add_rmsnorm_f32/residual", residual, wantResidual, 1e-4)
}

func probeVulkanMatVecAddRMSNormQ8() error {
	x := probeVulkanX()
	rows, cols := 4, 5
	q := tensor.QuantizeQ8Row(probeVulkanW(), rows, cols)
	residual := probeVulkanResidual(rows)
	normWeight := probeVulkanNormWeight(rows)
	wantOut, wantResidual := probeMatVecAddRMSNormWant(x, nil, q, nil, nil, rows, cols, residual, normWeight)
	out := make([]float32, rows)
	if err := VulkanMatVecAddRMSNormQ8(out, residual, x, q, normWeight); err != nil {
		return err
	}
	if err := compareFloat32("matvec_add_rmsnorm_q8/out", out, wantOut, 1e-4); err != nil {
		return err
	}
	return compareFloat32("matvec_add_rmsnorm_q8/residual", residual, wantResidual, 1e-4)
}

func probeVulkanMatVecAddRMSNormQ6() error {
	x := probeVulkanX()
	rows, cols := 4, 5
	q := tensor.QuantizeQ6Row(probeVulkanW(), rows, cols)
	residual := probeVulkanResidual(rows)
	normWeight := probeVulkanNormWeight(rows)
	wantOut, wantResidual := probeMatVecAddRMSNormWant(x, nil, nil, q, nil, rows, cols, residual, normWeight)
	out := make([]float32, rows)
	if err := VulkanMatVecAddRMSNormQ6(out, residual, x, q, normWeight); err != nil {
		return err
	}
	if err := compareFloat32("matvec_add_rmsnorm_q6/out", out, wantOut, 1e-4); err != nil {
		return err
	}
	return compareFloat32("matvec_add_rmsnorm_q6/residual", residual, wantResidual, 1e-4)
}

func probeVulkanMatVecAddRMSNormQ4() error {
	x := probeVulkanX()
	rows, cols := 4, 5
	q := tensor.QuantizeQ4Row(probeVulkanW(), rows, cols)
	residual := probeVulkanResidual(rows)
	normWeight := probeVulkanNormWeight(rows)
	wantOut, wantResidual := probeMatVecAddRMSNormWant(x, nil, nil, nil, q, rows, cols, residual, normWeight)
	out := make([]float32, rows)
	if err := VulkanMatVecAddRMSNormQ4(out, residual, x, q, normWeight); err != nil {
		return err
	}
	if err := compareFloat32("matvec_add_rmsnorm_q4/out", out, wantOut, 1e-4); err != nil {
		return err
	}
	return compareFloat32("matvec_add_rmsnorm_q4/residual", residual, wantResidual, 1e-4)
}

func probeVulkanSwiGLUGateUpF32() error {
	x, gate, up, _, rows, cols, _ := probeVulkanSwiGLUInput()
	want := make([]float32, rows)
	tensor.FusedSwiGLUF32Scratch(make([]float32, 2), x, gate, up, make([]float32, 2*rows), rows, cols, 2, want)
	out := make([]float32, rows)
	if err := VulkanSwiGLUGateUpF32(out, x, gate, up, rows, cols); err != nil {
		return err
	}
	return compareFloat32("swiglu_gate_up_f32", out, want, 1e-4)
}

func probeVulkanSwiGLUGateUpQ8() error {
	x, gateW, upW, _, rows, cols, _ := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ8Row(gateW, rows, cols)
	up := tensor.QuantizeQ8Row(upW, rows, cols)
	want := probeQuantSwiGLUGateUpWant(x, rows, func(g, u []float32) {
		tensor.MatVecQ8(g, x, gate)
		tensor.MatVecQ8(u, x, up)
	})
	out := make([]float32, rows)
	if err := VulkanSwiGLUGateUpQ8(out, x, gate, up); err != nil {
		return err
	}
	return compareFloat32("swiglu_gate_up_q8", out, want, 1e-4)
}

func probeVulkanSwiGLUGateUpQ6() error {
	x, gateW, upW, _, rows, cols, _ := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ6Row(gateW, rows, cols)
	up := tensor.QuantizeQ6Row(upW, rows, cols)
	want := probeQuantSwiGLUGateUpWant(x, rows, func(g, u []float32) {
		tensor.MatVecQ6(g, x, gate)
		tensor.MatVecQ6(u, x, up)
	})
	out := make([]float32, rows)
	if err := VulkanSwiGLUGateUpQ6(out, x, gate, up); err != nil {
		return err
	}
	return compareFloat32("swiglu_gate_up_q6", out, want, 1e-4)
}

func probeVulkanSwiGLUGateUpQ4() error {
	x, gateW, upW, _, rows, cols, _ := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ4Row(gateW, rows, cols)
	up := tensor.QuantizeQ4Row(upW, rows, cols)
	want := probeQuantSwiGLUGateUpWant(x, rows, func(g, u []float32) {
		tensor.MatVecQ4(g, x, gate)
		tensor.MatVecQ4(u, x, up)
	})
	out := make([]float32, rows)
	if err := VulkanSwiGLUGateUpQ4(out, x, gate, up); err != nil {
		return err
	}
	return compareFloat32("swiglu_gate_up_q4", out, want, 1e-4)
}

func probeVulkanSwiGLUDownF32() error {
	x, gate, up, down, rows, cols, outRows := probeVulkanSwiGLUInput()
	want := make([]float32, outRows)
	tensor.FusedSwiGLUF32Scratch(want, x, gate, up, down, rows, cols, outRows, make([]float32, rows))
	out := make([]float32, outRows)
	if err := VulkanSwiGLUDownF32(out, x, gate, up, down, rows, cols, outRows); err != nil {
		return err
	}
	return compareFloat32("swiglu_down_f32", out, want, 1e-4)
}

func probeVulkanSwiGLUDownQ8() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ8Row(gateW, rows, cols)
	up := tensor.QuantizeQ8Row(upW, rows, cols)
	down := tensor.QuantizeQ8Row(downW, outRows, rows)
	want := make([]float32, outRows)
	tensor.FusedSwiGLUQ8(want, x, gate, up, down)
	out := make([]float32, outRows)
	if err := VulkanSwiGLUDownQ8(out, x, gate, up, down); err != nil {
		return err
	}
	return compareFloat32("swiglu_down_q8", out, want, 1e-4)
}

func probeVulkanSwiGLUDownQ6() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ6Row(gateW, rows, cols)
	up := tensor.QuantizeQ6Row(upW, rows, cols)
	down := tensor.QuantizeQ6Row(downW, outRows, rows)
	want := make([]float32, outRows)
	tensor.FusedSwiGLUQ6(want, x, gate, up, down)
	out := make([]float32, outRows)
	if err := VulkanSwiGLUDownQ6(out, x, gate, up, down); err != nil {
		return err
	}
	return compareFloat32("swiglu_down_q6", out, want, 1e-4)
}

func probeVulkanSwiGLUDownQ4() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ4Row(gateW, rows, cols)
	up := tensor.QuantizeQ4Row(upW, rows, cols)
	down := tensor.QuantizeQ4Row(downW, outRows, rows)
	want := make([]float32, outRows)
	tensor.FusedSwiGLUQ4(want, x, gate, up, down)
	out := make([]float32, outRows)
	if err := VulkanSwiGLUDownQ4(out, x, gate, up, down); err != nil {
		return err
	}
	return compareFloat32("swiglu_down_q4", out, want, 1e-4)
}

func probeVulkanSwiGLUDownAddRMSNormF32() error {
	x, gate, up, down, rows, cols, outRows := probeVulkanSwiGLUInput()
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_f32", outRows, func(add []float32) {
		tensor.FusedSwiGLUF32Scratch(add, x, gate, up, down, rows, cols, outRows, make([]float32, rows))
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormF32(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows)
	}, false)
}

func probeVulkanSwiGLUDownAddRMSNormF32OutOnly() error {
	x, gate, up, down, rows, cols, outRows := probeVulkanSwiGLUInput()
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_out_only_f32", outRows, func(add []float32) {
		tensor.FusedSwiGLUF32Scratch(add, x, gate, up, down, rows, cols, outRows, make([]float32, rows))
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormF32OutOnly(normOut, residual, x, gate, up, down, normWeight, rows, cols, outRows)
	}, true)
}

func probeVulkanSwiGLUDownAddRMSNormQ8() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ8Row(gateW, rows, cols)
	up := tensor.QuantizeQ8Row(upW, rows, cols)
	down := tensor.QuantizeQ8Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_q8", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ8(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ8(normOut, residual, x, gate, up, down, normWeight)
	}, false)
}

func probeVulkanSwiGLUDownAddRMSNormQ8OutOnly() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ8Row(gateW, rows, cols)
	up := tensor.QuantizeQ8Row(upW, rows, cols)
	down := tensor.QuantizeQ8Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_out_only_q8", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ8(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ8OutOnly(normOut, residual, x, gate, up, down, normWeight)
	}, true)
}

func probeVulkanSwiGLUDownAddRMSNormQ6() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ6Row(gateW, rows, cols)
	up := tensor.QuantizeQ6Row(upW, rows, cols)
	down := tensor.QuantizeQ6Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_q6", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ6(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ6(normOut, residual, x, gate, up, down, normWeight)
	}, false)
}

func probeVulkanSwiGLUDownAddRMSNormQ6OutOnly() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ6Row(gateW, rows, cols)
	up := tensor.QuantizeQ6Row(upW, rows, cols)
	down := tensor.QuantizeQ6Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_out_only_q6", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ6(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ6OutOnly(normOut, residual, x, gate, up, down, normWeight)
	}, true)
}

func probeVulkanSwiGLUDownAddRMSNormQ4() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ4Row(gateW, rows, cols)
	up := tensor.QuantizeQ4Row(upW, rows, cols)
	down := tensor.QuantizeQ4Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_q4", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ4(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ4(normOut, residual, x, gate, up, down, normWeight)
	}, false)
}

func probeVulkanSwiGLUDownAddRMSNormQ4OutOnly() error {
	x, gateW, upW, downW, rows, cols, outRows := probeVulkanSwiGLUInput()
	gate := tensor.QuantizeQ4Row(gateW, rows, cols)
	up := tensor.QuantizeQ4Row(upW, rows, cols)
	down := tensor.QuantizeQ4Row(downW, outRows, rows)
	return probeVulkanSwiGLUDownAddRMSNorm("swiglu_down_add_rmsnorm_out_only_q4", outRows, func(add []float32) {
		tensor.FusedSwiGLUQ4(add, x, gate, up, down)
	}, func(normOut, residual, normWeight []float32) error {
		return VulkanSwiGLUDownAddRMSNormQ4OutOnly(normOut, residual, x, gate, up, down, normWeight)
	}, true)
}

func probeVulkanVisionMatRowsBiasF32() error {
	xs, w, bias, rows, cols := probeVisionMatRowsBiasInput()
	out := makeProbeRows(len(xs), rows)
	if err := VulkanMatRowsBiasF32(out, xs, w, bias, rows, cols); err != nil {
		return err
	}
	want := makeProbeRows(len(xs), rows)
	tensor.MatRowsBias(want, xs, w, bias, rows, cols)
	return compareRows("vision_matrows_bias_f32", out, want, 1e-4)
}

func probeVulkanVisionMatRowsBiasAddRowsF32() error {
	xs, w, bias, rows, cols := probeVisionMatRowsBiasInput()
	add := [][]float32{
		{0.5, -1, 2, 0.25},
		{-0.75, 0.5, 0, 1.5},
		{0.25, 0.5, 0.75, 1},
	}
	out := makeProbeRows(len(xs), rows)
	if err := VulkanMatRowsBiasAddRowsF32(out, xs, w, bias, add, rows, cols); err != nil {
		return err
	}
	want := makeProbeRows(len(xs), rows)
	tensor.MatRowsBiasAddRows(want, xs, w, bias, add, rows, cols)
	return compareRows("vision_matrows_bias_add_rows_f32", out, want, 1e-4)
}

func probeVulkanVisionMatRowsBias3F32() error {
	xs := probeVisionRows5()
	wa := []float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
	}
	ba := []float32{0.25, -0.5}
	wb := []float32{
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	bb := []float32{1, -1, 0.75}
	wc := []float32{2, 0, -1, 1, -0.5}
	bc := []float32{-0.25}
	outA, outB, outC := makeProbeRows(len(xs), 2), makeProbeRows(len(xs), 3), makeProbeRows(len(xs), 1)
	if err := VulkanMatRowsBias3F32(outA, outB, outC, xs, wa, ba, wb, bb, wc, bc, 2, 3, 1, 5); err != nil {
		return err
	}
	wantA, wantB, wantC := makeProbeRows(len(xs), 2), makeProbeRows(len(xs), 3), makeProbeRows(len(xs), 1)
	tensor.MatRowsBias3(wantA, wantB, wantC, xs, wa, ba, wb, bb, wc, bc, 2, 3, 1, 5)
	if err := compareRows("vision_matrows_bias3_f32/a", outA, wantA, 1e-4); err != nil {
		return err
	}
	if err := compareRows("vision_matrows_bias3_f32/b", outB, wantB, 1e-4); err != nil {
		return err
	}
	return compareRows("vision_matrows_bias3_f32/c", outC, wantC, 1e-4)
}

func probeVulkanVisionAttentionF32() error {
	q, k, v, tokens, heads, headDim := probeVulkanVisionAttentionInput()
	out := makeProbeRows(tokens, heads*headDim)
	if err := VulkanVisionAttentionF32(out, q, k, v, tokens, heads, headDim); err != nil {
		return err
	}
	want := makeProbeRows(tokens, heads*headDim)
	visionAttentionProbe(want, q, k, v, tokens, heads, headDim)
	return compareRows("vision_attention_f32", out, want, 1e-4)
}

func probeVulkanVisionAttentionOutF32() error {
	q, k, v, tokens, heads, headDim := probeVulkanVisionAttentionInput()
	hidden := heads * headDim
	w := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
	}
	out := makeProbeRows(tokens, hidden)
	if err := VulkanVisionAttentionOutF32(out, q, k, v, w, bias, tokens, heads, headDim); err != nil {
		return err
	}
	head := makeProbeRows(tokens, hidden)
	want := makeProbeRows(tokens, hidden)
	visionAttentionProbe(head, q, k, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, w, bias, hidden, hidden)
	return compareRows("vision_attention_out_f32", out, want, 1e-4)
}

func probeVulkanVisionMatRowsGELU2F32() error {
	xs, w1, b1, w2, b2, hiddenRows, cols, outRows := probeVisionGELU2Input()
	out := makeProbeRows(len(xs), outRows)
	if err := VulkanMatRowsGELU2F32(out, xs, w1, b1, w2, b2, hiddenRows, cols, outRows); err != nil {
		return err
	}
	hidden := makeProbeRows(len(xs), hiddenRows)
	want := makeProbeRows(len(xs), outRows)
	tensor.MatRowsBias(hidden, xs, w1, b1, hiddenRows, cols)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(want, hidden, w2, b2, outRows, hiddenRows)
	return compareRows("vision_matrows_gelu2_f32", out, want, 1e-4)
}

func probeVulkanVisionProjectImageF32() error {
	const (
		gridT      = 2
		gridH      = 2
		gridW      = 4
		visionDim  = 5
		hiddenRows = 6
		outRows    = 3
	)
	tokens := gridT * gridH * gridW
	batches := gridT * (gridH / 2) * (gridW / 2)
	xs := makeProbeRows(tokens, visionDim)
	for i := range xs {
		for j := range xs[i] {
			xs[i][j] = float32((i*11+j*7)%23-11) / 9
		}
	}
	normW := []float32{1, 0.5, -0.25, 1.5, -1}
	normB := []float32{0.1, -0.2, 0.3, -0.4, 0.5}
	w1 := make([]float32, hiddenRows*visionDim*4)
	for i := range w1 {
		w1[i] = float32((i*5)%17-8) / 13
	}
	b1 := []float32{0.1, -0.2, 0.3, -0.4, 0.05, -0.15}
	w2 := make([]float32, outRows*hiddenRows)
	for i := range w2 {
		w2[i] = float32((i*3)%11-5) / 7
	}
	b2 := []float32{0.05, -0.1, 0.15}
	out := makeProbeRows(batches, outRows)
	const eps = float32(1e-5)
	if err := VulkanProjectImageF32(out, xs, normW, normB, w1, b1, w2, b2, gridT, gridH, gridW, visionDim, hiddenRows, outRows, eps); err != nil {
		return err
	}
	merged := makeProbeRows(batches, visionDim*4)
	for batch := 0; batch < batches; batch++ {
		blocksW := gridW / 2
		blocksPerT := (gridH / 2) * blocksW
		frame := batch / blocksPerT
		local := batch - frame*blocksPerT
		by := local / blocksW
		bx := local - by*blocksW
		base := frame*gridH*gridW + by*2*gridW + bx*2
		tensor.LayerNorm(merged[batch][:visionDim], xs[base], normW, normB, eps)
		tensor.LayerNorm(merged[batch][visionDim:2*visionDim], xs[base+1], normW, normB, eps)
		tensor.LayerNorm(merged[batch][2*visionDim:3*visionDim], xs[base+gridW], normW, normB, eps)
		tensor.LayerNorm(merged[batch][3*visionDim:4*visionDim], xs[base+gridW+1], normW, normB, eps)
	}
	hidden := makeProbeRows(batches, hiddenRows)
	want := makeProbeRows(batches, outRows)
	tensor.MatRowsBias(hidden, merged, w1, b1, hiddenRows, visionDim*4)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(want, hidden, w2, b2, outRows, hiddenRows)
	return compareRows("vision_project_image_f32", out, want, 2e-4)
}

func probeVulkanVisionLayerNormRowsF32() error {
	xs := probeVisionRows5()
	weight := []float32{1, 0.5, -0.25, 2, -1}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5}
	out := makeProbeRows(len(xs), 5)
	const eps = float32(3e-5)
	if err := VulkanLayerNormRowsF32(out, xs, weight, bias, len(xs), 5, eps); err != nil {
		return err
	}
	want := makeProbeRows(len(xs), 5)
	tensor.LayerNormRows(want, xs, weight, bias, eps)
	return compareRows("vision_layernorm_rows_f32", out, want, 1e-4)
}

func probeVulkanVisionAddLayerNormRowsF32() error {
	xs := probeVisionRows5()
	add := [][]float32{
		{0.5, -0.25, 1, -1, 0.75},
		{-0.5, 0.25, 0.5, 1, -0.75},
		{1, 0, -1, 0.5, -0.5},
	}
	weight := []float32{1, 0.5, -0.25, 2, -1}
	bias := []float32{0.1, -0.2, 0.3, -0.4, 0.5}
	out := makeProbeRows(len(xs), 5)
	const eps = float32(3e-5)
	if err := VulkanAddThenLayerNormRowsF32(out, xs, add, weight, bias, len(xs), 5, eps); err != nil {
		return err
	}
	want := makeProbeRows(len(xs), 5)
	tensor.AddThenLayerNormRows(want, xs, add, weight, bias, eps)
	return compareRows("vision_add_layernorm_rows_f32", out, want, 1e-4)
}

func probeVulkanVisionRoPEPairF32() error {
	gridH, gridW, heads, headDim := 2, 3, 2, 8
	tokens, hidden := gridH*gridW, heads*headDim
	q := probeVulkanVisionRows(tokens, hidden, 1, 3, 17, 0.11)
	k := probeVulkanVisionRows(tokens, hidden, 2, 5, 19, 0.09)
	cosH, sinH, cosW, sinW := probeVulkanVisionRoPETables(gridH, gridW, headDim)
	wantQ := cloneProbeRows(q)
	wantK := cloneProbeRows(k)
	visionRoPEPairProbe(wantQ, wantK, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	if err := VulkanVisionRoPEPairF32(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim); err != nil {
		return err
	}
	if err := compareRows("vision_rope_pair_f32/q", q, wantQ, 1e-4); err != nil {
		return err
	}
	return compareRows("vision_rope_pair_f32/k", k, wantK, 1e-4)
}

func probeVulkanVisionMatRowsGELU2AddLayerNormF32() error {
	xs := [][]float32{
		{1, 2, -1, 0.5, 3},
		{-1, 0.25, 2, -0.5, 1},
		{0.5, -1.5, 1, 2, -2},
	}
	residual := [][]float32{
		{0.5, -0.25, 1, -1},
		{-0.5, 0.75, -1, 0.25},
		{1, 0.5, -0.75, -0.25},
	}
	w1 := []float32{
		1, 1, 0, 0, 0.5,
		0, -1, 1, 0, -0.5,
		1, 0, 1, 0, 0.25,
		0, 1, 0, 1, -0.25,
		-0.5, 0.25, 1, -1, 0.75,
		0.5, -0.75, 0.25, 1, -1,
	}
	b1 := []float32{0.1, -0.2, 0.3, -0.4, 0.05, -0.15}
	w2 := []float32{
		1, 0, -1, 0.5, 0.25, -0.25,
		0, 1, 0.5, -0.5, 0.75, 0,
		0.25, -0.25, 0.75, 0, -0.5, 1,
		-0.75, 0.5, 0, 1, 0.25, -0.25,
	}
	b2 := []float32{0.05, -0.1, 0.15, -0.2}
	normW := []float32{1, 0.5, -0.25, 2}
	normB := []float32{0.1, -0.2, 0.3, -0.4}
	out := makeProbeRows(len(xs), 4)
	const eps = float32(3e-5)
	if err := VulkanMatRowsGELU2AddLayerNormF32(out, xs, residual, w1, b1, w2, b2, normW, normB, 6, 5, 4, eps); err != nil {
		return err
	}
	hidden := makeProbeRows(len(xs), 6)
	mlp := makeProbeRows(len(xs), 4)
	want := makeProbeRows(len(xs), 4)
	tensor.MatRowsBias(hidden, xs, w1, b1, 6, 5)
	tensor.GELUTanhRowsInPlace(hidden)
	tensor.MatRowsBias(mlp, hidden, w2, b2, 4, 6)
	tensor.AddThenLayerNormRows(want, residual, mlp, normW, normB, eps)
	return compareRows("vision_matrows_gelu2_add_layernorm_f32", out, want, 2e-4)
}

func probeVulkanVisionRoPEAttentionOutF32() error {
	gridH, gridW, heads, headDim := 2, 3, 2, 8
	tokens, hidden := gridH*gridW, heads*headDim
	q := probeVulkanVisionRows(tokens, hidden, 1, 3, 17, 0.11)
	k := probeVulkanVisionRows(tokens, hidden, 2, 5, 19, 0.09)
	v := probeVulkanVisionRows(tokens, hidden, 4, 7, 23, 0.07)
	w := make([]float32, hidden*hidden)
	bias := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*7)%29-14) * 0.025
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.015
	}
	cosH, sinH, cosW, sinW := probeVulkanVisionRoPETables(gridH, gridW, headDim)
	wantQ := cloneProbeRows(q)
	wantK := cloneProbeRows(k)
	visionRoPEPairProbe(wantQ, wantK, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	head := makeProbeRows(tokens, hidden)
	want := makeProbeRows(tokens, hidden)
	visionAttentionProbe(head, wantQ, wantK, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, w, bias, hidden, hidden)
	out := makeProbeRows(tokens, hidden)
	if err := VulkanVisionRoPEAttentionOutF32(out, q, k, v, w, bias, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim); err != nil {
		return err
	}
	return compareRows("vision_rope_attention_out_f32", out, want, 1e-4)
}

func probeVulkanVisionQKVRoPEAttentionOutF32() error {
	gridH, gridW, heads, headDim := 2, 3, 2, 8
	tokens, hidden := gridH*gridW, heads*headDim
	x := makeProbeRows(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			x[i][j] = float32((i+2)*(j+3)%31-15) * 0.04
		}
	}
	qw, qb := probeVulkanVisionLinear(hidden, 0.017, 0.011)
	kw, kb := probeVulkanVisionLinear(hidden, 0.019, 0.013)
	vw, vb := probeVulkanVisionLinear(hidden, 0.023, 0.007)
	ow, ob := probeVulkanVisionLinear(hidden, 0.015, 0.009)
	cosH, sinH, cosW, sinW := probeVulkanVisionRoPETables(gridH, gridW, headDim)
	q, k, v := makeProbeRows(tokens, hidden), makeProbeRows(tokens, hidden), makeProbeRows(tokens, hidden)
	tensor.MatRowsBias3(q, k, v, x, qw, qb, kw, kb, vw, vb, hidden, hidden, hidden, hidden)
	visionRoPEPairProbe(q, k, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim)
	head := makeProbeRows(tokens, hidden)
	want := makeProbeRows(tokens, hidden)
	visionAttentionProbe(head, q, k, v, tokens, heads, headDim)
	tensor.MatRowsBias(want, head, ow, ob, hidden, hidden)
	out := makeProbeRows(tokens, hidden)
	if err := VulkanVisionQKVRoPEAttentionOutF32(out, x, qw, qb, kw, kb, vw, vb, ow, ob, cosH, sinH, cosW, sinW, gridH, gridW, heads, headDim, hidden); err != nil {
		return err
	}
	return compareRows("vision_qkv_rope_attention_out_f32", out, want, 1e-4)
}

func probeVulkanSwiGLUDownAddRMSNorm(name string, outRows int, makeAdd func([]float32), run func(normOut, residual, normWeight []float32) error, outOnly bool) error {
	add := make([]float32, outRows)
	makeAdd(add)
	residual := probeVulkanResidual(outRows)
	wantResidual := append([]float32(nil), residual...)
	wantOut := make([]float32, outRows)
	normWeight := probeVulkanNormWeight(outRows)
	if outOnly {
		tensor.AddRMSNormOutOnly(wantOut, wantResidual, add, normWeight, 1e-6)
	} else {
		tensor.AddRMSNorm(wantOut, wantResidual, add, normWeight, 1e-6)
	}
	out := make([]float32, outRows)
	if err := run(out, residual, normWeight); err != nil {
		return err
	}
	if err := compareFloat32(name+"/out", out, wantOut, 1e-4); err != nil {
		return err
	}
	return compareFloat32(name+"/residual", residual, wantResidual, 1e-4)
}

func probeFusedQKVMRoPEWantF32(x, wa, wb, wc []float32, rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32, []float32) {
	outA, outB, outC := make([]float32, rowsA), make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec3(outA, outB, outC, x, wa, wb, wc, rowsA, rowsB, rowsC, cols)
	applyProbeMRoPE(outA, qHeads, headDim, cosTable, sinTable)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outA, outB, outC
}

func probeFusedQKVMRoPEWantQ8(x []float32, a, b, c *tensor.Q8Matrix, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32, []float32) {
	outA, outB, outC := make([]float32, a.Rows), make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec3Q8(outA, outB, outC, x, a, b, c)
	applyProbeMRoPE(outA, qHeads, headDim, cosTable, sinTable)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outA, outB, outC
}

func probeFusedQKVMRoPEWantQ6(x []float32, a, b, c *tensor.Q6Matrix, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32, []float32) {
	outA, outB, outC := make([]float32, a.Rows), make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec3Q6(outA, outB, outC, x, a, b, c)
	applyProbeMRoPE(outA, qHeads, headDim, cosTable, sinTable)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outA, outB, outC
}

func probeFusedQKVMRoPEWantQ4(x []float32, a, b, c *tensor.Q4Matrix, qHeads, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32, []float32) {
	outA, outB, outC := make([]float32, a.Rows), make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec3Q4(outA, outB, outC, x, a, b, c)
	applyProbeMRoPE(outA, qHeads, headDim, cosTable, sinTable)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outA, outB, outC
}

func probeFusedKVMRoPEWantF32(x, wb, wc []float32, rowsB, rowsC, cols, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32) {
	outB, outC := make([]float32, rowsB), make([]float32, rowsC)
	tensor.FusedMatVec2(outB, outC, x, wb, wc, rowsB, rowsC, cols)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outB, outC
}

func probeFusedKVMRoPEWantQ8(x []float32, b, c *tensor.Q8Matrix, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32) {
	outB, outC := make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec2Q8(outB, outC, x, b, c)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outB, outC
}

func probeFusedKVMRoPEWantQ6(x []float32, b, c *tensor.Q6Matrix, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32) {
	outB, outC := make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec2Q6(outB, outC, x, b, c)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outB, outC
}

func probeFusedKVMRoPEWantQ4(x []float32, b, c *tensor.Q4Matrix, kvHeads, headDim int, cosTable, sinTable []float32) ([]float32, []float32) {
	outB, outC := make([]float32, b.Rows), make([]float32, c.Rows)
	tensor.FusedMatVec2Q4(outB, outC, x, b, c)
	applyProbeMRoPE(outB, kvHeads, headDim, cosTable, sinTable)
	return outB, outC
}

func probeVulkanTextAttentionOut(name string, run func([]float32) error, wantFn func([]float32), qRows int) error {
	out := make([]float32, qRows)
	if err := run(out); err != nil {
		return err
	}
	want := make([]float32, qRows)
	wantFn(want)
	return compareFloat32(name, out, want, 1e-4)
}

func probeVulkanTextAttentionOutAddRMSNorm(name string, qRows int, makeAttOut func([]float32), run func(normOut, residual, normWeight []float32) error) error {
	attOut := make([]float32, qRows)
	makeAttOut(attOut)
	residual := probeVulkanTextResidual(qRows)
	wantResidual := append([]float32(nil), residual...)
	normWeight := probeVulkanNormWeight(qRows)
	wantNorm := make([]float32, qRows)
	tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)
	gotNorm := make([]float32, qRows)
	if err := run(gotNorm, residual, normWeight); err != nil {
		return err
	}
	if err := compareFloat32(name+"/norm", gotNorm, wantNorm, 1e-4); err != nil {
		return err
	}
	return compareFloat32(name+"/residual", residual, wantResidual, 1e-4)
}

func probeVulkanTextFirstTokenValueOutAddRMSNorm(name string, qRows, numHeads, kvHeads, headDim int, v []float32, project func(attOut, head []float32), run func(normOut, residual, normWeight []float32) error) error {
	head := make([]float32, qRows)
	group := numHeads / kvHeads
	for h := 0; h < numHeads; h++ {
		kvh := h / group
		copy(head[h*headDim:(h+1)*headDim], v[kvh*headDim:(kvh+1)*headDim])
	}
	attOut := make([]float32, qRows)
	project(attOut, head)
	residual := probeVulkanTextResidual(qRows)
	wantResidual := append([]float32(nil), residual...)
	normWeight := probeVulkanNormWeight(qRows)
	wantNorm := make([]float32, qRows)
	tensor.AddRMSNorm(wantNorm, wantResidual, attOut, normWeight, 1e-6)
	gotNorm := make([]float32, qRows)
	if err := run(gotNorm, residual, normWeight); err != nil {
		return err
	}
	if err := compareFloat32(name+"/norm", gotNorm, wantNorm, 1e-4); err != nil {
		return err
	}
	return compareFloat32(name+"/residual", residual, wantResidual, 1e-4)
}

func probeVulkanTextResidual(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*7)%13-6) * 0.08
	}
	return out
}

func textAttentionProbe(out, q, k, v []float32, cacheLen, numHeads, kvHeads, headDim int) {
	scale := float32(1 / math.Sqrt(float64(headDim)))
	group := numHeads / kvHeads
	kvDim := kvHeads * headDim
	scores := make([]float32, cacheLen)
	for h := 0; h < numHeads; h++ {
		kvh := h / group
		qBase := h * headDim
		kvHeadBase := kvh * headDim
		maxScore := float32(math.Inf(-1))
		for t := 0; t < cacheLen; t++ {
			kBase := t*kvDim + kvHeadBase
			var score float32
			for d := 0; d < headDim; d++ {
				score += q[qBase+d] * k[kBase+d]
			}
			score *= scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var denom float32
		for t := 0; t < cacheLen; t++ {
			w := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = w
			denom += w
		}
		for d := 0; d < headDim; d++ {
			var sum float32
			for t := 0; t < cacheLen; t++ {
				sum += scores[t] * v[t*kvDim+kvHeadBase+d]
			}
			out[qBase+d] = sum / denom
		}
	}
}

func visionRoPEPairProbe(q, k [][]float32, cosH, sinH, cosW, sinW []float32, gridH, gridW, heads, headDim int) {
	half := headDim / 2
	quarter := half / 2
	period := gridH * gridW
	for token := range q {
		pos := token % period
		hy := pos / gridW
		wx := pos % gridW
		for h := 0; h < heads; h++ {
			base := h * headDim
			for i := 0; i < quarter; i++ {
				rotateVisionRoPEPairProbe(q[token], k[token], base+i, base+quarter+i, cosH[hy*quarter+i], sinH[hy*quarter+i])
				rotateVisionRoPEPairProbe(q[token], k[token], base+half+i, base+half+quarter+i, cosW[wx*quarter+i], sinW[wx*quarter+i])
			}
		}
	}
}

func rotateVisionRoPEPairProbe(q, k []float32, a, b int, cs, sn float32) {
	qa, qb := q[a], q[b]
	ka, kb := k[a], k[b]
	q[a] = qa*cs - qb*sn
	q[b] = qb*cs + qa*sn
	k[a] = ka*cs - kb*sn
	k[b] = kb*cs + ka*sn
}

func visionAttentionProbe(out, q, k, v [][]float32, tokens, heads, headDim int) {
	scale := float32(1 / math.Sqrt(float64(headDim)))
	scores := make([]float32, tokens)
	for i := 0; i < tokens; i++ {
		for h := 0; h < heads; h++ {
			offset := h * headDim
			maxScore := float32(math.Inf(-1))
			for j := 0; j < tokens; j++ {
				var score float32
				for d := 0; d < headDim; d++ {
					score += q[i][offset+d] * k[j][offset+d]
				}
				score *= scale
				scores[j] = score
				if score > maxScore {
					maxScore = score
				}
			}
			var denom float32
			for j := 0; j < tokens; j++ {
				w := float32(math.Exp(float64(scores[j] - maxScore)))
				scores[j] = w
				denom += w
			}
			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < tokens; j++ {
					sum += scores[j] * v[j][offset+d]
				}
				out[i][offset+d] = sum / denom
			}
		}
	}
}

func probeVulkanX() []float32 {
	return []float32{2, -1, 0.5, 4, -3}
}

func probeVulkanW() []float32 {
	return []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
}

func probeVulkanFusedMatVecInput() ([]float32, []float32, []float32, []float32, int, int, int, int) {
	rowsA, rowsB, rowsC, cols := 4, 3, 2, 5
	x := probeVulkanX()
	makeW := func(rows, seed int) []float32 {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i+seed)%11-5)*0.125 + float32((i+seed)%3)*0.03125
		}
		return w
	}
	return x, makeW(rowsA, 1), makeW(rowsB, 4), makeW(rowsC, 7), rowsA, rowsB, rowsC, cols
}

func probeVulkanFusedMRoPEInput() ([]float32, []float32, []float32, []float32, int, int, int, int, int, int, int, []float32, []float32) {
	qHeads, kvHeads, headDim, cols := 2, 1, 32, 5
	rowsA, rowsB, rowsC := qHeads*headDim, kvHeads*headDim, kvHeads*headDim
	x := probeVulkanX()
	makeW := func(rows, seed int) []float32 {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i+seed)%19-9)*0.0625 + float32((i+seed)%5)*0.015625
		}
		return w
	}
	cosTable, sinTable := probeVulkanRoPETables(headDim)
	return x, makeW(rowsA, 2), makeW(rowsB, 5), makeW(rowsC, 8), rowsA, rowsB, rowsC, cols, qHeads, kvHeads, headDim, cosTable, sinTable
}

func probeVulkanTextAttentionInput() ([]float32, []float32, []float32, int, int, int, int) {
	cacheLen, numHeads, kvHeads, headDim := 7, 4, 2, 4
	qRows := numHeads * headDim
	kvDim := kvHeads * headDim
	q := make([]float32, qRows)
	k := make([]float32, cacheLen*kvDim)
	v := make([]float32, cacheLen*kvDim)
	for i := range q {
		q[i] = float32((i*3)%11-5) * 0.17
	}
	for i := range k {
		k[i] = float32((i*5)%13-6) * 0.11
		v[i] = float32((i*7)%17-8) * 0.09
	}
	return q, k, v, cacheLen, numHeads, kvHeads, headDim
}

func probeVulkanTextOutWeights(qRows int) ([]float32, []float32) {
	w := make([]float32, qRows*qRows)
	bias := make([]float32, qRows)
	for i := range w {
		w[i] = float32((i*5)%17-8) * 0.07
	}
	for i := range bias {
		bias[i] = float32(i%5-2) * 0.03
	}
	return w, bias
}

func probeVulkanTextFirstTokenKV() ([]float32, []float32, int, int, int) {
	numHeads, kvHeads, headDim := 4, 2, 4
	kvDim := kvHeads * headDim
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for i := range v {
		v[i] = float32((i*7)%17-8) * 0.09
		k[i] = float32((i*5)%13-6) * 0.11
	}
	return k, v, numHeads, kvHeads, headDim
}

func makeProbeRows(rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	data := make([]float32, rows*cols)
	for i := range out {
		out[i] = data[i*cols : (i+1)*cols]
	}
	return out
}

func cloneProbeRows(in [][]float32) [][]float32 {
	if len(in) == 0 {
		return nil
	}
	out := makeProbeRows(len(in), len(in[0]))
	for i := range in {
		copy(out[i], in[i])
	}
	return out
}

func probeVulkanVisionRows(rows, cols, rowSeed, colSeed, mod int, scale float32) [][]float32 {
	out := makeProbeRows(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out[i][j] = float32((i+rowSeed)*(j+colSeed)%mod-mod/2) * scale
		}
	}
	return out
}

func probeVulkanVisionRoPETables(gridH, gridW, headDim int) ([]float32, []float32, []float32, []float32) {
	quarter := headDim / 4
	cosH, sinH := probeVulkanVisionRoPETable(gridH, quarter, 0.17)
	cosW, sinW := probeVulkanVisionRoPETable(gridW, quarter, 0.23)
	return cosH, sinH, cosW, sinW
}

func probeVulkanVisionRoPETable(rows, quarter int, step float64) ([]float32, []float32) {
	cosv := make([]float32, rows*quarter)
	sinv := make([]float32, rows*quarter)
	for r := 0; r < rows; r++ {
		for i := 0; i < quarter; i++ {
			angle := float64(r+1) * float64(i+1) * step
			cosv[r*quarter+i] = float32(math.Cos(angle))
			sinv[r*quarter+i] = float32(math.Sin(angle))
		}
	}
	return cosv, sinv
}

func probeVulkanVisionLinear(hidden int, wScale, bScale float32) ([]float32, []float32) {
	w := make([]float32, hidden*hidden)
	b := make([]float32, hidden)
	for i := range w {
		w[i] = float32((i*7)%37-18) * wScale
	}
	for i := range b {
		b[i] = float32(i%11-5) * bScale
	}
	return w, b
}

func probeVisionRows5() [][]float32 {
	return [][]float32{
		{2, -1, 0.5, 4, -3},
		{-1, 3, 2, 0.25, 1},
		{0, 1, -2, 3, 0.5},
	}
}

func probeVisionMatRowsBiasInput() ([][]float32, []float32, []float32, int, int) {
	xs := probeVisionRows5()
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	bias := []float32{0.25, -0.5, 1, 2}
	return xs, w, bias, 4, 5
}

func probeVulkanVisionAttentionInput() ([][]float32, [][]float32, [][]float32, int, int, int) {
	tokens, heads, headDim := 4, 2, 4
	hidden := heads * headDim
	q := makeProbeRows(tokens, hidden)
	k := makeProbeRows(tokens, hidden)
	v := makeProbeRows(tokens, hidden)
	for i := 0; i < tokens; i++ {
		for j := 0; j < hidden; j++ {
			q[i][j] = float32((i+1)*(j+2)%7-3) * 0.25
			k[i][j] = float32((i+3)*(j+1)%11-5) * 0.2
			v[i][j] = float32((i+2)*(j+4)%13-6) * 0.15
		}
	}
	return q, k, v, tokens, heads, headDim
}

func probeVisionGELU2Input() ([][]float32, []float32, []float32, []float32, []float32, int, int, int) {
	xs := probeVisionRows5()
	w1 := []float32{
		1, 0, 0, 0, 0,
		0, 1, 1, 0, 0.5,
		0, 0, 1, 1, 0,
		-1, 2, 0, 0, 0,
	}
	b1 := []float32{0.25, -0.5, 1, -1}
	w2 := []float32{
		1, 0, 0.5, -0.25,
		0, 1, -1, 0.5,
		0.25, 0.25, 0.25, 0.25,
	}
	b2 := []float32{0.1, -0.2, 0.3}
	return xs, w1, b1, w2, b2, 4, 5, 3
}

func probeVulkanTopKInput() ([]float32, []float32, int, int, int) {
	rows, cols, k := 270, 5, 8
	x := probeVulkanX()
	w := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		base := float32((r%17)-8) * 0.125
		for c := 0; c < cols; c++ {
			w[r*cols+c] = base + float32((r+c)%7-3)*0.25
		}
	}
	setRow := func(r int, weight []float32) {
		copy(w[r*cols:(r+1)*cols], weight)
	}
	setRow(3, []float32{1, 0, 0, 0, 0})
	setRow(42, []float32{0, 0, 0, 1, 0})
	setRow(199, []float32{0, -2, 1, 1, -1})
	setRow(256, []float32{2, 0, 0, 1, -0.5})
	setRow(269, []float32{-1, 1, 0, 2, -2})
	setRow(128, []float32{0.5, -1, 0, 1, 1})
	return x, w, rows, cols, k
}

func probeVulkanVector() []float32 {
	const n = 128
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%17)-8)*0.25 + float32(i%5)*0.03125
	}
	return out
}

func probeVulkanAddVector(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%11)-5)*0.125 - float32(i%3)*0.0625
	}
	return out
}

func probeVulkanNormWeight(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 0.75 + float32(i%13)*0.03125
	}
	return out
}

func probeVulkanResidual(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i%7)-3)*0.1875 + float32(i)*0.015625
	}
	return out
}

func probeMatVecAddRMSNormWant(x, w []float32, q8 *tensor.Q8Matrix, q6 *tensor.Q6Matrix, q4 *tensor.Q4Matrix, rows, cols int, residual, normWeight []float32) ([]float32, []float32) {
	add := make([]float32, rows)
	switch {
	case q8 != nil:
		tensor.MatVecQ8(add, x, q8)
	case q6 != nil:
		tensor.MatVecQ6(add, x, q6)
	case q4 != nil:
		tensor.MatVecQ4(add, x, q4)
	default:
		tensor.MatVec(add, x, w, rows, cols)
	}
	wantResidual := append([]float32(nil), residual...)
	wantOut := make([]float32, rows)
	tensor.AddRMSNorm(wantOut, wantResidual, add, normWeight, 1e-6)
	return wantOut, wantResidual
}

func probeVulkanSwiGLUInput() ([]float32, []float32, []float32, []float32, int, int, int) {
	rows, cols, outRows := 6, 5, 3
	x := probeVulkanX()
	gate := make([]float32, rows*cols)
	up := make([]float32, rows*cols)
	down := make([]float32, outRows*rows)
	for i := range gate {
		gate[i] = float32((i%13)-6)*0.125 + float32(i%3)*0.03125
		up[i] = float32((i%11)-5)*0.09375 - float32(i%4)*0.015625
	}
	for i := range down {
		down[i] = float32((i%7)-3)*0.15625 + float32(i%5)*0.03125
	}
	return x, gate, up, down, rows, cols, outRows
}

func probeQuantSwiGLUGateUpWant(x []float32, rows int, matvecs func(g, u []float32)) []float32 {
	g := make([]float32, rows)
	u := make([]float32, rows)
	matvecs(g, u)
	for i := range g {
		g[i] = tensor.SiLU(g[i]) * u[i]
	}
	return g
}

func probeVulkanHeadVector(heads, dim int) []float32 {
	out := make([]float32, heads*dim)
	for i := range out {
		out[i] = float32((i%19)-9)*0.125 + float32(i%7)*0.015625
	}
	return out
}

func probeVulkanRoPETables(dim int) ([]float32, []float32) {
	half := dim / 2
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := 0; i < half; i++ {
		angle := float64(i+1) * 0.03125
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}
	return cosTable, sinTable
}

func applyProbeMRoPE(x []float32, heads, dim int, cosTable, sinTable []float32) {
	half := dim / 2
	for h := 0; h < heads; h++ {
		base := h * dim
		for i := 0; i < half; i++ {
			cs, sn := cosTable[i], sinTable[i]
			a, b := x[base+i], x[base+half+i]
			x[base+i] = a*cs - b*sn
			x[base+half+i] = b*cs + a*sn
		}
	}
}

func matVecLogits(x, w []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	tensor.MatVec(out, x, w, rows, cols)
	return out
}

func probeVulkanTopKFromLogits(logits []float32, k int) []VulkanTokenScore {
	out := make([]VulkanTokenScore, len(logits))
	for i, score := range logits {
		out[i] = VulkanTokenScore{Token: i, Score: score}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].Token < out[j].Token
		}
		return out[i].Score > out[j].Score
	})
	return out[:k]
}

func compareFloat32(name string, got, want []float32, tol float64) error {
	if len(got) < len(want) {
		return fmt.Errorf("%s len=%d want at least %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			return fmt.Errorf("%s[%d]=%.6f want %.6f", name, i, got[i], want[i])
		}
	}
	return nil
}

func compareFusedMatVec3(name string, gotA, gotB, gotC, wantA, wantB, wantC []float32) error {
	if err := compareFloat32(name+"/a", gotA, wantA, 1e-4); err != nil {
		return err
	}
	if err := compareFloat32(name+"/b", gotB, wantB, 1e-4); err != nil {
		return err
	}
	return compareFloat32(name+"/c", gotC, wantC, 1e-4)
}

func compareFusedMatVec2(name string, gotB, gotC, wantB, wantC []float32) error {
	if err := compareFloat32(name+"/b", gotB, wantB, 1e-4); err != nil {
		return err
	}
	return compareFloat32(name+"/c", gotC, wantC, 1e-4)
}

func compareRows(name string, got, want [][]float32, tol float64) error {
	if len(got) < len(want) {
		return fmt.Errorf("%s rows=%d want at least %d", name, len(got), len(want))
	}
	for i := range want {
		if err := compareFloat32(fmt.Sprintf("%s[%d]", name, i), got[i], want[i], tol); err != nil {
			return err
		}
	}
	return nil
}

func compareTokenScore(name string, gotToken int, gotScore float32, wantToken int, wantScore float32, tol float64) error {
	if gotToken != wantToken || math.Abs(float64(gotScore-wantScore)) > tol {
		return fmt.Errorf("%s token=%d score=%.6f want token=%d score=%.6f", name, gotToken, gotScore, wantToken, wantScore)
	}
	return nil
}

func compareVulkanTopK(name string, got, want []VulkanTokenScore, tol float64) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i].Token != want[i].Token || math.Abs(float64(got[i].Score-want[i].Score)) > tol {
			return fmt.Errorf("%s[%d]=token:%d score:%.6f want token:%d score:%.6f", name, i, got[i].Token, got[i].Score, want[i].Token, want[i].Score)
		}
	}
	return nil
}

func probeVulkanChainedRMSNormMatVecF32() error {
	x := []float32{2, -1, 0.5, 4, -3}
	normWeight := []float32{1, 1, 1, 1, 1}
	w := []float32{
		1, 0, 0, 0, 0,
		0, 1, 2, 0, 0,
		-1, 0, 0, 0.25, 1,
		0.5, 0.5, 0.5, 0.5, 0.5,
	}
	out := make([]float32, 4)
	if err := VulkanChainedRMSNormMatVecF32(out, x, normWeight, w, 5, 4, 5); err != nil {
		return err
	}
	normOut := make([]float32, 5)
	tensor.RMSNorm(normOut, x, normWeight, 1e-6)
	want := make([]float32, 4)
	tensor.MatVec(want, normOut, w, 4, 5)
	return compareFloat32("chained_rmsnorm_matvec_f32", out, want, 1e-3)
}

func probeVulkanChainedMatVecAddRMSNormMatVecF32() error {
	// matvec1: 4x5, AddRMSNorm: 4, matvec2: 3x4
	out := make([]float32, 3)
	x := []float32{1, 2, 3, 4, 5}
	w1 := []float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
	}
	residual := []float32{1, 1, 1, 1}
	normWeight := []float32{1, 1, 1, 1}
	w2 := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	if err := VulkanChainedMatVecAddRMSNormMatVecF32(out, x, residual, w1, normWeight, w2, 4, 5, 3, 4); err != nil {
		return err
	}
	return nil
}

func probeVulkanChainedRMSNormQKVMRoPEF32() error {
	n := 64
	cols := n
	rowsA := 64
	rowsB := 32
	rowsC := 32
	kvHeads := 2
	headDim := 16
	half := headDim / 2

	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.3))
	}
	normWeight := make([]float32, n)
	for i := range normWeight {
		normWeight[i] = 1.0
	}
	wa := make([]float32, rowsA*cols)
	wb := make([]float32, rowsB*cols)
	wc := make([]float32, rowsC*cols)
	for i := range wa {
		wa[i] = float32(math.Cos(float64(i)*0.1)) * 0.01
	}
	for i := range wb {
		wb[i] = float32(math.Sin(float64(i)*0.1)) * 0.01
	}
	for i := range wc {
		wc[i] = float32(math.Cos(float64(i)*0.15)) * 0.01
	}
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		angle := float64(i) / float64(half) * 0.5
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}

	outA := make([]float32, rowsA)
	outB := make([]float32, rowsB)
	outC := make([]float32, rowsC)
	if err := VulkanChainedRMSNormFusedQKVMRoPEF32(outA, outB, outC, x, normWeight, wa, wb, wc, cosTable, sinTable, n, rowsA, rowsB, rowsC, cols, kvHeads, headDim); err != nil {
		return err
	}
	return nil
}

func probeVulkanChainedRMSNormQKVMRoPEQ8() error {
	n := 64
	cols := n
	rowsA := 64
	rowsB := 32
	rowsC := 32
	kvHeads := 2
	headDim := 16
	half := headDim / 2

	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.3))
	}
	normWeight := make([]float32, n)
	for i := range normWeight {
		normWeight[i] = 1.0
	}
	wa := make([]float32, rowsA*cols)
	wb := make([]float32, rowsB*cols)
	wc := make([]float32, rowsC*cols)
	for i := range wa {
		wa[i] = float32(math.Cos(float64(i)*0.1)) * 0.01
	}
	for i := range wb {
		wb[i] = float32(math.Sin(float64(i)*0.1)) * 0.01
	}
	for i := range wc {
		wc[i] = float32(math.Cos(float64(i)*0.15)) * 0.01
	}
	qa := tensor.QuantizeQ8Row(wa, rowsA, cols)
	qb := tensor.QuantizeQ8Row(wb, rowsB, cols)
	qc := tensor.QuantizeQ8Row(wc, rowsC, cols)
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		angle := float64(i) / float64(half) * 0.5
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}

	outA := make([]float32, rowsA)
	outB := make([]float32, rowsB)
	outC := make([]float32, rowsC)
	if err := VulkanChainedRMSNormFusedQKVMRoPEQ8(outA, outB, outC, x, normWeight, cosTable, sinTable, qa, qb, qc, n, kvHeads, headDim); err != nil {
		return err
	}
	return nil
}

func probeVulkanChainedQKVAttentionOutAddRMSNormF32() error {
	// Small dims: hidden=8, numHeads=2, kvHeads=1, headDim=4
	hidden := 8
	numHeads := 2
	kvHeads := 1
	headDim := 4
	qRows := numHeads * headDim
	kvRows := kvHeads * headDim
	cacheLen := 3

	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32(i+1) * 0.1
	}
	wa := make([]float32, qRows*hidden)
	wb := make([]float32, kvRows*hidden)
	wc := make([]float32, kvRows*hidden)
	for i := range wa {
		wa[i] = 0.01
	}
	for i := range wb {
		wb[i] = 0.01
	}
	for i := range wc {
		wc[i] = 0.01
	}
	half := headDim / 2
	cosTable := make([]float32, half)
	sinTable := make([]float32, half)
	for i := range cosTable {
		angle := float64(i) / float64(half) * 0.5
		cosTable[i] = float32(math.Cos(angle))
		sinTable[i] = float32(math.Sin(angle))
	}
	w := make([]float32, qRows*qRows)
	for i := range w {
		w[i] = 0.01
	}
	bias := make([]float32, qRows)
	normWeight := make([]float32, qRows)
	for i := range normWeight {
		normWeight[i] = 1.0
	}
	kvDim := kvRows
	kCache := make([]float32, (cacheLen+1)*kvDim)
	vCache := make([]float32, (cacheLen+1)*kvDim)
	for i := range kCache {
		kCache[i] = 0.1
		vCache[i] = 0.1
	}
	normOut := make([]float32, qRows)
	residual := make([]float32, qRows)
	for i := range residual {
		residual[i] = 0.05
	}
	outK := make([]float32, kvRows)
	outV := make([]float32, kvRows)
	if err := VulkanChainedQKVMRoPEAttentionOutAddRMSNormF32(
		normOut, residual, x,
		wa, wb, wc,
		cosTable, sinTable,
		w, bias, normWeight,
		kCache, vCache,
		0, cacheLen, hidden, numHeads, kvHeads, headDim,
		outK, outV,
	); err != nil {
		return err
	}
	return nil
}
