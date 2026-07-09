content = open('internal/model/model.go', 'r').read()

# Add Q8 op constant after F32 constant
old_const = 'vulkanOpChainedQKVAttentionOutAddRMSNormF32\n)'
new_const = 'vulkanOpChainedQKVAttentionOutAddRMSNormF32\n\tvulkanOpChainedQKVAttentionOutAddRMSNormQ8\n)'
if 'vulkanOpChainedQKVAttentionOutAddRMSNormQ8\n)' not in content:
    content = content.replace(old_const, new_const)

# Add Q8 name mapping
old_name = 'vulkanOpChainedQKVAttentionOutAddRMSNormF32: \"chained_qkv_attention_out_norm_f32\",\n}'
new_name = 'vulkanOpChainedQKVAttentionOutAddRMSNormF32: \"chained_qkv_attention_out_norm_f32\",\n\tvulkanOpChainedQKVAttentionOutAddRMSNormQ8:  \"chained_qkv_attention_out_norm_q8\",\n}'
if 'chained_qkv_attention_out_norm_q8' not in content:
    content = content.replace(old_name, new_name)

# Add Q8 wiring after F32 chain in attentionWithNorm
old_wire = '\t\treturn nil, true\n\t}\n\tqkvHasRoPE := false'
new_wire = '\t\treturn nil, true\n\t}\n\t// Try Q8 chain\n\tif hasRoPE && tl.q8.q != nil && tl.q6.q == nil && tl.q4.q == nil &&\n\t\trt.vulkanChainedQKVAttentionOutAddRMSNormQ8(normOut, residual, x, cache, tl, ropeCos, ropeSin, normWeight, c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim) {\n\t\treturn nil, true\n\t}\n\tqkvHasRoPE := false'
if 'vulkanChainedQKVAttentionOutAddRMSNormQ8(' not in content:
    content = content.replace(old_wire, new_wire)

# Add Q8 method before vulkanTextFirstTokenValueOutAddRMSNorm
insert_before = 'func (rt *Runtime) vulkanTextFirstTokenValueOutAddRMSNorm('
q8_method = '''// vulkanChainedQKVAttentionOutAddRMSNormQ8 is the Q8 variant.
func (rt *Runtime) vulkanChainedQKVAttentionOutAddRMSNormQ8(normOut, residual, x []float32, cache *kvCache, tl *textLayer, cosTable, sinTable, normWeight []float32, numHeads, kvHeads, headDim int) bool {
\tif !rt.vulkanOpEnabled(vulkanOpChainedQKVAttentionOutAddRMSNormQ8) || tl == nil || cache == nil || cache.len <= 0 || numHeads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim > 256 || headDim%2 != 0 {
\t\treturn false
\t}
\tqRows := numHeads * headDim
\tkvRows := kvHeads * headDim
\thidden := len(x)
\tif hidden <= 0 || qRows <= 0 || kvRows <= 0 {
\t\treturn false
\t}
\tif len(normOut) < qRows || len(residual) < qRows || len(x) < hidden || len(normWeight) < qRows {
\t\treturn false
\t}
\tif tl.q8.q == nil || tl.q8.k == nil || tl.q8.v == nil {
\t\treturn false
\t}
\tif !q8FusedMatVec3ShapeOK(make([]float32, qRows), make([]float32, kvRows), make([]float32, kvRows), x, tl.q8.q, tl.q8.k, tl.q8.v, qRows, kvRows, kvRows, hidden) {
\t\treturn false
\t}
\tif !f32MatVecWeightsReady(tl.w.o, qRows, qRows) {
\t\treturn false
\t}
\tif !fusedMRoPEShapeOK(make([]float32, qRows), make([]float32, kvRows), numHeads, kvHeads, headDim, cosTable, sinTable) {
\t\treturn false
\t}
\tif !textAttentionOutWorkReady(cache.len, numHeads, headDim, qRows, true) {
\t\treturn false
\t}
\tkCache, vCache := cache.vulkanBufferSlices()
\tnewK := make([]float32, kvRows)
\tnewV := make([]float32, kvRows)
\tif err := backend.VulkanChainedQKVMRoPEAttentionOutAddRMSNormQ8(
\t\tnormOut, residual, x,
\t\ttl.q8.q, tl.q8.k, tl.q8.v,
\t\tcosTable, sinTable,
\t\ttl.w.o, rt.zeroBias(qRows), normWeight,
\t\tkCache, vCache,
\t\tcache.epoch, cache.len, hidden, numHeads, kvHeads, headDim,
\t\tnewK, newV,
\t); err == nil {
\t\tcache.append(newK, newV)
\t\treturn true
\t} else {
\t\trt.disableVulkanOp(vulkanOpChainedQKVAttentionOutAddRMSNormQ8, err)
\t}
\treturn false
}

'''
if 'func (rt *Runtime) vulkanChainedQKVAttentionOutAddRMSNormQ8(' not in content:
    content = content.replace(insert_before, q8_method + insert_before)

with open('internal/model/model.go', 'w', newline='\r\n') as f:
    f.write(content)
print('model.go updated')