package backend

// Placeholder GLSL shader sources. These are populated when Vulkan GPU
// dispatch is enabled; empty strings cause compileVulkanGLSL to fail
// gracefully at runtime rather than blocking compilation.
var (
	vulkanArgmaxF32GLSL                     = ""
	vulkanArgmaxQuantizedF32GLSL            = ""
	vulkanBlockTopKF32GLSL                  = ""
	vulkanBlockTopKQuantizedF32GLSL         = ""
	vulkanAddRMSNormF32GLSL                 = ""
	vulkanFusedQKVMRoPEF32GLSL              = ""
	vulkanFusedQKVMRoPEQ4GLSL               = ""
	vulkanFusedQKVMRoPEQ6GLSL               = ""
	vulkanFusedQKVMRoPEQ8GLSL               = ""
	vulkanMRoPEF32GLSL                      = ""
	vulkanMRoPEPairF32GLSL                  = ""
	vulkanTextAttentionOutAddRMSNormF32GLSL = ""
)

const vulkanRMSNormF32PlanGLSL = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint n = pc.rows;
  float ss = 0.0;
  for (uint i = lid; i < n; i += 256) ss += x[i] * x[i];
  scratch[lid] = ss;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  float scale = inversesqrt(scratch[0] / float(n) + 1e-6);
  for (uint i = lid; i < n; i += 256) outv[i] = x[i] * scale * w[i];
}`
