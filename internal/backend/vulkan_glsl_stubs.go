package backend

// Placeholder GLSL shader sources. These are populated when Vulkan GPU
// dispatch is enabled; empty strings cause compileVulkanGLSL to fail
// gracefully at runtime rather than blocking compilation.
var (
	vulkanArgmaxF32GLSL                 = ""
	vulkanArgmaxQuantizedF32GLSL        = ""
	vulkanBlockTopKF32GLSL              = ""
	vulkanBlockTopKQuantizedF32GLSL     = ""
	vulkanAddRMSNormF32GLSL             = ""
	vulkanFusedQKVMRoPEF32GLSL          = ""
	vulkanFusedQKVMRoPEQ4GLSL           = ""
	vulkanFusedQKVMRoPEQ6GLSL           = ""
	vulkanFusedQKVMRoPEQ8GLSL           = ""
	vulkanMRoPEF32GLSL                  = ""
	vulkanMRoPEPairF32GLSL              = ""
	vulkanRMSNormF32PlanGLSL            = ""
	vulkanTextAttentionOutAddRMSNormF32GLSL = ""
)
