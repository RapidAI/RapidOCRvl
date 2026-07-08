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
	vulkanFusedQKVMRoPEF32GLSL              = vulkanFusedQKVMRoPEF32GLSLImpl
	vulkanFusedQKVMRoPEQ4GLSL               = ""
	vulkanFusedQKVMRoPEQ6GLSL               = ""
	vulkanFusedQKVMRoPEQ8GLSL               = vulkanFusedQKVMRoPEQ8GLSLImpl
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

const vulkanFusedQKVMRoPEF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rowsA; uint rowsB; uint rowsC; uint cols; uint packed; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer WA { float wa[]; };
layout(set=0,binding=2) readonly buffer WB { float wb[]; };
layout(set=0,binding=3) readonly buffer WC { float wc[]; };
layout(set=0,binding=4) readonly buffer Cos { float cosT[]; };
layout(set=0,binding=5) readonly buffer Sin { float sinT[]; };
layout(set=0,binding=6) buffer OA { float outA[]; };
layout(set=0,binding=7) buffer OB { float outB[]; };
layout(set=0,binding=8) buffer OC { float outC[]; };
shared float scratch[256];
float dotRowA(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) s += wa[row * pc.cols + c] * x[c];
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0];
}
float dotRowB(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) s += wb[row * pc.cols + c] * x[c];
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0];
}
float dotRowC(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  for (uint c = lid; c < pc.cols; c += 256) s += wc[row * pc.cols + c] * x[c];
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0];
}
void main() {
  uint g = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint hd = pc.packed & 0xFFFFu;
  uint hdimHalf = hd / 2u;
  uint qGroups = pc.rowsA / 2u;
  uint kGroups = pc.rowsB / 2u;
  if (g < qGroups) {
    uint h = g / hdimHalf;
    uint d = g - h * hdimHalf;
    uint r1 = h * hd + d;
    uint r2 = r1 + hdimHalf;
    float v1 = dotRowA(r1);
    float v2 = dotRowA(r2);
    float c = cosT[d];
    float s = sinT[d];
    if (lid == 0) { outA[r1] = v1 * c - v2 * s; outA[r2] = v1 * s + v2 * c; }
  } else if (g < qGroups + kGroups) {
    uint g2 = g - qGroups;
    uint h = g2 / hdimHalf;
    uint d = g2 - h * hdimHalf;
    uint r1 = h * hd + d;
    uint r2 = r1 + hdimHalf;
    float v1 = dotRowB(r1);
    float v2 = dotRowB(r2);
    float c = cosT[d];
    float s = sinT[d];
    if (lid == 0) { outB[r1] = v1 * c - v2 * s; outB[r2] = v1 * s + v2 * c; }
  } else {
    uint r = g - qGroups - kGroups;
    float v = dotRowC(r);
    if (lid == 0) outC[r] = v;
  }
}`

const vulkanFusedQKVMRoPEQ8GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rowsA; uint rowsB; uint rowsC; uint cols; uint packed; } pc;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer DA { uint da[]; };
layout(set=0,binding=2) readonly buffer DB { uint db[]; };
layout(set=0,binding=3) readonly buffer DC { uint dc[]; };
layout(set=0,binding=4) readonly buffer SA { float sa[]; };
layout(set=0,binding=5) readonly buffer SB { float sb[]; };
layout(set=0,binding=6) readonly buffer SC { float sc[]; };
layout(set=0,binding=7) readonly buffer Cos { float cosT[]; };
layout(set=0,binding=8) readonly buffer Sin { float sinT[]; };
layout(set=0,binding=9) buffer OA { float outA[]; };
layout(set=0,binding=10) buffer OB { float outB[]; };
layout(set=0,binding=11) buffer OC { float outC[]; };
shared float scratch[256];
float dotRowA(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint base = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    uint w = da[(base + c) >> 2];
    int v = int(bitfieldExtract(w, int(((base + c) & 3u) * 8u), 8));
    s += float(v) * x[c];
  }
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sa[row];
}
float dotRowB(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint base = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    uint w = db[(base + c) >> 2];
    int v = int(bitfieldExtract(w, int(((base + c) & 3u) * 8u), 8));
    s += float(v) * x[c];
  }
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sb[row];
}
float dotRowC(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint base = row * pc.cols;
  for (uint c = lid; c < pc.cols; c += 256) {
    uint w = dc[(base + c) >> 2];
    int v = int(bitfieldExtract(w, int(((base + c) & 3u) * 8u), 8));
    s += float(v) * x[c];
  }
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sc[row];
}
void main() {
  uint g = gl_WorkGroupID.x;
  uint lid = gl_LocalInvocationID.x;
  uint hd = pc.packed & 0xFFFFu;
  uint hdimHalf = hd / 2u;
  uint qGroups = pc.rowsA / 2u;
  uint kGroups = pc.rowsB / 2u;
  if (g < qGroups) {
    uint h = g / hdimHalf;
    uint d = g - h * hdimHalf;
    uint r1 = h * hd + d;
    uint r2 = r1 + hdimHalf;
    float v1 = dotRowA(r1);
    float v2 = dotRowA(r2);
    float c = cosT[d];
    float s = sinT[d];
    if (lid == 0) { outA[r1] = v1 * c - v2 * s; outA[r2] = v1 * s + v2 * c; }
  } else if (g < qGroups + kGroups) {
    uint g2 = g - qGroups;
    uint h = g2 / hdimHalf;
    uint d = g2 - h * hdimHalf;
    uint r1 = h * hd + d;
    uint r2 = r1 + hdimHalf;
    float v1 = dotRowB(r1);
    float v2 = dotRowB(r2);
    float c = cosT[d];
    float s = sinT[d];
    if (lid == 0) { outB[r1] = v1 * c - v2 * s; outB[r2] = v1 * s + v2 * c; }
  } else {
    uint r = g - qGroups - kGroups;
    float v = dotRowC(r);
    if (lid == 0) outC[r] = v;
  }
}`
