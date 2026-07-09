package backend

// Placeholder GLSL shader sources. These are populated when Vulkan GPU
// dispatch is enabled; empty strings cause compileVulkanGLSL to fail
// gracefully at runtime rather than blocking compilation.
var (
	vulkanArgmaxF32GLSL                     = vulkanArgmaxF32GLSLImpl
	vulkanArgmaxQuantizedF32GLSL            = vulkanArgmaxQuantizedF32GLSLImpl
	vulkanBlockTopKF32GLSL                  = vulkanBlockTopKF32GLSLImpl
	vulkanBlockTopKQuantizedF32GLSL         = vulkanBlockTopKQuantizedF32GLSLImpl
	vulkanAddRMSNormF32GLSL                 = vulkanAddRMSNormF32GLSLImpl
	vulkanFusedQKVMRoPEF32GLSL              = vulkanFusedQKVMRoPEF32GLSLImpl
	vulkanFusedQKVMRoPEQ4GLSL               = vulkanFusedQKVMRoPEQ4GLSLImpl
	vulkanFusedQKVMRoPEQ6GLSL               = vulkanFusedQKVMRoPEQ6GLSLImpl
	vulkanFusedQKVMRoPEQ8GLSL               = vulkanFusedQKVMRoPEQ8GLSLImpl
	vulkanMRoPEF32GLSL                      = vulkanMRoPEF32GLSLImpl
	vulkanMRoPEPairF32GLSL                  = vulkanMRoPEPairF32GLSLImpl
	vulkanTextAttentionOutAddRMSNormF32GLSL = vulkanTextAttentionOutAddRMSNormF32GLSLImpl
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
    uint idx = base + c;
    uint w = da[idx >> 2];
    uint raw = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    int v = int(raw);
    if ((raw & 0x80u) != 0u) v -= 256;
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
    uint idx = base + c;
    uint w = db[idx >> 2];
    uint raw = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    int v = int(raw);
    if ((raw & 0x80u) != 0u) v -= 256;
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
    uint idx = base + c;
    uint w = dc[idx >> 2];
    uint raw = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    int v = int(raw);
    if ((raw & 0x80u) != 0u) v -= 256;
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

const vulkanFusedQKVMRoPEQ6GLSLImpl = `#version 450
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
float q6A(uint bit) {
  uint word = da[bit >> 5];
  uint v = (word >> (bit & 31u)) & 63u;
  return float(int(v) - 32);
}
float q6B(uint bit) {
  uint word = db[bit >> 5];
  uint v = (word >> (bit & 31u)) & 63u;
  return float(int(v) - 32);
}
float q6C(uint bit) {
  uint word = dc[bit >> 5];
  uint v = (word >> (bit & 31u)) & 63u;
  return float(int(v) - 32);
}
float dotRowA(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint rowStride = ((pc.cols * 6u + 7u) >> 3) << 3;
  uint rowBits = row * rowStride;
  for (uint c = lid; c < pc.cols; c += 256) s += q6A(rowBits + c * 6u) * x[c];
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sa[row];
}
float dotRowB(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint rowStride = ((pc.cols * 6u + 7u) >> 3) << 3;
  uint rowBits = row * rowStride;
  for (uint c = lid; c < pc.cols; c += 256) s += q6B(rowBits + c * 6u) * x[c];
  scratch[lid] = s; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sb[row];
}
float dotRowC(uint row) {
  uint lid = gl_LocalInvocationID.x; float s = 0.0;
  uint rowStride = ((pc.cols * 6u + 7u) >> 3) << 3;
  uint rowBits = row * rowStride;
  for (uint c = lid; c < pc.cols; c += 256) s += q6C(rowBits + c * 6u) * x[c];
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
const vulkanFusedQKVMRoPEQ4GLSLImpl = `#version 450
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
uint nibbleStride() { return ((pc.cols + 1u) / 2u) * 2u; }
float dotRowA(uint row) {
  uint lid = gl_LocalInvocationID.x; float sum = 0.0;
  uint base = row * nibbleStride();
  for (uint c = lid; c < pc.cols; c += 256) {
    uint idx = base + c;
    uint w = da[idx >> 3];
    uint v = (w >> ((idx & 7u) * 4u)) & 15u;
    sum += float(int(v) - 8) * x[c];
  }
  scratch[lid] = sum; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sa[row];
}
float dotRowB(uint row) {
  uint lid = gl_LocalInvocationID.x; float sum = 0.0;
  uint base = row * nibbleStride();
  for (uint c = lid; c < pc.cols; c += 256) {
    uint idx = base + c;
    uint w = db[idx >> 3];
    uint v = (w >> ((idx & 7u) * 4u)) & 15u;
    sum += float(int(v) - 8) * x[c];
  }
  scratch[lid] = sum; barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  return scratch[0] * sb[row];
}
float dotRowC(uint row) {
  uint lid = gl_LocalInvocationID.x; float sum = 0.0;
  uint base = row * nibbleStride();
  for (uint c = lid; c < pc.cols; c += 256) {
    uint idx = base + c;
    uint w = dc[idx >> 3];
    uint v = (w >> ((idx & 7u) * 4u)) & 15u;
    sum += float(int(v) - 8) * x[c];
  }
  scratch[lid] = sum; barrier();
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
const vulkanAddRMSNormF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) buffer Dst { float dstv[]; };
layout(set=0,binding=1) readonly buffer Add { float addv[]; };
layout(set=0,binding=2) readonly buffer W { float w[]; };
layout(set=0,binding=3) buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint n = pc.rows;
  float v = (lid < n) ? (dstv[lid] + addv[lid]) : 0.0;
  scratch[lid] = v * v;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  float scale = inversesqrt(scratch[0] / float(n) + 1e-6);
  if (lid < n) { dstv[lid] = v; outv[lid] = v * scale * w[lid]; }
}`
const vulkanMRoPEF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer Cos { float cosT[]; };
layout(set=0,binding=2) readonly buffer Sin { float sinT[]; };
layout(set=0,binding=3) buffer O { float outv[]; };
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint dim = pc.cols;
  uint hdimHalf = dim / 2u;
  uint heads = pc.rows / dim;
  uint pairs = heads * hdimHalf;
  if (lid < pairs) {
    uint h = lid / hdimHalf;
    uint d = lid - h * hdimHalf;
    uint i1 = h * dim + d;
    uint i2 = i1 + hdimHalf;
    float a = x[i1];
    float b = x[i2];
    float c = cosT[d];
    float sn = sinT[d];
    x[i1] = a * c - b * sn;
    x[i2] = b * c + a * sn;
  }
}`
const vulkanMRoPEPairF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=0) buffer Q { float q[]; };
layout(set=0,binding=1) buffer K { float k[]; };
layout(set=0,binding=2) readonly buffer Cos { float cosT[]; };
layout(set=0,binding=3) readonly buffer Sin { float sinT[]; };
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint dim = pc.cols & 0xFFFFu;
  uint kvHeads = pc.cols >> 16;
  uint hdimHalf = dim / 2u;
  uint qPairs = pc.rows * hdimHalf;
  uint kPairs = kvHeads * hdimHalf;
  if (lid < qPairs) {
    uint h = lid / hdimHalf;
    uint d = lid - h * hdimHalf;
    uint i1 = h * dim + d;
    uint i2 = i1 + hdimHalf;
    float a = q[i1];
    float b = q[i2];
    float c = cosT[d];
    float sn = sinT[d];
    q[i1] = a * c - b * sn;
    q[i2] = b * c + a * sn;
  } else {
    uint j = lid - qPairs;
    if (j < kPairs) {
      uint h = j / hdimHalf;
      uint d = j - h * hdimHalf;
      uint i1 = h * dim + d;
      uint i2 = i1 + hdimHalf;
      float a = k[i1];
      float b = k[i2];
      float c = cosT[d];
      float sn = sinT[d];
      k[i1] = a * c - b * sn;
      k[i2] = b * c + a * sn;
    }
  }
}`
const vulkanArgmaxF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=2) readonly buffer O { float outv[]; };
layout(set=0,binding=3) buffer R { float result[]; };
shared float sval[256];
shared uint sidx[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint rows = pc.rows;
  float bestVal = -1.0/0.0;
  uint bestIdx = 0u;
  for (uint i = lid; i < rows; i += 256) {
    float v = outv[i];
    if (v > bestVal) { bestVal = v; bestIdx = i; }
  }
  sval[lid] = bestVal;
  sidx[lid] = bestIdx;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) {
      float v0 = sval[lid];
      float v1 = sval[lid + stride];
      if (v1 > v0) { sval[lid] = v1; sidx[lid] = sidx[lid + stride]; }
    }
    barrier();
  }
  if (lid == 0) {
    result[0] = sval[0];
    result[1] = float(sidx[0]);
  }
}`
const vulkanBlockTopKF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=2) readonly buffer O { float outv[]; };
layout(set=0,binding=3) buffer R { float result[]; };
shared float sval[256];
shared uint sidx[256];
shared uint taken[8]; // 256-bit mask of selected rows
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint block = gl_WorkGroupID.x;
  uint base = block * 256u;
  uint count = min(pc.rows - base, 256u);
  // initialize taken mask
  if (lid < 8u) taken[lid] = 0u;
  barrier();
  for (uint sel = 0u; sel < 64u; sel++) {
    float bestVal = -1.0/0.0;
    uint bestIdx = 0xFFFFFFFFu;
    uint i = lid;
    if (i < count) {
      uint absIdx = base + i;
      uint word = taken[i >> 5];
      if ((word & (1u << (i & 31u))) == 0u) {
        float v = outv[absIdx];
        if (v > bestVal) { bestVal = v; bestIdx = absIdx; }
      }
    }
    sval[lid] = bestVal;
    sidx[lid] = bestIdx;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) {
        float v0 = sval[lid];
        float v1 = sval[lid + stride];
        uint i0 = sidx[lid];
        uint i1 = sidx[lid + stride];
        bool take1 = (v1 > v0) || (v1 == v0 && (i0 == 0xFFFFFFFFu || (i1 != 0xFFFFFFFFu && i1 < i0)));
        if (take1) { sval[lid] = v1; sidx[lid] = i1; }
      }
      barrier();
    }
    if (lid == 0) {
      uint winner = sidx[0];
      float winVal = sval[0];
      uint off = block * 64u * 2u + sel * 2u;
      result[off] = winVal;
      result[off + 1u] = float(winner);
      if (winner != 0xFFFFFFFFu) {
        uint li = winner - base;
        taken[li >> 5] |= (1u << (li & 31u));
      }
    }
    barrier();
  }
}`
const vulkanArgmaxQuantizedF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=3) readonly buffer O { float outv[]; };
layout(set=0,binding=4) buffer R { float result[]; };
shared float sval[256];
shared uint sidx[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint rows = pc.rows;
  float bestVal = -1.0/0.0;
  uint bestIdx = 0u;
  for (uint i = lid; i < rows; i += 256) {
    float v = outv[i];
    if (v > bestVal) { bestVal = v; bestIdx = i; }
  }
  sval[lid] = bestVal;
  sidx[lid] = bestIdx;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (lid < stride) {
      float v0 = sval[lid];
      float v1 = sval[lid + stride];
      if (v1 > v0) { sval[lid] = v1; sidx[lid] = sidx[lid + stride]; }
    }
    barrier();
  }
  if (lid == 0) {
    result[0] = sval[0];
    result[1] = float(sidx[0]);
  }
}`
const vulkanBlockTopKQuantizedF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=3) readonly buffer O { float outv[]; };
layout(set=0,binding=4) buffer R { float result[]; };
shared float sval[256];
shared uint sidx[256];
shared uint taken[8];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint block = gl_WorkGroupID.x;
  uint base = block * 256u;
  uint count = min(pc.rows - base, 256u);
  if (lid < 8u) taken[lid] = 0u;
  barrier();
  for (uint sel = 0u; sel < 64u; sel++) {
    float bestVal = -1.0/0.0;
    uint bestIdx = 0xFFFFFFFFu;
    uint i = lid;
    if (i < count) {
      uint absIdx = base + i;
      uint word = taken[i >> 5];
      if ((word & (1u << (i & 31u))) == 0u) {
        float v = outv[absIdx];
        if (v > bestVal) { bestVal = v; bestIdx = absIdx; }
      }
    }
    sval[lid] = bestVal;
    sidx[lid] = bestIdx;
    barrier();
    for (uint stride = 128; stride > 0; stride >>= 1) {
      if (lid < stride) {
        float v0 = sval[lid];
        float v1 = sval[lid + stride];
        uint i0 = sidx[lid];
        uint i1 = sidx[lid + stride];
        bool take1 = (v1 > v0) || (v1 == v0 && (i0 == 0xFFFFFFFFu || (i1 != 0xFFFFFFFFu && i1 < i0)));
        if (take1) { sval[lid] = v1; sidx[lid] = i1; }
      }
      barrier();
    }
    if (lid == 0) {
      uint winner = sidx[0];
      float winVal = sval[0];
      uint off = block * 64u * 2u + sel * 2u;
      result[off] = winVal;
      result[off + 1u] = float(winner);
      if (winner != 0xFFFFFFFFu) {
        uint li = winner - base;
        taken[li >> 5] |= (1u << (li & 31u));
      }
    }
    barrier();
  }
}`
const vulkanTextAttentionOutAddRMSNormF32GLSLImpl = `#version 450
layout(local_size_x = 256) in;
layout(push_constant) uniform Push { uint rows; uint cols; } pc;
layout(set=0,binding=6) readonly buffer Proj { float projv[]; };
layout(set=0,binding=7) buffer Res { float resv[]; };
layout(set=0,binding=8) readonly buffer W { float w[]; };
layout(set=0,binding=9) writeonly buffer O { float outv[]; };
shared float scratch[256];
void main() {
  uint lid = gl_LocalInvocationID.x;
  uint n = pc.rows;
  float v = (lid < n) ? (resv[lid] + projv[lid]) : 0.0;
  scratch[lid] = v * v;
  barrier();
  for (uint stride = 128; stride > 0; stride >>= 1) { if (lid < stride) scratch[lid] += scratch[lid + stride]; barrier(); }
  float scale = inversesqrt(scratch[0] / float(n) + 1e-6);
  if (lid < n) { resv[lid] = v; outv[lid] = v * scale * w[lid]; }
}`