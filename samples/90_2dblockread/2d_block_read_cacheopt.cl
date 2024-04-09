
enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0,
  LSC_LDCC_L1UC_L3UC = 1, // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C = 2,  // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC = 3,  // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C = 4,   // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC = 5,  // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C = 6,   // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C = 7, // Override to L1 invalidate-after-read, and L3 cached
};

__kernel void
subgroup_block_read_cacheopts_u8_m1k32v2(__global int *a, __global int *out,
                                         int W, int H, int P,
                                         enum LSC_LDCC cache_control) {
  ushort2 v = intel_subgroup_block_read_cacheopts_u8_m1k32v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u8_m1k32v2(__global int *a, __global int *out,
                                         int W, int H, int P,
                                         enum LSC_LDCC cache_control) {
  ushort2 b = intel_subgroup_block_read_cacheopts_u8_m1k32v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u8_m4k32v2(__global int *a, __global int *out,
                                         int W, int H, int P,
                                         enum LSC_LDCC cache_control) {
  ushort8 b = intel_subgroup_block_read_cacheopts_u8_m1k32v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u8_m2k32v2(__global int *a, __global int *out,
                                         int W, int H, int P,
                                         enum LSC_LDCC cache_control) {
  ushort16 b = intel_subgroup_block_read_cacheopts_u8_m8k32v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u16_m1k16v2(__global int *a, __global int *out,
                                          int W, int H, int P,
                                          enum LSC_LDCC cache_control) {
  ushort2 b = intel_subgroup_block_read_cacheopts_u16_m1k16v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u16_m2k16v2(__global int *a, __global int *out,
                                          int W, int H, int P,
                                          enum LSC_LDCC cache_control) {
  ushort4 b = intel_subgroup_block_read_cacheopts_u16_m2k16v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u16_m4k16v2(__global int *a, __global int *out,
                                          int W, int H, int P,
                                          enum LSC_LDCC cache_control) {
  ushort8 b = intel_subgroup_block_read_cacheopts_u16_m4k16v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void
subgroup_block_read_cacheopts_u16_m8k16v2(__global int *a, __global int *out,
                                          int W, int H, int P,
                                          enum LSC_LDCC cache_control) {
  ushort16 b = intel_subgroup_block_read_cacheopts_u16_m8k16v2(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void subgroup_block_transform_u8_k32(__global int *a,
                                              __global int *out, int W, int H,
                                              int P,
                                              enum LSC_LDCC cache_control) {
  uint8 b = intel_subgroup_block_read_cacheopts_transform_u8_k32(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void subgroup_block_transform_u16_k16(__global int *a,
                                               __global int *out, int W, int H,
                                               int P,
                                               enum LSC_LDCC cache_control) {
  uint8 b = intel_subgroup_block_read_cacheopts_transform_u16_k16(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void subgroup_block_transpose_u32_k8(__global int *a,
                                              __global int *out, int W, int H,
                                              int P,
                                              enum LSC_LDCC cache_control) {
  uint8 b = intel_subgroup_block_read_cacheopts_transpose_u32_k8(
      a, W, H, P, (int2)(0, 0), cache_control);
};

__kernel void subgroup_block_transpose_u64_k4(__global int *a,
                                              __global int *out, int W, int H,
                                              int P,
                                              enum LSC_LDCC cache_control) {
  ulong4 b = intel_subgroup_block_read_cacheopts_transpose_u64_k4(
      a, W, H, P, (int2)(0, 0), cache_control);
};
