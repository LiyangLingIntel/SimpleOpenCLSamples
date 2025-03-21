// build with: ocloc -internal_options "-cl-ext=+all,+cl_intel_subgroup_extended_block_read_cacheopts" -file 2d_block_read_cacheopt.cl -device pvc

enum LSC_LDCC {
  LSC_LDCC_DEFAULT = 0,
  LSC_LDCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
  LSC_LDCC_L1UC_L3C = 2,   // Override to L1 uncached and L3 cached
  LSC_LDCC_L1C_L3UC = 3,   // Override to L1 cached and L3 uncached
  LSC_LDCC_L1C_L3C = 4,    // Override to L1 cached and L3 cached
  LSC_LDCC_L1S_L3UC = 5,   // Override to L1 streaming load and L3 uncached
  LSC_LDCC_L1S_L3C = 6,    // Override to L1 streaming load and L3 cached
  LSC_LDCC_L1IAR_L3C = 7,  // Override to L1 invalidate-after-read, and L3 cached
};

// Notes: intel_subgroup_block_read_cacheopts require cachecontrol operand to be immediate
// which means we cannot pass it as a variable. We need to use the enum directly in the function call.

__kernel void subgroup_block_read_cacheopts_u8_m1k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 b = intel_subgroup_block_read_cacheopts_u8_m1k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u8_m2k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 b = intel_subgroup_block_read_cacheopts_u8_m2k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u8_m4k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 b = intel_subgroup_block_read_cacheopts_u8_m4k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u8_m8k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort16 b = intel_subgroup_block_read_cacheopts_u8_m8k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u16_m1k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 b = intel_subgroup_block_read_cacheopts_u16_m1k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u16_m2k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 b = intel_subgroup_block_read_cacheopts_u16_m2k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u16_m4k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 b = intel_subgroup_block_read_cacheopts_u16_m4k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_u16_m8k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort16 b = intel_subgroup_block_read_cacheopts_u16_m8k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_transform_u8_k32(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_cacheopts_transform_u8_k32(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_transform_u16_k16(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_cacheopts_transform_u16_k16(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_transpose_u32_k8(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_cacheopts_transpose_u32_k8(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_read_cacheopts_transpose_u64_k4(__global int *a, __global int *out, int W, int H, int P) {
  ulong4 b = intel_subgroup_block_read_cacheopts_transpose_u64_k4(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};
