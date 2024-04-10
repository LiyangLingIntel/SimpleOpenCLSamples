// build with: ocloc -internal_options "-cl-ext=+all,+cl_intel_subgroup_extended_block_read_cacheopts" -file 2d_block_prefetch.cl -device pvc

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

// Notes: intel_subgroup_block_prefetch require cachecontrol operand to be immediate
// which means we cannot pass it as a variable. We need to use the enum directly in the function call.

__kernel void subgroup_block_prefetch_u8_m1k32v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u8_m1k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u8_m2k32v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u8_m2k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u8_m4k32v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u8_m4k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u8_m8k32v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u8_m8k32v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u16_m1k16v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u16_m1k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u16_m2k16v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u16_m2k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u16_m4k16v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u16_m4k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_u16_m8k16v2(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_u16_m8k16v2(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_transform_u8_k32(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_transform_u8_k32(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_transform_u16_k16(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_transform_u16_k16(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_transpose_u32_k8(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_transpose_u32_k8(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

__kernel void subgroup_block_prefetch_transpose_u64_k4(__global int *a, __global int *out, int W, int H, int P) {
  intel_subgroup_block_prefetch_transpose_u64_k4(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
};

// Define block prefetches which are supported by the hardware but are not in the headers:
// TODO: remove these when the headers are updated

// void __builtin_IB_subgroup_block_read_prefetch_u16_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v2(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                             int pitch_minus_one, int2 coord,
//                                                             enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                             int pitch_minus_one, int2 coord,
//                                                             enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                            int pitch_minus_one, int2 coord,
//                                                            enum LSC_LDCC cache_control);
// void __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one,
//                                                             int pitch_minus_one, int2 coord,
//                                                             enum LSC_LDCC cache_control);

// #define DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(FUNC_NAME, TYPE, INTERNAL_FUNC)                           \
//   TYPE FUNC_NAME(__global void *base_address, int width, int height, int pitch, int2 coord,                     \
//                  enum LSC_LDCC cache_control) {                                                                 \
//     long baseoffset = as_long(base_address);                                                                    \
//     int width_minus_one = width - 1;                                                                            \
//     int height_minus_one = height - 1;                                                                          \
//     int pitch_minus_one = pitch - 1;                                                                            \
//     return INTERNAL_FUNC(baseoffset, width_minus_one, height_minus_one, pitch_minus_one, coord, cache_control); \
//   }
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m1k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m1k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m2k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m2k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m4k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m4k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m8k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m8k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m16k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m16k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u16_m32k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u32_m8k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u32_m8k16v1);
// DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC_CACHEOPTS(intel_subgroup_block_prefetch_u32_m16k16v1, void,
//                                               __builtin_IB_subgroup_block_read_prefetch_u32_m16k16v1);

// __kernel void subgroup_block_prefetch_u16_m1k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m1k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u16_m2k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m2k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u16_m4k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m4k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u16_m8k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m8k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u16_m16k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m16k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u16_m32k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u16_m32k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u32_m8k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u32_m8k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };

// __kernel void subgroup_block_prefetch_u32_m16k16v1(__global int *a, __global int *out, int W, int H, int P) {
//   intel_subgroup_block_prefetch_u32_m16k16v1(a, W, H, P, (int2)(0, 0), LSC_LDCC_DEFAULT);
// };