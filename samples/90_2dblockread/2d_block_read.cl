
__kernel void subgroup_block_read_u8_m1k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 b = intel_subgroup_block_read_u8_m1k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m8k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 b = intel_subgroup_block_read_u8_m2k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m4k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 b = intel_subgroup_block_read_u8_m4k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m2k32v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort16 b = intel_subgroup_block_read_u8_m8k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m1k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 b = intel_subgroup_block_read_u16_m1k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m2k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 b = intel_subgroup_block_read_u16_m2k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m4k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 b = intel_subgroup_block_read_u16_m4k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m8k16v2(__global int *a, __global int *out, int W, int H, int P) {
  ushort16 b = intel_subgroup_block_read_u16_m8k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_transform_u8_k32(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_transform_u8_k32(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_transform_u16_k16(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_transform_u16_k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_transpose_u32_k8(__global int *a, __global int *out, int W, int H, int P) {
  uint8 b = intel_subgroup_block_read_transpose_u32_k8(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_transpose_u64_k4(__global int *a, __global int *out, int W, int H, int P) {
  ulong4 b = intel_subgroup_block_read_transpose_u64_k4(a, W, H, P, (int2)(0, 0));
};

// Define block reads which are supported by the hardware but are not in the headers:
// TODO: remove these when the headers are updated

ushort __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(long baseoffset, int width_minus_one, int height_minus_one,
                                                         int pitch_minus_one, int2 coord);
ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(long baseoffset, int width_minus_one, int height_minus_one,
                                                          int pitch_minus_one, int2 coord);
ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(long baseoffset, int width_minus_one, int height_minus_one,
                                                          int pitch_minus_one, int2 coord);
ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long baseoffset, int width_minus_one, int height_minus_one,
                                                          int pitch_minus_one, int2 coord);
ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(long baseoffset, int width_minus_one, int height_minus_one,
                                                            int pitch_minus_one, int2 coord);

#define DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(FUNC_NAME, TYPE, INTERNAL_FUNC)                      \
  TYPE FUNC_NAME(__global void *base_address, int width, int height, int pitch, int2 coord) {    \
    long baseoffset = as_long(base_address);                                                     \
    int width_minus_one = width - 1;                                                             \
    int height_minus_one = height - 1;                                                           \
    int pitch_minus_one = pitch - 1;                                                             \
    return INTERNAL_FUNC(baseoffset, width_minus_one, height_minus_one, pitch_minus_one, coord); \
  }

DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(intel_subgroup_block_read_u16_m1k16v1, ushort,
                                    __builtin_IB_subgroup_block_read_flat_u16_m1k16v1);
DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(intel_subgroup_block_read_u16_m2k16v1, ushort2,
                                    __builtin_IB_subgroup_block_read_flat_u16_m2k16v1);
DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(intel_subgroup_block_read_u16_m4k16v1, ushort4,
                                    __builtin_IB_subgroup_block_read_flat_u16_m4k16v1);
DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(intel_subgroup_block_read_u16_m8k16v1, ushort8,
                                    __builtin_IB_subgroup_block_read_flat_u16_m8k16v1);
DEFN_INTEL_SUB_GROUP_BLOCK_READ_LSC(intel_subgroup_block_read_u16_m16k16v1, ushort16,
                                    __builtin_IB_subgroup_block_read_flat_u16_m16k16v1);

__kernel void subgroup_block_read_u16_u16_m1k16v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort b = intel_subgroup_block_read_u16_m1k16v1(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m2k16v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 b = intel_subgroup_block_read_u16_m2k16v1(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m4k16v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 b = intel_subgroup_block_read_u16_m4k16v1(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m8k16v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 b = intel_subgroup_block_read_u16_m8k16v1(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m16k16v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort16 b = intel_subgroup_block_read_u16_m16k16v1(a, W, H, P, (int2)(0, 0));
};