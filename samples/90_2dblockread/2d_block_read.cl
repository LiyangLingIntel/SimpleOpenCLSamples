
__kernel void subgroup_block_read_u8_m1k32v2(__global int *a, __global int *out,
                                             int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort2 b = intel_subgroup_block_read_u8_m1k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m8k32v2(__global int *a, __global int *out,
                                             int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort4 b = intel_subgroup_block_read_u8_m2k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m4k32v2(__global int *a, __global int *out,
                                             int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort8 b = intel_subgroup_block_read_u8_m4k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m2k32v2(__global int *a, __global int *out,
                                             int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort16 b = intel_subgroup_block_read_u8_m8k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m1k16v2(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort2 b = intel_subgroup_block_read_u16_m1k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m2k16v2(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort4 b = intel_subgroup_block_read_u16_m2k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m4k16v2(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort8 b = intel_subgroup_block_read_u16_m4k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m8k16v2(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort16 b = intel_subgroup_block_read_u16_m8k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transform_u8_k32(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  uint8 b =
      intel_subgroup_block_read_transform_u8_k32(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transform_u16_k16(__global int *a,
                                               __global int *out, int W, int H,
                                               int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  uint8 b =
      intel_subgroup_block_read_transform_u16_k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transpose_u32_k8(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  uint8 b =
      intel_subgroup_block_read_transpose_u32_k8(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transpose_u64_k4(__global int *a,
                                              __global int *out, int W, int H,
                                              int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ulong4 b =
      intel_subgroup_block_read_transpose_u64_k4(a, W, H, P, (int2)(0, 0));
};

// Define block reads which are supported by the hardware but are not in the
// headers:
typedef ushort __attribute__((ext_vector_type(32))) ushort32;
typedef ushort __attribute__((ext_vector_type(64))) ushort64;
typedef uint __attribute__((ext_vector_type(32))) uint32;

ushort __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(long baseoffset,
                                                         int width_minus_one,
                                                         int height_minus_one,
                                                         int pitch_minus_one,
                                                         int2 coord);
ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(long baseoffset,
                                                          int width_minus_one,
                                                          int height_minus_one,
                                                          int pitch_minus_one,
                                                          int2 coord);
ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(long baseoffset,
                                                          int width_minus_one,
                                                          int height_minus_one,
                                                          int pitch_minus_one,
                                                          int2 coord);
ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(long baseoffset,
                                                          int width_minus_one,
                                                          int height_minus_one,
                                                          int pitch_minus_one,
                                                          int2 coord);
ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
    long baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, int2 coord);

ushort intel_subgroup_block_read_u16_m1k16(const __global void *base_address,
                                           int width, int height, int pitch,
                                           int2 coord) {
  return __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort2 intel_subgroup_block_read_u16_m2k16(const __global void *base_address,
                                            int width, int height, int pitch,
                                            int2 coord) {
  return __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort4 intel_subgroup_block_read_u16_m4k16(const __global void *base_address,
                                            int width, int height, int pitch,
                                            int2 coord) {
  return __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort8 intel_subgroup_block_read_u16_m8k16(const __global void *base_address,
                                            int width, int height, int pitch,
                                            int2 coord) {
  return __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}
ushort16 intel_subgroup_block_read_u16_m16k16(const __global void *base_address,
                                              int width, int height, int pitch,
                                              int2 coord) {
  return __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
      as_long(base_address), width - 1, height - 1, pitch - 1, coord);
}

__kernel void subgroup_block_read_u16_m1k16(__global int *a, __global int *out,
                                            int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort b = intel_subgroup_block_read_u16_m1k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m2k16(__global int *a, __global int *out,
                                            int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort2 b = intel_subgroup_block_read_u16_m2k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m4k16(__global int *a, __global int *out,
                                            int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort4 b = intel_subgroup_block_read_u16_m4k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m8k16(__global int *a, __global int *out,
                                            int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort8 b = intel_subgroup_block_read_u16_m8k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m16k16(__global int *a, __global int *out,
                                             int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort16 b = intel_subgroup_block_read_u16_m16k16(a, W, H, P, (int2)(0, 0));
};