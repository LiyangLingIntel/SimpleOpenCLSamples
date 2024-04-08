
__kernel void subgroup_block_read_u8_m1k32v2(__global int *a, __global int *out, int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort2 b = intel_subgroup_block_read_u8_m1k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m8k32v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort4 b = intel_subgroup_block_read_u8_m2k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m4k32v2(__global int *a, __global int *out, int W, int H, int P) {
  int gid = get_global_id(0);
  int slid = get_sub_group_local_id();
  ushort8 b = intel_subgroup_block_read_u8_m4k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u8_m2k32v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort16 b = intel_subgroup_block_read_u8_m8k32v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m1k16v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort2 b = intel_subgroup_block_read_u16_m1k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m2k16v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort4 b = intel_subgroup_block_read_u16_m2k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m4k16v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort8 b = intel_subgroup_block_read_u16_m4k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_read_u16_m8k16v2(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ushort16 b = intel_subgroup_block_read_u16_m8k16v2(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transform_u8_k32(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    uint8 b = intel_subgroup_block_read_transform_u8_k32(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transform_u16_k16(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    uint8 b = intel_subgroup_block_read_transform_u16_k16(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transpose_u32_k8(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    uint8 b = intel_subgroup_block_read_transpose_u32_k8(a, W, H, P, (int2)(0, 0));
};

__kernel void subgroup_block_transpose_u64_k4(__global int *a, __global int *out, int W, int H, int P) {
    int gid = get_global_id(0);
    int slid = get_sub_group_local_id();
    ulong4 b = intel_subgroup_block_read_transpose_u64_k4(a, W, H, P, (int2)(0, 0));
};


// 