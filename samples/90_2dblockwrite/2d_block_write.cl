
__kernel void simd_subgroup_block_write(__global int *a, __global int *out) {
  global uint* out_ui = (global uint*)out;
  uint val;
  intel_sub_group_block_write(out_ui, val);
};

__kernel void simd_subgroup_block_write2(__global int *a, __global int *out) {
  global uint* out_ui = (global uint*)out;
  uint2 val;
  intel_sub_group_block_write2(out_ui, val);
};

__kernel void simd_subgroup_block_write4(__global int *a, __global int *out) {
  global uint* out_ui = (global uint*)out;
  uint4 val;
  intel_sub_group_block_write4(out_ui, val);
};

__kernel void simd_subgroup_block_write8(__global int *a, __global int *out) {
  global uint* out_ui = (global uint*)out;
  uint8 val;
  intel_sub_group_block_write8(out_ui, val);
};