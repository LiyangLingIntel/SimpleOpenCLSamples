// build with: ocloc -internal_options "-cl-ext=+all,+cl_intel_subgroup_extended_block_read_cacheopts" -file 2d_block_write.cl -device pvc

enum LSC_STCC {
  LSC_STCC_DEFAULT = 0,
  LSC_STCC_L1UC_L3UC = 1,  // Override to L1 uncached and L3 uncached
  LSC_STCC_L1UC_L3WB = 2,  // Override to L1 uncached and L3 written back
  LSC_STCC_L1WT_L3UC = 3,  // Override to L1 written through and L3 uncached
  LSC_STCC_L1WT_L3WB = 4,  // Override to L1 written through and L3 written back
  LSC_STCC_L1S_L3UC = 5,   // Override to L1 streaming and L3 uncached
  LSC_STCC_L1S_L3WB = 6,   // Override to L1 streaming and L3 written back
  LSC_STCC_L1WB_L3WB = 7,  // Override to L1 written through and L3 written back
};

// Notes: intel_subgroup_block_write_cacheopts require cachecontrol operand to be immediate
// which means we cannot pass it as a variable. We need to use the enum directly in the function call.

__kernel void subgroup_block_write_cacheopts_u8_m1k32v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort val;
  intel_subgroup_block_write_cacheopts_u8_m1k32v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u8_m2k32v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort2 val;
  intel_subgroup_block_write_cacheopts_u8_m2k32v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u8_m4k32v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort4 val;
  intel_subgroup_block_write_cacheopts_u8_m4k32v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u8_m8k32v1(__global int *a, __global int *out, int W, int H, int P) {
  ushort8 val;
  intel_subgroup_block_write_cacheopts_u8_m8k32v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u16_m1k16v1(__global int *a, __global int *out, int W, int H, int P,
                                                        enum LSC_STCC LSC_STCC_DEFAULT) {
  ushort val;
  intel_subgroup_block_write_cacheopts_u16_m1k16v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u16_m2k16v1(__global int *a, __global int *out, int W, int H, int P,
                                                        enum LSC_STCC LSC_STCC_DEFAULT) {
  ushort2 val;
  intel_subgroup_block_write_cacheopts_u16_m2k16v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u16_m4k16v1(__global int *a, __global int *out, int W, int H, int P,
                                                        enum LSC_STCC LSC_STCC_DEFAULT) {
  ushort4 val;
  intel_subgroup_block_write_cacheopts_u16_m4k16v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};

__kernel void subgroup_block_write_cacheopts_u16_m8k16v1(__global int *a, __global int *out, int W, int H, int P,
                                                        enum LSC_STCC LSC_STCC_DEFAULT) {
  ushort8 val;
  intel_subgroup_block_write_cacheopts_u16_m8k16v1(out, W, H, P, (int2)(0, 0), val, LSC_STCC_DEFAULT);
};