
void dummy_store_float(global float* C, float8 v, int rowStart, int colStart, int stride) {
  global uint* C_ui = (global uint*)C;
  uint8 v_ui = as_uint8(v);

  uint offset = rowStart * stride + colStart;

  intel_sub_group_block_write(C_ui + offset, v_ui.s0);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s1);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s2);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s3);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s4);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s5);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s6);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s7);
  offset += stride;
}

void dummy_store_int(global int* C, int8 v, int rowStart, int colStart, int stride) {
  global uint* C_ui = (global uint*)C;
  uint8 v_ui = as_uint8(v);

  uint offset = rowStart * stride + colStart;

  intel_sub_group_block_write(C_ui + offset, v_ui.s0);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s1);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s2);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s3);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s4);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s5);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s6);
  offset += stride;
  intel_sub_group_block_write(C_ui + offset, v_ui.s7);
  offset += stride;
}

kernel void sub_group_i8_i8_dpas(global int* C, global ushort* A, global ushort* B, int K) {
  int8 sum = 0;
  short8 aData;
  int8 bData;
  sum = intel_sub_group_i8_i8_matrix_mad_k32(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_int(C, sum, 0, 0, N);
}

kernel void sub_group_i8_u8_dpas(global int* C, global ushort* A, global ushort* B, int K) {
  int8 sum = 0;
  short8 aData;
  uint8 bData;
  sum = intel_sub_group_i8_u8_matrix_mad_k32(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_int(C, sum, 0, 0, N);
}

kernel void sub_group_u8_i8_dpas(global int* C, global ushort* A, global ushort* B, int K) {
  int8 sum = 0;
  ushort8 aData;
  int8 bData;
  sum = intel_sub_group_u8_i8_matrix_mad_k32(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_int(C, sum, 0, 0, N);
}

kernel void sub_group_u8_u8_dpas(global int* C, global ushort* A, global ushort* B, int K) {
  int8 sum = 0;
  ushort8 aData;
  uint8 bData;
  sum = intel_sub_group_u8_u8_matrix_mad_k32(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_int(C, sum, 0, 0, N);
}

kernel void sub_group_bf16_bf16_dpas(global float* C, global ushort* A, global ushort* B, int K) {
  float8 sum = 0;
  short8 aData;
  int8 bData;
  sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_float(C, sum, 0, 0, N);
}

kernel void sub_group_f16_f16_dpas(global float* C, global ushort* A, global ushort* B, int K) {
  float8 sum = 0;
  short8 aData;
  int8 bData;
  sum = intel_sub_group_f16_f16_matrix_mad_k16(aData, bData, sum);
  const int N = get_global_size(0);
  dummy_store_float(C, sum, 0, 0, N);
}

kernel void sub_group_tf32_tf32_dpas(global float* C, global ushort* A, global ushort* B, int K) {
  const int N = get_global_size(0);
  float8 sum = 0;
  short8 aData;
  int8 bData;
  sum = intel_sub_group_tf32_tf32_matrix_mad_k8_f32(aData, bData, sum);
  dummy_store_float(C, sum, 0, 0, N);
}