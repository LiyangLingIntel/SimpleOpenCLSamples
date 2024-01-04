float bfloat16_to_float(ushort u)
{
#if defined(cl_intel_bfloat16_conversions)
    return intel_convert_as_bfloat16_float(u);
#else
    return as_float(u << 16);
#endif
}

kernel void bfloat16_naive(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_global_id(1);
    int n = get_global_id(0);

    float sum = 0;
    for (int k = 0; k < K; k++) {
        sum += bfloat16_to_float(A[m * K + k]) * bfloat16_to_float(B[k * N + n]);
    }

    C[m * N + n] = sum;
}

#if defined(cl_intel_subgroup_matrix_multiply_accumulate)

// M rows x K columns
static int __load_a_row_major_bf16_m1(global ushort* A, int rowStart, int colStart, int stride)
{
    int ret;

    int offset = rowStart * stride + colStart + get_sub_group_local_id() * 2;

    ret = as_int(vload2(0, A + offset));

    return ret; 
}

// K rows x N columns:
// Each work-item loads K values and converts to VNNI.
// Stride is in units of elements.
static int8 __load_b_row_major_bf16_k16(global ushort* B, int rowStart, int colStart, int stride)
{
    int8 ret;

    int offset = rowStart * stride + colStart + get_sub_group_local_id();

    // Note: this could probably use block loads?
    ushort row0  = B[offset]; offset += stride;
    ushort row1  = B[offset]; offset += stride;
    ushort row2  = B[offset]; offset += stride;
    ushort row3  = B[offset]; offset += stride;
    ushort row4  = B[offset]; offset += stride;
    ushort row5  = B[offset]; offset += stride;
    ushort row6  = B[offset]; offset += stride;
    ushort row7  = B[offset]; offset += stride;
    ushort row8  = B[offset]; offset += stride;
    ushort row9  = B[offset]; offset += stride;
    ushort row10 = B[offset]; offset += stride;
    ushort row11 = B[offset]; offset += stride;
    ushort row12 = B[offset]; offset += stride;
    ushort row13 = B[offset]; offset += stride;
    ushort row14 = B[offset]; offset += stride;
    ushort row15 = B[offset]; offset += stride;

    ret.s0 = as_int((ushort2)(row0,  row1 ));
    ret.s1 = as_int((ushort2)(row2,  row3 ));
    ret.s2 = as_int((ushort2)(row4,  row5 ));
    ret.s3 = as_int((ushort2)(row6,  row7 ));
    ret.s4 = as_int((ushort2)(row8,  row9 ));
    ret.s5 = as_int((ushort2)(row10, row11));
    ret.s6 = as_int((ushort2)(row12, row13));
    ret.s7 = as_int((ushort2)(row14, row15));

    return ret;
}

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(8, 1, 1)))
kernel void bfloat16_dpas_basic(global float* C, global ushort* A, global ushort* B, int K)
{
    const int N = get_global_size(0);
    int m = get_group_id(1);
    int n = get_group_id(0) * get_local_size(0);

    float sum = 0;
    for (int k = 0; k < K; k += 16) {
        int     aData = __load_a_row_major_bf16_m1(A, m, k, K);
        int8    bData = __load_b_row_major_bf16_k16(B, k, n, N);
        sum = intel_sub_group_bf16_bf16_matrix_mad_k16(aData, bData, sum);
    }

    C[m * N + n + get_sub_group_local_id()] = sum;
}

#else

#pragma message("cl_intel_subgroup_matrix_multiply_accumulate is unsupported!")

kernel void bfloat16_dpas_basic(global float* C, global ushort* A, global ushort* B, int K) {}

#endif