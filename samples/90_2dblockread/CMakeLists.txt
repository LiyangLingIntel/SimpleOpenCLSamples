add_opencl_sample(
    TEST
    NUMBER 90
    TARGET 2dblockread
    SOURCES main.cpp
    KERNELS 2d_block_read.cl 2d_block_read_cacheopt.cl)
