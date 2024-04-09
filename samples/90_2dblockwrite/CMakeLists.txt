add_opencl_sample(
    TEST
    NUMBER 90
    TARGET 2dblockwrite
    SOURCES main.cpp
    KERNELS 2d_block_write.cl 2d_block_write_cacheopt.cl)
