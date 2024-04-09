#include <CL/opencl.hpp>
#include <popl/popl.hpp>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "bfloat16.hpp"
#include "util.hpp"

bool zeroData = false;
bool identityData = false;
bool fixedData = false;

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

std::string makeTestName(const std::string& func, size_t U, size_t M, size_t K, size_t V, bool transpose,
                         bool transform) {
  std::ostringstream ret;
  ret << func;
  if (!transpose && !transform) {
    ret << "_u" << U << "_m" << M << "k" << K << "v" << V;
  } else {
    ret << (transpose ? "_transpose" : "");
    ret << (transform ? "_transform" : "");
    ret << "_u" << U << "_k" << K;
  }
  return ret.str();
}

template <typename T>
static void fill_matrix(std::vector<T>& M, size_t numRows, size_t numCols) {
  if (zeroData) {
    std::generate(std::begin(M), std::end(M), [&] { return 0.0f; });
  } else if (identityData) {
    std::generate(std::begin(M), std::end(M), [&] { return 1.0f; });
  } else if (fixedData) {
    for (size_t r = 0; r < numRows; r++) {
      for (size_t c = 0; c < numCols; c++) {
        M[r * numCols + c] = static_cast<float>(r + c);
      }
    }
  } else {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    std::generate(std::begin(M), std::end(M), [&] { return dist(rng); });
  }
}

template <typename T>
static void vnni_matrix(std::vector<T>& dst, const std::vector<T>& src, size_t numRows, size_t numCols, size_t factor) {
  for (size_t r = 0; r < numRows / factor; r++) {
    for (size_t c = 0; c < numCols; c++) {
      for (size_t k = 0; k < factor; k++) {
        dst[r * numCols * factor + c * factor + k] = src[(r * factor + k) * numCols + c];
      }
    }
  }
}

template <int BitWidth, int BlockM, int BlockK, int ArrayLength, bool transpose, bool transform>
static void subgroup_block_prefetch(cl::Context& context, cl::Program& program, cl::CommandQueue& queue, cl::Buffer& A,
                                    cl::Buffer& B, size_t W, size_t H, size_t P) {
  printf("%80s: ", makeTestName(__FUNCTION__, BitWidth, BlockM, BlockK, ArrayLength, transpose, transform).c_str());
  fflush(stdout);

  std::string kernelName = "subgroup_block_prefetch";
  if (transpose && transform) {
    printf("unsupported.\n");
  } else if (!transpose && !transform) {
    kernelName += "_u" + std::to_string(BitWidth);
    kernelName += "_m" + std::to_string(BlockM);
    kernelName += "k" + std::to_string(BlockK);
    kernelName += "v" + std::to_string(ArrayLength);
  } else {
    kernelName += transpose ? "_transpose" : "";
    kernelName += transform ? "_transform" : "";
    kernelName += "_u" + std::to_string(BitWidth);
    kernelName += "_k" + std::to_string(BlockK);
  }

  cl::Kernel kernel{program, kernelName.c_str()};
  if (kernel() == nullptr) {
    printf("unsupported.\n");
  } else {
    kernel.setArg(0, A);
    kernel.setArg(1, B);
    kernel.setArg(2, static_cast<cl_int>(W));
    kernel.setArg(3, static_cast<cl_int>(H));
    kernel.setArg(4, static_cast<cl_int>(P));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{W * H});
    printf(" done!\n");
  }
}

int main(int argc, char** argv) {
  int platformIndex = 0;
  int deviceIndex = 0;

  std::string fileName("2d_block_prefetch.cl");
  std::string buildOptions;
  size_t matrixSize = 256;

  {
    popl::OptionParser op("Supported Options");
    op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
    op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
    op.add<popl::Value<std::string>>("", "file", "Kernel File Name", fileName, &fileName);
    op.add<popl::Value<std::string>>("", "options", "Program Build Options", buildOptions, &buildOptions);
    op.add<popl::Value<size_t>>("m", "matrixsize", "Matrix Size", matrixSize, &matrixSize);

    bool printUsage = false;
    try {
      op.parse(argc, argv);
    } catch (std::exception& e) {
      fprintf(stderr, "Error: %s\n\n", e.what());
      printUsage = true;
    }

    if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
      fprintf(stderr,
              "Usage: 2dblockread [options]\n"
              "%s",
              op.help().c_str());
      return -1;
    }
  }

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  printf("Running on platform: %s\n", platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str());

  std::vector<cl::Device> devices;
  platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  printf("Running on device: %s\n", devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str());

  cl::Context context{devices[deviceIndex]};
  cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

  printf("Reading program source from file: %s\n", fileName.c_str());
  std::string kernelString = readStringFromFile(fileName.c_str());

  cl::Program program{context, kernelString};
  program.build();
  for (auto& device : program.getInfo<CL_PROGRAM_DEVICES>()) {
    printf("Program build log for device %s:\n", device.getInfo<CL_DEVICE_NAME>().c_str());
    printf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
  }

  const auto M = matrixSize;
  const auto N = matrixSize;
  const auto K = matrixSize;

  std::vector<bfloat16> A_vec(M * K);
  std::vector<float> C_ref(M * N);

  printf("Initializing source matrices...\n");
  fill_matrix(A_vec, M, K);

  printf("Creating source buffers...\n");
  cl::Buffer A{context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A_vec.size() * sizeof(A_vec[0]), A_vec.data()};
  cl::Buffer C{context, CL_MEM_WRITE_ONLY, C_ref.size() * sizeof(C_ref[0])};

  printf("Running tests...\n");

  subgroup_block_prefetch<8, 1, 32, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<8, 2, 32, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<8, 4, 32, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<8, 8, 32, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 1, 16, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 2, 16, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 4, 16, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 8, 16, 2, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<8, 0, 32, 0, false, true>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 0, 16, 0, false, true>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<32, 0, 8, 0, true, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<64, 0, 4, 0, true, false>(context, program, commandQueue, A, C, M, K, M);

  subgroup_block_prefetch<16, 1, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 2, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 4, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 8, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 16, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<16, 32, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<32, 8, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);
  subgroup_block_prefetch<32, 16, 16, 1, false, false>(context, program, commandQueue, A, C, M, K, M);

  printf("Done.\n");

  return 0;
}
