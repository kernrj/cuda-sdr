//
// Created by Rick Kern on 1/3/23.
//

#include "CudaAllocatorFactory.h"

#include "CudaAllocator.h"

using namespace std;

std::shared_ptr<IAllocator> CudaAllocatorFactory::createCudaAllocator(
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    size_t alignment,
    bool useHostMemory) {
  return make_shared<CudaAllocator>(cudaDevice, cudaStream, alignment, useHostMemory);
}
