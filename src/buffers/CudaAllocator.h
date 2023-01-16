//
// Created by Rick Kern on 1/3/23.
//

#ifndef GPUSDR_CUDAALLOCATOR_H
#define GPUSDR_CUDAALLOCATOR_H

#include <cuda_runtime.h>

#include "buffers/IAllocator.h"

class CudaAllocator : public IAllocator {
 public:
  CudaAllocator(int32_t cudaDevice, cudaStream_t cudaStream, size_t alignment, bool useHostMemory);
  ~CudaAllocator() override = default;

  std::shared_ptr<uint8_t> allocate(size_t size, size_t* sizeOut) override;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  const size_t mAlignment;
  const bool mUseHostMemory;
};

#endif  // GPUSDR_CUDAALLOCATOR_H
