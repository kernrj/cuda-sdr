//
// Created by Rick Kern on 1/3/23.
//

#ifndef GPUSDR_CUDAALLOCATORFACTORY_H
#define GPUSDR_CUDAALLOCATORFACTORY_H

#include "buffers/ICudaAllocatorFactory.h"

class CudaAllocatorFactory : public ICudaAllocatorFactory {
 public:
  ~CudaAllocatorFactory() override = default;

  std::shared_ptr<IAllocator> createCudaAllocator(
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      size_t alignment,
      bool useHostMemory) override;
};

#endif  // GPUSDR_CUDAALLOCATORFACTORY_H
