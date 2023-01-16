//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_CUDABUFFERCOPIER_H
#define GPUSDR_CUDABUFFERCOPIER_H

#include <cuda_runtime.h>

#include <cstdint>

#include "buffers/IBufferCopier.h"

class CudaBufferCopier : public IBufferCopier {
 public:
  CudaBufferCopier(int32_t cudaDevice, cudaStream_t cudaStream, cudaMemcpyKind memcpyKind);
  ~CudaBufferCopier() override = default;

  void copy(void* dst, const void* src, size_t length) override;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  const cudaMemcpyKind mMemcpyKind;
};

#endif  // GPUSDR_CUDABUFFERCOPIER_H
