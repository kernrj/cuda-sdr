//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_CUDAMEMSET_H
#define GPUSDRPIPELINE_CUDAMEMSET_H

#include <cuda_runtime.h>

#include "buffers/IMemSet.h"

class CudaMemSet : public IMemSet {
 public:
  CudaMemSet(int32_t cudaDevice, cudaStream_t cudaStream);
  ~CudaMemSet() override = default;

  void memSet(void* data, uint8_t value, size_t byteCount) override;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
};

#endif  // GPUSDRPIPELINE_CUDAMEMSET_H
