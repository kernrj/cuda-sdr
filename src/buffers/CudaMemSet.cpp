//
// Created by Rick Kern on 1/14/23.
//

#include "CudaMemSet.h"

#include "util/CudaDevicePushPop.h"

CudaMemSet::CudaMemSet(int32_t cudaDevice, cudaStream_t cudaStream)
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

void CudaMemSet::memSet(void* data, uint8_t value, size_t byteCount) {
  CudaDevicePushPop deviceSetter(mCudaDevice);
  SAFE_CUDA(cudaMemsetAsync(data, value, byteCount, mCudaStream));
}
