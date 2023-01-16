//
// Created by Rick Kern on 1/4/23.
//

#include "CudaBufferCopier.h"

#include "GSErrors.h"
#include "util/CudaDevicePushPop.h"

CudaBufferCopier::CudaBufferCopier(int32_t cudaDevice, cudaStream_t cudaStream, cudaMemcpyKind memcpyKind)
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mMemcpyKind(memcpyKind) {}

void CudaBufferCopier::copy(void* dst, const void* src, size_t length) {
  CudaDevicePushPop setAndRestore(mCudaDevice);
  SAFE_CUDA(cudaMemcpyAsync(dst, src, length, mMemcpyKind, mCudaStream));
}
