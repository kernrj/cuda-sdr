//
// Created by Rick Kern on 1/3/23.
//

#include "CudaAllocator.h"

#include <functional>
#include <stdexcept>

#include "util/CudaDevicePushPop.h"

using namespace std;

CudaAllocator::CudaAllocator(int32_t cudaDevice, cudaStream_t cudaStream, size_t alignment, bool useHostMemory)
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mAlignment(alignment),
      mUseHostMemory(useHostMemory) {
  if (mAlignment < 1) {
    throw invalid_argument("alignment must be a positive integer");
  }
}

shared_ptr<uint8_t> CudaAllocator::allocate(size_t size, size_t* sizeOut) {
  CudaDevicePushPop cudaDeviceSetter(mCudaDevice);

  if (size == 0) {
    if (sizeOut != nullptr) {
      *sizeOut = 0;
    }

    return {};
  }

  const char* memoryType = mUseHostMemory ? "host" : "GPU";
  printf(
      "CUDA [%s] memory allocation. Size [%zd] alignment [%zd] stream [%p]\n",
      memoryType,
      size,
      mAlignment,
      mCudaStream);

  uint8_t* rawBuffer = nullptr;

  const size_t usableLength = (size + mAlignment - 1) / mAlignment * mAlignment;
  const size_t allocSize = (mAlignment - 1) + usableLength;

  if (mUseHostMemory) {
    SAFE_CUDA(cudaHostAlloc(&rawBuffer, allocSize, cudaHostAllocDefault));
  } else {
    SAFE_CUDA(cudaMallocAsync(&rawBuffer, allocSize, mCudaStream));
  }

  auto startAddress =
      reinterpret_cast<uint8_t*>(reinterpret_cast<intptr_t>(rawBuffer + mAlignment - 1) / mAlignment * mAlignment);

  if (sizeOut != nullptr) {
    // The number of bytes from the start of the returned address (which may be greater than the allocated address).
    *sizeOut = usableLength;
  }

  std::function<void(uint8_t * data)> deleter;
  if (mUseHostMemory) {
    deleter = [cudaDevice = mCudaDevice, cudaStream = mCudaStream, rawBuffer](uint8_t* data) {
      CudaDevicePushPop deviceSetter(cudaDevice);
      cudaStreamSynchronize(cudaStream);
      SAFE_CUDA(cudaFreeHost(rawBuffer));
    };
  } else {
    deleter = [cudaStream = mCudaStream](uint8_t* buffer) { cudaFreeAsync(buffer, cudaStream); };
  }

  auto sptr = shared_ptr<uint8_t>(rawBuffer, deleter);
  auto alignedSptr = shared_ptr<uint8_t>(sptr, startAddress);

  return alignedSptr;
}
