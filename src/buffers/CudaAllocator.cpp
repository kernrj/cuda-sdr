/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "CudaAllocator.h"

#include <functional>

#include "CudaMemory.h"
#include "Memory.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

CudaAllocator::CudaAllocator(int32_t cudaDevice, cudaStream_t cudaStream, size_t alignment, bool useHostMemory) noexcept
    : mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mAlignment(max<size_t>(1, alignment)),
      mUseHostMemory(useHostMemory) {}

Result<IMemory> CudaAllocator::allocate(size_t size) noexcept {
  CUDA_DEV_PUSH_POP_OR_RET_RESULT(mCudaDevice);

  if (size == 0) {
    return makeRefResultNonNull<IMemory>(new (nothrow) Memory(nullptr, 0, nullptr, nullptr));
  }

  const char* memoryType = mUseHostMemory ? "host" : "GPU";
  gslog(
      GSLOG_DEBUG,
      "CUDA [%s] memory allocation. Size [%zd] alignment [%zd] stream [%p]",
      memoryType,
      size,
      mAlignment,
      mCudaStream);

  if (size > INT64_MAX) {
    gslog(GSLOG_ERROR, "Buffer size [%zu] is too large", size);
    return ERR_RESULT(Status_InvalidArgument);
  }

  uint8_t* rawBuffer = nullptr;

  const size_t usableLength = (size + mAlignment - 1) / mAlignment * mAlignment;
  const size_t allocSize = (mAlignment - 1) + usableLength;
  cudaError_t cudaAllocResult = cudaSuccess;
  CudaMemory::Deleter deleter = nullptr;

  if (mUseHostMemory) {
    cudaAllocResult = cudaHostAlloc(&rawBuffer, allocSize, cudaHostAllocDefault);
    deleter = cudaHostMemDeleter;
  } else {
    cudaAllocResult = cudaMallocAsync(&rawBuffer, allocSize, mCudaStream);
    deleter = cudaGpuMemDeleter;
  }

  if (cudaAllocResult != cudaSuccess) {
    gslog(
        GSLOG_ERROR,
        "Failed to allocate CUDA memory with size [%zu] on GPU [%d] stream [%p] host memory? [%d]",
        allocSize,
        mCudaDevice,
        mCudaStream,
        mUseHostMemory);

    return ERR_RESULT(Status_OutOfMemory);
  }

  auto startAddress =
      reinterpret_cast<uint8_t*>(reinterpret_cast<intptr_t>(rawBuffer + mAlignment - 1) / mAlignment * mAlignment);

  IMemory* memory = new (nothrow) CudaMemory(startAddress, usableLength, mCudaDevice, mCudaStream, deleter, rawBuffer);

  return makeRefResultNonNull(memory);
}

void CudaAllocator::cudaHostMemDeleter(
    [[maybe_unused]] uint8_t* alignedAddress, // Not used when freeing. Base address (context) is used.
    void* context,  // Start-address of the allocated memory. alignedAddress can be greater.
    int32_t cudaDevice,
    cudaStream_t cudaStream) noexcept {
  void* startAddress = context;

  CUDA_DEV_PUSH_POP_OR_RET(cudaDevice, );
  cudaStreamSynchronize(cudaStream);
  SAFE_CUDA_WARN_ONLY(cudaFreeHost(startAddress));
}

void CudaAllocator::cudaGpuMemDeleter(
    [[maybe_unused]] uint8_t* alignedAddress, // Not used when freeing. Base address (context) is used.
    void* context,  // Start-address of the allocated memory. alignedAddress can be greater.
    int32_t cudaDevice,
    cudaStream_t cudaStream) noexcept {
  void* startAddress = context;
  CUDA_DEV_PUSH_POP_OR_RET(cudaDevice, );

  SAFE_CUDA_WARN_ONLY(cudaFreeAsync(startAddress, cudaStream));
}
