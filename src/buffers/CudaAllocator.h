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

#ifndef GPUSDR_CUDAALLOCATOR_H
#define GPUSDR_CUDAALLOCATOR_H

#include <cuda_runtime.h>

#include "buffers/IAllocator.h"

class CudaAllocator final : public IAllocator {
 public:
  CudaAllocator(int32_t cudaDevice, cudaStream_t cudaStream, size_t alignment, bool useHostMemory) noexcept;

  Result<IMemory> allocate(size_t size) noexcept final;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  const size_t mAlignment;
  const bool mUseHostMemory;

 private:
  static void cudaHostMemDeleter(uint8_t* data, void* context, int32_t cudaDevice, cudaStream_t cudaStream) noexcept;
  static void cudaGpuMemDeleter(uint8_t* data, void* context, int32_t cudaDevice, cudaStream_t cudaStream) noexcept;

  REF_COUNTED(CudaAllocator);
};

#endif  // GPUSDR_CUDAALLOCATOR_H
