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

#ifndef GPUSDRPIPELINE_CUDAMEMORY_H
#define GPUSDRPIPELINE_CUDAMEMORY_H

#include <cuda_runtime.h>

#include "IMemory.h"

class CudaMemory final : public IMemory {
 public:
  using Deleter = void (*)(uint8_t* data, void* context, int32_t cudaDevice, cudaStream_t cudaStream) noexcept;

 public:
  CudaMemory(
      uint8_t* data,
      size_t capacity,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      Deleter deleter,
      void* deleterContext) noexcept;

  [[nodiscard]] uint8_t* data() noexcept final;
  [[nodiscard]] const uint8_t* data() const noexcept final;
  [[nodiscard]] size_t capacity() const noexcept final;
  [[nodiscard]] int32_t cudaDevice() const noexcept;
  [[nodiscard]] cudaStream_t cudaStream() const noexcept;

 private:
  uint8_t* const mData;
  const size_t mCapacity;
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  const Deleter mDeleter;
  void* const mDeleterContext;

 private:
  ~CudaMemory() final;

  REF_COUNTED_NO_DESTRUCTOR(CudaMemory);
};

#endif  // GPUSDRPIPELINE_CUDAMEMORY_H
