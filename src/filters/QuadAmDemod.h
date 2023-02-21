/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#ifndef GPUSDR_SRC_QUADAMDEMOD_H_
#define GPUSDR_SRC_QUADAMDEMOD_H_

#include <cuda_runtime.h>

#include "Factories.h"
#include "filters/BaseFilter.h"

class QuadAmDemod final : public BaseFilter {
 public:
  static Result<Filter> create(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories) noexcept;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;

 private:
  QuadAmDemod(
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IRelocatableResizableBufferFactory* relocatableBufferFactory,
      IBufferSliceFactory* bufferSliceFactory,
      IMemSet* memSet) noexcept;

  REF_COUNTED(QuadAmDemod);
};

#endif  // GPUSDR_SRC_QUADAMDEMOD_H_
