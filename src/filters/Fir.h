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

#ifndef GPUSDR_FIRFF_H
#define GPUSDR_FIRFF_H

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "Factories.h"
#include "buffers/IBufferSliceFactory.h"
#include "buffers/ICudaAllocatorFactory.h"
#include "buffers/ICudaBufferCopierFactory.h"
#include "buffers/IRelocatableCudaBufferFactory.h"
#include "filters/BaseFilter.h"

class Fir final : public BaseFilter {
 public:
  static Result<Filter> create(
      SampleType tapType,
      SampleType elementType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IFactories* factories) noexcept;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  [[nodiscard]] Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

 private:
  static const size_t mAlignment;

  const SampleType mTapType;
  const SampleType mElementType;

  ConstRef<IAllocator> mAllocator;
  ConstRef<IBufferCopier> mHostToDeviceBufferCopier;

  size_t mDecimation;
  Ref<IMemory> mTaps;
  size_t mTapCount;
  cudaStream_t mCudaStream;
  int32_t mCudaDevice;
  const size_t mElementSize;

 private:
  Status setTaps(const float* tapsReversed, size_t tapCount) noexcept;
  [[nodiscard]] size_t getNumOutputElements() const noexcept;
  [[nodiscard]] size_t getNumInputElements() const noexcept;

  Fir(SampleType tapType,
      SampleType elementType,
      size_t decimation,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IAllocator* allocator,
      IBufferCopier* hostToDeviceBufferCopier,
      IRelocatableResizableBufferFactory* relocatableBufferFactory,
      IBufferSliceFactory* bufferSliceFactory,
      IMemSet* memSet) noexcept;

  REF_COUNTED(Fir);
};

#endif  // GPUSDR_FIRCCF_H
