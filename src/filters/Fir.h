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

class Fir : public BaseFilter {
 public:
  Fir(FirType firType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      IFactories* factories);

  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;
  void readOutput(const std::vector<std::shared_ptr<IBuffer>>& portOutputs) override;

 private:
  static const size_t mAlignment;

  const FirType mFirType;
  IFactories* const mFactories;

  const std::shared_ptr<IAllocator> mCudaAllocator;

  size_t mDecimation;
  std::shared_ptr<float> mTaps;
  int32_t mTapCount;
  cudaStream_t mCudaStream;
  int32_t mCudaDevice;
  const size_t mElementSize;

 private:
  void setTaps(const float* tapsReversed, size_t tapCount);
  [[nodiscard]] size_t getNumOutputElements() const;
  [[nodiscard]] size_t getNumInputElements() const;
};

#endif  // GPUSDR_FIRCCF_H
