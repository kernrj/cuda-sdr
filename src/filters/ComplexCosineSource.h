/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_COMPLEXCOSINESOURCE_H_
#define SDRTEST_SRC_COMPLEXCOSINESOURCE_H_

#include <cuda_runtime.h>

#include "buffers/IBuffer.h"
#include "filters/Filter.h"

class ComplexCosineSource final : public virtual Source {
 public:
  ComplexCosineSource(float sampleRate, float frequency, int32_t cudaDevice, cudaStream_t cudaStream) noexcept;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;

 private:
  float mSampleRate;
  float mFrequency;
  float mIndexToRadiansMultiplier;
  int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  float mPhi;
  size_t mAlignment;

  REF_COUNTED(ComplexCosineSource);
};

#endif  // SDRTEST_SRC_COMPLEXCOSINESOURCE_H_
