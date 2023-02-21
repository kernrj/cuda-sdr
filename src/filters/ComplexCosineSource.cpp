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

#include "ComplexCosineSource.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr.h>

#include "GSErrors.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

ComplexCosineSource::ComplexCosineSource(
    float sampleRate,
    float frequency,
    int32_t cudaDevice,
    cudaStream_t cudaStream) noexcept
    : mSampleRate(sampleRate),
      mFrequency(frequency),
      mIndexToRadiansMultiplier(static_cast<float>(2.0 * M_PI * mFrequency / mSampleRate)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mPhi(0.0f),
      mAlignment(32) {}

size_t ComplexCosineSource::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(port == 0, 0, "Port [%zu] is out of range", port);
  return SIZE_MAX;
}

size_t ComplexCosineSource::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(cuComplex);
}

Status ComplexCosineSource::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  const auto& output = portOutputBuffers[0];
  const size_t numOutputElements = output->range()->remaining() / sizeof(cuComplex);
  const float phiEnd = mPhi + static_cast<float>(numOutputElements) * mIndexToRadiansMultiplier;

  SAFE_CUDA_OR_RET_STATUS(
      gsdrCosineC(mPhi, phiEnd, output->writePtr<cuComplex>(), numOutputElements, mCudaDevice, mCudaStream));

  mPhi = fmod(phiEnd, 2.0f * M_PIf);

  const size_t outputNumBytes = numOutputElements * sizeof(cuComplex);
  FWD_IF_ERR(output->range()->increaseEndOffset(outputNumBytes));

  return Status_Success;
}
