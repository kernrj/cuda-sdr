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

#include "CosineSource.h"

#include <cuda_runtime.h>
#include <gsdr/gsdr.h>

#include "GSErrors.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

CosineSource::CosineSource(float sampleRate, float frequency, int32_t cudaDevice, cudaStream_t cudaStream) noexcept
    : mSampleRate(sampleRate),
      mFrequency(frequency),
      mIndexToRadiansMultiplier(static_cast<float>(2.0 * M_PI * mFrequency / mSampleRate)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mPhi(0.0f),
      mAlignment(32) {}

size_t CosineSource::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Port [%zu] is out of range", port);

  return SIZE_MAX;
}

size_t CosineSource::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(float);
}

Status CosineSource::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  const auto& output = portOutputBuffers[0];
  const size_t numOutputElements = output->range()->remaining() / sizeof(float);
  const float phiEnd = mPhi + static_cast<float>(numOutputElements) * mIndexToRadiansMultiplier;

  SAFE_CUDA_OR_RET_STATUS(
      gsdrCosineF(mPhi, phiEnd, output->writePtr<float>(), numOutputElements, mCudaDevice, mCudaStream));

  mPhi = fmod(phiEnd, 2.0f * M_PIf);

  const size_t outputNumBytes = numOutputElements * sizeof(float);
  FWD_IF_ERR(output->range()->increaseEndOffset(outputNumBytes));

  return Status_Success;
}
