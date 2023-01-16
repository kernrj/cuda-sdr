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

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <gsdr/gsdr.h>

#include "util/CudaDevicePushPop.h"

using namespace std;

CosineSource::CosineSource(float sampleRate, float frequency, int32_t cudaDevice, cudaStream_t cudaStream)
    : mSampleRate(sampleRate),
      mFrequency(frequency),
      mIndexToRadiansMultiplier(static_cast<float>(2.0 * M_PI * mFrequency / mSampleRate)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mPhi(0.0f),
      mAlignment(32) {}

size_t CosineSource::getOutputDataSize(size_t port) {
  if (port > 0) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  // This is arbitrary, but something > 0 so calling code allocated buffers of a non-zero size.
  return 16 * 1024 * getOutputSizeAlignment(port) * sizeof(cuComplex);
}

size_t CosineSource::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(cuComplex); }

void CosineSource::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  const auto& output = portOutputs[0];
  const size_t numOutputElements = output->range()->remaining() / sizeof(cuComplex);
  const float phiEnd = mPhi + static_cast<float>(numOutputElements) * mIndexToRadiansMultiplier;

  SAFE_CUDA(gsdrCosineC(mPhi, phiEnd, output->writePtr<cuComplex>(), numOutputElements, mCudaDevice, mCudaStream));

  mPhi = fmod(phiEnd, 2.0f * M_PIf);

  const size_t outputNumBytes = numOutputElements * sizeof(cuComplex);
  output->range()->increaseEndOffset(outputNumBytes);
}
