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

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "Buffer.h"
#include "CosineSource.h"
#include "CudaDevicePushPop.h"

using namespace std;

__global__ void k_ComplexCosine(float indexToRadiansMultiplier, float phi, cuComplex* values) {
  const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
  const float theta = phi + static_cast<float>(x) * indexToRadiansMultiplier;

  cuComplex result;
  sincosf(theta, &result.x, &result.y);

  values[x] = result;
}

CosineSource::CosineSource(float sampleRate, float frequency, int32_t cudaDevice, cudaStream_t cudaStream)
    : mSampleRate(sampleRate), mFrequency(frequency),
      mIndexToRadiansMultiplier(static_cast<float>(2.0 * M_PI / (mSampleRate / mFrequency))), mCudaDevice(cudaDevice),
      mCudaStream(cudaStream), mPhi(0.0f), mAlignment(32) {}

size_t CosineSource::getOutputDataSize(size_t port) {
  if (port > 0) {
    throw runtime_error("Port [" + to_string(port) + "] is out of range");
  }

  return 16 * 1024 * mAlignment * sizeof(cuComplex);
}

size_t CosineSource::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(cuComplex); }

void CosineSource::readOutput(const vector<shared_ptr<Buffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestoreDevice(mCudaDevice);

  const auto& output = portOutputs[0];

  const size_t maxOutputNumValues = output->remaining() / sizeof(cuComplex);
  const size_t blockCount = maxOutputNumValues / mAlignment;

  const dim3 blocks(blockCount);
  const dim3 threads(mAlignment);

  k_ComplexCosine<<<blocks, threads, 0, mCudaStream>>>(mIndexToRadiansMultiplier, mPhi, output->writePtr<cuComplex>());

  const size_t outputNumValues = blockCount * mAlignment;
  mPhi += static_cast<float>(outputNumValues) * mIndexToRadiansMultiplier;
  mPhi = fmod(mPhi, 2.0f * M_PIf);

  const size_t outputNumBytes = outputNumValues * sizeof(cuComplex);

  output->increaseEndOffset(outputNumBytes);
}
