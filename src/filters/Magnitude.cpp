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

#include "Magnitude.h"

#include <cuComplex.h>
#include <gsdr/gsdr.h>

#include "util/CudaDevicePushPop.h"

using namespace std;

const size_t Magnitude::mAlignment = 32;

Magnitude::Magnitude(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t Magnitude::getOutputDataSize(size_t port) { return getAvailableNumInputElements() * sizeof(float); }

size_t Magnitude::getAvailableNumInputElements() const {
  return getPortInputBuffer(0)->range()->used() / sizeof(cuComplex);
}

size_t Magnitude::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(float); }

void Magnitude::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(float);
  const size_t numElements = min(numInputElements, maxNumOutputElements);

  SAFE_CUDA(gsdrMagnitude(
      getPortInputBuffer(0)->readPtr<cuComplex>(),
      outputBuffer->writePtr<float>(),
      numElements,
      mCudaDevice,
      mCudaStream));

  const size_t readNumBytes = numElements * sizeof(cuComplex);
  const size_t writtenNumBytes = numElements * sizeof(float);
  outputBuffer->range()->increaseEndOffset(writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(0, readNumBytes);
}
