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

#include "AddConstToVectorLength.h"

#include <cuComplex.h>
#include <cuda.h>
#include <gsdr/gsdr.h>

#include "CudaErrors.h"

using namespace std;

const size_t AddConstToVectorLength::mAlignment = 32;

AddConstToVectorLength::AddConstToVectorLength(
    float addValueToMagnitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mAddValueToMagnitude(addValueToMagnitude),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t AddConstToVectorLength::getOutputDataSize(size_t port) {
  return getAvailableNumInputElements() * sizeof(cuComplex);
}

size_t AddConstToVectorLength::getAvailableNumInputElements() const {
  return getPortInputBuffer(0)->range()->used() / sizeof(cuComplex);
}

size_t AddConstToVectorLength::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(cuComplex); }

void AddConstToVectorLength::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(cuComplex);
  const size_t processNumElements = min(maxNumOutputElements, numInputElements);

  SAFE_CUDA(gsdrAddToMagnitude(
      getPortInputBuffer(0)->readPtr<cuComplex>(),
      mAddValueToMagnitude,
      outputBuffer->writePtr<cuComplex>(),
      processNumElements,
      mCudaDevice,
      mCudaStream));

  const size_t writtenNumBytes = processNumElements * sizeof(cuComplex);

  outputBuffer->range()->increaseEndOffset(writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes);
}
