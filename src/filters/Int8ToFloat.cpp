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

#include "Int8ToFloat.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <gsdr/conversion.h>

#include <cstdint>

#include "CudaErrors.h"
#include "Factories.h"

using namespace std;

const size_t Int8ToFloat::mAlignment = 32;

Int8ToFloat::Int8ToFloat(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t Int8ToFloat::getOutputDataSize(size_t port) {
  if (port != 0) {
    throw invalid_argument("Port [" + to_string(port) + "] is out of range");
  }

  return getPortInputBuffer(0)->range()->used();
}

size_t Int8ToFloat::getOutputSizeAlignment(size_t port) { return mAlignment; }

void Int8ToFloat::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw invalid_argument("One output port is required");
  }

  const auto inputBuffer = getPortInputBuffer(0);
  const auto& outputBuffer = portOutputs[0];

  const size_t elementCount = min(inputBuffer->range()->used(), outputBuffer->range()->remaining() / sizeof(float));

  SAFE_CUDA(gsdrInt8ToNormFloat(
      getPortInputBuffer(0)->readPtr<int8_t>(),
      outputBuffer->writePtr<float>(),
      elementCount,
      mCudaDevice,
      mCudaStream));

  outputBuffer->range()->increaseEndOffset(elementCount * sizeof(float));
  consumeInputBytesAndMoveUsedToStart(0, elementCount * sizeof(int8_t));
}
