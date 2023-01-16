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

#include "AddConst.h"

#include <gsdr/gsdr.h>

#include "CudaErrors.h"

using namespace std;

const size_t AddConst::mAlignment = 32;

AddConst::AddConst(float addConst, int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mAddConst(addConst),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t AddConst::getOutputDataSize(size_t port) { return getAvailableNumInputElements() * sizeof(float); }
size_t AddConst::getAvailableNumInputElements() const { return getPortInputBuffer(0)->range()->used() / sizeof(float); }
size_t AddConst::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(float); }

void AddConst::readOutput(const std::vector<std::shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  const auto inputBuffer = getPortInputBuffer(0);
  const auto& outputBuffer = portOutputs[0];

  const size_t numInputElements = getAvailableNumInputElements();
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(float);
  const size_t processNumElements = min(numInputElements, maxNumOutputElements);

  SAFE_CUDA(gsdrAddConstFF(
      inputBuffer->readPtr<float>(),
      mAddConst,
      outputBuffer->writePtr<float>(),
      processNumElements,
      mCudaDevice,
      mCudaStream));

  const size_t writtenNumBytes = processNumElements * sizeof(float);
  outputBuffer->range()->increaseEndOffset(writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes);
}
