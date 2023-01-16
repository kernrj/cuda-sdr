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

#include "QuadDemod.h"

#include <cuComplex.h>
#include <gsdr/gsdr.h>

#include "CudaErrors.h"
#include "Factories.h"

using namespace std;

using InputType = cuComplex;
using OutputType = float;

QuadDemod::QuadDemod(float gain, int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, 32, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mGain(gain) {}

size_t QuadDemod::getOutputSizeAlignment(size_t port) { return 32 * sizeof(OutputType); }

size_t QuadDemod::getOutputDataSize(size_t port) {
  auto buffer = getPortInputBuffer(port);
  const size_t numInputElements = buffer->range()->used() / sizeof(InputType);
  const size_t numOutputElements = numInputElements - 1;

  return numOutputElements * sizeof(OutputType);
}

void QuadDemod::readOutput(const vector<std::shared_ptr<IBuffer>>& portOutputs) {
  const shared_ptr<IBuffer> inputBuffer = getPortInputBuffer(0);
  const shared_ptr<IBuffer>& outputBuffer = portOutputs[0];

  const size_t availableNumInputElements = inputBuffer->range()->used() / sizeof(InputType);
  const size_t maxNumOutputElementsInBuffer = outputBuffer->range()->remaining() / sizeof(OutputType);
  const size_t maxNumOutputElementsAvailable = availableNumInputElements - 1;
  const size_t numOutputElements = min(maxNumOutputElementsAvailable, maxNumOutputElementsInBuffer);

  CHECK_CUDA("Before quadDemod");
  gsdrQuadFrequencyDemod(
      inputBuffer->readPtr<InputType>(),
      outputBuffer->writePtr<OutputType>(),
      mGain,
      numOutputElements,
      mCudaDevice,
      mCudaStream);
  CHECK_CUDA("After quadDemod");

  const size_t writtenNumBytes = numOutputElements * sizeof(OutputType);
  const size_t discardNumInputBytes = numOutputElements * sizeof(InputType);

  outputBuffer->range()->increaseEndOffset(writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(0, discardNumInputBytes);
}
