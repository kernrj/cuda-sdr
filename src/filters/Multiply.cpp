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

#include "Multiply.h"

#include <cuComplex.h>
#include <cuda.h>
#include <gsdr/gsdr.h>

#include "Factories.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

const size_t MultiplyCcc::mAlignment = 32;

MultiplyCcc::MultiplyCcc(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        2,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t MultiplyCcc::getOutputDataSize(size_t port) { return getAvailableNumInputElements() * sizeof(cuComplex); }

size_t MultiplyCcc::getAvailableNumInputElements() const {
  const size_t port0NumElements = getPortInputBuffer(0)->range()->used() / sizeof(cuComplex);
  const size_t port1NumElements = getPortInputBuffer(1)->range()->used() / sizeof(cuComplex);
  const size_t numInputElements = min(port0NumElements, port1NumElements);

  return numInputElements;
}

size_t MultiplyCcc::getOutputSizeAlignment(size_t port) { return mAlignment * sizeof(cuComplex); }

void MultiplyCcc::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputs[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(cuComplex);
  const size_t numElements = min(numInputElements, maxNumOutputElements);

  gsdrMultiplyCC(
      getPortInputBuffer(0)->readPtr<cuComplex>(),
      getPortInputBuffer(1)->readPtr<cuComplex>(),
      portOutputs[0]->writePtr<cuComplex>(),
      numElements,
      mCudaDevice,
      mCudaStream);

  const size_t writtenNumBytes = numElements * sizeof(cuComplex);
  outputBuffer->range()->increaseEndOffset(writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes);
  consumeInputBytesAndMoveUsedToStart(1, writtenNumBytes);
}
