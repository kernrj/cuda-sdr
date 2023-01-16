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

#include "Fir.h"

#include <cuComplex.h>
#include <gsdr/gsdr.h>

#include <stdexcept>

#include "Factories.h"
#include "GSErrors.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

const size_t Fir::mAlignment = 32;

static size_t getElementSize(FirType firType) {
  switch (firType) {
    case FirType_FloatTapsFloatSignal:
      return sizeof(float);

    case FirType_FloatTapsComplexFloatSignal:
      return sizeof(cuComplex);

    default:
      THROW("Unknown FirType [" << firType << "]");
  }
}

Fir::Fir(
    FirType firType,
    size_t decimation,
    const float* taps,
    size_t tapCount,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mFirType(firType),
      mFactories(factories),
      mCudaAllocator(
          mFactories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, mAlignment, false)),
      mDecimation(max(static_cast<size_t>(1), decimation)),
      mTapCount(0),
      mCudaStream(cudaStream),
      mCudaDevice(cudaDevice),
      mElementSize(getElementSize(firType)) {
  setTaps(taps, tapCount);
}

void Fir::setTaps(const float* tapsReversed, size_t tapCount) {
  CudaDevicePushPop setAndRestore(mCudaDevice);

  if (mTapCount < tapCount) {
    auto data = mCudaAllocator->allocate(tapCount * sizeof(float), nullptr);
    mTaps = shared_ptr<float>(data, reinterpret_cast<float*>(data.get()));
  }

  SAFE_CUDA(cudaMemcpyAsync(mTaps.get(), tapsReversed, tapCount * sizeof(float), cudaMemcpyHostToDevice, mCudaStream));

  mTapCount = static_cast<int32_t>(tapCount);
}

size_t Fir::getNumOutputElements() const {
  /*
   * decimation = 2
   * tap count = 7
   * For 1 output, 8 inputs are needed
   * For 2 outputs, 10 inputs are needed
   * 3: 12
   * 4: 14
   *
   * i i i i i i i i i i i i i i i i i i i i i
   * - - - - - - -
   *   s s s s s s s
   *     - - - - - - -
   *       s s s s s s s
   *         - - - - - - -
   *           s s s s s s s
   *             - - - - - - -
   *               s s s s s s s
   *
   * decimation = 3
   * tap count = 5
   * i i i i i i i i i i i i i i i i i i i i i
   * - - - - -
   *   s s s s s
   *     s s s s s
   *       - - - - -
   *         s s s s s
   *           s s s s s
   *             - - - - -
   *               s s s s s
   *                 s s s s s
   *
   * 1: 7
   * 2: 10
   * 3: 13
   *
   * tapCount - decimation + 1
   */

  const size_t numInputElements = getNumInputElements();
  const size_t numInputsElementsForFirstOutput = mTapCount - mDecimation + 1;
  if (numInputElements < numInputsElementsForFirstOutput) {
    return 0;
  }

  return (numInputElements - (mTapCount - 1)) / mDecimation;
}

size_t Fir::getNumInputElements() const { return getPortInputBuffer(0)->range()->used() / mElementSize; }

size_t Fir::getOutputDataSize(size_t port) { return getNumOutputElements() * mElementSize; }

size_t Fir::getOutputSizeAlignment(size_t port) {
  if (port != 0) {
    THROW("Output port [" << port << "] is out of range");
  }

  return mAlignment * mElementSize;
}

void Fir::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    THROW("Must have one output port");
  }

  if (getPortInputBuffer(0)->range()->used() < mTapCount) {
    return;
  }

  CudaDevicePushPop setAndRestore(mCudaDevice);

  const auto& outputBuffer = portOutputs[0];

  const size_t maxOutputsInOutputBuffer = outputBuffer->range()->remaining() / mElementSize;
  const size_t availableNumOutputs = getNumOutputElements();
  const size_t numOutputs = min(availableNumOutputs, maxOutputsInOutputBuffer);
  const size_t numBlocks = (numOutputs + 31) / 32;

  if (numOutputs == 0) {
    return;
  }

  switch (mFirType) {
    case FirType_FloatTapsFloatSignal:
      gsdrFirFF(
          mDecimation,
          mTaps.get(),
          mTapCount,
          getPortInputBuffer(0)->readPtr<float>(),
          outputBuffer->writePtr<float>(),
          numOutputs,
          mCudaDevice,
          mCudaStream);
      break;
    case FirType_FloatTapsComplexFloatSignal:
      gsdrFirFC(
          mDecimation,
          mTaps.get(),
          mTapCount,
          getPortInputBuffer(0)->readPtr<cuComplex>(),
          outputBuffer->writePtr<cuComplex>(),
          numOutputs,
          mCudaDevice,
          mCudaStream);
      break;
  }

  const size_t numOutputBytes = numOutputs * mElementSize;
  outputBuffer->range()->increaseEndOffset(numOutputBytes);

  const size_t numInputElementsToDiscard = numOutputs * mDecimation;
  const size_t numInputBytesToDiscard = numInputElementsToDiscard * mElementSize;
  consumeInputBytesAndMoveUsedToStart(0, numInputBytesToDiscard);

  /*
   * dec = 3
   * tap size = 4
   *
   * i = available input element
   * . = future input element
   *
   * i i i i i i i i . . . . . . . . . . . . . .
   * o - - -     o - - -     o - - -     o - - -
   *       o - - -     o - - -     o - - -
   *
   * 2 outputs available, need to keep last two elements.
   * availableNumInputs - generatedNumOutputs * decimation
   * 8 - 2 * 3 = 2
   * Or, discard generatedNumOutputs * decimation inputs.
   * have 8, but discard 6
   *
   * i i i i i i i i i i i i i . . . . . . . . .
   * o - - - - -
   *       o - - - - -
   *             o - - - - -
   *                   o - - - - -
   *                         o - - - - -
   *                               o - - - - -
   *                                     o - - - - -
   *
   * 13 - 3 * 3 = 4
   * Or, have 13, discard 9
   */
}
