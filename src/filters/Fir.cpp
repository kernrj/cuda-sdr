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
#include "GSLog.h"
#include "util/CudaDevicePushPop.h"

using namespace std;

const size_t Fir::mAlignment = 32;

static size_t getSampleSize(SampleType sampleType) {
  switch (sampleType) {
    case SampleType_Float:
      return sizeof(float);
    case SampleType_FloatComplex:
      return sizeof(cuComplex);
    case SampleType_Int8Complex:
      return sizeof(int8_t);
    default:
      GS_FAIL("Unknown SampleType [" << sampleType << "]");
  }
}

Result<Filter> Fir::create(
    SampleType tapType,
    SampleType elementType,
    size_t decimation,
    const float* taps,
    size_t tapCount,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories) noexcept {
  Ref<IAllocator> allocator;
  Ref<IBufferCopier> deviceToDeviceBufferCopier;
  Ref<IBufferCopier> hostToDeviceBufferCopier;
  Ref<IRelocatableResizableBufferFactory> relocatableBufferFactory;
  Ref<IMemSet> memSet;

  UNWRAP_OR_FWD_RESULT(
      allocator,
      factories->getCudaAllocatorFactory()->createCudaAllocator(cudaDevice, cudaStream, mAlignment, false));
  UNWRAP_OR_FWD_RESULT(
      deviceToDeviceBufferCopier,
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyDeviceToDevice));
  UNWRAP_OR_FWD_RESULT(
      hostToDeviceBufferCopier,
      factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, cudaMemcpyHostToDevice));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream));
  UNWRAP_OR_FWD_RESULT(
      relocatableBufferFactory,
      factories->createRelocatableResizableBufferFactory(allocator.get(), deviceToDeviceBufferCopier.get()));

  Fir* fir = new (nothrow)
      Fir(tapType,
          elementType,
          decimation,
          cudaDevice,
          cudaStream,
          allocator.get(),
          hostToDeviceBufferCopier.get(),
          relocatableBufferFactory.get(),
          factories->getBufferSliceFactory(),
          memSet.get());
  NON_NULL_OR_RET(fir);

  Status status = fir->setTaps(taps, tapCount);
  if (status != Status_Success) {
    fir->unref();
    return errResult<Filter>(status);
  }

  return makeRefResultNonNull<Filter>(fir);
}

Fir::Fir(
    SampleType tapType,
    SampleType elementType,
    size_t decimation,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IAllocator* allocator,
    IBufferCopier* hostToDeviceBufferCopier,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, memSet),
      mTapType(tapType),
      mElementType(elementType),
      mAllocator(allocator),
      mHostToDeviceBufferCopier(hostToDeviceBufferCopier),
      mDecimation(max(static_cast<size_t>(1), decimation)),
      mTapCount(0),
      mCudaStream(cudaStream),
      mCudaDevice(cudaDevice),
      mElementSize(getSampleSize(elementType)) {}

Status Fir::setTaps(const float* tapsReversed, size_t tapCount) noexcept {
  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCudaDevice);

  if (mTapCount < tapCount) {
    UNWRAP_OR_FWD_STATUS(mTaps, mAllocator->allocate(tapCount * sizeof(float)));
  }

  const size_t bufferLength = tapCount * getSampleSize(mTapType);
  FWD_IF_ERR(mHostToDeviceBufferCopier->copy(mTaps->data(), tapsReversed, bufferLength));

  mTapCount = static_cast<int32_t>(tapCount);

  return Status_Success;
}

size_t Fir::getNumOutputElements() const noexcept {
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

size_t Fir::getNumInputElements() const noexcept {
  Ref<const IBuffer> inputBuffer;
  if (!inputPortsInitialized()) {
    return 0;
  }

  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);
  return inputBuffer->range()->used() / mElementSize;
}

size_t Fir::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return getNumOutputElements() * mElementSize;
}

size_t Fir::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);

  return mAlignment * mElementSize;
}

Status Fir::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "Must have one output port");

  IBuffer* outputBuffer = portOutputBuffers[0];
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));

  if (inputBuffer->range()->used() < mTapCount) {
    return Status_Success;
  }

  const size_t maxOutputsInOutputBuffer = outputBuffer->range()->remaining() / mElementSize;
  const size_t availableNumOutputs = getNumOutputElements();
  const size_t numOutputs = min(availableNumOutputs, maxOutputsInOutputBuffer);

  if (numOutputs == 0) {
    return Status_Success;
  }

  if (mTapType == SampleType_Float && mElementType == SampleType_Float) {
    SAFE_CUDA_OR_RET_STATUS(gsdrFirFF(
        mDecimation,
        mTaps->as<float>(),
        mTapCount,
        inputBuffer->readPtr<float>(),
        outputBuffer->writePtr<float>(),
        numOutputs,
        mCudaDevice,
        mCudaStream));
  } else if (mTapType == SampleType_Float && mElementType == SampleType_FloatComplex) {
    SAFE_CUDA_OR_RET_STATUS(gsdrFirFC(
        mDecimation,
        mTaps->as<float>(),
        mTapCount,
        inputBuffer->readPtr<cuComplex>(),
        outputBuffer->writePtr<cuComplex>(),
        numOutputs,
        mCudaDevice,
        mCudaStream));
  } else if (mTapType == SampleType_FloatComplex && mElementType == SampleType_FloatComplex) {
    SAFE_CUDA_OR_RET_STATUS(gsdrFirCC(
        mDecimation,
        mTaps->as<cuComplex>(),
        mTapCount,
        inputBuffer->readPtr<cuComplex>(),
        outputBuffer->writePtr<cuComplex>(),
        numOutputs,
        mCudaDevice,
        mCudaStream));
  } else if (mTapType == SampleType_FloatComplex && mElementType == SampleType_Float) {
    SAFE_CUDA_OR_RET_STATUS(gsdrFirCF(
        mDecimation,
        mTaps->as<cuComplex>(),
        mTapCount,
        inputBuffer->readPtr<float>(),
        outputBuffer->writePtr<cuComplex>(),
        numOutputs,
        mCudaDevice,
        mCudaStream));
  }

  const size_t numOutputBytes = numOutputs * mElementSize;
  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(numOutputBytes));

  const size_t numInputElementsToDiscard = numOutputs * mDecimation;
  const size_t numInputBytesToDiscard = numInputElementsToDiscard * mElementSize;
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, numInputBytesToDiscard));

  return Status_Success;

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

size_t Fir::preferredInputBufferSize(size_t port) noexcept { return 1 << 20; }
