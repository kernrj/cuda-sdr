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

Result<Filter> MultiplyCcc::create(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream));
  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false));

  return makeRefResultNonNull<Filter>(new (nothrow) MultiplyCcc(
      cudaDevice,
      cudaStream,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get()));
}

MultiplyCcc::MultiplyCcc(
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 2, memSet),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t MultiplyCcc::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(port == 0, 0, "Port [%zu] is out of range", port);
  return getAvailableNumInputElements() * sizeof(cuComplex);
}

size_t MultiplyCcc::getAvailableNumInputElements() const {
  if (!inputPortsInitialized()) {
    return 0;
  }

  Ref<const IBuffer> inputBuffer0;
  UNWRAP_OR_RETURN(inputBuffer0, getPortInputBuffer(0), 0);
  Ref<const IBuffer> inputBuffer1;
  UNWRAP_OR_RETURN(inputBuffer1, getPortInputBuffer(1), 0);

  const size_t port0NumElements = inputBuffer0->range()->used() / sizeof(cuComplex);
  const size_t port1NumElements = inputBuffer1->range()->used() / sizeof(cuComplex);
  const size_t numInputElements = min(port0NumElements, port1NumElements);

  return numInputElements;
}

size_t MultiplyCcc::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(cuComplex);
}

size_t MultiplyCcc::preferredInputBufferSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(port <= 1, 0, "Output port [%zu] is out of range", port);

  if (!inputPortsInitialized()) {
    return 8192 * sizeof(cuComplex);
  }

  Ref<const IBuffer> inputBuffer0;
  UNWRAP_OR_RETURN(inputBuffer0, getPortInputBuffer(0), 0);
  Ref<const IBuffer> inputBuffer1;
  UNWRAP_OR_RETURN(inputBuffer1, getPortInputBuffer(1), 0);

  const auto range0 = inputBuffer0->range();
  const auto range1 = inputBuffer1->range();

  constexpr size_t maxSize = 100 << 20;

  if (range0->used() == 0 && range1->used() == 0) {
    return 8192 * sizeof(cuComplex);
  } else if (range0->used() >= range1->used()) {
    switch (port) {
      case 0:
        return 0;
      case 1:
        return min(maxSize, range0->used() - range1->used());
      default:
        GS_FAIL("Cannot get preferred input buffer size. Unknown port [" << port << "]");
    }
  } else {
    switch (port) {
      case 0:
        return min(maxSize, range1->used() - range0->used());
      case 1:
        return 0;
      default:
        GS_FAIL("Cannot get preferred input buffer size. Unknown port [" << port << "]");
    }
  }
}

Status MultiplyCcc::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");
  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCudaDevice);

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputBuffers[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(cuComplex);
  const size_t numElements = min(numInputElements, maxNumOutputElements);
  Ref<IBuffer> inputBuffer0;
  Ref<IBuffer> inputBuffer1;

  UNWRAP_OR_FWD_STATUS(inputBuffer0, getPortInputBuffer(0));
  UNWRAP_OR_FWD_STATUS(inputBuffer1, getPortInputBuffer(1));

  gsdrMultiplyCC(
      inputBuffer0->readPtr<cuComplex>(),
      inputBuffer1->readPtr<cuComplex>(),
      portOutputBuffers[0]->writePtr<cuComplex>(),
      numElements,
      mCudaDevice,
      mCudaStream);

  const size_t writtenNumBytes = numElements * sizeof(cuComplex);
  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(1, writtenNumBytes));

  return Status_Success;
}
