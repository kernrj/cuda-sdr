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

Result<Filter> Magnitude::create(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream));

  return makeRefResultNonNull<Filter>(new (nothrow) Magnitude(
      cudaDevice,
      cudaStream,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get()));
}

Magnitude::Magnitude(
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, memSet),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t Magnitude::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return getAvailableNumInputElements() * sizeof(float);
}

size_t Magnitude::getAvailableNumInputElements() const noexcept {
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);

  return inputBuffer->range()->used() / sizeof(cuComplex);
}

size_t Magnitude::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(float);
}

Status Magnitude::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount > 0, "One output port is required");

  CUDA_DEV_PUSH_POP_OR_RET_STATUS(mCudaDevice);

  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputBuffers[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(float);
  const size_t numElements = min(numInputElements, maxNumOutputElements);

  SAFE_CUDA_OR_RET_STATUS(gsdrMagnitude(
      inputBuffer->readPtr<cuComplex>(),
      outputBuffer->writePtr<float>(),
      numElements,
      mCudaDevice,
      mCudaStream));

  const size_t readNumBytes = numElements * sizeof(cuComplex);
  const size_t writtenNumBytes = numElements * sizeof(float);
  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, readNumBytes));

  return Status_Success;
}

size_t Magnitude::preferredInputBufferSize(size_t port) noexcept { return 1 << 20; }
